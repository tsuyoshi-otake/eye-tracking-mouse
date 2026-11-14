//! What to do with each camera frame, free of any system call.
//!
//! The engine owns what is known about the person (their face shape, their
//! calibration, the tuning) and the one thing going on at a time: pointing, or a
//! calibration run (of the gaze, its check, or the head). The pipeline feeds it
//! frames and carries out what it answers; the engine never moves the pointer itself.
//!
//! Contract:
//! - During a calibration run no pointer action and no sound is answered.
//! - What is remembered from earlier frames (a blink in progress, the cursor's
//!   fine position) is dropped when frames stop arriving for longer than
//!   `MAX_FRAME_GAP`, when `interrupt` is called and when a calibration run
//!   starts, so a break can never end in a click.
//! - A calibration run ends exactly once: with `Output::finished` in the frame
//!   it ends, or with `cancel_calibration` returning true.
//! - A head calibration that succeeds puts its `HeadGain` into the tuning
//!   before the frame that reports it returns: the next frame points with it.

use crate::control::calibration::{Calibration, CalibrationSession, Failure, Step, View};
use crate::control::controller::{Activity, Controller, Frame, Makeup, PointingMode, Sound, Tuning};
use crate::control::head_calibration::{HeadCalibration, HeadStep};
use crate::control::head_pointer::HeadGain;
use crate::geometry::RectI;
use crate::vision::landmarks::FaceMesh;
use crate::vision::pose::{ReferenceBuilder, ReferenceShape};
use crate::vision::{self, FaceMeasurement};

/// Frames averaged into a face shape while there is no calibration to take it from.
const REFERENCE_FRAMES: usize = 20;
/// Frames further apart than this many seconds do not belong together.
const MAX_FRAME_GAP: f64 = 0.5;

/// One camera frame and the desktop as it is now.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    pub time: f64,
    /// The face found in the frame, if any.
    pub mesh: Option<&'a FaceMesh>,
    /// Size of the camera image the mesh was found in.
    pub frame_size: (u32, u32),
    /// Where the cursor is; `None` while the desktop cannot be reached.
    pub cursor: Option<(i32, i32)>,
    /// The monitors, in desktop pixels.
    pub screens: &'a [RectI],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerAction {
    Move(i32, i32),
    /// Put the cursor on this pixel and left-click.
    Click(i32, i32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Pointing(Activity),
    Calibrating(View),
}

/// What a frame of pointing was decided from: the face as measured, and what the cursor's move was made of.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trace {
    pub face: FaceMeasurement,
    pub makeup: Makeup,
}

/// What a calibration run that succeeded leaves behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    /// A calibration of the gaze, made or checked: its typical error in pixels.
    Gaze { error_px: f32 },
    /// The head pointer's speeds and steadiness for this person, in use from the next frame.
    Head(HeadGain),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Output {
    /// Set in a frame of pointing in which a face was measured. Nothing acts on it.
    pub trace: Option<Trace>,
    pub pointer: Option<PointerAction>,
    pub sound: Option<Sound>,
    pub mode: Mode,
    /// Set in the frame a calibration run ends: what it leaves behind, or why it failed.
    pub finished: Option<Result<Outcome, Failure>>,
}

/// The run going on, if any.
enum Session {
    Gaze(CalibrationSession),
    Head(HeadCalibration),
}

/// Why a calibration cannot be used for gaze right now. The face shape in it stays good.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GazeProblem {
    /// The screen that was calibrated is not among the monitors any more.
    ScreenChanged,
    /// The camera delivers a different image size than it did at calibration.
    CameraChanged,
}

pub fn gaze_problem(calibration: &Calibration, frame_size: (u32, u32), screens: &[RectI]) -> Option<GazeProblem> {
    if !screens.contains(&calibration.area) {
        Some(GazeProblem::ScreenChanged)
    } else if calibration.gaze.neutral().frame_size != frame_size {
        Some(GazeProblem::CameraChanged)
    } else {
        None
    }
}

pub struct Engine {
    tuning: Tuning,
    calibration: Option<Calibration>,
    /// The face shape picked up on the fly while there is no calibration.
    learned: Option<ReferenceShape>,
    builder: ReferenceBuilder,
    controller: Controller,
    session: Option<Session>,
    last_frame: f64,
}

impl Engine {
    pub fn new(tuning: Tuning, calibration: Option<Calibration>) -> Engine {
        Engine {
            tuning,
            calibration,
            learned: None,
            builder: ReferenceBuilder::default(),
            controller: Controller::default(),
            session: None,
            last_frame: f64::NEG_INFINITY,
        }
    }

    pub fn calibration(&self) -> Option<&Calibration> {
        self.calibration.as_ref()
    }

    /// Whether anything reads the gaze estimate: the eyes carry the cursor in eye and hybrid mode, never in head mode.
    pub fn uses_gaze(&self) -> bool {
        self.tuning.pointing != PointingMode::Head
    }

    pub fn pointing(&self) -> PointingMode {
        self.tuning.pointing
    }

    /// For tests: what the program itself needs to know it reads from `Output::finished`.
    #[cfg(test)]
    pub fn head_gain(&self) -> HeadGain {
        self.tuning.head_gain
    }

    /// Changes what moves the cursor. Pointing starts afresh, as after any break.
    pub fn set_pointing(&mut self, pointing: PointingMode) {
        if self.tuning.pointing != pointing {
            self.tuning.pointing = pointing;
            self.interrupt();
        }
    }

    /// For tests: what the program itself needs to know it reads from `Output::mode`.
    #[cfg(test)]
    pub fn is_calibrating(&self) -> bool {
        self.session.is_some()
    }

    /// Starts a calibration run on the screen `area`; one already under way is abandoned.
    pub fn start_calibration(&mut self, area: RectI, frame_size: (u32, u32), time: f64) {
        // Pointing resumes afresh whichever way the run ends.
        self.interrupt();
        self.session = Some(Session::Gaze(CalibrationSession::new(area, frame_size, time)));
    }

    /// Starts a head calibration on the screen `area`; a run already under way is abandoned.
    pub fn start_head_calibration(&mut self, area: RectI, time: f64) {
        self.interrupt();
        self.session = Some(Session::Head(HeadCalibration::new(area, time)));
    }

    /// Starts a check of the calibration there is, which adds to it. The calibration must be
    /// for the screen `area` and for camera frames of `frame_size`; false, and nothing is
    /// started, when there is none or it is for another screen or camera.
    pub fn start_check(&mut self, area: RectI, frame_size: (u32, u32), time: f64) -> bool {
        let fits = |c: &&Calibration| c.area == area && c.gaze.neutral().frame_size == frame_size;
        let Some(calibration) = self.calibration.as_ref().filter(fits).cloned() else {
            return false;
        };
        self.interrupt();
        // The calibration stays in use as it is until the check hands back the one that learned from it.
        self.session = Some(Session::Gaze(CalibrationSession::check(calibration, time)));
        true
    }

    /// Abandons the calibration run without a result. False if none was under way.
    pub fn cancel_calibration(&mut self) -> bool {
        self.session.take().is_some()
    }

    /// Tells the engine that the frames to come do not continue the ones before.
    pub fn interrupt(&mut self) {
        self.controller.restart();
    }

    pub fn frame(&mut self, input: &Input) -> Output {
        if input.time - self.last_frame > MAX_FRAME_GAP {
            self.interrupt();
        }
        self.last_frame = input.time;
        match self.session.take() {
            Some(session) => self.calibrate(session, input),
            None => self.point(input),
        }
    }

    fn calibrate(&mut self, session: Session, input: &Input) -> Output {
        let quiet = |mode, finished| Output { trace: None, pointer: None, sound: None, mode, finished };
        let ended = |outcome| quiet(Mode::Pointing(Activity::default()), Some(outcome));
        match session {
            Session::Gaze(session) => match session.step(input.time, input.mesh) {
                Step::Running(session) => {
                    let view = session.view(input.time);
                    self.session = Some(Session::Gaze(session));
                    quiet(Mode::Calibrating(view), None)
                }
                Step::Done(calibration) => {
                    let error_px = calibration.error_px;
                    self.calibration = Some(calibration);
                    ended(Ok(Outcome::Gaze { error_px }))
                }
                Step::Failed(failure) => ended(Err(failure)),
            },
            Session::Head(session) => {
                // Measured against the person's face shape, as in pointing.
                let face = input.mesh.and_then(|mesh| self.measure(mesh));
                match session.step(input.time, face.as_ref().map(|face| &face.pose)) {
                    HeadStep::Running(session) => {
                        let view = session.view(input.time);
                        self.session = Some(Session::Head(session));
                        quiet(Mode::Calibrating(view), None)
                    }
                    HeadStep::Done(gain) => {
                        self.tuning.head_gain = gain;
                        ended(Ok(Outcome::Head(gain)))
                    }
                    HeadStep::Failed(failure) => ended(Err(failure)),
                }
            }
        }
    }

    fn point(&mut self, input: &Input) -> Output {
        let face = input.mesh.and_then(|mesh| self.measure(mesh));
        let calibration =
            self.calibration.as_ref().filter(|c| gaze_problem(c, input.frame_size, input.screens).is_none());
        let frame =
            Frame { time: input.time, face: face.as_ref(), cursor: input.cursor, screens: input.screens, calibration };
        let decision = self.controller.step(&frame, &self.tuning);
        let pointer = match (decision.click_at, decision.move_to) {
            (Some((x, y)), _) => Some(PointerAction::Click(x, y)),
            (None, Some((x, y))) => Some(PointerAction::Move(x, y)),
            (None, None) => None,
        };
        let trace = face.map(|face| Trace { face, makeup: decision.makeup });
        Output { trace, pointer, sound: decision.sound, mode: Mode::Pointing(decision.activity), finished: None }
    }

    /// Measures the face against the person's shape: the calibrated one, or
    /// one averaged from the first frames that show a face.
    fn measure(&mut self, mesh: &FaceMesh) -> Option<FaceMeasurement> {
        if let Some(calibration) = &self.calibration {
            return vision::measure(&calibration.reference, mesh);
        }
        if self.learned.is_none() {
            self.builder.add(&mesh.points);
            if self.builder.frames() >= REFERENCE_FRAMES {
                // A shape that cannot be built is collected again from scratch.
                self.learned = self.builder.build();
                self.builder = ReferenceBuilder::default();
            }
        }
        vision::measure(self.learned.as_ref()?, mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::calibration::Phase;
    use crate::control::calibration::simulation::{calibrated, following, sweeping};
    use crate::control::gaze::simulation::{FRAME_SIZE, Person, Posture, SCREEN};
    use crate::geometry::Vec2;

    const MIDDLE: Vec2 = Vec2::new(1280.0, 540.0);

    /// A person in front of the camera and a cursor that goes where the engine sends it.
    struct Desk {
        engine: Engine,
        person: Person,
        screens: Vec<RectI>,
        time: f64,
        cursor: (i32, i32),
        clicks: Vec<(i32, i32)>,
        sounds: Vec<Sound>,
    }

    impl Desk {
        fn new(calibration: Option<Calibration>) -> Desk {
            Desk {
                engine: Engine::new(Tuning::default(), calibration),
                person: Person::new(0.5),
                screens: vec![SCREEN],
                time: 0.0,
                cursor: (1280, 540),
                clicks: Vec::new(),
                sounds: Vec::new(),
            }
        }

        /// The next frame, a thirtieth of a second after the last. `look` is
        /// the head's posture, where the eyes point and how open they are; `None` is an empty chair.
        fn frame(&mut self, look: Option<(&Posture, Vec2, f32)>) -> Output {
            self.time += 1.0 / 30.0;
            let mesh = look.map(|(posture, gaze, openness)| self.person.look(self.time, posture, gaze, openness));
            let output = self.engine.frame(&Input {
                time: self.time,
                mesh: mesh.as_ref(),
                frame_size: FRAME_SIZE,
                cursor: Some(self.cursor),
                screens: &self.screens,
            });
            match output.pointer {
                Some(PointerAction::Move(x, y)) => self.cursor = (x, y),
                Some(PointerAction::Click(x, y)) => {
                    self.cursor = (x, y);
                    self.clicks.push((x, y));
                }
                None => {}
            }
            self.sounds.extend(output.sound);
            output
        }

        /// Frames for `seconds` with the head still, looking at `gaze`.
        fn gaze_at(&mut self, seconds: f64, gaze: Vec2, openness: f32) {
            for _ in 0..(seconds * 30.0).round() as u32 {
                self.frame(Some((&following(gaze), gaze, openness)));
            }
        }

        /// Looks at `gaze` with the eyes alone; the head stays facing the camera.
        fn eyes_at(&mut self, seconds: f64, gaze: Vec2) {
            for _ in 0..(seconds * 30.0).round() as u32 {
                self.frame(Some((&Posture::default(), gaze, 1.0)));
            }
        }

        /// Goes through a calibration run as an attentive person would. Returns how it ended.
        fn calibrate(&mut self) -> Result<f32, Failure> {
            self.engine.start_calibration(SCREEN, FRAME_SIZE, self.time);
            self.through_the_run()
        }

        /// Goes through the run the engine has started as an attentive person would.
        fn through_the_run(&mut self) -> Result<f32, Failure> {
            self.through_the_run_to_its_end().map(|outcome| match outcome {
                Outcome::Gaze { error_px } => error_px,
                Outcome::Head(gain) => panic!("a gaze run ended with {gain:?}"),
            })
        }

        fn through_the_run_to_its_end(&mut self) -> Result<Outcome, Failure> {
            let mut gaze = MIDDLE;
            let mut sweep = false;
            loop {
                assert!(self.time < 300.0, "the run must end");
                let posture = if sweep { sweeping(self.time) } else { following(gaze) };
                let output = self.frame(Some((&posture, gaze, 1.0)));
                assert_eq!((output.pointer, output.sound), (None, None), "the pointer is left alone during the run");
                match (output.mode, output.finished) {
                    (Mode::Calibrating(view), None) => {
                        assert!(self.engine.is_calibrating());
                        gaze = Vec2::new(view.target.0 as f32, view.target.1 as f32);
                        sweep = view.phase == Phase::Sweep;
                    }
                    (Mode::Pointing(_), Some(outcome)) => {
                        assert!(!self.engine.is_calibrating());
                        return outcome;
                    }
                    other => panic!("a frame of a run either continues it or ends it, not {other:?}"),
                }
            }
        }

        fn distance_to(&self, place: Vec2) -> f32 {
            Vec2::new(self.cursor.0 as f32, self.cursor.1 as f32).distance(place)
        }
    }

    #[test]
    fn without_a_calibration_the_face_shape_is_learned_and_the_head_then_moves_the_cursor() {
        let mut desk = Desk::new(None);
        for frame in 1..=30 {
            let output = desk.frame(Some((&Posture::default(), MIDDLE, 1.0)));
            let expected = if frame < REFERENCE_FRAMES { Activity::NoFace } else { Activity::Tracking };
            assert_eq!(output.mode, Mode::Pointing(expected), "frame {frame}");
        }
        assert!(desk.distance_to(MIDDLE) < 4.0, "a still head leaves the cursor where it is: {:?}", desk.cursor);
        // Positive yaw turns the nose toward the image's right, which is the person's left.
        for step in 1..=15 {
            let posture = Posture { yaw: 0.2 * step as f32 / 15.0, ..Posture::default() };
            desk.frame(Some((&posture, MIDDLE, 1.0)));
        }
        assert!(desk.cursor.0 < 1280 - 150, "the cursor went to {:?}", desk.cursor);
        assert!((desk.cursor.1 - 540).abs() < 60, "the cursor went to {:?}", desk.cursor);
        assert!(desk.clicks.is_empty() && desk.sounds.is_empty());
    }

    #[test]
    fn a_check_needs_a_calibration_for_this_screen_and_camera_and_then_adds_to_it() {
        let mut none = Desk::new(None);
        assert!(!none.engine.start_check(SCREEN, FRAME_SIZE, 0.0) && !none.engine.is_calibrating());
        let before = calibrated();
        let mut desk = Desk::new(Some(before.clone()));
        desk.gaze_at(1.0, MIDDLE, 1.0);
        assert!(!desk.engine.start_check(RectI::new(0, 0, 1920, 1080), FRAME_SIZE, desk.time), "another screen");
        assert!(!desk.engine.start_check(SCREEN, (640, 480), desk.time), "another camera");
        assert!(!desk.engine.is_calibrating());

        // A check that is cancelled leaves the calibration as it was.
        assert!(desk.engine.start_check(SCREEN, FRAME_SIZE, desk.time));
        desk.gaze_at(3.0, MIDDLE, 1.0);
        assert!(desk.engine.cancel_calibration());
        assert_eq!(desk.engine.calibration(), Some(&before));

        // One that is gone through replaces it with the one that learned, twice over.
        for round in 1..=2 {
            assert!(desk.engine.start_check(SCREEN, FRAME_SIZE, desk.time));
            let error = desk.through_the_run().expect("an attentive person gets through the check");
            let after = desk.engine.calibration().unwrap();
            assert_eq!((after.error_px, after.training.checked), (error, round * 12));
            assert!(after.training.lessons.len() > before.training.lessons.len() + (round - 1) * 200);
        }
        let far = Vec2::new(2200.0, 300.0);
        desk.engine.set_pointing(PointingMode::Eye);
        desk.gaze_at(1.5, far, 1.0);
        assert!(desk.distance_to(far) < 200.0, "pointing goes on with the new calibration: {:?}", desk.cursor);
    }

    #[test]
    fn a_calibration_run_leaves_the_pointer_alone_and_then_lets_the_gaze_move_it() {
        let mut desk = Desk::new(None);
        desk.engine.set_pointing(PointingMode::Eye);
        desk.gaze_at(1.0, MIDDLE, 1.0);
        let far = Vec2::new(2200.0, 300.0);
        desk.gaze_at(1.5, far, 1.0);
        assert!(desk.distance_to(far) > 600.0, "without a calibration the gaze moves nothing: {:?}", desk.cursor);

        let error = desk.calibrate().expect("an attentive person calibrates");
        assert!((5.0..150.0).contains(&error), "typical error {error}");
        assert_eq!(desk.engine.calibration().map(|c| (c.area, c.error_px)), Some((SCREEN, error)));

        desk.gaze_at(0.5, MIDDLE, 1.0);
        desk.gaze_at(1.5, far, 1.0);
        assert!(desk.distance_to(far) < 200.0, "the cursor went to {:?}", desk.cursor);
        assert!(desk.clicks.is_empty());
    }

    #[test]
    fn switching_to_eye_pointing_takes_effect_on_the_next_frames_and_back_again() {
        let mut desk = Desk::new(Some(calibrated()));
        let (near, far) = (Vec2::new(1450.0, 640.0), Vec2::new(2200.0, 300.0));
        desk.eyes_at(1.0, MIDDLE);
        desk.eyes_at(1.0, near);
        desk.eyes_at(1.0, far);
        assert!(desk.distance_to(MIDDLE) < 40.0, "in head mode no look, near or far, moves the cursor: {:?}", desk.cursor);
        assert!(!desk.engine.uses_gaze(), "head mode reads no gaze");

        desk.engine.set_pointing(PointingMode::Eye);
        assert!(desk.engine.uses_gaze());
        desk.eyes_at(2.0, near);
        assert!(desk.distance_to(near) < 120.0, "in eye mode the cursor is where the person looks: {:?}", desk.cursor);

        desk.engine.set_pointing(PointingMode::Head);
        let held = desk.cursor;
        desk.eyes_at(1.0, MIDDLE);
        desk.eyes_at(1.0, far);
        let moved = Vec2::new(desk.cursor.0 as f32, desk.cursor.1 as f32).distance(Vec2::new(held.0 as f32, held.1 as f32));
        assert!(moved < 40.0, "back in head mode the eyes move nothing: {held:?} to {:?}", desk.cursor);
        assert!(desk.clicks.is_empty());
    }

    #[test]
    fn a_long_blink_clicks_once_when_nothing_interrupts_it() {
        let mut desk = Desk::new(Some(calibrated()));
        desk.gaze_at(1.5, MIDDLE, 1.0);
        desk.gaze_at(0.45, MIDDLE, 0.0);
        assert_eq!(desk.sounds, [Sound::Armed]);
        desk.gaze_at(0.5, MIDDLE, 1.0);
        assert_eq!(desk.clicks.len(), 1, "{:?}", desk.clicks);
    }

    #[test]
    fn a_blink_that_was_armed_before_a_break_never_clicks_after_it() {
        // The camera stalls with the eyes shut and delivers again while they are open,
        // well before the blink would have been given up on its own.
        let mut stalled = Desk::new(Some(calibrated()));
        stalled.gaze_at(1.5, MIDDLE, 1.0);
        stalled.gaze_at(0.45, MIDDLE, 0.0);
        assert_eq!(stalled.sounds, [Sound::Armed]);
        stalled.time += 0.6;
        stalled.gaze_at(1.0, MIDDLE, 1.0);
        assert_eq!(stalled.clicks, [], "after a stalled camera");

        // The same blink across a pause (the pipeline interrupts, then frames continue at once).
        let mut paused = Desk::new(Some(calibrated()));
        paused.gaze_at(1.5, MIDDLE, 1.0);
        paused.gaze_at(0.45, MIDDLE, 0.0);
        paused.engine.interrupt();
        paused.gaze_at(1.0, MIDDLE, 1.0);
        assert_eq!(paused.clicks, [], "after a pause");

        // And across a calibration run that is given up straight away.
        let mut calibrating = Desk::new(Some(calibrated()));
        calibrating.gaze_at(1.5, MIDDLE, 1.0);
        calibrating.gaze_at(0.45, MIDDLE, 0.0);
        calibrating.engine.start_calibration(SCREEN, FRAME_SIZE, calibrating.time);
        assert!(calibrating.engine.cancel_calibration());
        calibrating.gaze_at(1.0, MIDDLE, 1.0);
        assert_eq!(calibrating.clicks, [], "after a cancelled calibration");
    }

    #[test]
    fn the_gaze_is_ignored_when_the_screen_or_the_camera_is_not_the_calibrated_one() {
        let calibration = calibrated();
        let other = RectI::new(0, 0, 1920, 1080);
        assert_eq!(gaze_problem(&calibration, FRAME_SIZE, &[SCREEN]), None);
        assert_eq!(gaze_problem(&calibration, FRAME_SIZE, &[RectI::new(-1920, 0, 0, 1080), SCREEN]), None);
        assert_eq!(gaze_problem(&calibration, FRAME_SIZE, &[other]), Some(GazeProblem::ScreenChanged));
        assert_eq!(gaze_problem(&calibration, FRAME_SIZE, &[]), Some(GazeProblem::ScreenChanged));
        assert_eq!(gaze_problem(&calibration, (640, 480), &[SCREEN]), Some(GazeProblem::CameraChanged));

        let far = Vec2::new(1700.0, 300.0);
        let mut same = Desk::new(Some(calibration.clone()));
        same.engine.set_pointing(PointingMode::Eye);
        same.gaze_at(1.0, MIDDLE, 1.0);
        same.gaze_at(1.5, far, 1.0);
        assert!(same.distance_to(far) < 200.0, "on the calibrated screen the gaze moves the cursor: {:?}", same.cursor);

        let mut changed = Desk::new(Some(calibration));
        changed.engine.set_pointing(PointingMode::Eye);
        changed.screens = vec![other];
        changed.cursor = (960, 540);
        changed.gaze_at(1.0, MIDDLE, 1.0);
        changed.gaze_at(1.5, far, 1.0);
        assert!(changed.distance_to(Vec2::new(960.0, 540.0)) < 120.0, "the cursor went to {:?}", changed.cursor);
    }

    /// Goes through a head calibration as a person would who turns this far (yaw, pitch in
    /// radians) with ease, at a radian a second. Returns how it ended.
    fn head_calibration(desk: &mut Desk, comfortable: (f32, f32)) -> Result<Outcome, Failure> {
        desk.engine.start_head_calibration(SCREEN, desk.time);
        let mut facing = (0.0f32, 0.0f32);
        let mut goal = (0.0, 0.0);
        loop {
            assert!(desk.time < 120.0, "the run must end");
            let step = 1.0 / 30.0;
            let toward = |now: f32, to: f32| now + (to - now).clamp(-step, step);
            facing = (toward(facing.0, goal.0), toward(facing.1, goal.1));
            let posture = Posture { yaw: facing.0, pitch: facing.1, ..Posture::default() };
            let output = desk.frame(Some((&posture, MIDDLE, 1.0)));
            assert_eq!((output.pointer, output.sound), (None, None), "the pointer is left alone during the run");
            match (output.mode, output.finished) {
                (Mode::Calibrating(view), None) => {
                    // Positive yaw turns the nose toward the person's left; positive pitch nods down.
                    goal = match view.phase {
                        Phase::HeadLeft => (comfortable.0, 0.0),
                        Phase::HeadRight => (-comfortable.0, 0.0),
                        Phase::HeadUp => (0.0, -comfortable.1),
                        Phase::HeadDown => (0.0, comfortable.1),
                        _ => (0.0, 0.0),
                    };
                }
                (Mode::Pointing(_), Some(outcome)) => return outcome,
                other => panic!("a frame of a run either continues it or ends it, not {other:?}"),
            }
        }
    }

    #[test]
    fn a_head_calibration_puts_its_speeds_to_use_at_once_and_leaves_the_gaze_alone() {
        let mut desk = Desk::new(Some(calibrated()));
        desk.eyes_at(1.0, MIDDLE);
        let before = desk.engine.calibration().cloned();
        let Ok(Outcome::Head(gain)) = head_calibration(&mut desk, (0.35, 0.25)) else { panic!("the run failed") };
        assert_eq!(desk.engine.head_gain(), gain, "in use from the next frame");
        assert_eq!(desk.engine.calibration().cloned(), before, "the gaze calibration is not touched");
        assert!(gain.fast > Tuning::default().head_gain.fast && gain.slow <= gain.fast, "{gain:?}");
        assert!((0.5..=8.0).contains(&gain.steadiness), "{gain:?}");

        // A turn moves the cursor by what the new speeds say: farther than the defaults would.
        let turn = |desk: &mut Desk| {
            desk.cursor = (1280, 540);
            desk.eyes_at(1.0, MIDDLE);
            for step in 1..=15 {
                let posture = Posture { yaw: 0.1 * step as f32 / 15.0, ..Posture::default() };
                desk.frame(Some((&posture, MIDDLE, 1.0)));
            }
            desk.eyes_at(1.0, MIDDLE);
            (desk.cursor.0 - 1280).abs()
        };
        let fitted = turn(&mut desk);
        let mut plain = Desk::new(Some(calibrated()));
        let usual = turn(&mut plain);
        assert!(fitted as f32 > usual as f32 * 1.3, "{fitted} px with {gain:?} against {usual} px with the defaults");
        assert!(desk.clicks.is_empty());
    }

    #[test]
    fn a_head_calibration_that_fails_or_is_cancelled_keeps_the_speeds_there_were() {
        let mut desk = Desk::new(None);
        desk.eyes_at(1.0, MIDDLE);
        let usual = desk.engine.head_gain();
        // Nobody turns: the first turn waits until the stage gives up.
        assert_eq!(head_calibration(&mut desk, (0.0, 0.0)), Err(Failure::HeadNotMoved));
        assert_eq!(desk.engine.head_gain(), usual);
        desk.engine.start_head_calibration(SCREEN, desk.time);
        let during = desk.frame(Some((&Posture::default(), MIDDLE, 1.0)));
        assert!(matches!(during.mode, Mode::Calibrating(view) if view.phase == Phase::HeadRest), "{during:?}");
        assert!(desk.engine.cancel_calibration() && !desk.engine.is_calibrating());
        assert_eq!(desk.engine.head_gain(), usual);
    }

    #[test]
    fn a_cancelled_run_reports_nothing_and_a_failed_one_keeps_the_calibration_before_it() {
        let before = calibrated();
        let mut desk = Desk::new(Some(before.clone()));
        desk.gaze_at(1.0, MIDDLE, 1.0);
        assert!(!desk.engine.cancel_calibration(), "nothing to cancel yet");

        desk.engine.start_calibration(SCREEN, FRAME_SIZE, desk.time);
        let during = desk.frame(Some((&Posture::default(), MIDDLE, 1.0)));
        assert!(matches!(during.mode, Mode::Calibrating(view) if view.phase == Phase::Rest), "{during:?}");
        assert!(desk.engine.cancel_calibration());
        assert!(!desk.engine.cancel_calibration());
        let after = desk.frame(Some((&Posture::default(), MIDDLE, 1.0)));
        assert!(matches!(after.mode, Mode::Pointing(_)) && after.finished.is_none(), "{after:?}");

        // The person walks away during the next run.
        desk.engine.start_calibration(SCREEN, FRAME_SIZE, desk.time);
        let ended = loop {
            assert!(desk.time < 60.0, "the run must end");
            if let Some(outcome) = desk.frame(None).finished {
                break outcome;
            }
        };
        assert_eq!(ended, Err(Failure::FaceLost));
        assert_eq!(desk.engine.calibration(), Some(&before));
    }
}
