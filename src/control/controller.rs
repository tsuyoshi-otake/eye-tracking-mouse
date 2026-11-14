//! Decides, frame by frame, where the cursor goes and when it clicks.
//!
//! One call per camera frame. Who owns the cursor, in order of priority:
//!
//! 1. Whoever else moves it (the hand on the mouse): the controller holds off.
//! 2. A long blink: the cursor stays where it was when the eyes closed, and
//!    a click lands there when they open.
//! 3. The gaze: a rest on a place far from the cursor jumps the cursor there.
//! 4. The head: everything else is small head movement nudging the cursor.
//!
//! That is the head mode. In eye mode 3 and 4 give way to the gaze estimate
//! being the cursor's place in every frame, and in hybrid mode the head's
//! movement is added to that as a correction.
//!
//! The controller never touches the system; it returns what should be done.

use std::collections::VecDeque;

use crate::geometry::{RectI, Vec2, clamp_to_union};
use crate::vision::FaceMeasurement;

use super::blink::{BlinkClicker, BlinkEvent, BlinkSettings};
use super::calibration::Calibration;
use super::eye_pointer::EyePointer;
use super::fixation::FixationDetector;
use super::guard::{ManualGuard, Verdict};
use super::head_pointer::{HeadGain, HeadPointer};
use super::hybrid::Correction;

/// Seconds of cursor history kept, for going back to before the lids dropped.
const REWIND: f64 = 0.15;
/// The cursor goes back at most this many pixels; anything more was a deliberate move.
const REWIND_LIMIT: f32 = 25.0;
/// Seconds after the eyes reopen during which the face is still settling and is not used.
const SETTLE: f64 = 0.2;
/// A gaze rest counts as a new place when it is this share of the jump distance from the last one.
const NEW_PLACE_SHARE: f32 = 0.8;
/// A gaze estimate up to this share of the screen's size beyond its edge still means the edge.
const EDGE_MARGIN: f32 = 0.15;

/// What moves the cursor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PointingMode {
    /// Small head movements nudge the cursor; a gaze rest jumps it to where the person looks.
    #[default]
    Head,
    /// The eyes move the cursor directly: every frame's gaze estimate is where
    /// it sits. Needs a calibration; without one the head carries pointing.
    Eye,
    /// The eyes carry the cursor as in `Eye`, and small head movements move it
    /// away from where the eyes put it, to make up for what the estimate is
    /// off by. Looking somewhere else drops that correction. Needs a
    /// calibration; without one the head carries pointing.
    Hybrid,
}

/// What the person can adjust.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    /// Head, eyes, or both.
    pub pointing: PointingMode,
    /// Jump the cursor to where the person looks. Only in head mode: in eye
    /// mode every frame's estimate is already where the cursor is.
    pub gaze_jump: bool,
    /// Click with a long blink.
    pub blink_click: bool,
    pub head_gain: HeadGain,
    /// How firmly the gaze-driven cursor is held in eye and hybrid mode, like `head_gain`'s steadiness.
    pub eye_steadiness: f32,
    /// Cursor pixels per eye span of head movement in hybrid mode, at any speed. There the head
    /// only corrects what the eyes carried, so it is slower than `head_gain` and does not speed
    /// up with a quick movement: measured on a person, the quick gain threw the cursor by up to
    /// 460 px in one frame.
    pub hybrid_speed: f32,
    /// In head mode the gaze moves the cursor only when it rests at least this many pixels
    /// away from it. In hybrid mode the head's correction is dropped once the eyes have
    /// carried the cursor this far from where the correction began.
    pub jump_distance: f32,
    pub blink: BlinkSettings,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            pointing: PointingMode::Head,
            gaze_jump: true,
            blink_click: true,
            head_gain: HeadGain { slow: 700.0, fast: 2200.0, steadiness: 2.0 },
            eye_steadiness: 1.5,
            hybrid_speed: 450.0,
            jump_distance: 250.0,
            blink: BlinkSettings { hold: 0.4, limit: 1.5, closed_ratio: 0.5, open_ratio: 0.6 },
        }
    }
}

/// One camera frame's worth of input.
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    pub time: f64,
    /// `None` when no face was measured in this frame.
    pub face: Option<&'a FaceMeasurement>,
    /// Where the cursor is now; `None` while the desktop cannot be reached.
    pub cursor: Option<(i32, i32)>,
    /// The monitors, in desktop pixels.
    pub screens: &'a [RectI],
    pub calibration: Option<&'a Calibration>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sound {
    /// The eyes have been shut long enough: opening them now clicks.
    Armed,
    /// The click was made.
    Clicked,
    /// The pending click was abandoned.
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Activity {
    /// The face drives the cursor.
    #[default]
    Tracking,
    NoFace,
    /// Something else moved the cursor; waiting for it to rest.
    Yielded,
    /// The cursor or the screens cannot be reached.
    Blocked,
}

/// What the frame's pointing was made of. Nothing acts on it; it is for
/// looking into afterwards why the cursor went where it did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Makeup {
    /// The gaze model's estimate in desktop pixels, when it was asked.
    pub estimate: Option<Vec2>,
    /// Where the eyes put the cursor after filtering (eye and hybrid mode).
    pub eyes: Option<Vec2>,
    /// What the head added in this frame, in pixels.
    pub nudge: Vec2,
}

/// What to do after this frame. At most one of `move_to` and `click_at` is set.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Decision {
    pub makeup: Makeup,
    pub move_to: Option<(i32, i32)>,
    /// Put the cursor on this pixel and left-click.
    pub click_at: Option<(i32, i32)>,
    pub sound: Option<Sound>,
    pub activity: Activity,
}

#[derive(Clone, Debug, Default)]
pub struct Controller {
    guard: ManualGuard,
    head: HeadPointer,
    eye: EyePointer,
    correction: Correction,
    blink: BlinkClicker,
    fixation: FixationDetector,
    /// Where the cursor should be, finer than a pixel. `None` until it is taken over from the system.
    position: Option<Vec2>,
    /// Recent positions, oldest first.
    trail: VecDeque<(f64, Vec2)>,
    /// Where a click would land: the position from just before the eyes closed.
    anchor: Option<Vec2>,
    /// The face is not used before this time.
    steady_from: f64,
}

fn sound_for(event: Option<BlinkEvent>) -> Option<Sound> {
    match event {
        Some(BlinkEvent::Armed) => Some(Sound::Armed),
        Some(BlinkEvent::Cancelled) => Some(Sound::Cancelled),
        Some(BlinkEvent::Click) => Some(Sound::Clicked),
        None => None,
    }
}

/// Where a gaze rest at `place` may put the cursor: on the calibrated screen,
/// or nowhere if the person is looking well past it.
fn landing(area: RectI, place: Vec2) -> Option<Vec2> {
    let (left, top, right, bottom) = (area.left as f32, area.top as f32, (area.right - 1) as f32, (area.bottom - 1) as f32);
    let margin = Vec2::new(right - left, bottom - top) * EDGE_MARGIN;
    let near = place.x >= left - margin.x && place.x <= right + margin.x && place.y >= top - margin.y && place.y <= bottom + margin.y;
    near.then(|| Vec2::new(place.x.clamp(left, right), place.y.clamp(top, bottom)))
}

impl Controller {
    /// Gives up the cursor: a pending click is dropped and nothing carries over to when control returns.
    fn suspend(&mut self, time: f64, tuning: &Tuning) -> Option<Sound> {
        let sound = sound_for(self.blink.step(time, None, &tuning.blink)).filter(|_| tuning.blink_click);
        self.head.reset();
        self.eye.reset();
        self.correction.reset();
        self.fixation.reset();
        self.trail.clear();
        self.anchor = None;
        self.position = None;
        sound
    }

    pub fn step(&mut self, frame: &Frame, tuning: &Tuning) -> Decision {
        let time = frame.time;
        let Some(actual) = frame.cursor else {
            return Decision { sound: self.suspend(time, tuning), activity: Activity::Blocked, ..Decision::default() };
        };
        let probe = match self.guard.observe(time, actual) {
            Verdict::Go { .. } => None,
            Verdict::Hold => Some(None),
            // Past the edge of the screens there is nothing to try; the guard tries the other side next.
            Verdict::Probe((x, y)) => Some(clamp_to_union(frame.screens, f64::from(x), f64::from(y)).filter(|pixel| *pixel == (x, y))),
        };
        if let Some(move_to) = probe {
            let sound = self.suspend(time, tuning);
            return Decision { sound, activity: Activity::Yielded, move_to, ..Decision::default() };
        }
        // After a hold the position is gone, and the cursor is taken over from wherever it is now.
        let mut position = self.position.unwrap_or(Vec2::new(actual.0 as f32, actual.1 as f32));
        let Some(face) = frame.face else {
            let sound = sound_for(self.blink.step(time, None, &tuning.blink)).filter(|_| tuning.blink_click);
            self.head.reset();
            self.trail.clear();
            self.anchor = None;
            self.position = Some(position);
            return Decision { sound, activity: Activity::NoFace, ..Decision::default() };
        };

        // The lids are watched even when blink clicks are off: shut eyes say nothing about gaze.
        let was_closed = self.blink.is_closed();
        let event = self.blink.step(time, Some(face.eyes.openness), &tuning.blink).filter(|_| tuning.blink_click);
        let closed = self.blink.is_closed();
        if closed && !was_closed {
            // The lids began to drop a moment ago and may have dragged the cursor along.
            let earlier = self.trail.front().map_or(position, |&(_, earlier)| earlier);
            self.anchor = Some(if earlier.distance(position) <= REWIND_LIMIT { earlier } else { position });
        }
        if closed {
            self.steady_from = time + SETTLE;
        }
        let clicking = event == Some(BlinkEvent::Click);
        if clicking {
            position = self.anchor.unwrap_or(position);
        }
        if !closed {
            self.anchor = None;
        }

        let mut makeup = Makeup::default();
        if !closed && time >= self.steady_from {
            match tuning.pointing {
                PointingMode::Head => {
                    makeup.nudge = self.head.step(time, &face.pose, &tuning.head_gain);
                    position = position + makeup.nudge;
                    if let Some(calibration) = frame.calibration.filter(|_| tuning.gaze_jump) {
                        let estimate = calibration.gaze.predict(face);
                        makeup.estimate = Some(estimate);
                        let rest = self.fixation.step(time, estimate, tuning.jump_distance * NEW_PLACE_SHARE);
                        let place = rest.and_then(|place| landing(calibration.area, place));
                        if let Some(place) = place.filter(|place| place.distance(position) >= tuning.jump_distance) {
                            position = place;
                        }
                    }
                }
                PointingMode::Eye => match frame.calibration {
                    // The estimate is the position, kept on the calibrated
                    // screen. Looking past it holds the cursor where it is.
                    Some(calibration) => {
                        let estimate = calibration.gaze.predict(face);
                        makeup.estimate = Some(estimate);
                        if let Some(place) = landing(calibration.area, estimate) {
                            position = self.eye.step(time, place, tuning.eye_steadiness);
                            makeup.eyes = Some(position);
                        }
                    }
                    // No calibration yet: the head carries pointing until there is one.
                    None => {
                        makeup.nudge = self.head.step(time, &face.pose, &tuning.head_gain);
                        position = position + makeup.nudge;
                    }
                },
                PointingMode::Hybrid => {
                    // Correcting is fine work; without a calibration the head carries everything, as in head mode.
                    let fine = HeadGain { slow: tuning.hybrid_speed, fast: tuning.hybrid_speed, ..tuning.head_gain };
                    let gain = if frame.calibration.is_some() { fine } else { tuning.head_gain };
                    makeup.nudge = self.head.step(time, &face.pose, &gain);
                    makeup.estimate = frame.calibration.map(|calibration| calibration.gaze.predict(face));
                    let seen = frame.calibration.zip(makeup.estimate).and_then(|(calibration, estimate)| landing(calibration.area, estimate));
                    position = match seen {
                        Some(place) => {
                            let eyes = self.eye.step(time, place, tuning.eye_steadiness);
                            makeup.eyes = Some(eyes);
                            self.correction.step(position, eyes, makeup.nudge, tuning.jump_distance)
                        }
                        // No calibration, or looking past the screen: the eyes hold their part and the head goes on.
                        None => position + makeup.nudge,
                    };
                }
            }
            self.trail.push_back((time, position));
            while self.trail.front().is_some_and(|&(then, _)| time - then > REWIND) {
                self.trail.pop_front();
            }
        } else {
            // Whatever the head does meanwhile must not move the cursor afterwards. The gaze
            // history is kept, so looking at the same place after a blink is not a new rest.
            self.head.reset();
            self.trail.clear();
        }

        let Some(pixel) = clamp_to_union(frame.screens, f64::from(position.x), f64::from(position.y)) else {
            // No screen to be on: start over from the cursor once there is one.
            self.position = None;
            return Decision { sound: sound_for(event), activity: Activity::Blocked, makeup, ..Decision::default() };
        };
        // At an edge the position stops with the cursor, so turning back takes effect at once.
        if (pixel.0 as f32 - position.x).abs() > 0.5 {
            position.x = pixel.0 as f32;
        }
        if (pixel.1 as f32 - position.y).abs() > 0.5 {
            position.y = pixel.1 as f32;
        }
        self.position = Some(position);
        let mut decision = Decision { sound: sound_for(event), makeup, ..Decision::default() };
        if clicking {
            decision.click_at = Some(pixel);
            self.guard.commanded(pixel);
        } else if pixel != actual {
            decision.move_to = Some(pixel);
            self.guard.commanded(pixel);
        }
        decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::calibration::simulation::calibrated;
    use crate::control::gaze::simulation::{Person, Posture, SCREEN};
    use crate::vision;
    use crate::vision::eyes::EyeSignals;
    use crate::vision::pose::HeadPose;

    /// A desk with a simulated system cursor that goes wherever the controller says.
    struct Desk {
        controller: Controller,
        tuning: Tuning,
        screens: Vec<RectI>,
        cursor: Option<(i32, i32)>,
        frame: u32,
        moves: u32,
        clicks: Vec<(i32, i32)>,
        sounds: Vec<Sound>,
        activity: Activity,
    }

    impl Desk {
        fn new() -> Desk {
            Desk {
                controller: Controller::default(),
                tuning: Tuning::default(),
                screens: vec![SCREEN],
                cursor: Some((1280, 540)),
                frame: 0,
                moves: 0,
                clicks: Vec::new(),
                sounds: Vec::new(),
                activity: Activity::Tracking,
            }
        }

        fn time(&self) -> f64 {
            f64::from(self.frame) / 30.0
        }

        fn at(&self) -> (i32, i32) {
            self.cursor.unwrap()
        }

        fn feed(&mut self, face: Option<FaceMeasurement>, calibration: Option<&Calibration>) {
            let frame = Frame { time: self.time(), face: face.as_ref(), cursor: self.cursor, screens: &self.screens, calibration };
            let decision = self.controller.step(&frame, &self.tuning);
            assert!(decision.move_to.is_none() || decision.click_at.is_none(), "{decision:?}");
            if let Some(pixel) = decision.move_to.or(decision.click_at) {
                assert!(self.screens.iter().any(|screen| screen.contains(pixel.0, pixel.1)), "{pixel:?} is on no screen");
                self.cursor = Some(pixel);
            }
            self.moves += u32::from(decision.move_to.is_some());
            self.clicks.extend(decision.click_at);
            self.sounds.extend(decision.sound);
            self.activity = decision.activity;
            self.frame += 1;
        }

        /// Feeds frames of a face turned from `from` to `to` radians of yaw, eyes `openness` open.
        fn turn(&mut self, frames: u32, from: f32, to: f32, openness: f32) {
            for step in 1..=frames {
                let yaw = from + (to - from) * step as f32 / frames as f32;
                self.feed(Some(face(self.time(), yaw, openness)), None);
            }
        }

        fn rest(&mut self, frames: u32, yaw: f32, openness: f32) {
            self.turn(frames, yaw, yaw, openness);
        }
    }

    const OPEN: f32 = 0.3;
    const SHUT: f32 = 0.04;

    fn face(time: f64, yaw: f32, openness: f32) -> FaceMeasurement {
        FaceMeasurement {
            time,
            pose: HeadPose { yaw, pitch: 0.0, roll: 0.0, position: Vec2::new(640.0, 360.0), scale: 200.0 },
            eyes: EyeSignals { iris: Vec2::ZERO, lid: 0.0, openness: [openness; 2] },
        }
    }

    #[test]
    fn a_still_head_sends_nothing_and_a_turning_head_moves_the_cursor() {
        let mut desk = Desk::new();
        desk.rest(60, 0.0, OPEN);
        assert_eq!((desk.moves, desk.at()), (0, (1280, 540)));
        desk.turn(30, 0.0, 0.1, OPEN);
        desk.rest(30, 0.1, OPEN);
        // The nose went toward the image's right, which is the person's left.
        assert!(desk.at().0 < 1280 - 50 && desk.at().1 == 540, "{:?}", desk.at());
        assert_eq!(desk.activity, Activity::Tracking);
    }

    #[test]
    fn a_long_blink_clicks_where_the_cursor_was_before_the_lids_dropped() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        // The head dips a little as the lids come down.
        desk.turn(3, 0.0, 0.012, 0.2);
        let dragged = desk.at();
        assert!(dragged.0 < 1280, "the dip moved the cursor: {dragged:?}");
        desk.rest(14, 0.012, SHUT);
        assert_eq!(desk.sounds, [Sound::Armed]);
        assert!(desk.clicks.is_empty());
        assert_eq!(desk.at(), dragged, "nothing moves while the eyes are shut");
        desk.rest(1, 0.012, OPEN);
        assert_eq!(desk.clicks, [(1280, 540)]);
        assert_eq!(desk.at(), (1280, 540));
        assert_eq!(desk.sounds, [Sound::Armed, Sound::Clicked], "the click is announced");
        // The head settles back; that is not a pointing movement.
        desk.turn(5, 0.012, 0.0, OPEN);
        assert_eq!(desk.at(), (1280, 540));
        assert_eq!(desk.clicks.len(), 1);
    }

    #[test]
    fn an_ordinary_blink_neither_clicks_nor_moves_the_cursor_back() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.turn(3, 0.0, 0.012, 0.2);
        let dragged = desk.at();
        desk.rest(5, 0.012, SHUT);
        desk.rest(30, 0.012, OPEN);
        assert!(desk.clicks.is_empty() && desk.sounds.is_empty());
        assert_eq!(desk.at(), dragged);
    }

    #[test]
    fn a_deliberate_move_just_before_the_blink_is_not_undone() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.turn(4, 0.0, 0.08, OPEN);
        let moved = desk.at();
        assert!(moved.0 < 1280 - 40, "{moved:?}");
        desk.rest(14, 0.08, SHUT);
        desk.rest(1, 0.08, OPEN);
        assert_eq!(desk.clicks, [moved]);
    }

    #[test]
    fn head_movement_made_with_the_eyes_shut_is_ignored() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.turn(5, 0.0, 0.15, SHUT);
        desk.rest(60, 0.15, OPEN);
        assert_eq!(desk.at(), (1280, 540));
    }

    #[test]
    fn with_blink_clicks_off_a_long_blink_does_nothing() {
        let mut desk = Desk::new();
        desk.tuning.blink_click = false;
        desk.rest(45, 0.0, OPEN);
        desk.rest(20, 0.0, SHUT);
        desk.rest(30, 0.0, OPEN);
        desk.rest(70, 0.0, SHUT);
        desk.rest(10, 0.0, OPEN);
        assert!(desk.clicks.is_empty() && desk.sounds.is_empty());
    }

    #[test]
    fn a_hand_on_the_mouse_takes_over_and_control_resumes_from_where_it_left_the_cursor() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.turn(10, 0.0, 0.05, OPEN);
        // The mouse drags the cursor away while the head keeps turning.
        for step in 0..20 {
            desk.cursor = Some((300 + step * 10, 200));
            desk.turn(1, 0.05, 0.06, OPEN);
            assert_eq!(desk.activity, Activity::Yielded);
        }
        let left_at = desk.at();
        let moves = desk.moves;
        desk.turn(40, 0.06, 0.2, OPEN);
        assert_eq!((desk.at(), desk.moves), (left_at, moves), "still yielding 1.3 s after the mouse stopped");
        desk.rest(10, 0.2, OPEN);
        assert_eq!(desk.activity, Activity::Tracking);
        let taken_at = (left_at.0 + 1, left_at.1);
        assert_eq!(desk.at(), taken_at, "taking over moves the cursor by the one pixel that asks whether it may");
        desk.turn(15, 0.2, 0.25, OPEN);
        assert!(desk.at().0 < left_at.0 && (desk.at().0 - left_at.0).abs() < 200, "{:?}", desk.at());
    }

    #[test]
    fn touching_the_mouse_cancels_a_pending_click() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.rest(15, 0.0, SHUT);
        assert_eq!(desk.sounds, [Sound::Armed]);
        desk.cursor = Some((900, 900));
        desk.rest(1, 0.0, SHUT);
        assert_eq!(desk.sounds, [Sound::Armed, Sound::Cancelled]);
        desk.rest(90, 0.0, OPEN);
        assert!(desk.clicks.is_empty());
        assert_eq!(desk.at(), (901, 900), "the pixel tried on taking over, and nothing else");
    }

    #[test]
    fn while_the_pointer_is_on_another_computer_nothing_but_single_pixels_is_sent() {
        // A program sharing the mouse with another computer takes every move sent here
        // and the cursor stays where it was parked.
        let parked = (1200, 400);
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        let (mut sent, moves) = (Vec::new(), desk.moves);
        for frame in 0..600 {
            desk.cursor = Some(parked);
            let yaw = 0.2 * (f32::from(frame as u16) / 20.0).sin();
            desk.turn(1, yaw, yaw, OPEN);
            if desk.at() != parked {
                sent.push((desk.at().0 - parked.0, desk.at().1 - parked.1));
            }
        }
        // 20 seconds of a turning head: one pixel every 1.5 s, to the right and to the left in turn.
        assert!((4..=14).contains(&sent.len()) && sent.len() as u32 == desk.moves - moves, "{sent:?}");
        assert!(sent.iter().all(|step| step.0.abs() == 1 && step.1 == 0), "{sent:?}");
        assert!(sent.windows(2).all(|pair| pair[0].0 + pair[1].0 == 0), "{sent:?}");
        assert!(desk.clicks.is_empty());
        // Back on this computer the cursor obeys again, and the head has it after one more pixel.
        desk.rest(60, 0.0, OPEN);
        assert_eq!(desk.activity, Activity::Tracking);
    }

    #[test]
    fn losing_the_face_holds_the_cursor_and_cancels_a_pending_click() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.rest(15, 0.0, SHUT);
        for _ in 0..30 {
            desk.feed(None, None);
            assert_eq!(desk.activity, Activity::NoFace);
        }
        assert_eq!(desk.sounds, [Sound::Armed, Sound::Cancelled]);
        // The face comes back turned: that is where pointing starts from, not a movement.
        desk.rest(60, 0.2, OPEN);
        assert!(desk.clicks.is_empty());
        assert_eq!((desk.at(), desk.moves), ((1280, 540), 0));
    }

    #[test]
    fn an_unreachable_desktop_blocks_everything_until_it_is_back() {
        let mut desk = Desk::new();
        desk.rest(45, 0.0, OPEN);
        desk.cursor = None;
        desk.turn(30, 0.0, 0.2, OPEN);
        assert_eq!((desk.activity, desk.moves), (Activity::Blocked, 0));
        desk.cursor = Some((1280, 540));
        desk.rest(30, 0.2, OPEN);
        assert_eq!((desk.activity, desk.at(), desk.moves), (Activity::Tracking, (1280, 540), 0));
        // Without any screen there is nowhere to point either.
        desk.screens.clear();
        desk.turn(10, 0.2, 0.3, OPEN);
        assert_eq!((desk.activity, desk.moves), (Activity::Blocked, 0));
    }

    #[test]
    fn the_cursor_stops_at_the_edge_and_turns_back_at_once() {
        let mut desk = Desk::new();
        desk.cursor = Some((40, 540));
        desk.rest(45, 0.0, OPEN);
        desk.turn(20, 0.0, 0.5, OPEN);
        desk.rest(30, 0.5, OPEN);
        assert_eq!(desk.at(), (0, 540));
        // Far past the edge by now, had the position not stopped with the cursor.
        desk.turn(10, 0.5, 0.46, OPEN);
        desk.rest(30, 0.46, OPEN);
        assert!(desk.at().0 > 15, "{:?}", desk.at());
    }

    #[test]
    fn the_cursor_never_lands_in_the_gap_beside_a_smaller_second_screen() {
        let mut desk = Desk::new();
        desk.screens = vec![RectI::new(0, 0, 2560, 1080), RectI::new(0, 1080, 1500, 1980)];
        desk.cursor = Some((1400, 1500));
        desk.rest(45, 0.0, OPEN);
        // Toward the person's right, where the lower screen ends but the upper one goes on.
        desk.turn(30, 0.0, -0.4, OPEN);
        desk.rest(30, -0.4, OPEN);
        assert_eq!(desk.at(), (1499, 1500), "`feed` has checked every pixel on the way");
    }

    /// A calibrated person in front of the desk.
    struct Sitter {
        person: Person,
        calibration: Calibration,
    }

    impl Sitter {
        fn new() -> Sitter {
            Sitter { person: Person::new(1.0), calibration: calibrated() }
        }

        fn look(&mut self, desk: &mut Desk, frames: u32, posture: &Posture, target: Vec2, openness: f32) {
            for _ in 0..frames {
                let mesh = self.person.look(desk.time(), posture, target, openness);
                let face = vision::measure(&self.calibration.reference, &mesh);
                desk.feed(face, Some(&self.calibration));
            }
        }
    }

    fn distance(cursor: (i32, i32), target: Vec2) -> f32 {
        Vec2::new(cursor.0 as f32, cursor.1 as f32).distance(target)
    }

    #[test]
    fn looking_at_a_far_place_brings_the_cursor_there_within_half_a_second() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        let (middle, far) = (Vec2::new(1280.0, 540.0), Vec2::new(2200.0, 300.0));
        sitter.look(&mut desk, 30, &Posture::default(), middle, 1.0);
        assert!(distance(desk.at(), middle) < 40.0, "looking near the cursor leaves it alone: {:?}", desk.at());
        sitter.look(&mut desk, 15, &Posture::default(), far, 1.0);
        assert!(distance(desk.at(), far) < 110.0, "{:?}", desk.at());
        sitter.look(&mut desk, 15, &Posture::default(), Vec2::new(300.0, 900.0), 1.0);
        assert!(distance(desk.at(), Vec2::new(300.0, 900.0)) < 110.0, "{:?}", desk.at());
    }

    #[test]
    fn a_glance_at_something_near_the_cursor_does_not_move_it() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        sitter.look(&mut desk, 30, &Posture::default(), Vec2::new(1280.0, 540.0), 1.0);
        let before = desk.at();
        sitter.look(&mut desk, 30, &Posture::default(), Vec2::new(1400.0, 600.0), 1.0);
        assert!(distance(desk.at(), Vec2::new(before.0 as f32, before.1 as f32)) < 30.0, "{:?} from {before:?}", desk.at());
    }

    #[test]
    fn the_gaze_does_nothing_when_switched_off_or_not_calibrated() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        desk.tuning.gaze_jump = false;
        sitter.look(&mut desk, 30, &Posture::default(), Vec2::new(1280.0, 540.0), 1.0);
        sitter.look(&mut desk, 30, &Posture::default(), Vec2::new(2200.0, 300.0), 1.0);
        assert!(distance(desk.at(), Vec2::new(1280.0, 540.0)) < 40.0, "{:?}", desk.at());
        desk.tuning.gaze_jump = true;
        for _ in 0..30 {
            let mesh = sitter.person.look(desk.time(), &Posture::default(), Vec2::new(300.0, 900.0), 1.0);
            desk.feed(vision::measure(&sitter.calibration.reference, &mesh), None);
        }
        assert!(distance(desk.at(), Vec2::new(1280.0, 540.0)) < 40.0, "{:?}", desk.at());
    }

    #[test]
    fn after_tuning_with_the_head_a_blink_does_not_throw_the_cursor_back_to_the_gaze() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        let far = Vec2::new(2000.0, 400.0);
        sitter.look(&mut desk, 30, &Posture::default(), Vec2::new(1280.0, 540.0), 1.0);
        sitter.look(&mut desk, 20, &Posture::default(), far, 1.0);
        assert!(distance(desk.at(), far) < 110.0, "{:?}", desk.at());
        // A quick, large head turn carries the cursor well away while the eyes stay on the place.
        for step in 1..=8 {
            let posture = Posture { yaw: 0.0275 * step as f32, ..Posture::default() };
            sitter.look(&mut desk, 1, &posture, far, 1.0);
        }
        let turned = Posture { yaw: 0.22, ..Posture::default() };
        sitter.look(&mut desk, 30, &turned, far, 1.0);
        let tuned = desk.at();
        assert!(distance(tuned, far) > 300.0, "the head moved the cursor only to {tuned:?}");
        // An ordinary blink, then a long one that clicks.
        sitter.look(&mut desk, 5, &turned, far, 0.0);
        sitter.look(&mut desk, 30, &turned, far, 1.0);
        assert!(distance(desk.at(), Vec2::new(tuned.0 as f32, tuned.1 as f32)) < 20.0, "{:?} from {tuned:?}", desk.at());
        sitter.look(&mut desk, 15, &turned, far, 0.0);
        sitter.look(&mut desk, 30, &turned, far, 1.0);
        assert_eq!(desk.clicks.len(), 1);
        assert!(distance(desk.clicks[0], Vec2::new(tuned.0 as f32, tuned.1 as f32)) < 20.0, "{:?} from {tuned:?}", desk.clicks);
        assert!(distance(desk.at(), Vec2::new(tuned.0 as f32, tuned.1 as f32)) < 20.0, "{:?} from {tuned:?}", desk.at());
    }

    #[test]
    fn looking_past_the_calibrated_screen_moves_nothing_and_just_past_it_means_its_edge() {
        let area = RectI::new(0, 0, 2560, 1080);
        assert_eq!(landing(area, Vec2::new(4000.0, 500.0)), None);
        assert_eq!(landing(area, Vec2::new(1000.0, -400.0)), None);
        assert_eq!(landing(area, Vec2::new(2700.0, 1100.0)), Some(Vec2::new(2559.0, 1079.0)));
        assert_eq!(landing(area, Vec2::new(-100.0, 300.0)), Some(Vec2::new(0.0, 300.0)));
        assert_eq!(landing(area, Vec2::new(1000.0, 500.0)), Some(Vec2::new(1000.0, 500.0)));
    }

    /// Feeds the desk frames of the person looking at `target`, without a calibration.
    fn look_uncalibrated(sitter: &mut Sitter, desk: &mut Desk, frames: u32, posture: &Posture, target: Vec2) {
        for _ in 0..frames {
            let mesh = sitter.person.look(desk.time(), posture, target, 1.0);
            desk.feed(vision::measure(&sitter.calibration.reference, &mesh), None);
        }
    }

    #[test]
    fn in_eye_mode_the_cursor_tracks_the_gaze_even_within_the_jump_distance() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        desk.tuning.pointing = PointingMode::Eye;
        // In head mode a look this close to the cursor moves nothing; in eye
        // mode the estimate is the position, however near it is.
        let near = Vec2::new(1450.0, 640.0);
        sitter.look(&mut desk, 60, &Posture::default(), near, 1.0);
        assert!(distance(desk.at(), near) < 120.0, "{:?}", desk.at());
        let far = Vec2::new(2100.0, 250.0);
        sitter.look(&mut desk, 60, &Posture::default(), far, 1.0);
        assert!(distance(desk.at(), far) < 120.0, "{:?}", desk.at());
    }

    #[test]
    fn eye_mode_without_a_calibration_points_with_the_head() {
        // The same person twice, measured with the same noise: once in each mode.
        let run = |pointing: PointingMode| {
            let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
            desk.tuning.pointing = pointing;
            look_uncalibrated(&mut sitter, &mut desk, 30, &Posture::default(), Vec2::new(1280.0, 540.0));
            let rested = desk.at();
            for step in 1..=20 {
                look_uncalibrated(&mut sitter, &mut desk, 1, &Posture { yaw: 0.01 * step as f32, ..Posture::default() }, Vec2::new(1280.0, 540.0));
            }
            (rested, desk.at())
        };
        let (rested, turned) = run(PointingMode::Eye);
        assert!(distance(rested, Vec2::new(1280.0, 540.0)) < 3.0, "a still head moves next to nothing: {rested:?}");
        assert!(turned.0 < 1280 - 50, "the head still points while there is no calibration: {turned:?}");
        assert_eq!((rested, turned), run(PointingMode::Head), "pixel for pixel what head mode does");
    }

    #[test]
    fn in_hybrid_mode_the_eyes_carry_the_cursor_and_the_head_corrects_it_until_the_eyes_move_on() {
        // The same person twice, measured with the same noise: the eyes alone, and both.
        let run = |pointing: PointingMode| {
            let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
            desk.tuning.pointing = pointing;
            let target = Vec2::new(1700.0, 420.0);
            sitter.look(&mut desk, 60, &Posture::default(), target, 1.0);
            let carried = desk.at();
            // The eyes stay on the place while the head turns a little and rests.
            for step in 1..=20 {
                sitter.look(&mut desk, 1, &Posture { yaw: 0.004 * step as f32, ..Posture::default() }, target, 1.0);
            }
            let turned = Posture { yaw: 0.08, ..Posture::default() };
            sitter.look(&mut desk, 30, &turned, target, 1.0);
            let corrected = desk.at();
            sitter.look(&mut desk, 60, &turned, Vec2::new(600.0, 800.0), 1.0);
            (carried, corrected, desk.at())
        };
        let (eye, hybrid) = (run(PointingMode::Eye), run(PointingMode::Hybrid));
        let apart = |one: (i32, i32), other: (i32, i32)| distance(one, Vec2::new(other.0 as f32, other.1 as f32));
        assert!(apart(hybrid.0, eye.0) < 3.0, "with a still head the eyes alone decide: {:?} against {:?}", hybrid.0, eye.0);
        assert!(distance(hybrid.0, Vec2::new(1700.0, 420.0)) < 120.0, "{:?}", hybrid.0);
        // The turn of the head moved the cursor to the person's left, which the eyes alone would not have.
        assert!(hybrid.1.0 < eye.1.0 - 25, "hybrid {:?} against eye {:?}", hybrid.1, eye.1);
        assert!(apart(eye.1, eye.0) < 40.0, "the eyes alone stay: {:?} from {:?}", eye.1, eye.0);
        // Looking somewhere else leaves the correction behind.
        assert!(apart(hybrid.2, eye.2) < 3.0, "hybrid {:?} against eye {:?}", hybrid.2, eye.2);
    }

    #[test]
    fn in_hybrid_mode_a_quick_turn_of_the_head_does_not_throw_the_cursor_as_it_does_in_head_mode() {
        // The same quick turn while the eyes stay on one place: how far the cursor went from where it rested.
        let thrown = |pointing: PointingMode| {
            let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
            desk.tuning.pointing = pointing;
            desk.tuning.gaze_jump = false;
            let target = Vec2::new(1700.0, 420.0);
            sitter.look(&mut desk, 60, &Posture::default(), target, 1.0);
            let rested = desk.at();
            for step in 1..=5 {
                sitter.look(&mut desk, 1, &Posture { yaw: 0.03 * step as f32, ..Posture::default() }, target, 1.0);
            }
            sitter.look(&mut desk, 20, &Posture { yaw: 0.15, ..Posture::default() }, target, 1.0);
            distance(desk.at(), Vec2::new(rested.0 as f32, rested.1 as f32))
        };
        let (head, hybrid) = (thrown(PointingMode::Head), thrown(PointingMode::Hybrid));
        assert!(hybrid > 20.0, "the head still corrects: {hybrid} px");
        assert!(hybrid < head * 0.4, "hybrid {hybrid} px against head mode's {head} px");
    }

    #[test]
    fn hybrid_mode_without_a_calibration_points_with_the_head() {
        let run = |pointing: PointingMode| {
            let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
            desk.tuning.pointing = pointing;
            look_uncalibrated(&mut sitter, &mut desk, 30, &Posture::default(), Vec2::new(1280.0, 540.0));
            for step in 1..=20 {
                look_uncalibrated(&mut sitter, &mut desk, 1, &Posture { yaw: 0.01 * step as f32, ..Posture::default() }, Vec2::new(1280.0, 540.0));
            }
            desk.at()
        };
        assert!(run(PointingMode::Hybrid).0 < 1280 - 50);
        assert_eq!(run(PointingMode::Hybrid), run(PointingMode::Head), "pixel for pixel what head mode does");
    }

    #[test]
    fn the_blink_click_still_works_in_eye_mode() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        desk.tuning.pointing = PointingMode::Eye;
        let target = Vec2::new(1900.0, 400.0);
        sitter.look(&mut desk, 60, &Posture::default(), target, 1.0);
        let resting = desk.at();
        assert!(distance(resting, target) < 120.0, "{resting:?}");
        sitter.look(&mut desk, 15, &Posture::default(), target, 0.0);
        sitter.look(&mut desk, 5, &Posture::default(), target, 1.0);
        assert_eq!(desk.clicks.len(), 1);
        assert!(distance(desk.clicks[0], Vec2::new(resting.0 as f32, resting.1 as f32)) < 20.0, "{:?} from {resting:?}", desk.clicks);
    }

    #[test]
    fn a_noisy_estimate_still_leaves_a_steady_eye_mode_cursor() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        desk.tuning.pointing = PointingMode::Eye;
        let target = Vec2::new(900.0, 700.0);
        sitter.look(&mut desk, 90, &Posture::default(), target, 1.0);
        let mut path = 0.0;
        for _ in 0..60 {
            let before = Vec2::new(desk.at().0 as f32, desk.at().1 as f32);
            sitter.look(&mut desk, 1, &Posture::default(), target, 1.0);
            path += distance(desk.at(), before);
        }
        assert!(distance(desk.at(), target) < 120.0, "{:?}", desk.at());
        assert!(path < 200.0, "a resting gaze rattles the cursor by {path} px over 2 s");
    }

    #[test]
    fn the_hand_on_the_mouse_still_wins_in_eye_mode() {
        let (mut sitter, mut desk) = (Sitter::new(), Desk::new());
        desk.tuning.pointing = PointingMode::Eye;
        sitter.look(&mut desk, 45, &Posture::default(), Vec2::new(1280.0, 540.0), 1.0);
        for step in 0..20 {
            desk.cursor = Some((300 + step * 10, 200));
            let mesh = sitter.person.look(desk.time(), &Posture::default(), Vec2::new(1280.0, 540.0), 1.0);
            desk.feed(vision::measure(&sitter.calibration.reference, &mesh), Some(&sitter.calibration));
            assert_eq!(desk.activity, Activity::Yielded);
        }
        assert_eq!(desk.at(), (490, 200));
    }
}
