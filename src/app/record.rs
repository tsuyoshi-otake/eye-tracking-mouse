//! Records for finding out afterwards why the program did what it did: one
//! of a calibration run (why the gaze estimate is as far off as it is), one of
//! pointing (why the cursor went where it went).
//!
//! They hold what the tracker measured in every frame next to what the person
//! was shown, or what the cursor was told: numbers only, never a picture. They
//! are made only when the settings ask for them, and nothing in the program
//! reads them back.

use std::fmt::Write;

use crate::control::calibration::{Phase, View};
use crate::control::controller::{Activity, PointingMode};
use crate::geometry::Vec2;

use super::engine::{Mode, Output, PointerAction};
use super::settings::pointing_word;
use crate::geometry::RectI;
use crate::vision::landmarks::FaceMesh;

const HEADER: &str = "eye-tracking-mouse calibration record 1";

pub struct CalibrationRecord {
    text: String,
}

impl CalibrationRecord {
    /// Starts the record of a run on the screen `area`, from a camera image of `frame_size`.
    pub fn begin(area: RectI, frame_size: (u32, u32)) -> CalibrationRecord {
        let mut text = format!("{HEADER}\n");
        let _ = writeln!(text, "area = {} {} {} {}", area.left, area.top, area.right, area.bottom);
        let _ = writeln!(text, "camera = {} {}", frame_size.0, frame_size.1);
        text.push_str("# frame = time, stage, place shown (x y), share of the stage done, then x y z of every point\n");
        CalibrationRecord { text }
    }

    /// Adds a frame in which a face was found: what was on the page, and the face's points in image pixels.
    pub fn add(&mut self, view: &View, mesh: &FaceMesh) {
        let stage = match view.phase {
            Phase::Rest => "rest",
            Phase::Places => "places",
            Phase::Sweep => "sweep",
            Phase::Check => "check",
        };
        let _ = write!(self.text, "frame = {:.4} {stage} {} {} {:.3}", mesh.time, view.target.0, view.target.1, view.dwell);
        for point in &mesh.points {
            let _ = write!(self.text, " {:.2} {:.2} {:.2}", point[0], point[1], point[2]);
        }
        self.text.push('\n');
    }

    /// Closes the record with how the run ended and gives the text to keep.
    pub fn finish(mut self, outcome: &str) -> String {
        let _ = writeln!(self.text, "outcome = {outcome}");
        self.text
    }
}

const POINTING_HEADER: &str = "eye-tracking-mouse pointing record 1";
/// Frames in one record of pointing: ten minutes at thirty a second.
const POINTING_FRAMES: usize = 18_000;

pub struct PointingRecord {
    text: String,
    frames: usize,
}

impl Default for PointingRecord {
    fn default() -> Self {
        let mut text = format!("{POINTING_HEADER}\n");
        text.push_str("# frame = time, mode, activity, cursor before (x y), what was done (none/move/click x y),\n");
        text.push_str("#   head yaw pitch roll, head place in the image (x y) and size, iris (x y), lid,\n");
        text.push_str("#   gaze estimate (x y), the eyes' filtered place (x y), the head's nudge (x y); nan = not there\n");
        PointingRecord { text, frames: 0 }
    }
}

impl PointingRecord {
    /// Adds a frame of pointing: where the cursor was, and what the engine answered.
    pub fn add(&mut self, time: f64, pointing: PointingMode, cursor: Option<(i32, i32)>, output: &Output) {
        let Mode::Pointing(activity) = output.mode else { return };
        let activity = match activity {
            Activity::Tracking => "tracking",
            Activity::NoFace => "noface",
            Activity::Yielded => "yielded",
            Activity::Blocked => "blocked",
        };
        let (cx, cy) = cursor.unwrap_or((-1, -1));
        let (done, x, y) = match output.pointer {
            None => ("none", -1, -1),
            Some(PointerAction::Move(x, y)) => ("move", x, y),
            Some(PointerAction::Click(x, y)) => ("click", x, y),
        };
        let _ = write!(self.text, "frame = {time:.4} {} {activity} {cx} {cy} {done} {x} {y}", pointing_word(pointing));
        let mut numbers = [f32::NAN; 15];
        if let Some(trace) = &output.trace {
            let (pose, eyes, makeup) = (&trace.face.pose, &trace.face.eyes, &trace.makeup);
            let pair = |place: Option<Vec2>| place.map_or([f32::NAN; 2], |place| [place.x, place.y]);
            let (estimate, carried) = (pair(makeup.estimate), pair(makeup.eyes));
            numbers = [
                pose.yaw, pose.pitch, pose.roll, pose.position.x, pose.position.y, pose.scale,
                eyes.iris.x, eyes.iris.y, eyes.lid,
                estimate[0], estimate[1], carried[0], carried[1], makeup.nudge.x, makeup.nudge.y,
            ];
        }
        for number in numbers {
            let _ = write!(self.text, " {number:.4}");
        }
        self.text.push('\n');
        self.frames += 1;
    }

    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// True when the record should be put away and a new one begun.
    pub fn is_full(&self) -> bool {
        self.frames >= POINTING_FRAMES
    }

    pub fn finish(self) -> String {
        self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::controller::Makeup;
    use crate::vision::FaceMeasurement;
    use crate::vision::eyes::EyeSignals;
    use crate::vision::pose::HeadPose;

    use super::super::engine::Trace;

    #[test]
    fn a_pointing_record_has_a_line_per_frame_of_pointing_and_none_for_a_calibration() {
        let mut record = PointingRecord::default();
        assert!(record.is_empty() && !record.is_full());
        let face = FaceMeasurement {
            time: 3.5,
            pose: HeadPose { yaw: 0.1, pitch: -0.05, roll: 0.0, position: Vec2::new(640.0, 360.5), scale: 250.0 },
            eyes: EyeSignals { iris: Vec2::new(0.02, -0.03), lid: -0.15, openness: [0.3; 2] },
        };
        let makeup = Makeup { estimate: Some(Vec2::new(1700.5, 420.0)), eyes: None, nudge: Vec2::new(-2.5, 0.25) };
        let output = Output {
            trace: Some(Trace { face, makeup }),
            pointer: Some(PointerAction::Move(1698, 420)),
            sound: None,
            mode: Mode::Pointing(Activity::Tracking),
            finished: None,
        };
        record.add(3.5, PointingMode::Hybrid, Some((1700, 420)), &output);
        record.add(3.6, PointingMode::Eye, None, &Output { trace: None, pointer: None, mode: Mode::Pointing(Activity::NoFace), ..output });
        // A frame of a calibration run is not pointing.
        let view = View { phase: Phase::Rest, target: (0, 0), dwell: 0.0, progress: 0.0, face_visible: true, answer: None, score: None };
        record.add(3.7, PointingMode::Eye, None, &Output { mode: Mode::Calibrating(view), ..output });
        assert!(!record.is_empty());
        let text = record.finish();
        let lines: Vec<&str> = text.lines().filter(|line| !line.starts_with('#')).collect();
        assert_eq!(
            lines,
            [
                POINTING_HEADER,
                "frame = 3.5000 hybrid tracking 1700 420 move 1698 420 0.1000 -0.0500 0.0000 640.0000 360.5000 250.0000 0.0200 -0.0300 -0.1500 1700.5000 420.0000 NaN NaN -2.5000 0.2500",
                "frame = 3.6000 eye noface -1 -1 none -1 -1 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN",
            ]
        );
    }

    #[test]
    fn a_record_has_a_line_per_frame_with_what_was_shown_and_every_point() {
        let mut record = CalibrationRecord::begin(RectI::new(0, 0, 1920, 1080), (1280, 720));
        let view = View { phase: Phase::Places, target: (154, 86), dwell: 0.25, progress: 0.3, face_visible: true, answer: None, score: None };
        let mesh = FaceMesh { points: vec![[1.0, 2.5, -3.126]; 4], presence: 1.0, time: 12.5 };
        record.add(&view, &mesh);
        record.add(&View { phase: Phase::Sweep, ..view }, &mesh);
        let text = record.finish("done 48.6");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[..3], [HEADER, "area = 0 0 1920 1080", "camera = 1280 720"]);
        assert!(lines[3].starts_with('#'));
        assert_eq!(lines[4], "frame = 12.5000 places 154 86 0.250 1.00 2.50 -3.13 1.00 2.50 -3.13 1.00 2.50 -3.13 1.00 2.50 -3.13");
        assert!(lines[5].starts_with("frame = 12.5000 sweep 154 86 "));
        assert_eq!(lines[6..], ["outcome = done 48.6"]);
    }
}
