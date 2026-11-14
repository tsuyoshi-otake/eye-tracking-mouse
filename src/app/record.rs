//! A record of one calibration run, for finding out afterwards why the gaze
//! estimate is as far off as it is.
//!
//! It holds what the tracker measured in every frame of the run next to what
//! the person was shown at that moment: numbers only, never a picture. It is
//! made only when the settings ask for it, and nothing in the program reads
//! it back.

use std::fmt::Write;

use crate::control::calibration::{Phase, View};
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

#[cfg(test)]
mod tests {
    use super::*;

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
