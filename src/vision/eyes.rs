//! What the eyes are doing, measured in each eye's own frame.
//!
//! Every length is divided by that eye's width (corner to corner) and taken
//! along or across the corner line, so the numbers do not change when the head
//! moves closer, shifts or rolls.

use crate::geometry::Vec2;

use super::landmarks::{EyeIndices, LEFT_EYE, Point3, RIGHT_EYE, xy};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeSignals {
    /// Iris centre relative to the middle of the eye: x toward the image's
    /// right along the corner line, y downward across it. Mean of both eyes.
    pub iris: Vec2,
    /// Upper lid relative to the corner line (negative is above it). It drops
    /// when the gaze drops, so it helps with vertical gaze. Mean of both eyes.
    pub lid: f32,
    /// Lid gap over eye width for the right and the left eye: about 0.3 open, near 0 shut.
    pub openness: [f32; 2],
}

struct OneEye {
    iris: Vec2,
    lid: f32,
    openness: f32,
}

fn measure_eye(points: &[Point3], eye: &EyeIndices) -> Option<OneEye> {
    let (first, second) = (xy(points, eye.corners.0), xy(points, eye.corners.1));
    let width = first.distance(second);
    if !(width.is_finite() && width > 1e-3) {
        return None;
    }
    let along = (second - first) * (1.0 / width);
    let across = Vec2::new(-along.y, along.x);
    let middle = (first + second) * 0.5;
    let relative = |index: usize| {
        let offset = xy(points, index) - middle;
        Vec2::new(offset.dot(along), offset.dot(across)) * (1.0 / width)
    };
    let gaps: f32 = eye.lid_pairs.iter().map(|&(upper, lower)| xy(points, upper).distance(xy(points, lower))).sum();
    let measured = OneEye { iris: relative(eye.iris), lid: relative(eye.upper_lid).y, openness: gaps / (2.0 * width) };
    (measured.iris.is_finite() && measured.lid.is_finite() && measured.openness.is_finite()).then_some(measured)
}

pub fn measure(points: &[Point3]) -> Option<EyeSignals> {
    let right = measure_eye(points, &RIGHT_EYE)?;
    let left = measure_eye(points, &LEFT_EYE)?;
    Some(EyeSignals {
        iris: (right.iris + left.iris) * 0.5,
        lid: (right.lid + left.lid) * 0.5,
        openness: [right.openness, left.openness],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::landmarks::LANDMARK_COUNT;
    use crate::vision::synthetic::{SyntheticHead, View};

    fn signals(view: &View, target: Option<[f32; 3]>, openness: f32) -> EyeSignals {
        measure(&SyntheticHead::new().render(view, target, openness)).unwrap()
    }

    /// A target 6.7 eye spans (60 cm) in front of the face, offset sideways and vertically.
    fn target(x: f32, y: f32) -> Option<[f32; 3]> {
        Some([x, y, -6.7])
    }

    #[test]
    fn the_iris_follows_the_gaze_right_and_down() {
        let centre = signals(&View::default(), target(0.0, 0.0), 1.0);
        let right = signals(&View::default(), target(2.0, 0.0), 1.0);
        let down = signals(&View::default(), target(0.0, 1.5), 1.0);
        assert!(centre.iris.x.abs() < 0.01, "{centre:?}");
        assert!(right.iris.x > centre.iris.x + 0.08 && (right.iris.y - centre.iris.y).abs() < 0.01, "{right:?}");
        assert!(down.iris.y > centre.iris.y + 0.05 && (down.iris.x - centre.iris.x).abs() < 0.01, "{down:?}");
        assert!(down.lid > centre.lid + 0.02, "the upper lid follows the gaze down");
    }

    #[test]
    fn distance_position_and_roll_do_not_change_the_signals() {
        let plain = signals(&View::default(), None, 1.0);
        let moved = signals(&View { roll: 0.5, scale: 120.0, position: Vec2::new(200.0, 500.0), ..View::default() }, None, 1.0);
        assert!(plain.iris.distance(moved.iris) < 1e-4 && (plain.lid - moved.lid).abs() < 1e-4);
        assert!((plain.openness[0] - moved.openness[0]).abs() < 1e-4);
    }

    #[test]
    fn openness_is_about_a_third_open_and_near_zero_shut() {
        let open = signals(&View::default(), None, 1.0);
        let shut = signals(&View::default(), None, 0.0);
        let half = signals(&View::default(), None, 0.5);
        for eye in 0..2 {
            assert!((0.2..0.4).contains(&open.openness[eye]), "{open:?}");
            assert!(shut.openness[eye] < 0.01, "{shut:?}");
            assert!((half.openness[eye] / open.openness[eye] - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn a_collapsed_eye_gives_no_measurement() {
        assert!(measure(&vec![[10.0, 10.0, 0.0]; LANDMARK_COUNT]).is_none());
    }
}
