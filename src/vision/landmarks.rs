//! The face landmark model's contract: its tensors and what its points mean.
//!
//! The model is MediaPipe Face Mesh V2. It takes a 256x256 crop of a face
//! (RGB, values 0..1) and returns 478 points; "left" and "right" below are the
//! person's own, so the right eye is the one nearer the left edge of the image.

use crate::geometry::Vec2;

pub const MODEL_ONNX: &[u8] = include_bytes!("../../assets/face_landmarks.onnx");
pub const INPUT_NAME: &str = "image";
pub const INPUT_SIDE: usize = 256;
pub const INPUT_SHAPE: [i64; 4] = [1, 3, INPUT_SIDE as i64, INPUT_SIDE as i64];
pub const INPUT_LEN: usize = 3 * INPUT_SIDE * INPUT_SIDE;
pub const OUTPUT_LANDMARKS: &str = "landmarks";
pub const OUTPUT_FLAG: &str = "face_flag";
pub const LANDMARK_COUNT: usize = 478;
/// The model's own threshold for "this crop contains a face".
pub const PRESENCE_THRESHOLD: f32 = 0.5;

/// x right, y down, z away from the camera, all in the same unit.
pub type Point3 = [f32; 3];

/// One face in camera-image pixels.
#[derive(Clone, Debug, Default)]
pub struct FaceMesh {
    pub points: Vec<Point3>,
    /// Probability that the tracked crop really shows a face.
    pub presence: f32,
    /// Seconds, on the clock of the frame it came from.
    pub time: f64,
}

pub fn xy(points: &[Point3], index: usize) -> Vec2 {
    Vec2::new(points[index][0], points[index][1])
}

/// Landmarks of one eye, ordered so `corners.0` is nearer the image's left edge.
pub struct EyeIndices {
    pub corners: (usize, usize),
    pub iris: usize,
    pub upper_lid: usize,
    pub lower_lid: usize,
    /// Two upper-lid points and the two lower-lid points below them.
    pub lid_pairs: [(usize, usize); 2],
}

pub const RIGHT_EYE: EyeIndices = EyeIndices {
    corners: (33, 133),
    iris: 468,
    upper_lid: 159,
    lower_lid: 145,
    lid_pairs: [(160, 144), (158, 153)],
};

pub const LEFT_EYE: EyeIndices = EyeIndices {
    corners: (362, 263),
    iris: 473,
    upper_lid: 386,
    lower_lid: 374,
    lid_pairs: [(385, 380), (387, 373)],
};

/// Outer eye corners. Their distance, the "eye span" (about 9 cm), is the unit of head-relative lengths.
pub const EYE_SPAN: (usize, usize) = (33, 263);

/// Points that barely move with expression: nose bridge and sides, eye
/// corners, forehead, temples and cheekbones. Head pose is fitted to these.
pub const RIGID: [usize; 49] = [
    6, 168, 197, 195, 5, 4, 1, // nose bridge to tip
    98, 327, 129, 358, // nose sides
    33, 133, 362, 263, // eye corners
    10, 151, 9, 8, 109, 338, 67, 297, 69, 299, 108, 337, // forehead
    127, 356, 234, 454, 93, 323, 162, 389, 21, 251, 54, 284, // temples
    116, 345, 123, 352, 50, 280, 117, 346, 118, 347, // cheekbones
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_tables_stay_inside_the_mesh_and_do_not_repeat() {
        let mut seen = std::collections::HashSet::new();
        for index in RIGID {
            assert!(index < LANDMARK_COUNT);
            assert!(seen.insert(index), "rigid index {index} is listed twice");
        }
        for eye in [&RIGHT_EYE, &LEFT_EYE] {
            let lids = eye.lid_pairs.iter().flat_map(|&(a, b)| [a, b]);
            for index in [eye.corners.0, eye.corners.1, eye.iris, eye.upper_lid, eye.lower_lid].into_iter().chain(lids) {
                assert!(index < LANDMARK_COUNT);
            }
        }
        assert!(RIGID.contains(&EYE_SPAN.0) && RIGID.contains(&EYE_SPAN.1));
    }

    #[test]
    fn the_embedded_model_is_an_onnx_file_of_the_expected_size() {
        assert_eq!(MODEL_ONNX.len(), 4_816_578);
        assert_eq!(INPUT_LEN, 196_608);
    }
}
