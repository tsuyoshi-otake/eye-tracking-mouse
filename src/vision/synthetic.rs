//! A crude 3-D head for tests: enough anatomy to exercise the pose and gaze maths.
//!
//! Head frame: x toward the image's right, y down, z away from the camera,
//! origin at the mean of the rigid landmarks, lengths in eye spans.

use crate::geometry::Vec2;

use super::landmarks::{EYE_SPAN, EyeIndices, LANDMARK_COUNT, LEFT_EYE, Point3, RIGHT_EYE, RIGID};

const EYEBALL_RADIUS: f32 = 0.133;
const EYE_X: f32 = 0.335;

/// How the camera sees the head.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Image pixels per eye span.
    pub scale: f32,
    /// Image position of the head origin.
    pub position: Vec2,
}

impl Default for View {
    fn default() -> Self {
        View { yaw: 0.0, pitch: 0.0, roll: 0.0, scale: 216.0, position: Vec2::new(640.0, 360.0) }
    }
}

pub struct SyntheticHead {
    points: Vec<[f32; 3]>,
    /// Eyeball centres: right eye, left eye.
    eyeballs: [[f32; 3]; 2],
}

fn surface(x: f32, y: f32) -> f32 {
    0.45 * x * x + 0.25 * y * y
}

impl SyntheticHead {
    pub fn new() -> Self {
        let mut state = 12345u32;
        let mut random = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0
        };
        let mut points: Vec<[f32; 3]> = (0..LANDMARK_COUNT)
            .map(|_| {
                let (x, y) = (random() * 0.75, random() * 0.9);
                [x, y, surface(x, y)]
            })
            .collect();
        // The nose stands out toward the camera.
        for (index, y, z) in [(168, -0.18, -0.03), (6, -0.1, -0.05), (197, -0.02, -0.08), (195, 0.05, -0.12), (5, 0.15, -0.2), (4, 0.25, -0.28), (1, 0.3, -0.3)] {
            points[index] = [0.0, y, z];
        }
        for (eye, side) in [(&RIGHT_EYE, -1.0f32), (&LEFT_EYE, 1.0)] {
            let outer = if side < 0.0 { eye.corners.0 } else { eye.corners.1 };
            let inner = if side < 0.0 { eye.corners.1 } else { eye.corners.0 };
            points[outer] = [side * 0.5, 0.0, 0.1];
            points[inner] = [side * 0.17, 0.0, 0.1];
        }
        let mut head = SyntheticHead { points, eyeballs: [[-EYE_X, 0.0, 0.05 + EYEBALL_RADIUS], [EYE_X, 0.0, 0.05 + EYEBALL_RADIUS]] };
        // Put the origin at the mean of the rigid points.
        let mut mean = [0.0f32; 3];
        for &index in &RIGID {
            for (total, value) in mean.iter_mut().zip(head.points[index]) {
                *total += value / RIGID.len() as f32;
            }
        }
        for p in head.points.iter_mut().chain(head.eyeballs.iter_mut()) {
            for (value, offset) in p.iter_mut().zip(mean) {
                *value -= offset;
            }
        }
        debug_assert!((head.points[EYE_SPAN.1][0] - head.points[EYE_SPAN.0][0] - 1.0).abs() < 1e-6);
        head
    }

    /// Head-to-camera rotation: yaw about the vertical axis, then pitch, then roll.
    pub fn rotation(view: &View) -> [[f32; 3]; 3] {
        let (sy, cy) = view.yaw.sin_cos();
        let (sp, cp) = view.pitch.sin_cos();
        let (sr, cr) = view.roll.sin_cos();
        let tilted = [[cy, 0.0, -sy], [-sp * sy, cp, -sp * cy], [cp * sy, sp, cp * cy]];
        [
            [cr * tilted[0][0] - sr * tilted[1][0], cr * tilted[0][1] - sr * tilted[1][1], cr * tilted[0][2] - sr * tilted[1][2]],
            [sr * tilted[0][0] + cr * tilted[1][0], sr * tilted[0][1] + cr * tilted[1][1], sr * tilted[0][2] + cr * tilted[1][2]],
            tilted[2],
        ]
    }

    /// The mesh the landmark model would report.
    ///
    /// `target` is the point both eyes look at, relative to the head origin
    /// along the camera's axes, in eye spans; `None` looks straight out of the
    /// face. `openness` scales the lid gap: 1 is open, 0 is shut.
    pub fn render(&self, view: &View, target: Option<[f32; 3]>, openness: f32) -> Vec<Point3> {
        let r = Self::rotation(view);
        let rotate = |p: [f32; 3]| [0, 1, 2].map(|row| r[row][0] * p[0] + r[row][1] * p[1] + r[row][2] * p[2]);
        let unrotate = |p: [f32; 3]| [0, 1, 2].map(|column| r[0][column] * p[0] + r[1][column] * p[1] + r[2][column] * p[2]);
        let mut head_points = self.points.clone();
        for (eye, centre) in [(&RIGHT_EYE, self.eyeballs[0]), (&LEFT_EYE, self.eyeballs[1])] {
            let gaze = match target {
                Some(target) => {
                    let eye_position = rotate(centre);
                    let to_target = [target[0] - eye_position[0], target[1] - eye_position[1], target[2] - eye_position[2]];
                    let length = to_target.iter().map(|v| v * v).sum::<f32>().sqrt();
                    unrotate(to_target.map(|v| v / length))
                }
                None => [0.0, 0.0, -1.0],
            };
            place_eye(&mut head_points, eye, centre, gaze, openness);
        }
        head_points
            .iter()
            .map(|&p| {
                let q = rotate(p);
                [view.position.x + view.scale * q[0], view.position.y + view.scale * q[1], view.scale * q[2]]
            })
            .collect()
    }
}

/// Iris on the eyeball where the gaze leaves it; lids follow the iris a little and close with `openness`.
fn place_eye(points: &mut [[f32; 3]], eye: &EyeIndices, centre: [f32; 3], gaze: [f32; 3], openness: f32) {
    let iris = [centre[0] + EYEBALL_RADIUS * gaze[0], centre[1] + EYEBALL_RADIUS * gaze[1], centre[2] + EYEBALL_RADIUS * gaze[2]];
    points[eye.iris] = iris;
    let surface_z = centre[2] - EYEBALL_RADIUS;
    let lower = centre[1] + 0.05 + 0.2 * EYEBALL_RADIUS * gaze[1];
    let upper_open = centre[1] - 0.05 + 0.6 * EYEBALL_RADIUS * gaze[1];
    let upper = lower - openness * (lower - upper_open);
    points[eye.upper_lid] = [centre[0], upper, surface_z];
    points[eye.lower_lid] = [centre[0], lower, surface_z];
    for (pair, offset) in eye.lid_pairs.iter().zip([-0.055f32, 0.055]) {
        points[pair.0] = [centre[0] + offset, lower - 0.9 * (lower - upper), surface_z];
        points[pair.1] = [centre[0] + offset, lower, surface_z];
    }
}
