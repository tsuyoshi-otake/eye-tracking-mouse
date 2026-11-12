//! Head pose from the rigid landmarks.
//!
//! The person's own face shape, captured once (`ReferenceShape`), is fitted to
//! each frame's 2-D landmarks as a scaled, rotated and shifted orthographic
//! projection. Only the image positions of the current frame are used, which
//! are far steadier than the model's depth estimate.
//!
//! Angles are relative to the pose in which the reference was captured:
//! positive yaw turns the nose toward the image's right, positive pitch toward
//! its bottom, positive roll turns the face clockwise on screen.

use crate::geometry::{Vec2, solve_spd};

use super::landmarks::{EYE_SPAN, Point3, RIGID};

/// The rigid landmarks of one person: centred on their mean, eye line along x,
/// in units of the eye span.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceShape {
    points: Vec<[f32; 3]>,
    /// Sum over points of p * p^T, row-major 3x3.
    gram: [f64; 9],
}

impl ReferenceShape {
    /// Accepts stored points; `None` if they cannot support a pose fit.
    pub fn from_points(points: Vec<[f32; 3]>) -> Option<ReferenceShape> {
        if points.len() != RIGID.len() || points.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let mut gram = [0.0f64; 9];
        for p in &points {
            for row in 0..3 {
                for column in 0..3 {
                    gram[row * 3 + column] += f64::from(p[row]) * f64::from(p[column]);
                }
            }
        }
        // A flat or collapsed shape cannot tell rotations apart.
        let depth_spread = gram[8] / points.len() as f64;
        solve_spd(&gram, &[1.0, 1.0, 1.0], 3)?;
        (depth_spread > 1e-4).then_some(ReferenceShape { points, gram })
    }

    pub fn points(&self) -> &[[f32; 3]] {
        &self.points
    }
}

/// Averages several frames of a face held still into a `ReferenceShape`.
#[derive(Clone, Debug, Default)]
pub struct ReferenceBuilder {
    sum: Vec<[f64; 3]>,
    frames: usize,
}

impl ReferenceBuilder {
    pub fn add(&mut self, mesh: &[Point3]) {
        let Some(shape) = normalized_rigid(mesh) else {
            return;
        };
        if self.sum.is_empty() {
            self.sum = vec![[0.0; 3]; RIGID.len()];
        }
        for (total, p) in self.sum.iter_mut().zip(&shape) {
            for axis in 0..3 {
                total[axis] += f64::from(p[axis]);
            }
        }
        self.frames += 1;
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn build(&self) -> Option<ReferenceShape> {
        if self.frames == 0 {
            return None;
        }
        let n = self.frames as f64;
        ReferenceShape::from_points(self.sum.iter().map(|p| p.map(|v| (v / n) as f32)).collect())
    }
}

/// The rigid points centred, with the eye line turned onto the x axis and
/// scaled so the eye span is 1.
fn normalized_rigid(mesh: &[Point3]) -> Option<Vec<[f32; 3]>> {
    let (a, b) = (mesh[EYE_SPAN.0], mesh[EYE_SPAN.1]);
    let span = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
    if !(span.is_finite() && span > 1e-3) {
        return None;
    }
    let roll = (b[1] - a[1]).atan2(b[0] - a[0]);
    let (sin, cos) = roll.sin_cos();
    let mut mean = [0.0f32; 3];
    for &index in &RIGID {
        for axis in 0..3 {
            mean[axis] += mesh[index][axis] / RIGID.len() as f32;
        }
    }
    Some(
        RIGID
            .iter()
            .map(|&index| {
                let d = [mesh[index][0] - mean[0], mesh[index][1] - mean[1], mesh[index][2] - mean[2]];
                [(d[0] * cos + d[1] * sin) / span, (-d[0] * sin + d[1] * cos) / span, d[2] / span]
            })
            .collect(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadPose {
    /// Radians; see the module docs for the signs.
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    /// Mean of the rigid landmarks, in image pixels.
    pub position: Vec2,
    /// Image pixels per eye span: grows as the face comes closer.
    pub scale: f32,
}

impl HeadPose {
    /// Where the nose points, as the image-plane part of a unit vector.
    pub fn forward(&self) -> Vec2 {
        Vec2::new(self.yaw.sin(), self.pitch.sin())
    }
}

/// Fits the reference shape to this frame's landmarks.
pub fn estimate(reference: &ReferenceShape, mesh: &[Point3]) -> Option<HeadPose> {
    let count = RIGID.len() as f32;
    let mut position = Vec2::ZERO;
    for &index in &RIGID {
        position = position + Vec2::new(mesh[index][0], mesh[index][1]) * (1.0 / count);
    }
    // Least squares for the two rows of the 2x3 matrix that maps the shape onto the image.
    let (mut rhs_x, mut rhs_y) = ([0.0f64; 3], [0.0f64; 3]);
    for (p, &index) in reference.points.iter().zip(&RIGID) {
        let (dx, dy) = (f64::from(mesh[index][0] - position.x), f64::from(mesh[index][1] - position.y));
        for axis in 0..3 {
            rhs_x[axis] += f64::from(p[axis]) * dx;
            rhs_y[axis] += f64::from(p[axis]) * dy;
        }
    }
    let row_x = solve_spd(&reference.gram, &rhs_x, 3)?;
    let row_y = solve_spd(&reference.gram, &rhs_y, 3)?;
    let (len_x, len_y) = (norm(&row_x), norm(&row_y));
    // A rigid head projects with the same scale on both axes; anything else is a bad mesh.
    if !(len_x > 1e-6 && len_y > 1e-6 && (0.6..=1.6).contains(&(len_x / len_y))) {
        return None;
    }
    // Make the rows a pair of perpendicular unit vectors, sharing the correction evenly.
    let (ux, uy) = (scaled(&row_x, 1.0 / len_x), scaled(&row_y, 1.0 / len_y));
    let overlap: f64 = (0..3).map(|i| ux[i] * uy[i]).sum();
    let rx: Vec<f64> = (0..3).map(|i| ux[i] - 0.5 * overlap * uy[i]).collect();
    let ry: Vec<f64> = (0..3).map(|i| uy[i] - 0.5 * overlap * ux[i]).collect();
    let (rx, ry) = (scaled(&rx, 1.0 / norm(&rx)), scaled(&ry, 1.0 / norm(&ry)));
    let pose = HeadPose {
        yaw: (-rx[2]).clamp(-1.0, 1.0).asin() as f32,
        pitch: (-ry[2]).clamp(-1.0, 1.0).asin() as f32,
        roll: ry[0].atan2(rx[0]) as f32,
        position,
        scale: ((len_x + len_y) * 0.5) as f32,
    };
    (pose.yaw.is_finite() && pose.pitch.is_finite() && pose.roll.is_finite() && position.is_finite()).then_some(pose)
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn scaled(v: &[f64], k: f64) -> Vec<f64> {
    v.iter().map(|x| x * k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::synthetic::{SyntheticHead, View};

    fn reference(head: &SyntheticHead) -> ReferenceShape {
        let mut builder = ReferenceBuilder::default();
        builder.add(&head.render(&View::default(), None, 1.0));
        builder.build().unwrap()
    }

    fn close(actual: f32, expected: f32, tolerance: f32) -> bool {
        (actual - expected).abs() <= tolerance
    }

    #[test]
    fn the_reference_is_centred_with_a_unit_eye_span_along_x() {
        let head = SyntheticHead::new();
        let view = View { roll: 0.4, scale: 180.0, position: Vec2::new(500.0, 300.0), ..View::default() };
        let mut builder = ReferenceBuilder::default();
        builder.add(&head.render(&view, None, 1.0));
        let shape = builder.build().unwrap();
        let mean: f32 = shape.points().iter().map(|p| p[0] + p[1] + p[2]).sum::<f32>() / 49.0;
        assert!(mean.abs() < 1e-4);
        let at = |index: usize| shape.points()[RIGID.iter().position(|&i| i == index).unwrap()];
        let (a, b) = (at(EYE_SPAN.0), at(EYE_SPAN.1));
        assert!(close(b[0] - a[0], 1.0, 1e-4) && close(b[1] - a[1], 0.0, 1e-4));
    }

    #[test]
    fn the_capture_pose_reads_as_zero() {
        let head = SyntheticHead::new();
        let pose = estimate(&reference(&head), &head.render(&View::default(), None, 1.0)).unwrap();
        assert!(close(pose.yaw, 0.0, 1e-4) && close(pose.pitch, 0.0, 1e-4) && close(pose.roll, 0.0, 1e-4));
        assert!(close(pose.scale, View::default().scale, 0.05));
    }

    #[test]
    fn single_axis_turns_are_recovered_with_the_documented_signs() {
        let head = SyntheticHead::new();
        let shape = reference(&head);
        for angle in [-0.35f32, -0.1, 0.05, 0.3] {
            let yawed = estimate(&shape, &head.render(&View { yaw: angle, ..View::default() }, None, 1.0)).unwrap();
            assert!(close(yawed.yaw, angle, 2e-3) && close(yawed.pitch, 0.0, 2e-3), "{yawed:?}");
            let pitched = estimate(&shape, &head.render(&View { pitch: angle, ..View::default() }, None, 1.0)).unwrap();
            assert!(close(pitched.pitch, angle, 2e-3) && close(pitched.yaw, 0.0, 2e-3), "{pitched:?}");
            let rolled = estimate(&shape, &head.render(&View { roll: angle, ..View::default() }, None, 1.0)).unwrap();
            assert!(close(rolled.roll, angle, 2e-3) && close(rolled.yaw, 0.0, 2e-3), "{rolled:?}");
        }
    }

    #[test]
    fn a_positive_yaw_moves_the_nose_tip_right_and_a_positive_pitch_moves_it_down() {
        let head = SyntheticHead::new();
        let straight = head.render(&View::default(), None, 1.0);
        let yawed = head.render(&View { yaw: 0.2, ..View::default() }, None, 1.0);
        let pitched = head.render(&View { pitch: 0.2, ..View::default() }, None, 1.0);
        assert!(yawed[1][0] > straight[1][0] + 5.0);
        assert!(pitched[1][1] > straight[1][1] + 5.0);
    }

    #[test]
    fn combined_turns_shifts_and_distance_changes_are_separated() {
        let head = SyntheticHead::new();
        let shape = reference(&head);
        let view = View { yaw: 0.2, pitch: -0.15, roll: 0.1, scale: 150.0, position: Vec2::new(400.0, 500.0) };
        let pose = estimate(&shape, &head.render(&view, None, 1.0)).unwrap();
        let r = SyntheticHead::rotation(&view);
        assert!(close(pose.forward().x, -r[0][2], 2e-3) && close(pose.forward().y, -r[1][2], 2e-3), "{pose:?}");
        assert!(close(pose.roll, r[1][0].atan2(r[0][0]), 2e-3));
        assert!(close(pose.scale, 150.0, 0.2));
        // The rigid points are centred in the head frame, so their mean follows the head position.
        assert!(pose.position.distance(view.position) < 0.5);
    }

    #[test]
    fn averaging_frames_cancels_landmark_noise() {
        let head = SyntheticHead::new();
        let clean = head.render(&View::default(), None, 1.0);
        let mut builder = ReferenceBuilder::default();
        let mut state = 7u32;
        for _ in 0..40 {
            let mut noisy = clean.clone();
            for p in &mut noisy {
                for v in p.iter_mut() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    *v += ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0;
                }
            }
            builder.add(&noisy);
        }
        assert_eq!(builder.frames(), 40);
        let averaged = builder.build().unwrap();
        let exact = reference(&head);
        let worst = averaged.points().iter().zip(exact.points()).flat_map(|(a, b)| (0..3).map(move |i| (a[i] - b[i]).abs()));
        assert!(worst.fold(0.0f32, f32::max) < 0.004);
    }

    #[test]
    fn degenerate_shapes_and_meshes_are_refused() {
        assert!(ReferenceShape::from_points(vec![[0.0; 3]; RIGID.len()]).is_none());
        assert!(ReferenceShape::from_points(vec![[1.0, 2.0, 3.0]; 5]).is_none());
        let flat: Vec<[f32; 3]> = (0..RIGID.len()).map(|i| [(i % 7) as f32 - 3.0, (i / 7) as f32 - 3.0, 0.0]).collect();
        assert!(ReferenceShape::from_points(flat).is_none());
        let head = SyntheticHead::new();
        let collapsed = vec![[5.0, 5.0, 0.0]; crate::vision::landmarks::LANDMARK_COUNT];
        assert!(estimate(&reference(&head), &collapsed).is_none());
        assert!(ReferenceBuilder::default().build().is_none());
    }
}
