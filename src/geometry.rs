//! Small math helpers shared by `vision`, `control` and `platform`.

use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn dot(self, o: Vec2) -> f32 {
        self.x * o.x + self.y * o.y
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn distance(self, o: Vec2) -> f32 {
        (self - o).length()
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        Vec2::new(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, k: f32) -> Vec2 {
        Vec2::new(self.x * k, self.y * k)
    }
}

/// An integer rectangle in desktop pixels; `right` and `bottom` are exclusive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RectI {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

impl RectI {
    pub const fn new(left: i32, top: i32, right: i32, bottom: i32) -> Self {
        Self { left, top, right, bottom }
    }

    pub fn width(&self) -> i32 {
        self.right - self.left
    }

    pub fn height(&self) -> i32 {
        self.bottom - self.top
    }

    pub fn is_empty(&self) -> bool {
        self.width() <= 0 || self.height() <= 0
    }

    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    /// Smallest rectangle containing both.
    pub fn union(&self, o: &RectI) -> RectI {
        RectI::new(
            self.left.min(o.left),
            self.top.min(o.top),
            self.right.max(o.right),
            self.bottom.max(o.bottom),
        )
    }
}

/// The pixel inside any of `rects` that is nearest to (`x`, `y`).
///
/// A desktop made of several monitors is not a rectangle, so clamping to the
/// bounding box could leave the pointer in a region that has no screen.
/// Returns `None` for a non-finite point or when there is no usable rectangle.
pub fn clamp_to_union(rects: &[RectI], x: f64, y: f64) -> Option<(i32, i32)> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let mut best: Option<((i32, i32), f64)> = None;
    for r in rects.iter().filter(|r| !r.is_empty()) {
        let cx = x.round().clamp(r.left as f64, (r.right - 1) as f64);
        let cy = y.round().clamp(r.top as f64, (r.bottom - 1) as f64);
        let d = (cx - x).powi(2) + (cy - y).powi(2);
        if best.is_none_or(|(_, best_d)| d < best_d) {
            best = Some(((cx as i32, cy as i32), d));
        }
    }
    best.map(|(p, _)| p)
}

/// Solves `A x = b` for a symmetric positive definite `A` (row-major `n` x `n`)
/// by Cholesky factorisation. Returns `None` when `A` is not positive definite.
pub fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), n * n, "matrix must be n x n");
    assert_eq!(b.len(), n, "right-hand side must have n entries");
    // Lower-triangular factor L with A = L * L^T.
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    // Forward substitution: L y = b.
    let mut x = b.to_vec();
    for i in 0..n {
        for k in 0..i {
            x[i] -= l[i * n + k] * x[k];
        }
        x[i] /= l[i * n + i];
    }
    // Back substitution: L^T x = y.
    for i in (0..n).rev() {
        for k in i + 1..n {
            x[i] -= l[k * n + i] * x[k];
        }
        x[i] /= l[i * n + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd_recovers_a_known_solution() {
        // A = M^T M + I for a fixed M, so it is symmetric positive definite.
        let a = [6.0, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 4.0];
        let expected = [1.5, -2.0, 0.25];
        let b: Vec<f64> = (0..3).map(|i| (0..3).map(|j| a[i * 3 + j] * expected[j]).sum()).collect();
        let x = solve_spd(&a, &b, 3).expect("positive definite");
        for (got, want) in x.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn solve_spd_rejects_an_indefinite_matrix() {
        assert!(solve_spd(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0], 2).is_none());
        assert!(solve_spd(&[0.0], &[1.0], 1).is_none());
    }

    // The developer's own layout: an ultrawide above a narrower second screen.
    const TWO_SCREENS: [RectI; 2] =
        [RectI::new(0, 0, 2560, 1080), RectI::new(0, 1080, 1500, 1980)];

    #[test]
    fn clamp_keeps_points_that_are_already_on_a_screen() {
        assert_eq!(clamp_to_union(&TWO_SCREENS, 100.4, 200.6), Some((100, 201)));
        assert_eq!(clamp_to_union(&TWO_SCREENS, 1499.0, 1500.0), Some((1499, 1500)));
    }

    #[test]
    fn clamp_never_lands_in_the_gap_beside_the_smaller_screen() {
        // (2000, 1500) is inside the bounding box but on no monitor.
        let (x, y) = clamp_to_union(&TWO_SCREENS, 2000.0, 1500.0).unwrap();
        assert!(TWO_SCREENS.iter().any(|r| r.contains(x, y)), "({x}, {y}) is off-screen");
        assert_eq!((x, y), (2000, 1079));
    }

    #[test]
    fn clamp_pulls_far_points_to_the_nearest_edge_pixel() {
        assert_eq!(clamp_to_union(&TWO_SCREENS, -50.0, -50.0), Some((0, 0)));
        assert_eq!(clamp_to_union(&TWO_SCREENS, 9999.0, 500.0), Some((2559, 500)));
        assert_eq!(clamp_to_union(&TWO_SCREENS, 700.0, 9999.0), Some((700, 1979)));
    }

    #[test]
    fn clamp_refuses_garbage_input() {
        assert_eq!(clamp_to_union(&TWO_SCREENS, f64::NAN, 0.0), None);
        assert_eq!(clamp_to_union(&TWO_SCREENS, 0.0, f64::INFINITY), None);
        assert_eq!(clamp_to_union(&[], 0.0, 0.0), None);
        assert_eq!(clamp_to_union(&[RectI::new(5, 5, 5, 9)], 0.0, 0.0), None);
    }

    #[test]
    fn rect_union_and_contains_agree_on_exclusive_edges() {
        let u = TWO_SCREENS[0].union(&TWO_SCREENS[1]);
        assert_eq!(u, RectI::new(0, 0, 2560, 1980));
        assert!(u.contains(2559, 1979));
        assert!(!u.contains(2560, 0));
    }
}
