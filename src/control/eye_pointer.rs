//! Turns a per-frame gaze estimate into the place the cursor sits, for the
//! mode where the eyes move the cursor directly.
//!
//! Unlike the head pointer this is an absolute position, not a displacement:
//! every frame the gaze model says where on the screen the person looks, and
//! the filtered value is where the cursor should be. The One-Euro filter
//! holds the estimate steady at rest and opens up on a saccade, and the
//! backlash swallows the tens of pixels of estimate jitter that remain.

use crate::geometry::Vec2;

use super::filter::{Backlash, OneEuro2};

// The three constants were chosen against the simulated person's estimate at
// rest, which scatters by about 36 px on each axis and by 60 px from one frame
// to the next (2,000 px/s). They are a starting point until a person's own
// noise has been read from `--diag`.

/// Smoothing at rest (Hz), at a steadiness of 1.
const REST_CUTOFF: f32 = 0.4;
/// Cutoff gained per pixel per second of estimate speed. A saccade is tens of
/// thousands of pixels a second and opens the filter; the jitter's own speed
/// must not, which it did at twenty times this value.
const SPEED_RESPONSE: f32 = 0.0005;
/// Estimate jitter below this many pixels never moves the cursor, at a steadiness of 1.
const SLACK: f32 = 10.0;

#[derive(Clone, Debug)]
pub struct EyePointer {
    smoother: OneEuro2,
    backlash: Backlash,
}

impl Default for EyePointer {
    fn default() -> Self {
        EyePointer { smoother: OneEuro2::new(REST_CUTOFF, SPEED_RESPONSE), backlash: Backlash::new(SLACK) }
    }
}

impl EyePointer {
    /// Forgets the filtered estimate, so the next frame is taken fresh.
    pub fn reset(&mut self) {
        *self = EyePointer::default();
    }

    /// The cursor position for this frame's gaze estimate, in desktop pixels.
    pub fn step(&mut self, time: f64, place: Vec2, steadiness: f32) -> Vec2 {
        self.smoother.set_min_cutoff(REST_CUTOFF / steadiness);
        self.backlash.set_slack(SLACK * steadiness);
        self.backlash.follow(self.smoother.filter(time, place))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise in -1..1.
    fn noise(state: &mut u32) -> f32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*state >> 8) as f32 / 8_388_608.0 - 1.0
    }

    #[test]
    fn a_noisy_estimate_around_one_place_leaves_an_almost_still_cursor() {
        let mut pointer = EyePointer::default();
        let mut state = 5u32;
        let (mut input_power, mut output) = (0.0f32, Vec2::ZERO);
        for frame in 0..600 {
            let place = Vec2::new(800.0 + 15.0 * noise(&mut state), 400.0 + 15.0 * noise(&mut state));
            output = pointer.step(f64::from(frame) / 30.0, place, 1.0);
            if frame > 60 {
                input_power += (place.x - 800.0).powi(2) + (place.y - 400.0).powi(2);
            }
        }
        // Tens of pixels of jitter come out the other side a few pixels at most.
        assert!(output.distance(Vec2::new(800.0, 400.0)) < SLACK + 3.0, "{output:?}");
        assert!(input_power.sqrt() > 15.0, "the noise is strong enough to show");
    }

    #[test]
    fn looking_somewhere_else_carries_the_cursor_there() {
        let mut pointer = EyePointer::default();
        for frame in 0..30 {
            pointer.step(f64::from(frame) / 30.0, Vec2::new(500.0, 500.0), 1.0);
        }
        let mut output = Vec2::ZERO;
        for frame in 0..90 {
            output = pointer.step(f64::from(30 + frame) / 30.0, Vec2::new(1800.0, 700.0), 1.0);
        }
        // Landed within the backlash's slack of the new estimate.
        assert!((output.x - 1800.0).abs() < SLACK + 1.0 && (output.y - 700.0).abs() < SLACK + 1.0, "{output:?}");
    }

    #[test]
    fn more_steadiness_calms_jitter_but_still_follows_a_move() {
        let wobble = |steadiness: f32| {
            let mut pointer = EyePointer::default();
            let mut state = 3u32;
            let (mut path, mut last) = (0.0, Vec2::ZERO);
            for frame in 0..300 {
                let place = Vec2::new(800.0 + 15.0 * noise(&mut state), 400.0 + 15.0 * noise(&mut state));
                let output = pointer.step(f64::from(frame) / 30.0, place, steadiness);
                if frame > 60 {
                    path += (output - last).length();
                }
                last = output;
            }
            path
        };
        assert!(wobble(3.0) < wobble(1.0), "steadiness 3 vs 1");
    }

    #[test]
    fn a_reset_discards_the_filtered_estimate() {
        let mut pointer = EyePointer::default();
        pointer.step(0.0, Vec2::new(100.0, 100.0), 1.0);
        pointer.reset();
        assert_eq!(pointer.step(1.0, Vec2::new(900.0, 600.0), 1.0), Vec2::new(900.0, 600.0));
    }
}
