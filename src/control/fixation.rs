//! Notices when the gaze settles on a new place.
//!
//! Per-frame gaze estimates from a webcam scatter by tens of pixels, so a
//! single sample says little. The detector keeps the last `WINDOW` seconds:
//! while the eye is travelling the samples are spread out; once they bunch up
//! the gaze is resting, and their mean is a usable estimate of where. A rest
//! is reported once, and only if it is far from the previous one (the caller
//! says how far), so reading along a line or glancing nearby reports nothing.

use std::collections::VecDeque;

use crate::geometry::Vec2;

/// Seconds of samples that must agree.
const WINDOW: f64 = 0.2;
const MIN_SAMPLES: usize = 5;
/// RMS distance from their mean (pixels) below which the samples count as one resting place.
const MAX_SPREAD: f32 = 110.0;

#[derive(Clone, Debug, Default)]
pub struct FixationDetector {
    samples: VecDeque<(f64, Vec2)>,
    place: Option<Vec2>,
}

impl FixationDetector {
    pub fn reset(&mut self) {
        self.samples.clear();
        self.place = None;
    }

    /// Feeds one gaze estimate (screen pixels). Returns the centre of a new
    /// resting place the first time the gaze is found there; a rest closer
    /// than `apart` pixels to the previous one is still the same place.
    pub fn step(&mut self, time: f64, gaze: Vec2, apart: f32) -> Option<Vec2> {
        if !gaze.is_finite() {
            return None;
        }
        // Out-of-order or long-interrupted input cannot be compared with what came before.
        if self.samples.back().is_some_and(|&(last, _)| time <= last || time - last > WINDOW) {
            self.samples.clear();
        }
        self.samples.push_back((time, gaze));
        while self.samples.front().is_some_and(|&(first, _)| time - first > WINDOW) {
            self.samples.pop_front();
        }
        let covered = time - self.samples.front().map_or(time, |&(first, _)| first);
        if self.samples.len() < MIN_SAMPLES || covered < WINDOW * 0.7 {
            return None;
        }
        let count = self.samples.len() as f32;
        let mean = self.samples.iter().fold(Vec2::ZERO, |sum, &(_, p)| sum + p) * (1.0 / count);
        let spread = (self.samples.iter().map(|&(_, p)| (p - mean).dot(p - mean)).sum::<f32>() / count).sqrt();
        if spread > MAX_SPREAD {
            return None;
        }
        match self.place {
            Some(place) if place.distance(mean) < apart => {
                // Follow slow drift so a long stay does not creep into a "new" place.
                self.place = Some(place + (mean - place) * 0.1);
                None
            }
            _ => {
                self.place = Some(mean);
                Some(mean)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rig {
        detector: FixationDetector,
        frame: u32,
        noise: u32,
        found: Vec<Vec2>,
    }

    impl Rig {
        fn new() -> Rig {
            Rig { detector: FixationDetector::default(), frame: 0, noise: 3, found: Vec::new() }
        }

        /// Looks at `target` for `frames` frames with uniform noise of the given amplitude.
        fn look(&mut self, target: Vec2, frames: u32, amplitude: f32) -> &mut Rig {
            for _ in 0..frames {
                let mut jitter = || {
                    self.noise = self.noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((self.noise >> 8) as f32 / 8_388_608.0 - 1.0) * amplitude
                };
                let sample = target + Vec2::new(jitter(), jitter());
                self.found.extend(self.detector.step(f64::from(self.frame) / 30.0, sample, 200.0));
                self.frame += 1;
            }
            self
        }
    }

    #[test]
    fn the_first_rest_is_reported_once_near_its_true_place() {
        let mut rig = Rig::new();
        rig.look(Vec2::new(500.0, 400.0), 60, 60.0);
        assert_eq!(rig.found.len(), 1);
        assert!(rig.found[0].distance(Vec2::new(500.0, 400.0)) < 40.0, "{:?}", rig.found[0]);
    }

    #[test]
    fn a_far_jump_is_reported_within_a_third_of_a_second() {
        let mut rig = Rig::new();
        rig.look(Vec2::new(300.0, 300.0), 30, 60.0);
        let before = rig.found.len();
        rig.look(Vec2::new(1800.0, 700.0), 10, 60.0);
        assert_eq!(rig.found.len(), before + 1, "found {:?}", rig.found);
        assert!(rig.found[before].distance(Vec2::new(1800.0, 700.0)) < 45.0);
    }

    #[test]
    fn nearby_glances_and_reading_report_nothing_new() {
        let mut rig = Rig::new();
        rig.look(Vec2::new(800.0, 500.0), 30, 60.0);
        for step in 1..=6 {
            rig.look(Vec2::new(800.0 + step as f32 * 25.0, 500.0), 12, 60.0);
        }
        rig.look(Vec2::new(800.0, 560.0), 30, 60.0);
        assert_eq!(rig.found.len(), 1, "found {:?}", rig.found);
    }

    #[test]
    fn a_wandering_gaze_never_counts_as_resting() {
        let mut rig = Rig::new();
        for frame in 0..90 {
            let sweep = Vec2::new(200.0 + frame as f32 * 50.0 % 1500.0, 300.0 + (frame % 7) as f32 * 90.0);
            rig.look(sweep, 1, 0.0);
        }
        assert!(rig.found.is_empty(), "found {:?}", rig.found);
    }

    #[test]
    fn interruptions_and_garbage_start_the_window_over() {
        let mut rig = Rig::new();
        rig.look(Vec2::new(400.0, 400.0), 4, 0.0);
        rig.frame += 30;
        rig.look(Vec2::new(400.0, 400.0), 3, 0.0);
        assert!(rig.found.is_empty(), "seven samples, but never a full window in a row");
        assert_eq!(rig.detector.step(100.0, Vec2::new(f32::NAN, 0.0), 200.0), None);
        rig.look(Vec2::new(400.0, 400.0), 10, 0.0);
        assert_eq!(rig.found.len(), 1);
        rig.detector.reset();
        rig.look(Vec2::new(400.0, 400.0), 10, 0.0);
        assert_eq!(rig.found.len(), 2, "after a reset the same place is new again");
    }
}
