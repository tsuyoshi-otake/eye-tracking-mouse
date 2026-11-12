//! Smoothing for noisy measurements.

use crate::geometry::Vec2;

/// A gap this long between samples means the signal was interrupted: start over.
const MAX_GAP: f64 = 0.5;
/// Cutoff (Hz) of the low-pass on the speed estimate.
const RATE_CUTOFF: f32 = 1.0;

/// One-Euro filter (Casiez, Roussel and Vogel, 2012): a low-pass whose cutoff
/// rises with speed, so a resting signal is steady and a moving one barely lags.
#[derive(Clone, Debug)]
pub struct OneEuro {
    /// Cutoff (Hz) at rest: lower is steadier.
    min_cutoff: f32,
    /// Cutoff gained per unit of speed: higher follows fast movement more closely.
    beta: f32,
    /// Time, filtered value and filtered speed of the previous sample.
    previous: Option<(f64, f32, f32)>,
}

fn smoothing(dt: f32, cutoff: f32) -> f32 {
    let tau = 1.0 / (2.0 * std::f32::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuro {
    pub const fn new(min_cutoff: f32, beta: f32) -> Self {
        OneEuro { min_cutoff, beta, previous: None }
    }

    pub fn filter(&mut self, time: f64, value: f32) -> f32 {
        let filtered = match self.previous {
            Some((last_time, last, last_rate)) if time > last_time && time - last_time <= MAX_GAP => {
                let dt = (time - last_time) as f32;
                let rate = last_rate + smoothing(dt, RATE_CUTOFF) * ((value - last) / dt - last_rate);
                let cutoff = self.min_cutoff + self.beta * rate.abs();
                (last + smoothing(dt, cutoff) * (value - last), rate)
            }
            // A repeated timestamp adds no information.
            Some((last_time, last, last_rate)) if time == last_time => (last, last_rate),
            _ => (value, 0.0),
        };
        self.previous = Some((time, filtered.0, filtered.1));
        filtered.0
    }
}

/// A pair of One-Euro filters for a point.
#[derive(Clone, Debug)]
pub struct OneEuro2 {
    x: OneEuro,
    y: OneEuro,
}

impl OneEuro2 {
    pub const fn new(min_cutoff: f32, beta: f32) -> Self {
        OneEuro2 { x: OneEuro::new(min_cutoff, beta), y: OneEuro::new(min_cutoff, beta) }
    }

    /// Changes the cutoff at rest; what was filtered so far is kept.
    pub fn set_min_cutoff(&mut self, min_cutoff: f32) {
        self.x.min_cutoff = min_cutoff;
        self.y.min_cutoff = min_cutoff;
    }

    pub fn filter(&mut self, time: f64, value: Vec2) -> Vec2 {
        Vec2::new(self.x.filter(time, value.x), self.y.filter(time, value.y))
    }
}

/// Mechanical backlash: the output follows the input only once the input has
/// moved more than `slack` away from it. Jitter smaller than the slack never
/// reaches the output, so a resting pointer stays exactly where it is.
#[derive(Clone, Debug)]
pub struct Backlash {
    slack: f32,
    output: Option<Vec2>,
}

impl Backlash {
    pub const fn new(slack: f32) -> Self {
        Backlash { slack, output: None }
    }

    pub fn set_slack(&mut self, slack: f32) {
        self.slack = slack;
    }

    pub fn follow(&mut self, input: Vec2) -> Vec2 {
        let output = match self.output {
            Some(output) => {
                let gap = input - output;
                let distance = gap.length();
                if distance > self.slack { output + gap * ((distance - self.slack) / distance) } else { output }
            }
            None => input,
        };
        self.output = Some(output);
        output
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
    fn a_resting_signal_is_steadied() {
        let mut filter = OneEuro::new(1.0, 0.0);
        let mut state = 1u32;
        let (mut input_power, mut output_power) = (0.0f32, 0.0f32);
        for frame in 0..600 {
            let disturbance = noise(&mut state);
            let out = filter.filter(f64::from(frame) / 30.0, 5.0 + disturbance);
            if frame > 60 {
                input_power += disturbance * disturbance;
                output_power += (out - 5.0) * (out - 5.0);
            }
        }
        // At 30 samples a second a 1 Hz cutoff leaves about a third of the noise amplitude.
        assert!(output_power.sqrt() < 0.4 * input_power.sqrt(), "{output_power} of {input_power}");
    }

    #[test]
    fn a_fast_movement_is_followed_closely_when_beta_is_set() {
        let run = |beta: f32| {
            let mut filter = OneEuro::new(1.0, beta);
            let mut lag = 0.0f32;
            for frame in 0..60 {
                let truth = frame as f32 * 10.0;
                lag = truth - filter.filter(f64::from(frame) / 30.0, truth);
            }
            lag
        };
        assert!(run(1.0) < run(0.0) * 0.2, "lag {} vs {}", run(1.0), run(0.0));
    }

    #[test]
    fn a_long_gap_restarts_from_the_new_value() {
        let mut filter = OneEuro::new(1.0, 0.0);
        filter.filter(0.0, 0.0);
        assert!(filter.filter(0.033, 10.0) < 5.0);
        assert_eq!(filter.filter(2.0, -3.0), -3.0, "two seconds later the old value is forgotten");
    }

    #[test]
    fn repeated_and_backward_timestamps_do_not_produce_nonsense() {
        let mut filter = OneEuro::new(1.0, 1.0);
        let first = filter.filter(1.0, 4.0);
        assert_eq!(filter.filter(1.0, 100.0), first);
        assert!(filter.filter(0.5, 7.0).is_finite());
    }

    #[test]
    fn backlash_swallows_small_jitter_and_trails_real_movement_by_the_slack() {
        let mut backlash = Backlash::new(1.0);
        assert_eq!(backlash.follow(Vec2::new(10.0, 10.0)), Vec2::new(10.0, 10.0));
        assert_eq!(backlash.follow(Vec2::new(10.6, 9.5)), Vec2::new(10.0, 10.0));
        let moved = backlash.follow(Vec2::new(15.0, 10.0));
        assert!((moved.x - 14.0).abs() < 1e-5 && (moved.y - 10.0).abs() < 1e-5);
        assert_eq!(backlash.follow(Vec2::new(14.5, 10.0)), moved, "reversing has to cross the slack first");
    }

    #[test]
    fn the_point_filter_treats_both_axes_alike() {
        let mut filter = OneEuro2::new(1.0, 0.0);
        filter.filter(0.0, Vec2::ZERO);
        let out = filter.filter(0.033, Vec2::new(4.0, 4.0));
        assert!(out.x > 0.0 && out.x < 4.0 && out.x == out.y);
        let uneven = filter.filter(0.066, Vec2::new(4.0, 8.0));
        assert!(uneven.y > uneven.x, "{uneven:?}");
    }
}
