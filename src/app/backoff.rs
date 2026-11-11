//! How long to wait before trying again something that keeps failing.

/// Seconds waited after the first failure.
const FIRST: f64 = 1.0;
/// The wait doubles with every failure up to this many seconds.
const LONGEST: f64 = 30.0;
/// Each wait is shifted by up to this share either way, so that retries do
/// not fall into step with anything periodic.
const SPREAD: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct Backoff {
    failures: u32,
    random: u32,
}

impl Backoff {
    /// `seed` only varies the shifts; any value will do.
    pub fn new(seed: u32) -> Backoff {
        // The generator below would stay at zero forever.
        Backoff { failures: 0, random: seed.max(1) }
    }

    /// Seconds to wait after one more failure: about 1, 2, 4, 8, 16, then 30 from there on.
    pub fn next_wait(&mut self) -> f64 {
        let base = (FIRST * f64::from(1u32 << self.failures.min(8))).min(LONGEST);
        self.failures = self.failures.saturating_add(1);
        // xorshift32
        self.random ^= self.random << 13;
        self.random ^= self.random >> 17;
        self.random ^= self.random << 5;
        let unit = f64::from(self.random) / f64::from(u32::MAX);
        base * (1.0 + SPREAD * (2.0 * unit - 1.0))
    }

    /// Call after a success: the next failure waits the shortest time again.
    pub fn reset(&mut self) {
        self.failures = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_double_up_to_half_a_minute_and_stay_there() {
        for seed in [0, 1, 7, 0xDEAD_BEEF, u32::MAX] {
            let mut backoff = Backoff::new(seed);
            for (failure, base) in [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0].into_iter().enumerate() {
                let wait = backoff.next_wait();
                assert!((base * 0.8..=base * 1.2).contains(&wait), "failure {failure} with seed {seed}: {wait}");
            }
            for _ in 0..10_000 {
                assert!(backoff.next_wait() <= 36.0, "a long run of failures never waits longer");
            }
        }
    }

    #[test]
    fn a_success_starts_the_waits_from_the_shortest_again() {
        let mut backoff = Backoff::new(3);
        for _ in 0..6 {
            backoff.next_wait();
        }
        backoff.reset();
        assert!(backoff.next_wait() <= 1.2);
        assert!(backoff.next_wait() >= 1.6);
    }

    #[test]
    fn waits_are_shifted_differently_each_time() {
        let mut backoff = Backoff::new(42);
        let mut waits = Vec::new();
        for _ in 0..20 {
            backoff.reset();
            waits.push(backoff.next_wait());
        }
        let (low, high) = waits.iter().fold((f64::MAX, f64::MIN), |(low, high), &wait| (low.min(wait), high.max(wait)));
        assert!(high - low > 0.1, "twenty first waits all fell within {low}..{high}");
        assert!(Backoff::new(1).next_wait() != Backoff::new(2).next_wait());
    }
}
