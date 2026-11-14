//! Yields the cursor to the hand on the mouse.
//!
//! The program knows where it last put the cursor. If the cursor is found
//! somewhere else, something else moved it (the physical mouse, a touchpad, a
//! remote session), and that wins: the program holds off until the cursor has
//! been left alone for `QUIET_TIME`, then carries on from wherever it is.
//!
//! Before it carries on it asks whether the cursor is its to move at all, with
//! a move of one pixel. A program that shares the mouse with another computer
//! (Mouse Without Borders) takes every move made here while the pointer is
//! over there, also this program's, and adds it to the pointer on the other
//! computer: a move to where the person looks would throw that pointer across
//! its screen every `QUIET_TIME`, away from the edge it has to cross to come
//! back. One pixel, to the right and to the left in turn, does not.

/// The cursor may differ from the commanded position by this many pixels
/// without counting as moved by someone else.
const TOLERANCE: i32 = 3;
/// Seconds the cursor must be left alone before control is taken back.
const QUIET_TIME: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Do not move the cursor.
    Hold,
    /// Move the cursor to this pixel next to it and nothing else: the next
    /// observation tells whether moves arrive.
    Probe((i32, i32)),
    /// The cursor is ours. `resync` means "start again from where the cursor is now".
    Go { resync: bool },
}

#[derive(Clone, Debug, Default)]
pub struct ManualGuard {
    expected: Option<(i32, i32)>,
    resume_at: Option<f64>,
    /// The move asked for in the frame before was a probe.
    probing: bool,
    /// The next probe goes to the left.
    leftward: bool,
}

impl ManualGuard {
    /// Call once per frame with the actual cursor position, before deciding on a move.
    pub fn observe(&mut self, time: f64, actual: (i32, i32)) -> Verdict {
        let first = self.expected.is_none();
        // A probe is one pixel long, so it has arrived only if the cursor is exactly there.
        let slack = if self.probing { 0 } else { TOLERANCE };
        let elsewhere = self.expected.is_some_and(|e| (e.0 - actual.0).abs() > slack || (e.1 - actual.1).abs() > slack);
        let probed = std::mem::take(&mut self.probing);
        self.expected = Some(actual);
        if elsewhere {
            self.resume_at = Some(time + QUIET_TIME);
            return Verdict::Hold;
        }
        match self.resume_at {
            _ if probed => {
                self.resume_at = None;
                Verdict::Go { resync: true }
            }
            Some(at) if time < at => Verdict::Hold,
            Some(_) => {
                let pixel = (actual.0 + if self.leftward { -1 } else { 1 }, actual.1);
                (self.probing, self.leftward, self.expected) = (true, !self.leftward, Some(pixel));
                Verdict::Probe(pixel)
            }
            None => Verdict::Go { resync: first },
        }
    }

    /// Call after asking the system to put the cursor at `position`.
    pub fn commanded(&mut self, position: (i32, i32)) {
        self.expected = Some(position);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_observation_starts_from_the_cursor() {
        let mut guard = ManualGuard::default();
        assert_eq!(guard.observe(0.0, (100, 100)), Verdict::Go { resync: true });
        assert_eq!(guard.observe(0.03, (100, 100)), Verdict::Go { resync: false });
    }

    #[test]
    fn our_own_moves_are_not_mistaken_for_the_mouse() {
        let mut guard = ManualGuard::default();
        guard.observe(0.0, (100, 100));
        guard.commanded((140, 90));
        assert_eq!(guard.observe(0.03, (140, 90)), Verdict::Go { resync: false });
        guard.commanded((150, 90));
        assert_eq!(guard.observe(0.06, (148, 91)), Verdict::Go { resync: false }, "a couple of pixels of rounding");
    }

    #[test]
    fn a_mouse_movement_holds_until_the_cursor_has_rested_and_then_resyncs_once() {
        let mut guard = ManualGuard::default();
        guard.observe(0.0, (100, 100));
        guard.commanded((140, 90));
        assert_eq!(guard.observe(0.03, (300, 200)), Verdict::Hold);
        // Still being moved half a second later: the quiet time starts again.
        assert_eq!(guard.observe(0.5, (420, 260)), Verdict::Hold);
        assert_eq!(guard.observe(1.9, (420, 260)), Verdict::Hold, "1.4 s after the last movement");
        // One pixel first; it arrives, so the cursor is ours again.
        assert_eq!(guard.observe(2.01, (420, 260)), Verdict::Probe((421, 260)));
        assert_eq!(guard.observe(2.04, (421, 260)), Verdict::Go { resync: true });
        assert_eq!(guard.observe(2.07, (421, 260)), Verdict::Go { resync: false });
    }

    #[test]
    fn a_mouse_that_moves_during_the_probe_holds_again() {
        let mut guard = ManualGuard::default();
        guard.observe(0.0, (100, 100));
        assert_eq!(guard.observe(0.03, (300, 200)), Verdict::Hold);
        assert_eq!(guard.observe(1.6, (300, 200)), Verdict::Probe((301, 200)));
        assert_eq!(guard.observe(1.63, (340, 200)), Verdict::Hold);
        assert_eq!(guard.observe(3.0, (340, 200)), Verdict::Hold);
        assert_eq!(guard.observe(3.2, (340, 200)), Verdict::Probe((339, 200)));
    }

    #[test]
    fn while_moves_do_not_arrive_only_single_pixels_are_tried_and_they_cancel_out() {
        // The pointer is on another computer: whatever is sent here goes there instead.
        let mut guard = ManualGuard::default();
        guard.observe(0.0, (100, 100));
        guard.commanded((500, 500));
        assert_eq!(guard.observe(0.03, (100, 100)), Verdict::Hold);
        let (mut time, mut sent) = (0.03, Vec::new());
        while time < 20.0 {
            time += 0.03;
            match guard.observe(time, (100, 100)) {
                Verdict::Hold => {}
                Verdict::Probe(pixel) => sent.push((time, pixel.0 - 100, pixel.1 - 100)),
                Verdict::Go { .. } => panic!("taken back at {time} although nothing arrives"),
            }
        }
        assert!(sent.len() >= 10, "{sent:?}");
        assert!(sent.windows(2).all(|pair| pair[1].0 - pair[0].0 >= QUIET_TIME), "{sent:?}");
        assert!(sent.windows(2).all(|pair| pair[0].1 + pair[1].1 == 0 && pair[0].2 == 0), "{sent:?}");
        // The pointer comes back: the other program puts the cursor at the edge, it rests, a pixel arrives.
        assert_eq!(guard.observe(20.0, (0, 300)), Verdict::Hold);
        let Verdict::Probe(pixel) = guard.observe(21.6, (0, 300)) else { panic!("a probe is due") };
        assert_eq!(guard.observe(21.63, pixel), Verdict::Go { resync: true });
    }
}
