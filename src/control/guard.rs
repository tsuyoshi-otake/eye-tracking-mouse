//! Yields the cursor to the hand on the mouse.
//!
//! The program knows where it last put the cursor. If the cursor is found
//! somewhere else, something else moved it (the physical mouse, a touchpad, a
//! remote session), and that wins: the program holds off until the cursor has
//! been left alone for `QUIET_TIME`, then carries on from wherever it is.

/// The cursor may differ from the commanded position by this many pixels
/// without counting as moved by someone else.
const TOLERANCE: i32 = 3;
/// Seconds the cursor must be left alone before control is taken back.
const QUIET_TIME: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Do not move the cursor.
    Hold,
    /// The cursor is ours. `resync` means "start again from where the cursor is now".
    Go { resync: bool },
}

#[derive(Clone, Debug, Default)]
pub struct ManualGuard {
    expected: Option<(i32, i32)>,
    resume_at: Option<f64>,
}

impl ManualGuard {
    /// Call once per frame with the actual cursor position, before deciding on a move.
    pub fn observe(&mut self, time: f64, actual: (i32, i32)) -> Verdict {
        let first = self.expected.is_none();
        let moved_by_other = self
            .expected
            .is_some_and(|e| (e.0 - actual.0).abs() > TOLERANCE || (e.1 - actual.1).abs() > TOLERANCE);
        self.expected = Some(actual);
        if moved_by_other {
            self.resume_at = Some(time + QUIET_TIME);
            return Verdict::Hold;
        }
        match self.resume_at {
            Some(at) if time < at => Verdict::Hold,
            Some(_) => {
                self.resume_at = None;
                Verdict::Go { resync: true }
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
        assert_eq!(guard.observe(2.01, (420, 260)), Verdict::Go { resync: true });
        assert_eq!(guard.observe(2.04, (420, 260)), Verdict::Go { resync: false });
    }

    #[test]
    fn a_move_the_system_did_not_carry_out_also_holds() {
        // For example the cursor is confined by another program: do not fight it every frame.
        let mut guard = ManualGuard::default();
        guard.observe(0.0, (100, 100));
        guard.commanded((500, 500));
        assert_eq!(guard.observe(0.03, (100, 100)), Verdict::Hold);
        assert_eq!(guard.observe(1.6, (100, 100)), Verdict::Go { resync: true });
    }
}
