//! Clicking with a deliberate long blink.
//!
//! Every path through the state machine ends back in `Open`:
//!
//! ```text
//! Open --both eyes shut--> Closed --held `hold` s--> Armed --eyes open--> Open (Click)
//!                            |                         |
//!                            | eyes open (a normal     | still shut after `limit` s
//!                            v  blink: nothing)        v
//!                           Open                     Spent (Cancelled) --eyes open--> Open
//! ```
//!
//! Losing the face returns to `Open` from anywhere and never clicks. An eye
//! counts as shut or open relative to its own slowly adapting open size, so
//! the thresholds fit small and large eyes alike. A wink (one eye) does nothing.

/// Seconds for the open-size estimate to follow an eye that opens wider, and one that narrows.
/// Slow both ways, so the estimate is the eye's usual size and not the widest it has just been:
/// measured on a person, an eye wanders between 0.6 and 1.0 of its widest while plainly open.
const BASELINE_RISE: f32 = 1.5;
const BASELINE_FALL: f32 = 3.0;
/// The open-size estimate never goes below this, so noise on a shut eye cannot count as "open".
const BASELINE_FLOOR: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlinkSettings {
    /// Seconds the eyes must stay shut before a click is armed.
    pub hold: f32,
    /// Seconds after which a still-shut pair of eyes cancels the click.
    pub limit: f32,
    /// An eye is shut below this fraction of its open size ...
    pub closed_ratio: f32,
    /// ... and open again above this fraction.
    pub open_ratio: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlinkEvent {
    /// The eyes have been shut long enough: opening them now clicks.
    Armed,
    Click,
    /// An armed click was abandoned.
    Cancelled,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Open,
    Closed { since: f64 },
    Armed { since: f64 },
    Spent,
}

#[derive(Clone, Debug)]
pub struct BlinkClicker {
    state: State,
    /// Open size of the right and the left eye, and when it was last updated.
    baseline: Option<([f32; 2], f64)>,
}

impl Default for BlinkClicker {
    fn default() -> Self {
        BlinkClicker { state: State::Open, baseline: None }
    }
}

impl BlinkClicker {
    /// Whether the eyes are currently counted as shut (the pointer should hold still).
    pub fn is_closed(&self) -> bool {
        self.state != State::Open
    }

    /// Feeds one frame. `openness` is `None` while no face is visible.
    pub fn step(&mut self, time: f64, openness: Option<[f32; 2]>, settings: &BlinkSettings) -> Option<BlinkEvent> {
        let Some(openness) = openness else {
            let was_armed = matches!(self.state, State::Armed { .. });
            self.state = State::Open;
            return was_armed.then_some(BlinkEvent::Cancelled);
        };
        let (baseline, last_update) = self.baseline.unwrap_or((openness.map(|o| o.max(BASELINE_FLOOR)), time));
        let ratio = [openness[0] / baseline[0], openness[1] / baseline[1]];
        let shut = ratio.iter().all(|&r| r < settings.closed_ratio);
        let open = ratio.iter().all(|&r| r > settings.open_ratio);

        let (next, event) = match self.state {
            State::Open if shut => (State::Closed { since: time }, None),
            State::Open => (State::Open, None),
            State::Closed { .. } if open => (State::Open, None),
            State::Closed { since } if time - since >= f64::from(settings.hold) => {
                (State::Armed { since }, Some(BlinkEvent::Armed))
            }
            State::Armed { .. } if open => (State::Open, Some(BlinkEvent::Click)),
            State::Armed { since } if time - since >= f64::from(settings.limit) => {
                (State::Spent, Some(BlinkEvent::Cancelled))
            }
            State::Spent if open => (State::Open, None),
            unchanged => (unchanged, None),
        };
        self.state = next;

        // Learn the open size whenever the eyes are not in a blink, also from an eye that has
        // become narrower than it was: otherwise it could never count as open again.
        let mut learned = baseline;
        if next == State::Open {
            let dt = (time - last_update).clamp(0.0, 0.2) as f32;
            for eye in 0..2 {
                let seconds = if openness[eye] > learned[eye] { BASELINE_RISE } else { BASELINE_FALL };
                learned[eye] = (learned[eye] + (openness[eye] - learned[eye]) * (dt / seconds).min(1.0)).max(BASELINE_FLOOR);
            }
        }
        self.baseline = Some((learned, time));
        event
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SETTINGS: BlinkSettings = BlinkSettings { hold: 0.4, limit: 1.5, closed_ratio: 0.5, open_ratio: 0.6 };
    const OPEN: [f32; 2] = [0.30, 0.28];
    const SHUT: [f32; 2] = [0.06, 0.05];

    /// Feeds `seconds` of one eye state at 30 frames a second, collecting events.
    struct Rig {
        clicker: BlinkClicker,
        frame: u32,
        events: Vec<BlinkEvent>,
    }

    impl Rig {
        fn new() -> Rig {
            let mut rig = Rig { clicker: BlinkClicker::default(), frame: 0, events: Vec::new() };
            rig.feed(1.0, Some(OPEN));
            rig
        }

        fn feed(&mut self, seconds: f32, openness: Option<[f32; 2]>) -> &mut Rig {
            for _ in 0..(seconds * 30.0).round() as u32 {
                self.events.extend(self.clicker.step(f64::from(self.frame) / 30.0, openness, &SETTINGS));
                self.frame += 1;
            }
            self
        }
    }

    #[test]
    fn an_ordinary_blink_does_nothing() {
        let mut rig = Rig::new();
        rig.feed(0.2, Some(SHUT)).feed(0.5, Some(OPEN));
        assert!(rig.events.is_empty());
        assert!(!rig.clicker.is_closed());
    }

    #[test]
    fn a_long_blink_arms_after_the_hold_time_and_clicks_when_the_eyes_open() {
        let mut rig = Rig::new();
        rig.feed(0.3, Some(SHUT));
        assert!(rig.events.is_empty() && rig.clicker.is_closed());
        rig.feed(0.3, Some(SHUT));
        assert_eq!(rig.events, [BlinkEvent::Armed]);
        assert!(rig.clicker.is_closed());
        rig.feed(0.2, Some(OPEN));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Click]);
        assert!(!rig.clicker.is_closed());
    }

    #[test]
    fn eyes_kept_shut_past_the_limit_cancel_and_opening_them_then_does_not_click() {
        let mut rig = Rig::new();
        rig.feed(2.0, Some(SHUT));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Cancelled]);
        assert!(rig.clicker.is_closed(), "still shut, so the pointer keeps holding still");
        rig.feed(0.5, Some(OPEN));
        assert_eq!(rig.events.len(), 2);
        assert!(!rig.clicker.is_closed());
        // The next deliberate blink works as usual.
        rig.feed(0.6, Some(SHUT)).feed(0.2, Some(OPEN));
        assert_eq!(rig.events[2..], [BlinkEvent::Armed, BlinkEvent::Click]);
    }

    #[test]
    fn a_wink_or_half_closed_eyes_never_arm() {
        let mut rig = Rig::new();
        rig.feed(1.0, Some([SHUT[0], OPEN[1]])).feed(0.3, Some(OPEN));
        // Looking down narrows the eyes to roughly 60 % without shutting them.
        rig.feed(1.0, Some([0.18, 0.17])).feed(0.3, Some(OPEN));
        assert!(rig.events.is_empty());
    }

    #[test]
    fn losing_the_face_while_armed_cancels_without_a_click() {
        let mut rig = Rig::new();
        rig.feed(0.6, Some(SHUT)).feed(0.2, None);
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Cancelled]);
        assert!(!rig.clicker.is_closed());
        rig.feed(0.5, Some(OPEN));
        assert_eq!(rig.events.len(), 2);
    }

    #[test]
    fn losing_the_face_during_a_short_closure_is_silent() {
        let mut rig = Rig::new();
        rig.feed(0.2, Some(SHUT)).feed(0.2, None).feed(0.3, Some(OPEN));
        assert!(rig.events.is_empty());
    }

    #[test]
    fn half_open_eyes_between_the_thresholds_keep_the_current_state() {
        let mut rig = Rig::new();
        // Shut, then only half reopened (55 %): still counted as shut, so the click arms.
        rig.feed(0.2, Some(SHUT)).feed(0.3, Some([0.165, 0.155]));
        assert_eq!(rig.events, [BlinkEvent::Armed]);
        rig.feed(0.1, Some(OPEN));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Click]);
    }

    /// Measured on a person with `--diag`: the eyes rest at 0.27 and 0.20 of their widths, open
    /// to 0.29 and 0.23 now and then, drop to 0.05 and 0.03 when shut, and one of them comes
    /// back only to 0.15 for a second or more after a long blink.
    #[test]
    fn an_eye_that_reopens_narrower_than_its_widest_still_clicks() {
        let mut rig = Rig { clicker: BlinkClicker::default(), frame: 0, events: Vec::new() };
        rig.feed(2.0, Some([0.27, 0.20])).feed(0.5, Some([0.29, 0.23])).feed(1.0, Some([0.27, 0.20]));
        rig.feed(0.6, Some([0.05, 0.03])).feed(0.3, Some([0.26, 0.15]));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Click]);
        // And when the eye stays that narrow, the next long blink clicks as well.
        rig.feed(4.0, Some([0.26, 0.15])).feed(0.6, Some([0.05, 0.03])).feed(0.3, Some([0.26, 0.15]));
        assert_eq!(rig.events[2..], [BlinkEvent::Armed, BlinkEvent::Click]);
    }

    #[test]
    fn the_thresholds_follow_the_size_of_the_eyes() {
        // Narrow eyes: 0.16 open. A fixed threshold made for 0.30 eyes would call them half shut.
        let mut rig = Rig { clicker: BlinkClicker::default(), frame: 0, events: Vec::new() };
        rig.feed(2.0, Some([0.16, 0.16])).feed(0.6, Some([0.05, 0.05])).feed(0.2, Some([0.16, 0.16]));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Click]);
        // Starting with shut eyes must not teach a tiny open size.
        let mut rig = Rig { clicker: BlinkClicker::default(), frame: 0, events: Vec::new() };
        rig.feed(0.3, Some(SHUT)).feed(2.0, Some(OPEN)).feed(0.6, Some(SHUT)).feed(0.2, Some(OPEN));
        assert_eq!(rig.events, [BlinkEvent::Armed, BlinkEvent::Click]);
    }
}
