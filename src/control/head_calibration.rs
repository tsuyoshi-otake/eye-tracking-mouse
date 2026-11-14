//! The head calibration: how far this person turns their head with ease and how
//! still it rests, made into the head pointer's speeds and steadiness (`HeadGain`).
//!
//! The page of the gaze calibration shows it (`View`, the `Head…` phases). Stages,
//! each leading to the next or to a failure:
//!
//! ```text
//! Rest --> Left --> Right --> Up --> Down --> Done
//!   |        |        |        |       |
//!   +--------+--------+--------+-------+--> Failed
//! ```
//!
//! - Rest: facing the middle and holding still. The pointing signal (`Aim`) is
//!   recorded, to learn how much it shakes while the head rests.
//! - Left, Right, Up, Down: a dot near that edge of the screen. The person turns
//!   the face toward it as far as is comfortable and holds it there. The held
//!   signal is kept once it has stayed steady for `HOLD` seconds at least
//!   `MIN_TURN` past the rest, the way that was asked.
//!
//! What comes out: a quick turn through `REACH_SHARE` of the comfortable range
//! carries the cursor across the screen, on whichever axis needs the higher speed;
//! slow movement gets `SLOW_SHARE` of that. The steadiness is the least of
//! `STEADINESS_STEPS` with which the head pointer, at those speeds, keeps the
//! cursor within `STILL_PATH` while it replays the recorded rest.
//!
//! No stage waits forever: a face that stays away for `FACE_LIMIT` fails the run,
//! and so does a stage that takes longer than `STAGE_LIMIT`.

use crate::geometry::{RectI, Vec2};
use crate::vision::pose::HeadPose;

use super::calibration::{Failure, Phase, View};
use super::head_pointer::{Aim, HeadGain, HeadPointer, toward_screen};

/// Seconds at the start of the rest before its frames count: time to sit still.
const REST_SETTLE: f64 = 1.0;
/// Frames recorded at rest (three seconds of camera frames).
const REST_FRAMES: usize = 90;
/// Seconds after a turn is asked for before a hold counts: time to get there.
const TRAVEL: f64 = 0.8;
/// Seconds the head has to stay steady at the end of a turn.
const HOLD: f64 = 0.8;
/// How far a turn has to go past the rest, in eye spans (about five degrees).
const MIN_TURN: f32 = 0.08;
/// A held head stays within this many eye spans of where it is held, or within
/// `STEADY_SPREADS` times its shake at rest when that is more.
const STEADY_FLOOR: f32 = 0.02;
const STEADY_SPREADS: f32 = 4.0;
/// Frames on either side whose mean is the slow drift that `shake_only` takes out (half a second in all).
const SHAKE_REACH: usize = 7;
const STAGE_LIMIT: f64 = 20.0;
const FACE_LIMIT: f64 = 5.0;
/// The share of the comfortable range that a quick turn needs to cross the screen.
const REACH_SHARE: f32 = 1.0;
/// The slow speed as a share of the fast one (the defaults, 700 and 2200, are about a third).
const SLOW_SHARE: f32 = 1.0 / 3.0;
/// Cursor path per second, in pixels, that the shake may still cause. On one person's
/// pointing records (the stillest tenth of three-second stretches, 2026-11-14) this asked
/// for a steadiness of 3 to 4 at the speeds this calibration gives; 2 left 4 to 15 px/s.
const STILL_PATH: f32 = 3.0;
const STEADINESS_STEPS: [f32; 8] = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0];
/// Rest frames the replayed head pointer is given to settle before its path counts.
const POINTER_SETTLE: usize = 15;
/// The dots of the turns, as shares of the screen from its edges: where the outer places of the gaze calibration are.
const EDGE: (f32, f32) = (0.03, 0.05);
const TURNS: [Turn; 4] = [Turn::Left, Turn::Right, Turn::Up, Turn::Down];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Left,
    Right,
    Up,
    Down,
}

impl Turn {
    fn phase(self) -> Phase {
        match self {
            Turn::Left => Phase::HeadLeft,
            Turn::Right => Phase::HeadRight,
            Turn::Up => Phase::HeadUp,
            Turn::Down => Phase::HeadDown,
        }
    }

    /// How far `offset`, on the screen's axes, goes this way.
    fn reach(self, offset: Vec2) -> f32 {
        match self {
            Turn::Left => -offset.x,
            Turn::Right => offset.x,
            Turn::Up => -offset.y,
            Turn::Down => offset.y,
        }
    }

    /// Where the dot is, as shares of the screen across and down.
    fn share(self) -> (f32, f32) {
        match self {
            Turn::Left => (EDGE.0, 0.5),
            Turn::Right => (1.0 - EDGE.0, 0.5),
            Turn::Up => (0.5, EDGE.1),
            Turn::Down => (0.5, 1.0 - EDGE.1),
        }
    }
}

/// The pixel of the screen `area` at these shares across and down.
fn spot(area: RectI, (across, down): (f32, f32)) -> (i32, i32) {
    (
        area.left + (across * (area.width() - 1) as f32).round() as i32,
        area.top + (down * (area.height() - 1) as f32).round() as i32,
    )
}

/// The mean of `values` and how far they stray from it (root mean square).
fn centre_and_spread(values: &[Vec2]) -> (Vec2, f32) {
    let share = 1.0 / values.len() as f32;
    let centre = values.iter().fold(Vec2::ZERO, |sum, &value| sum + value * share);
    let spread = (values.iter().map(|&value| (value - centre).dot(value - centre)).sum::<f32>() * share).sqrt();
    (centre, spread)
}

/// The rest's frames with the slow drift taken out: each pose minus the mean of the
/// poses `SHAKE_REACH` frames around it, put back on the first pose. Nobody holds a head
/// perfectly still, and a slow drift moves the cursor as head pointing should; what the
/// steadiness is there for is the shake on top of it.
fn shake_only(poses: &[(f64, HeadPose)]) -> Vec<(f64, HeadPose)> {
    let Some(&(_, first)) = poses.first() else { return Vec::new() };
    (0..poses.len())
        .map(|index| {
            let near = &poses[index.saturating_sub(SHAKE_REACH)..(index + SHAKE_REACH + 1).min(poses.len())];
            let share = 1.0 / near.len() as f32;
            let mean = |part: fn(&HeadPose) -> Vec2| near.iter().fold(Vec2::ZERO, |sum, (_, pose)| sum + part(pose) * share);
            let (time, pose) = poses[index];
            let turn = |pose: &HeadPose| Vec2::new(pose.yaw, pose.pitch);
            let shaken = turn(&first) + turn(&pose) - mean(turn);
            let place = first.position + pose.position - mean(|pose| pose.position);
            (time, HeadPose { yaw: shaken.x, pitch: shaken.y, position: place, ..pose })
        })
        .collect()
}

/// What the rest established.
struct Rest {
    /// The resting signal, on the screen's axes.
    centre: Vec2,
    /// How far a held head may stray from where it is held, in eye spans.
    steady: f32,
    /// The rest's frames, shake only (`shake_only`), for replaying through the head pointer.
    shake: Vec<(f64, HeadPose)>,
}

impl Rest {
    fn new(poses: &[(f64, HeadPose)], signals: &[Vec2]) -> Rest {
        let (centre, _) = centre_and_spread(signals);
        let shake = shake_only(poses);
        let mut aim = Aim::default();
        let shaken: Vec<Vec2> = shake.iter().map(|(_, pose)| aim.step(pose)).collect();
        let (_, spread) = centre_and_spread(&shaken);
        Rest { centre, steady: STEADY_FLOOR.max(STEADY_SPREADS * spread), shake }
    }
}

/// The head staying put: since when, and the signals (screen axes) it has given meanwhile.
#[derive(Clone, Copy, Debug)]
struct Hold {
    since: f64,
    sum: Vec2,
    count: u32,
}

impl Hold {
    fn start(time: f64, signal: Vec2) -> Hold {
        Hold { since: time, sum: signal, count: 1 }
    }

    fn mean(&self) -> Vec2 {
        self.sum * (1.0 / self.count as f32)
    }

    /// The hold with this frame in it, or a new one from it when the head has moved on.
    fn follow(previous: Option<Hold>, time: f64, signal: Vec2, steady: f32) -> Hold {
        match previous {
            Some(hold) if hold.mean().distance(signal) <= steady => Hold { sum: hold.sum + signal, count: hold.count + 1, ..hold },
            _ => Hold::start(time, signal),
        }
    }
}

enum Stage {
    Rest { poses: Vec<(f64, HeadPose)>, signals: Vec<Vec2> },
    /// `held` has the held signal (screen axes) of each turn before `index`.
    Turn { rest: Rest, index: usize, hold: Option<Hold>, held: Vec<Vec2> },
}

pub enum HeadStep {
    Running(HeadCalibration),
    Done(HeadGain),
    Failed(Failure),
}

pub struct HeadCalibration {
    area: RectI,
    stage: Stage,
    stage_started: f64,
    face_seen: f64,
    face_visible: bool,
    aim: Aim,
}

impl HeadCalibration {
    /// A run on the screen `area` (desktop pixels): its dots are shown there, and the
    /// cursor's speeds are fitted to its size.
    pub fn new(area: RectI, time: f64) -> HeadCalibration {
        HeadCalibration {
            area,
            stage: Stage::Rest { poses: Vec::new(), signals: Vec::new() },
            stage_started: time,
            face_seen: time,
            face_visible: false,
            aim: Aim::default(),
        }
    }

    pub fn view(&self, time: f64) -> View {
        let (phase, target, dwell, finished) = match &self.stage {
            Stage::Rest { signals, .. } => {
                let settled = ((time - self.stage_started) / REST_SETTLE).clamp(0.0, 1.0) as f32;
                (Phase::HeadRest, spot(self.area, (0.5, 0.5)), 0.5 * settled + 0.5 * signals.len() as f32 / REST_FRAMES as f32, 0)
            }
            Stage::Turn { rest, index, hold, .. } => {
                let turn = TURNS[*index];
                let held = hold.filter(|hold| turn.reach(hold.mean() - rest.centre) >= MIN_TURN);
                let dwell = held.map_or(0.0, |hold| ((time - hold.since) / HOLD).clamp(0.0, 1.0) as f32);
                (turn.phase(), spot(self.area, turn.share()), dwell, index + 1)
            }
        };
        View {
            phase,
            target,
            dwell,
            progress: (finished as f32 + dwell) / (TURNS.len() + 1) as f32,
            face_visible: self.face_visible,
            answer: None,
            score: None,
        }
    }

    /// Feeds one camera frame; `pose` is `None` when no face was measured in it.
    pub fn step(mut self, time: f64, pose: Option<&HeadPose>) -> HeadStep {
        self.face_visible = pose.is_some();
        if time - self.stage_started >= STAGE_LIMIT {
            let failure = match self.stage {
                Stage::Rest { .. } => Failure::TooSlow,
                Stage::Turn { .. } => Failure::HeadNotMoved,
            };
            return HeadStep::Failed(failure);
        }
        let Some(pose) = pose else {
            return if time - self.face_seen >= FACE_LIMIT { HeadStep::Failed(Failure::FaceLost) } else { HeadStep::Running(self) };
        };
        self.face_seen = time;
        let signal = toward_screen(self.aim.step(pose));
        match self.stage {
            Stage::Rest { mut poses, mut signals } => {
                if time - self.stage_started >= REST_SETTLE {
                    poses.push((time, *pose));
                    signals.push(signal);
                }
                self.stage = if signals.len() < REST_FRAMES {
                    Stage::Rest { poses, signals }
                } else {
                    self.stage_started = time;
                    Stage::Turn { rest: Rest::new(&poses, &signals), index: 0, hold: None, held: Vec::new() }
                };
            }
            Stage::Turn { rest, index, hold, mut held } => {
                let turn = TURNS[index];
                let hold = (time - self.stage_started >= TRAVEL).then(|| Hold::follow(hold, time, signal, rest.steady));
                let kept = hold.filter(|hold| time - hold.since >= HOLD && turn.reach(hold.mean() - rest.centre) >= MIN_TURN);
                self.stage = match kept {
                    None => Stage::Turn { rest, index, hold, held },
                    Some(hold) => {
                        held.push(hold.mean());
                        if held.len() == TURNS.len() {
                            return HeadStep::Done(fit(self.area, &rest, &held));
                        }
                        self.stage_started = time;
                        Stage::Turn { rest, index: index + 1, hold: None, held }
                    }
                };
            }
        }
        HeadStep::Running(self)
    }
}

/// The speeds and the steadiness for a person whose held turns (screen axes, in the
/// order of `TURNS`) and rest are these, on the screen `area`.
fn fit(area: RectI, rest: &Rest, held: &[Vec2]) -> HeadGain {
    let [left, right, up, down] = [held[0], held[1], held[2], held[3]];
    let across = (right.x - left.x).max(MIN_TURN);
    let tall = (down.y - up.y).max(MIN_TURN);
    let fast = (area.width() as f32 / (REACH_SHARE * across)).max(area.height() as f32 / (REACH_SHARE * tall));
    let fast = fast.clamp(HeadGain::FAST.0, HeadGain::FAST.1);
    let slow = (fast * SLOW_SHARE).clamp(HeadGain::SLOW.0, HeadGain::SLOW.1);
    let fast = fast.max(slow);
    let still = |steadiness: f32| rest_path(&rest.shake, &HeadGain { slow, fast, steadiness }) <= STILL_PATH;
    let steadiness = STEADINESS_STEPS.into_iter().find(|&steadiness| still(steadiness)).unwrap_or(HeadGain::STEADINESS.1);
    HeadGain { slow, fast, steadiness }
}

/// How far, in pixels per second, the head pointer moves the cursor while it is fed these frames of a resting head.
fn rest_path(poses: &[(f64, HeadPose)], gain: &HeadGain) -> f32 {
    let mut pointer = HeadPointer::default();
    let mut path = 0.0;
    for (number, (time, pose)) in poses.iter().enumerate() {
        let step = pointer.step(*time, pose, gain);
        if number >= POINTER_SETTLE {
            path += step.length();
        }
    }
    let seconds = poses.last().zip(poses.get(POINTER_SETTLE)).map_or(0.0, |(last, first)| last.0 - first.0);
    if seconds > 0.0 { path / seconds as f32 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCREEN: RectI = RectI::new(0, 0, 2560, 1080);

    /// A person in front of the camera, by the head pose the tracker measures of them.
    /// `shake` is the size of the landmark jitter: image pixels of position, with 0.008
    /// (yaw) and 0.006 (pitch) radians per pixel of turn. At 1 it shakes about as much as
    /// the stillest stretches of one person's pointing records (0.006 to 0.008 eye spans).
    struct Sitter {
        time: f64,
        state: u32,
        shake: f32,
        /// Where the head is turned now, toward where it is going (yaw, pitch).
        facing: Vec2,
    }

    impl Sitter {
        fn new(shake: f32) -> Sitter {
            Sitter { time: 0.0, state: 7, shake, facing: Vec2::ZERO }
        }

        fn jitter(&mut self) -> f32 {
            self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((self.state >> 8) as f32 / 8_388_608.0 - 1.0) * self.shake
        }

        /// The next frame, a thirtieth of a second on, with the head turning toward `goal` at one radian a second.
        fn frame(&mut self, goal: Vec2) -> HeadPose {
            self.time += 1.0 / 30.0;
            let way = goal - self.facing;
            let step = 1.0 / 30.0;
            self.facing = if way.length() <= step { goal } else { self.facing + way * (step / way.length()) };
            let (jx, jy, jyaw, jpitch) = (self.jitter(), self.jitter(), self.jitter(), self.jitter());
            HeadPose {
                yaw: self.facing.x + jyaw * 0.008,
                pitch: self.facing.y + jpitch * 0.006,
                roll: 0.0,
                position: Vec2::new(640.0 + jx, 360.0 + jy),
                scale: 185.0,
            }
        }
    }

    /// Turns that the person finds comfortable, in radians: (yaw, pitch).
    /// Positive yaw turns the nose toward the image's right, the person's left; positive pitch nods down.
    fn goal(phase: Phase, comfortable: Vec2) -> Vec2 {
        match phase {
            Phase::HeadLeft => Vec2::new(comfortable.x, 0.0),
            Phase::HeadRight => Vec2::new(-comfortable.x, 0.0),
            Phase::HeadUp => Vec2::new(0.0, -comfortable.y),
            Phase::HeadDown => Vec2::new(0.0, comfortable.y),
            _ => Vec2::ZERO,
        }
    }

    /// Runs a head calibration with a person who does as asked, and every view of it.
    fn run(sitter: &mut Sitter, comfortable: Vec2) -> (HeadStep, Vec<View>) {
        let mut session = HeadCalibration::new(SCREEN, sitter.time);
        let mut views = Vec::new();
        loop {
            let view = session.view(sitter.time);
            views.push(view);
            let pose = sitter.frame(goal(view.phase, comfortable));
            match session.step(sitter.time, Some(&pose)) {
                HeadStep::Running(next) => session = next,
                ended => return (ended, views),
            }
            assert!(sitter.time < 120.0, "the run must end");
        }
    }

    fn fitted(shake: f32, comfortable: Vec2) -> HeadGain {
        match run(&mut Sitter::new(shake), comfortable).0 {
            HeadStep::Done(gain) => gain,
            HeadStep::Failed(failure) => panic!("{failure:?}"),
            HeadStep::Running(_) => unreachable!(),
        }
    }

    #[test]
    fn a_quick_turn_through_most_of_the_comfortable_range_crosses_the_screen() {
        let gain = fitted(0.2, Vec2::new(0.5, 0.3));
        // Turning 0.5 rad each way spans 2 sin 0.5 = 0.96 eye spans, nodding 0.3 rad each way 0.59.
        let across = 2560.0 / (REACH_SHARE * 2.0 * 0.5f32.sin());
        let tall = 1080.0 / (REACH_SHARE * 2.0 * 0.3f32.sin());
        assert!(across > tall, "the width decides here");
        assert!((gain.fast / across - 1.0).abs() < 0.03, "fast {} against {across}", gain.fast);
        assert!((gain.slow / (gain.fast * SLOW_SHARE) - 1.0).abs() < 1e-4, "{gain:?}");
    }

    #[test]
    fn a_person_who_turns_less_gets_a_faster_pointer_and_the_shorter_way_can_decide() {
        let wide = fitted(0.2, Vec2::new(0.5, 0.3));
        let narrow = fitted(0.2, Vec2::new(0.25, 0.15));
        assert!((narrow.fast / wide.fast - 1.95).abs() < 0.15, "{narrow:?} against {wide:?}");
        // Nodding hardly at all: the height of the screen needs the higher speed.
        let stiff = fitted(0.2, Vec2::new(0.5, 0.1));
        let tall = 1080.0 / (REACH_SHARE * 2.0 * 0.1f32.sin());
        assert!((stiff.fast / tall - 1.0).abs() < 0.05, "fast {} against {tall}", stiff.fast);
    }

    #[test]
    fn a_shaky_rest_gets_more_steadiness_and_the_cursor_then_stands_still_on_it() {
        let calm = fitted(0.2, Vec2::new(0.5, 0.3));
        let shaky = fitted(1.0, Vec2::new(0.5, 0.3));
        assert!(shaky.steadiness > calm.steadiness, "{shaky:?} against {calm:?}");
        assert!(STEADINESS_STEPS.contains(&calm.steadiness) && STEADINESS_STEPS.contains(&shaky.steadiness));
        assert!(shaky.steadiness < HeadGain::STEADINESS.1, "a shake like a real one is held before the last step: {shaky:?}");
        // A fresh rest of the same person, not the one the steadiness was chosen on, measured the same way.
        let mut sitter = Sitter { state: 12_345, ..Sitter::new(1.0) };
        let poses: Vec<(f64, HeadPose)> = (0..150)
            .map(|_| {
                let pose = sitter.frame(Vec2::ZERO);
                (sitter.time, pose)
            })
            .collect();
        let shake = shake_only(&poses);
        let path = rest_path(&shake, &shaky);
        assert!(path <= 2.0 * STILL_PATH, "{path} px/s with {shaky:?}");
        let loose = rest_path(&shake, &HeadGain { steadiness: 0.5, ..shaky });
        assert!(loose > 2.0 * STILL_PATH, "the shake is strong enough to need it: {loose} px/s");
    }

    #[test]
    fn taking_out_the_drift_leaves_the_shake_and_keeps_a_still_head_still() {
        // A head that drifts slowly to the side while it shakes: the drift goes, the shake stays.
        let mut sitter = Sitter::new(1.0);
        let poses: Vec<(f64, HeadPose)> = (0..90)
            .map(|frame| {
                let mut pose = sitter.frame(Vec2::ZERO);
                pose.yaw += 0.002 * frame as f32;
                (sitter.time, pose)
            })
            .collect();
        let shake = shake_only(&poses);
        assert_eq!(shake.len(), poses.len());
        let yaws = |poses: &[(f64, HeadPose)]| poses.iter().map(|(_, pose)| Vec2::new(pose.yaw, 0.0)).collect::<Vec<_>>();
        let (_, drifting) = centre_and_spread(&yaws(&poses));
        let (centre, shaking) = centre_and_spread(&yaws(&shake));
        assert!(drifting > 0.04 && shaking < 0.006, "spread {drifting} before, {shaking} after");
        assert!((centre.x - poses[0].1.yaw).abs() < 0.01, "put back where the rest began: {centre:?}");
        assert!(shake.iter().zip(&poses).all(|(a, b)| a.0 == b.0 && a.1.scale == b.1.scale), "times and sizes stay");
        // Nothing to take out of a head that does not move at all.
        let still: Vec<(f64, HeadPose)> = (0..30).map(|frame| (f64::from(frame) / 30.0, poses[0].1)).collect();
        assert!(shake_only(&still).iter().all(|(_, pose)| (pose.yaw - poses[0].1.yaw).abs() < 1e-6));
        assert!(shake_only(&[]).is_empty());
    }

    #[test]
    fn the_page_shows_each_dot_in_turn_and_the_ring_closes_while_the_head_holds() {
        let (ended, views) = run(&mut Sitter::new(0.2), Vec2::new(0.5, 0.3));
        assert!(matches!(ended, HeadStep::Done(_)));
        let mut phases: Vec<Phase> = views.iter().map(|view| view.phase).collect();
        phases.dedup();
        assert_eq!(phases, [Phase::HeadRest, Phase::HeadLeft, Phase::HeadRight, Phase::HeadUp, Phase::HeadDown]);
        let target = |phase| views.iter().find(|view| view.phase == phase).unwrap().target;
        assert_eq!(target(Phase::HeadRest), (1280, 540));
        assert_eq!(target(Phase::HeadLeft), (77, 540));
        assert_eq!(target(Phase::HeadRight), (2482, 540));
        assert_eq!(target(Phase::HeadUp), (1280, 54));
        assert_eq!(target(Phase::HeadDown), (1280, 1025));
        assert!(views.windows(2).all(|pair| pair[1].progress >= pair[0].progress - 1e-6), "the bar never goes back");
        for phase in [Phase::HeadLeft, Phase::HeadRight, Phase::HeadUp, Phase::HeadDown] {
            let dwell: Vec<f32> = views.iter().filter(|view| view.phase == phase).map(|view| view.dwell).collect();
            assert_eq!(dwell[0], 0.0, "{phase:?} starts with an open ring");
            assert!(*dwell.last().unwrap() > 0.9, "{phase:?} ends with the ring closed: {dwell:?}");
        }
        assert!(views.iter().all(|view| view.answer.is_none() && view.score.is_none()));
    }

    #[test]
    fn a_head_that_does_not_turn_or_turns_the_other_way_fails_the_run() {
        for wrong in [Vec2::ZERO, Vec2::new(-0.5, -0.3)] {
            let mut sitter = Sitter::new(0.2);
            let mut session = HeadCalibration::new(SCREEN, 0.0);
            let failure = loop {
                let phase = session.view(sitter.time).phase;
                // Asked to turn left, this person stays put or turns right.
                let way = if phase == Phase::HeadLeft { wrong } else { goal(phase, Vec2::new(0.5, 0.3)) };
                let pose = sitter.frame(way);
                match session.step(sitter.time, Some(&pose)) {
                    HeadStep::Running(next) => session = next,
                    HeadStep::Failed(failure) => break failure,
                    HeadStep::Done(gain) => panic!("{gain:?} from a head that did not turn left"),
                }
            };
            assert_eq!(failure, Failure::HeadNotMoved, "{wrong:?}");
            assert!(sitter.time > REST_SETTLE + 3.0 + STAGE_LIMIT - 0.1, "it waited for the turn: {}", sitter.time);
        }
    }

    #[test]
    fn a_face_that_stays_away_fails_the_run_and_a_short_absence_does_not() {
        let mut sitter = Sitter::new(0.2);
        let mut session = HeadCalibration::new(SCREEN, 0.0);
        for frame in 0..200 {
            // Away for a second during the rest: the run goes on and the page says so.
            let away = (60..90).contains(&frame);
            let pose = sitter.frame(Vec2::ZERO);
            session = match session.step(sitter.time, (!away).then_some(&pose)) {
                HeadStep::Running(next) => next,
                _ => panic!("ended at frame {frame}"),
            };
            assert_eq!(session.view(sitter.time).face_visible, !away);
        }
        let gone = sitter.time;
        let failure = loop {
            sitter.time += 1.0 / 30.0;
            match session.step(sitter.time, None) {
                HeadStep::Running(next) => session = next,
                HeadStep::Failed(failure) => break failure,
                HeadStep::Done(_) => unreachable!(),
            }
        };
        assert_eq!(failure, Failure::FaceLost);
        assert!((sitter.time - gone - FACE_LIMIT).abs() < 0.05, "after {} s", sitter.time - gone);
    }

    #[test]
    fn a_stop_on_the_way_before_the_travel_time_is_not_taken_for_the_turn() {
        // The person stops a little way out right after the dot appears, then goes on to where they can turn.
        let mut sitter = Sitter::new(0.2);
        let mut session = HeadCalibration::new(SCREEN, 0.0);
        let mut left_started = None;
        let gain = loop {
            let view = session.view(sitter.time);
            let mut way = goal(view.phase, Vec2::new(0.5, 0.3));
            if view.phase == Phase::HeadLeft {
                let started = *left_started.get_or_insert(sitter.time);
                if sitter.time - started < TRAVEL {
                    way = Vec2::new(0.15, 0.0);
                }
            }
            let pose = sitter.frame(way);
            match session.step(sitter.time, Some(&pose)) {
                HeadStep::Running(next) => session = next,
                HeadStep::Done(gain) => break gain,
                HeadStep::Failed(failure) => panic!("{failure:?}"),
            }
        };
        let full = fitted(0.2, Vec2::new(0.5, 0.3));
        assert!((gain.fast / full.fast - 1.0).abs() < 0.03, "{gain:?} against {full:?}");
    }
}
