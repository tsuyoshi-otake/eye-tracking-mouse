//! Turns head movement into cursor movement, like a mouse held by the nose.
//!
//! The pointing signal is where the nose is aimed (`Aim`): the head's position in the
//! image plus `LEVER` times the direction it faces, in eye spans. It is
//! smoothed, cleared of resting jitter, and its change since the previous
//! frame is scaled into pixels. The camera faces the user, so image-right is
//! the user's left: the horizontal axis is flipped on the way to the screen.

use crate::geometry::Vec2;
use crate::vision::pose::HeadPose;

use super::filter::{Backlash, OneEuro2};

/// How much turning the head counts compared with shifting it, in eye spans per unit of the facing vector.
const LEVER: f32 = 1.0;
/// Smoothing at rest (Hz) and how quickly it opens up with speed, at a steadiness of 1.
const REST_CUTOFF: f32 = 1.0;
const SPEED_RESPONSE: f32 = 10.0;
/// Jitter below this many eye spans never moves the cursor (about 0.1 mm of head movement), at a steadiness of 1.
const SLACK: f32 = 0.0012;
/// Signal speeds (eye spans per second) at which the slow gain ends and the fast gain is reached.
const SLOW_SPEED: f32 = 0.04;
const FAST_SPEED: f32 = 0.45;

/// Cursor pixels per eye span of pointing-signal movement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGain {
    /// For slow, precise movement.
    pub slow: f32,
    /// For quick, long movement.
    pub fast: f32,
    /// How firmly a resting cursor is held: the smoothing at rest and the jitter that is
    /// ignored both grow with it. Higher is calmer and a little slower to start moving.
    pub steadiness: f32,
}

impl HeadGain {
    /// The values each part may take, lowest and highest: what the settings file accepts and
    /// what the head calibration fits within. `fast` is never below `slow`.
    pub const SLOW: (f32, f32) = (50.0, 5000.0);
    pub const FAST: (f32, f32) = (50.0, 20000.0);
    pub const STEADINESS: (f32, f32) = (0.5, 8.0);
}

/// The raw pointing signal: where the nose is aimed, in eye spans. It is the head's
/// travel since the first frame plus `LEVER` times the direction it faces, so turning
/// and shifting the head both count. The head calibration measures this very signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct Aim {
    /// Head position of the previous frame, in image pixels.
    last_position: Option<Vec2>,
    /// Sideways and vertical head travel since the first frame, in eye spans.
    travel: Vec2,
}

impl Aim {
    /// The signal for this frame's pose.
    pub fn step(&mut self, pose: &HeadPose) -> Vec2 {
        if let Some(last) = self.last_position {
            // Dividing each step by the current scale keeps a lean toward the camera from reading as travel.
            self.travel = self.travel + (pose.position - last) * (1.0 / pose.scale);
        }
        self.last_position = Some(pose.position);
        self.travel + pose.forward() * LEVER
    }
}

#[derive(Clone, Debug)]
pub struct HeadPointer {
    smoother: OneEuro2,
    backlash: Backlash,
    aim: Aim,
    /// Time and cleaned signal of the previous frame.
    last_signal: Option<(f64, Vec2)>,
}

impl Default for HeadPointer {
    fn default() -> Self {
        HeadPointer {
            smoother: OneEuro2::new(REST_CUTOFF, SPEED_RESPONSE),
            backlash: Backlash::new(SLACK),
            aim: Aim::default(),
            last_signal: None,
        }
    }
}

impl HeadPointer {
    /// Forgets the previous frame, so whatever the head did meanwhile moves nothing.
    pub fn reset(&mut self) {
        *self = HeadPointer::default();
    }

    /// The cursor displacement, in pixels, for the head movement since the previous call.
    pub fn step(&mut self, time: f64, pose: &HeadPose, gain: &HeadGain) -> Vec2 {
        let raw = self.aim.step(pose);
        self.smoother.set_min_cutoff(REST_CUTOFF / gain.steadiness);
        self.backlash.set_slack(SLACK * gain.steadiness);
        let signal = self.backlash.follow(self.smoother.filter(time, raw));
        let previous = self.last_signal.replace((time, signal));
        let Some((last_time, last_signal)) = previous else {
            return Vec2::ZERO;
        };
        let dt = (time - last_time) as f32;
        if dt <= 0.0 {
            return Vec2::ZERO;
        }
        let change = signal - last_signal;
        let pace = ((change.length() / dt - SLOW_SPEED) / (FAST_SPEED - SLOW_SPEED)).clamp(0.0, 1.0);
        let blend = pace * pace * (3.0 - 2.0 * pace);
        let pixels_per_span = gain.slow + (gain.fast - gain.slow) * blend;
        toward_screen(change) * pixels_per_span
    }
}

/// A signal (or a change of it) on the screen's axes: the camera faces the person, so
/// image-right is the person's left, which is the screen's left.
pub fn toward_screen(signal: Vec2) -> Vec2 {
    Vec2::new(-signal.x, signal.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GAIN: HeadGain = HeadGain { slow: 700.0, fast: 2200.0, steadiness: 1.0 };

    fn pose(yaw: f32, pitch: f32, position: Vec2) -> HeadPose {
        HeadPose { yaw, pitch, roll: 0.0, position, scale: 200.0 }
    }

    /// Runs frames at 30 per second and returns the total displacement.
    fn run(pointer: &mut HeadPointer, start_frame: u32, frames: u32, pose_at: impl Fn(f32) -> HeadPose) -> Vec2 {
        let mut total = Vec2::ZERO;
        for frame in 0..frames {
            let progress = (frame + 1) as f32 / frames as f32;
            total = total + pointer.step(f64::from(start_frame + frame) / 30.0, &pose_at(progress), &GAIN);
        }
        total
    }

    const CENTRE: Vec2 = Vec2::new(640.0, 360.0);

    #[test]
    fn turning_the_nose_toward_image_right_moves_the_cursor_left_and_nodding_moves_it_down() {
        let mut pointer = HeadPointer::default();
        run(&mut pointer, 0, 10, |_| pose(0.0, 0.0, CENTRE));
        let turned = run(&mut pointer, 10, 30, |t| pose(0.1 * t, 0.0, CENTRE));
        assert!(turned.x < -50.0 && turned.y.abs() < 1.0, "{turned:?}");
        let nodded = run(&mut pointer, 40, 30, |t| pose(0.1, 0.1 * t, CENTRE));
        assert!(nodded.y > 50.0 && nodded.x.abs() < 15.0, "{nodded:?}");
    }

    #[test]
    fn shifting_the_head_moves_the_cursor_like_turning_it() {
        let mut pointer = HeadPointer::default();
        run(&mut pointer, 0, 10, |_| pose(0.0, 0.0, CENTRE));
        let shifted = run(&mut pointer, 10, 30, |t| pose(0.0, 0.0, CENTRE + Vec2::new(20.0 * t, 0.0)));
        assert!(shifted.x < -40.0, "0.1 eye spans toward image right: {shifted:?}");
    }

    /// The path the cursor travels while the head rests with strong jitter, and how far a turn carries it.
    fn wobble_and_reach(steadiness: f32) -> (f32, f32) {
        let gain = HeadGain { steadiness, ..GAIN };
        let mut pointer = HeadPointer::default();
        let mut state = 9u32;
        let mut wobble = 0.0;
        for frame in 0..300 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let jitter = (state >> 8) as f32 / 8_388_608.0 - 1.0;
            let step = pointer.step(f64::from(frame) / 30.0, &pose(0.004 * jitter, 0.0, CENTRE + Vec2::new(jitter, -jitter)), &gain);
            if frame > 30 {
                wobble += step.length();
            }
        }
        let mut reach = Vec2::ZERO;
        for frame in 0..60 {
            let yaw = 0.1 * (frame as f32 / 30.0).min(1.0);
            reach = reach + pointer.step(f64::from(300 + frame) / 30.0, &pose(yaw, 0.0, CENTRE), &gain);
        }
        (wobble, reach.length())
    }

    #[test]
    fn more_steadiness_calms_a_shaky_rest_and_a_turn_still_arrives() {
        let (loose_wobble, loose_reach) = wobble_and_reach(1.0);
        let (firm_wobble, firm_reach) = wobble_and_reach(3.0);
        assert!(loose_wobble > 5.0, "the jitter is strong enough to show: {loose_wobble}");
        assert!(firm_wobble < loose_wobble * 0.4, "{firm_wobble} against {loose_wobble}");
        assert!(firm_reach > loose_reach * 0.85, "{firm_reach} against {loose_reach}");
    }

    #[test]
    fn a_resting_head_with_landmark_jitter_does_not_move_the_cursor() {
        let mut pointer = HeadPointer::default();
        let mut state = 9u32;
        let mut total = Vec2::ZERO;
        let mut after_settling = Vec2::ZERO;
        for frame in 0..300 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let jitter = ((state >> 8) as f32 / 8_388_608.0 - 1.0) * 0.15;
            let step = pointer.step(f64::from(frame) / 30.0, &pose(0.0005 * jitter, 0.0, CENTRE + Vec2::new(jitter, -jitter)), &GAIN);
            total = total + step;
            if frame > 30 {
                after_settling = after_settling + Vec2::new(step.x.abs(), step.y.abs());
            }
        }
        assert!(total.length() < 2.0, "drift {total:?}");
        assert!(after_settling.length() < 2.0, "path length {after_settling:?}");
    }

    #[test]
    fn the_same_turn_travels_farther_when_made_quickly() {
        let turn = |frames: u32| {
            let mut pointer = HeadPointer::default();
            run(&mut pointer, 0, 10, |_| pose(0.0, 0.0, CENTRE));
            let moved = run(&mut pointer, 10, frames, |t| pose(0.2 * t, 0.0, CENTRE));
            // Let the smoothing finish.
            moved + run(&mut pointer, 10 + frames, 60, |_| pose(0.2, 0.0, CENTRE))
        };
        let (quick, slow) = (turn(6), turn(240));
        assert!(quick.x < slow.x * 1.8, "quick {quick:?} slow {slow:?}");
        // 0.2 rad is about 0.2 eye spans: between the slow and the fast gain.
        assert!((-460.0..-120.0).contains(&quick.x) && (-200.0..-100.0).contains(&slow.x), "quick {quick:?} slow {slow:?}");
    }

    #[test]
    fn a_reset_discards_movement_made_while_not_tracking() {
        let mut pointer = HeadPointer::default();
        run(&mut pointer, 0, 10, |_| pose(0.0, 0.0, CENTRE));
        pointer.reset();
        let after = run(&mut pointer, 100, 10, |_| pose(0.3, -0.2, CENTRE + Vec2::new(150.0, 80.0)));
        assert_eq!(after, Vec2::ZERO);
    }

    #[test]
    fn leaning_toward_the_camera_does_not_move_the_cursor() {
        let mut pointer = HeadPointer::default();
        let off_centre = Vec2::new(900.0, 200.0);
        run(&mut pointer, 0, 10, |_| pose(0.0, 0.0, off_centre));
        let leaned = run(&mut pointer, 10, 30, |t| HeadPose { scale: 200.0 + 60.0 * t, ..pose(0.0, 0.0, off_centre) });
        assert_eq!(leaned, Vec2::ZERO);
    }
}
