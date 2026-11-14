//! Lets the head correct a cursor that the eyes carry, for the mode that
//! uses both.
//!
//! The eyes say roughly where the cursor belongs; the estimate is off by tens
//! of pixels and no amount of looking makes it better. The head is exact but
//! slow over a distance. So the cursor is where the eyes put it plus a
//! correction the head has built up, and the correction belongs to the place
//! the person was looking at when it was made: once the eyes have carried the
//! cursor far enough from there, it is dropped.
//!
//! The correction itself is not stored. It is read back each frame as the
//! distance between the cursor and where the eyes put it, so whatever else
//! limits the cursor (a screen's edge, a click going back to before the lids
//! dropped) limits the correction with it.

use crate::geometry::Vec2;

#[derive(Clone, Copy, Debug, Default)]
pub struct Correction {
    /// Where the eyes put the cursor in the frame before this one.
    carried: Option<Vec2>,
    /// Where the eyes had put the cursor when the correction was last empty.
    began: Vec2,
}

impl Correction {
    /// Forgets the correction: the next frame starts where the eyes put the cursor.
    pub fn reset(&mut self) {
        *self = Correction::default();
    }

    /// The cursor position for this frame. `cursor` is where it was after the
    /// frame before, `eyes` where the eyes put it now, `nudge` what the head
    /// adds in this frame (all in pixels). The correction is dropped when
    /// `eyes` is `reach` pixels or more from where the correction began.
    pub fn step(&mut self, cursor: Vec2, eyes: Vec2, nudge: Vec2, reach: f32) -> Vec2 {
        let kept = self.carried.filter(|_| eyes.distance(self.began) < reach).map(|carried| cursor - carried);
        if kept.is_none() {
            self.began = eyes;
        }
        self.carried = Some(eyes);
        eyes + kept.unwrap_or(Vec2::ZERO) + nudge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REACH: f32 = 250.0;

    #[test]
    fn the_head_moves_the_cursor_away_from_where_the_eyes_put_it_and_it_stays_there() {
        let mut correction = Correction::default();
        let eyes = Vec2::new(800.0, 400.0);
        let mut cursor = correction.step(Vec2::new(10.0, 10.0), eyes, Vec2::ZERO, REACH);
        assert_eq!(cursor, eyes, "the first frame is where the eyes put it, wherever the cursor was");
        for _ in 0..20 {
            cursor = correction.step(cursor, eyes, Vec2::new(3.0, -1.5), REACH);
        }
        assert_eq!(cursor, Vec2::new(860.0, 370.0));
        // The head rests: the correction stays, also while the eyes' place wanders a little.
        let wandered = Vec2::new(812.0, 395.0);
        cursor = correction.step(cursor, wandered, Vec2::ZERO, REACH);
        assert_eq!(cursor, wandered + Vec2::new(60.0, -30.0));
    }

    #[test]
    fn looking_somewhere_else_drops_the_correction_and_nearby_does_not() {
        let mut correction = Correction::default();
        let eyes = Vec2::new(800.0, 400.0);
        let mut cursor = correction.step(eyes, eyes, Vec2::ZERO, REACH);
        cursor = correction.step(cursor, eyes, Vec2::new(40.0, 0.0), REACH);
        // Just short of the reach, in steps, as the eyes' filtered place travels.
        for x in [900.0, 1000.0, 1049.0] {
            cursor = correction.step(cursor, Vec2::new(x, 400.0), Vec2::ZERO, REACH);
            assert_eq!(cursor, Vec2::new(x + 40.0, 400.0));
        }
        cursor = correction.step(cursor, Vec2::new(1050.0, 400.0), Vec2::ZERO, REACH);
        assert_eq!(cursor, Vec2::new(1050.0, 400.0), "at the reach the correction is gone");
        // A new one is counted from here, not from the first place.
        cursor = correction.step(cursor, Vec2::new(1050.0, 400.0), Vec2::new(0.0, 25.0), REACH);
        cursor = correction.step(cursor, Vec2::new(820.0, 400.0), Vec2::ZERO, REACH);
        assert_eq!(cursor, Vec2::new(820.0, 425.0));
    }

    #[test]
    fn a_cursor_stopped_by_something_else_takes_the_correction_with_it() {
        let mut correction = Correction::default();
        let eyes = Vec2::new(30.0, 400.0);
        let mut cursor = correction.step(eyes, eyes, Vec2::ZERO, REACH);
        // The head pushes left past the screen's edge at 0; the caller stops the cursor there.
        for _ in 0..10 {
            cursor = correction.step(cursor, eyes, Vec2::new(-20.0, 0.0), REACH);
            cursor.x = cursor.x.max(0.0);
        }
        assert_eq!(cursor.x, 0.0);
        // Turning back takes effect at once: nothing was saved up beyond the edge.
        cursor = correction.step(cursor, eyes, Vec2::new(5.0, 0.0), REACH);
        assert_eq!(cursor.x, 5.0);
    }

    #[test]
    fn a_reset_starts_from_the_eyes_again() {
        let mut correction = Correction::default();
        let eyes = Vec2::new(800.0, 400.0);
        let cursor = correction.step(eyes, eyes, Vec2::new(50.0, 0.0), REACH);
        correction.reset();
        assert_eq!(correction.step(cursor, eyes, Vec2::ZERO, REACH), eyes);
    }
}
