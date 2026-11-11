//! The picture in the notification area: an eye whose colour tells the state at a glance.

use crate::control::controller::Activity;
use crate::platform::notify_icon::IconImage;

use super::pipeline::Status;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    /// The face is driving the cursor.
    Active,
    /// Running, but not driving the cursor right now.
    Attention,
    Paused,
    Calibrating,
    Stopped,
}

impl Look {
    pub fn of(status: &Status) -> Look {
        match status {
            Status::Pointing(Activity::Tracking) => Look::Active,
            Status::Starting | Status::Waiting(_) | Status::Pointing(_) => Look::Attention,
            Status::Paused => Look::Paused,
            Status::Calibrating => Look::Calibrating,
            Status::Stopped(_) => Look::Stopped,
        }
    }

    /// Red, green, blue.
    fn colour(self) -> [f32; 3] {
        match self {
            Look::Active => [46.0, 184.0, 92.0],
            Look::Attention => [242.0, 169.0, 0.0],
            Look::Paused => [138.0, 138.0, 138.0],
            Look::Calibrating => [45.0, 127.0, 249.0],
            Look::Stopped => [217.0, 59.0, 59.0],
        }
    }
}

const PUPIL: [f32; 3] = [28.0, 28.0, 28.0];
/// The pupil's radius as a share of the picture's side.
const PUPIL_SHARE: f32 = 0.2;

/// A disc in the look's colour with a dark pupil, its edges softened.
pub fn picture(size: usize, look: Look) -> IconImage {
    let centre = size as f32 * 0.5;
    let (outer, inner) = (centre - 0.5, size as f32 * PUPIL_SHARE);
    let colour = look.colour();
    let mut pixels = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let distance = (x as f32 + 0.5 - centre).hypot(y as f32 + 0.5 - centre);
            // How much of the pixel lies inside each circle.
            let alpha = (outer - distance + 0.5).clamp(0.0, 1.0);
            let pupil = (inner - distance + 0.5).clamp(0.0, 1.0);
            // The colour channels are stored already multiplied by the alpha.
            let channel = |index: usize| ((colour[index] + (PUPIL[index] - colour[index]) * pupil) * alpha).round() as u32;
            pixels.push(((alpha * 255.0).round() as u32) << 24 | channel(0) << 16 | channel(1) << 8 | channel(2));
        }
    }
    IconImage { size, pixels }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_picture_is_a_soft_edged_disc_with_a_dark_middle() {
        for size in [16usize, 20, 24, 32] {
            let image = picture(size, Look::Active);
            assert_eq!(image.pixels.len(), size * size);
            let at = |x: usize, y: usize| image.pixels[y * size + x];
            assert_eq!(at(0, 0), 0, "the corners are see-through");
            assert_eq!(at(size - 1, size - 1), 0);
            assert_eq!(at(size / 2, size / 2), 0xFF1C_1C1C, "the pupil");
            assert_eq!(at(size / 2, 1), 0xFF2E_B85C, "the ring in the look's colour");
            for &pixel in &image.pixels {
                let alpha = pixel >> 24;
                assert!((0..3).all(|shift| (pixel >> (shift * 8)) & 0xFF <= alpha), "{pixel:08X} is not premultiplied");
            }
            let soft = image.pixels.iter().filter(|&&pixel| !matches!(pixel >> 24, 0 | 255)).count();
            assert!(soft >= size, "only {soft} edge pixels are blended at size {size}");
        }
    }

    #[test]
    fn each_state_has_a_look_and_each_look_a_colour_of_its_own() {
        assert_eq!(Look::of(&Status::Pointing(Activity::Tracking)), Look::Active);
        assert_eq!(Look::of(&Status::Pointing(Activity::NoFace)), Look::Attention);
        assert_eq!(Look::of(&Status::Pointing(Activity::Yielded)), Look::Attention);
        assert_eq!(Look::of(&Status::Starting), Look::Attention);
        assert_eq!(Look::of(&Status::Waiting(String::new())), Look::Attention);
        assert_eq!(Look::of(&Status::Paused), Look::Paused);
        assert_eq!(Look::of(&Status::Calibrating), Look::Calibrating);
        assert_eq!(Look::of(&Status::Stopped(String::new())), Look::Stopped);
        let looks = [Look::Active, Look::Attention, Look::Paused, Look::Calibrating, Look::Stopped];
        let rings: std::collections::HashSet<u32> = looks.iter().map(|&look| picture(16, look).pixels[16 + 8]).collect();
        assert_eq!(rings.len(), looks.len());
    }
}
