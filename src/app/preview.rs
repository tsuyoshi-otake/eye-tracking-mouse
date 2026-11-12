//! The camera's picture in a window, for the person to check their seat, the
//! lighting, and whether the face and the eyes are found.
//!
//! The worker makes the pictures (`compose`); the thread that owns the windows
//! shows them (`PreviewWindow`). A picture only ever goes from memory to the
//! screen: none is stored or sent anywhere.

use crate::error::Result;
use crate::geometry::RectI;
use crate::image::BgraFrame;
use crate::platform::canvas::{Colour, Shape, Surface};
use crate::platform::window::{self, Kind, Window};
use crate::vision::landmarks::{FaceMesh, LEFT_EYE, Point3, RIGHT_EYE, RIGID};

use super::wording;

/// The picture is shrunk by a whole factor until it is at most this wide.
const MAX_WIDTH: usize = 640;
/// Marks, as 0x00RRGGBB: the points the head's pose is fitted to, the outline of the eyes, the irises.
const HEAD_MARK: u32 = 0x00FF_D60A;
const EYE_MARK: u32 = 0x002E_B85C;
const IRIS_MARK: u32 = 0x0000_E5FF;

const BACKGROUND: Colour = (18, 18, 22);
const CAPTION: Colour = (236, 236, 236);

/// Row by row from the top, each pixel 0x00RRGGBB.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u32>,
}

/// Colours the square of pixels around a point of the camera image, as far as it lies inside the picture.
fn mark(picture: &mut Picture, frame_width: usize, step: usize, point: Point3, reach: i32, colour: u32) {
    let x = (frame_width as f32 - 1.0 - point[0]) / step as f32;
    let y = point[1] / step as f32;
    let near = |value: f32, extent: usize| value >= -(reach as f32) && value <= (extent as i32 + reach) as f32;
    // Also turns away what is not a number at all.
    if !(near(x, picture.width) && near(y, picture.height)) {
        return;
    }
    let (x, y) = (x.round() as i32, y.round() as i32);
    for (dx, dy) in (-reach..=reach).flat_map(|dy| (-reach..=reach).map(move |dx| (dx, dy))) {
        let (x, y) = (x + dx, y + dy);
        if x >= 0 && y >= 0 && (x as usize) < picture.width && (y as usize) < picture.height {
            picture.pixels[y as usize * picture.width + x as usize] = colour;
        }
    }
}

/// Makes `picture` show `frame` the way a mirror would, shrunk, with the
/// points found on the face marked. The picture's buffer is reused.
pub fn compose(frame: &BgraFrame, mesh: Option<&FaceMesh>, picture: &mut Picture) {
    picture.pixels.clear();
    (picture.width, picture.height) = (0, 0);
    let step = frame.width.div_ceil(MAX_WIDTH).max(1);
    let (width, height) = (frame.width / step, frame.height / step);
    // Nothing to show, or a frame that does not hold what its size promises: the picture stays empty.
    if width == 0 || height == 0 || frame.data.len() < frame.width * frame.height * 4 {
        return;
    }
    (picture.width, picture.height) = (width, height);
    picture.pixels.reserve(width * height);
    for row in frame.data.chunks_exact(frame.width * 4).step_by(step).take(height) {
        for x in 0..width {
            // From the right, so that the person's right hand is on the right of the picture.
            let source = (frame.width - 1 - x * step) * 4;
            picture.pixels.push(u32::from(row[source + 2]) << 16 | u32::from(row[source + 1]) << 8 | u32::from(row[source]));
        }
    }
    let Some(mesh) = mesh else { return };
    let mut marks = |indices: &[usize], reach: i32, colour: u32| {
        for point in indices.iter().filter_map(|&index| mesh.points.get(index)) {
            mark(picture, frame.width, step, *point, reach, colour);
        }
    };
    marks(&RIGID, 1, HEAD_MARK);
    for eye in [&RIGHT_EYE, &LEFT_EYE] {
        let [(first_upper, first_lower), (second_upper, second_lower)] = eye.lid_pairs;
        let outline = [eye.corners.0, eye.corners.1, eye.upper_lid, eye.lower_lid, first_upper, first_lower];
        marks(&outline, 1, EYE_MARK);
        marks(&[second_upper, second_lower], 1, EYE_MARK);
        marks(&[eye.iris], 2, IRIS_MARK);
    }
}

/// What the preview window shows when its inside is `size` pixels: the
/// picture as large as fits without distorting it, and a line of text below.
pub fn scene<'a>(picture: &'a Picture, caption: &'a str, size: (i32, i32)) -> Vec<Shape<'a>> {
    let strip = (size.1 / 12).clamp(20, 48).min(size.1);
    let room = (size.0, size.1 - strip);
    let mut shapes = Vec::new();
    let (width, height) = (picture.width as i64, picture.height as i64);
    if width > 0 && height > 0 && room.0 > 0 && room.1 > 0 {
        // The larger of the two that fit: full width, or full height.
        let (shown_width, shown_height) = if i64::from(room.0) * height <= i64::from(room.1) * width {
            (room.0, (i64::from(room.0) * height / width) as i32)
        } else {
            ((i64::from(room.1) * width / height) as i32, room.1)
        };
        let (left, top) = ((room.0 - shown_width) / 2, (room.1 - shown_height) / 2);
        shapes.push(Shape::Picture {
            pixels: &picture.pixels,
            size: (picture.width, picture.height),
            area: RectI::new(left, top, left + shown_width, top + shown_height),
        });
    }
    let line = RectI::new(0, size.1 - strip, size.0, size.1);
    shapes.push(Shape::Text { text: caption, area: line, height: strip * 6 / 10, colour: CAPTION });
    shapes
}

pub struct PreviewWindow {
    window: Window,
    surface: Surface,
    picture: Picture,
    caption: String,
}

impl PreviewWindow {
    /// Opens the window on `screen` (desktop pixels) without taking the keyboard from what the person is doing.
    pub fn open(class: &str, screen: RectI) -> Result<PreviewWindow> {
        // A third of the screen wide, shaped for a 16:9 picture with the line of text below.
        let width = (screen.width() / 3).max(320);
        let height = width * 9 / 16 + width / 8;
        let (left, top) = (screen.left + screen.width() / 20, screen.top + screen.height() / 20);
        let frame = RectI::new(left, top, left + width, top + height);
        let window = window::open(class, wording::PREVIEW_TITLE, Kind::Framed, frame)?;
        let preview =
            PreviewWindow { window, surface: Surface::new(window), picture: Picture::default(), caption: String::new() };
        window::show(window, false);
        Ok(preview)
    }

    pub fn window(&self) -> Window {
        self.window
    }

    /// The picture on show, to be exchanged for a newer one.
    pub fn picture(&mut self) -> &mut Picture {
        &mut self.picture
    }

    /// Sets the line of text below the picture; `watching` says whether the camera is delivering.
    /// A picture from before the camera stopped is not left standing.
    pub fn describe(&mut self, caption: String, watching: bool) {
        self.caption = caption;
        if !watching {
            self.picture = Picture::default();
        }
    }

    pub fn draw(&mut self) {
        let size = window::client_size(self.window);
        self.surface.paint(BACKGROUND, &scene(&self.picture, &self.caption, size));
    }
}

impl Drop for PreviewWindow {
    fn drop(&mut self) {
        window::destroy(self.window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::landmarks::LANDMARK_COUNT;

    /// A frame whose pixel at (x, y) has blue = x, green = y and red = 200.
    fn gradient(width: usize, height: usize) -> BgraFrame {
        let mut frame = BgraFrame::new(width, height);
        for (index, pixel) in frame.data.chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&[(index % width) as u8, (index / width) as u8, 200, 0]);
        }
        frame
    }

    fn pixel(picture: &Picture, x: usize, y: usize) -> u32 {
        picture.pixels[y * picture.width + x]
    }

    #[test]
    fn the_picture_is_the_frame_mirrored_and_shrunk_by_a_whole_factor() {
        let mut picture = Picture::default();
        compose(&gradient(1280, 720), None, &mut picture);
        assert_eq!((picture.width, picture.height, picture.pixels.len()), (640, 360, 640 * 360));
        // Picture (0, 0) is the frame's top right pixel, x = 1279: blue 1279 % 256 = 255, green 0.
        assert_eq!(pixel(&picture, 0, 0), 0x00C8_00FF);
        // Picture (639, 100) is frame (1, 200).
        assert_eq!(pixel(&picture, 639, 100), 0x00C8_C801);

        // A frame that is small enough already keeps its size, and an odd width is not read past its end.
        compose(&gradient(100, 50), None, &mut picture);
        assert_eq!((picture.width, picture.height), (100, 50));
        assert_eq!(pixel(&picture, 99, 49), 0x00C8_3100);
        compose(&gradient(641, 3), None, &mut picture);
        assert_eq!((picture.width, picture.height, picture.pixels.len()), (320, 1, 320));
    }

    #[test]
    fn the_points_of_the_face_are_marked_where_a_mirror_would_show_them() {
        let frame = gradient(1280, 720);
        let mut mesh = FaceMesh { points: vec![[-500.0, -500.0, 0.0]; LANDMARK_COUNT], presence: 1.0, time: 0.0 };
        mesh.points[RIGHT_EYE.iris] = [279.0, 400.0, 0.0];
        mesh.points[LEFT_EYE.corners.1] = [1279.0, 0.0, 0.0];
        mesh.points[RIGID[0]] = [f32::NAN, f32::INFINITY, 0.0];
        let mut picture = Picture::default();
        compose(&frame, Some(&mesh), &mut picture);
        // Frame x 279 is 1000 from the right edge: picture x 500. Around it, a square five pixels wide.
        assert_eq!(pixel(&picture, 500, 200), IRIS_MARK);
        assert_eq!(pixel(&picture, 498, 198), IRIS_MARK);
        assert_eq!(pixel(&picture, 502, 202), IRIS_MARK);
        assert_ne!(pixel(&picture, 503, 200), IRIS_MARK);
        // A point on the frame's edge is marked as far as the picture goes.
        assert_eq!(pixel(&picture, 0, 0), EYE_MARK);
        assert_eq!(pixel(&picture, 1, 1), EYE_MARK);
        let marked = picture.pixels.iter().filter(|&&p| [IRIS_MARK, EYE_MARK, HEAD_MARK].contains(&p)).count();
        assert_eq!(marked, 25 + 4, "points outside the frame, or nowhere at all, leave no mark");

        // A mesh that is shorter than it should be is drawn as far as it goes.
        mesh.points.truncate(10);
        compose(&frame, Some(&mesh), &mut picture);
        assert_eq!(picture.pixels.len(), 640 * 360);
    }

    #[test]
    fn a_frame_without_its_pixels_gives_an_empty_picture() {
        let mut picture = Picture { width: 4, height: 1, pixels: vec![1, 2, 3, 4] };
        let broken = BgraFrame { width: 1280, height: 720, data: vec![0; 100], timestamp: 0.0 };
        compose(&broken, None, &mut picture);
        assert_eq!(picture, Picture::default());
        compose(&BgraFrame::default(), None, &mut picture);
        assert_eq!(picture, Picture::default());
    }

    #[test]
    fn the_window_shows_the_whole_picture_undistorted_above_the_caption() {
        let picture = Picture { width: 640, height: 360, pixels: vec![0; 640 * 360] };
        let shown = |size: (i32, i32)| match scene(&picture, "動作中", size)[..] {
            [Shape::Picture { area, size: (640, 360), .. }, Shape::Text { text: "動作中", area: line, .. }] => (area, line),
            ref other => panic!("{other:?}"),
        };
        // Wider than the picture: full height, centred sideways.
        let (area, line) = shown((1000, 408));
        assert_eq!(line, RectI::new(0, 374, 1000, 408));
        assert_eq!((area.top, area.bottom), (0, 374));
        assert_eq!(area.width(), 374 * 640 / 360);
        assert_eq!(area.left, 1000 - area.right, "the same margin on both sides");
        // Taller than the picture: full width, centred between the top and the caption.
        let (area, line) = shown((640, 808));
        assert_eq!((area.left, area.right, area.height()), (0, 640, 360));
        assert_eq!(area.top, line.top - area.bottom);

        // Without a picture, or without room for it, only the caption is left.
        assert!(matches!(scene(&Picture::default(), "一時停止中", (640, 400))[..], [Shape::Text { .. }]));
        assert!(matches!(scene(&picture, "x", (640, 10))[..], [Shape::Text { .. }]));
    }
}
