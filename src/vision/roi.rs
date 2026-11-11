//! The square region of the camera image that is handed to the landmark model.
//!
//! Follows MediaPipe's recipe: take the bounding box of the face, make it
//! square on its longer side, add 25 % on every side and turn it so the eyes
//! are level. Crop coordinates run 0..256 on both axes with pixel centres at
//! half-integers; the same mapping takes the model's answer back to the image.

use crate::geometry::Vec2;
use crate::image::BgraFrame;

use super::landmarks::{EYE_SPAN, INPUT_LEN, INPUT_SIDE, Point3, xy};

/// Box side divided by face size: 25 % margin on each side.
const MARGIN_SCALE: f32 = 1.5;
/// Smaller crops than this carry too little detail to track.
const MIN_SIZE: f32 = 48.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Roi {
    /// Centre in image pixels.
    pub center: Vec2,
    /// Side length in image pixels.
    pub size: f32,
    /// Angle of the crop's x axis in the image (radians, positive is clockwise on screen).
    pub rotation: f32,
}

impl Roi {
    /// From an upright face box, as a face detector reports it.
    pub fn from_box(left: f32, top: f32, width: f32, height: f32) -> Roi {
        Roi {
            center: Vec2::new(left + width * 0.5, top + height * 0.5),
            size: width.max(height) * MARGIN_SCALE,
            rotation: 0.0,
        }
    }

    /// From the landmarks of the previous frame.
    pub fn from_mesh(points: &[Point3]) -> Roi {
        let mut low = Vec2::new(f32::INFINITY, f32::INFINITY);
        let mut high = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for p in points {
            low = Vec2::new(low.x.min(p[0]), low.y.min(p[1]));
            high = Vec2::new(high.x.max(p[0]), high.y.max(p[1]));
        }
        let eye_line = xy(points, EYE_SPAN.1) - xy(points, EYE_SPAN.0);
        Roi {
            center: (low + high) * 0.5,
            size: (high.x - low.x).max(high.y - low.y) * MARGIN_SCALE,
            rotation: eye_line.y.atan2(eye_line.x),
        }
    }

    /// Whether the region is sane enough to crop from a frame of this size.
    pub fn is_usable(&self, frame_width: usize, frame_height: usize) -> bool {
        let (w, h) = (frame_width as f32, frame_height as f32);
        self.center.is_finite()
            && self.rotation.is_finite()
            && self.size >= MIN_SIZE
            && self.size <= 4.0 * w.max(h)
            && (0.0..=w).contains(&self.center.x)
            && (0.0..=h).contains(&self.center.y)
    }

    /// Image pixels per crop pixel.
    pub fn scale(&self) -> f32 {
        self.size / INPUT_SIDE as f32
    }

    /// Maps a crop coordinate (0..256) to image pixels.
    pub fn to_image(&self, x: f32, y: f32) -> Vec2 {
        let half = INPUT_SIDE as f32 * 0.5;
        let (dx, dy) = ((x - half) * self.scale(), (y - half) * self.scale());
        let (sin, cos) = self.rotation.sin_cos();
        Vec2::new(self.center.x + dx * cos - dy * sin, self.center.y + dx * sin + dy * cos)
    }

    /// Fills `out` with the crop as the model wants it: planar R, G, B in 0..1.
    /// Pixels outside the frame repeat the nearest edge pixel.
    pub fn extract(&self, frame: &BgraFrame, out: &mut [f32]) {
        assert_eq!(out.len(), INPUT_LEN);
        let plane = INPUT_SIDE * INPUT_SIDE;
        let (sin, cos) = self.rotation.sin_cos();
        let step = self.scale();
        // Image position of crop pixel (0, 0)'s centre, then one step per column and row.
        let origin = self.to_image(0.5, 0.5);
        for row in 0..INPUT_SIDE {
            let row_x = origin.x - row as f32 * step * sin;
            let row_y = origin.y + row as f32 * step * cos;
            for column in 0..INPUT_SIDE {
                let x = row_x + column as f32 * step * cos;
                let y = row_y + column as f32 * step * sin;
                let rgb = sample_rgb(frame, x, y);
                let at = row * INPUT_SIDE + column;
                out[at] = rgb[0];
                out[plane + at] = rgb[1];
                out[2 * plane + at] = rgb[2];
            }
        }
    }

    /// Converts the model's output (x, y, z triples in crop pixels) to image pixels.
    pub fn project(&self, raw: &[f32], points: &mut Vec<Point3>) {
        points.clear();
        points.extend(raw.chunks_exact(3).map(|p| {
            let image = self.to_image(p[0], p[1]);
            [image.x, image.y, p[2] * self.scale()]
        }));
    }
}

/// Bilinear sample at an image position where pixel (i, j) covers [i, i+1) x [j, j+1).
fn sample_rgb(frame: &BgraFrame, x: f32, y: f32) -> [f32; 3] {
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (tx, ty) = (fx - x0, fy - y0);
    let (x0, y0) = (x0 as i32, y0 as i32);
    let corners = [
        (frame.bgr_clamped(x0, y0), (1.0 - tx) * (1.0 - ty)),
        (frame.bgr_clamped(x0 + 1, y0), tx * (1.0 - ty)),
        (frame.bgr_clamped(x0, y0 + 1), (1.0 - tx) * ty),
        (frame.bgr_clamped(x0 + 1, y0 + 1), tx * ty),
    ];
    let mut rgb = [0.0f32; 3];
    for (bgr, weight) in corners {
        rgb[0] += f32::from(bgr[2]) * weight;
        rgb[1] += f32::from(bgr[1]) * weight;
        rgb[2] += f32::from(bgr[0]) * weight;
    }
    rgb.map(|v| v / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::landmarks::LANDMARK_COUNT;

    const SIDE: usize = INPUT_SIDE;

    /// A frame whose pixel (x, y) is blue = x, green = y, red = (x + y) mod 256.
    fn gradient_frame(width: usize, height: usize) -> BgraFrame {
        let mut frame = BgraFrame::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let at = (y * width + x) * 4;
                frame.data[at] = x as u8;
                frame.data[at + 1] = y as u8;
                frame.data[at + 2] = ((x + y) % 256) as u8;
            }
        }
        frame
    }

    fn crop(roi: &Roi, frame: &BgraFrame) -> Vec<f32> {
        let mut out = vec![0.0; INPUT_LEN];
        roi.extract(frame, &mut out);
        out
    }

    fn pixel(crop: &[f32], column: usize, row: usize) -> [f32; 3] {
        let at = row * SIDE + column;
        [crop[at] * 255.0, crop[SIDE * SIDE + at] * 255.0, crop[2 * SIDE * SIDE + at] * 255.0]
    }

    fn assert_close(actual: [f32; 3], expected: [f32; 3]) {
        for channel in 0..3 {
            assert!((actual[channel] - expected[channel]).abs() < 0.01, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn a_box_becomes_a_square_with_a_quarter_margin_on_each_side() {
        let roi = Roi::from_box(100.0, 50.0, 80.0, 120.0);
        assert_eq!(roi.center, Vec2::new(140.0, 110.0));
        assert_eq!(roi.size, 180.0);
        assert_eq!(roi.rotation, 0.0);
    }

    #[test]
    fn an_identity_region_copies_pixels_exactly_as_planar_rgb() {
        let frame = gradient_frame(SIDE, SIDE);
        let roi = Roi { center: Vec2::new(128.0, 128.0), size: 256.0, rotation: 0.0 };
        let out = crop(&roi, &frame);
        assert_close(pixel(&out, 0, 0), [0.0, 0.0, 0.0]);
        assert_close(pixel(&out, 200, 3), [203.0, 3.0, 200.0]);
        assert_close(pixel(&out, 255, 255), [254.0, 255.0, 255.0]);
    }

    #[test]
    fn a_quarter_turn_reads_the_frame_sideways() {
        let frame = gradient_frame(SIDE, SIDE);
        let roi = Roi { center: Vec2::new(128.0, 128.0), size: 256.0, rotation: std::f32::consts::FRAC_PI_2 };
        let out = crop(&roi, &frame);
        // The crop's x axis points down the image: crop (column, row) is image (255 - row, column).
        assert_close(pixel(&out, 10, 0), [9.0, 10.0, 255.0]);
        assert_close(pixel(&out, 0, 40), [215.0, 0.0, 215.0]);
    }

    #[test]
    fn a_half_size_region_averages_neighbouring_pixels() {
        let frame = gradient_frame(SIDE, SIDE);
        let roi = Roi { center: Vec2::new(64.0, 64.0), size: 128.0, rotation: 0.0 };
        let out = crop(&roi, &frame);
        // Crop pixel 4 has its centre at image x = 2.25: three quarters of the way
        // from the centre of pixel 1 (1.5) to the centre of pixel 2 (2.5).
        assert_close(pixel(&out, 4, 4), [3.5, 1.75, 1.75]);
    }

    #[test]
    fn pixels_outside_the_frame_repeat_the_edge() {
        let frame = gradient_frame(64, 64);
        let roi = Roi { center: Vec2::new(0.0, 32.0), size: 256.0, rotation: 0.0 };
        let out = crop(&roi, &frame);
        assert_close(pixel(&out, 0, 128), [32.0, 32.0, 0.0]);
        assert_close(pixel(&out, 255, 255), [126.0, 63.0, 63.0]);
    }

    #[test]
    fn projecting_undoes_the_crop_mapping() {
        let roi = Roi { center: Vec2::new(300.0, 200.0), size: 512.0, rotation: 0.3 };
        let mut points = Vec::new();
        roi.project(&[128.0, 128.0, 5.0, 256.0, 128.0, -2.0], &mut points);
        assert!((points[0][0] - 300.0).abs() < 1e-3 && (points[0][1] - 200.0).abs() < 1e-3);
        assert!((points[0][2] - 10.0).abs() < 1e-4);
        let along = Vec2::new(points[1][0] - 300.0, points[1][1] - 200.0);
        assert!((along.length() - 256.0).abs() < 1e-3);
        assert!((along.y.atan2(along.x) - 0.3).abs() < 1e-5);
    }

    #[test]
    fn a_mesh_gives_a_margin_box_turned_along_the_eye_line() {
        let mut points = vec![[100.0, 100.0, 0.0]; LANDMARK_COUNT];
        points[0] = [40.0, 60.0, 0.0];
        points[1] = [160.0, 220.0, 0.0];
        points[EYE_SPAN.0] = [60.0, 100.0, 0.0];
        points[EYE_SPAN.1] = [140.0, 180.0, 0.0];
        let roi = Roi::from_mesh(&points);
        assert_eq!(roi.center, Vec2::new(100.0, 140.0));
        assert_eq!(roi.size, 240.0);
        assert!((roi.rotation - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    }

    #[test]
    fn unusable_regions_are_rejected() {
        let good = Roi { center: Vec2::new(640.0, 360.0), size: 300.0, rotation: 0.1 };
        assert!(good.is_usable(1280, 720));
        assert!(!Roi { size: 10.0, ..good }.is_usable(1280, 720));
        assert!(!Roi { size: f32::NAN, ..good }.is_usable(1280, 720));
        assert!(!Roi { center: Vec2::new(-5.0, 360.0), ..good }.is_usable(1280, 720));
        assert!(!Roi { rotation: f32::INFINITY, ..good }.is_usable(1280, 720));
    }
}
