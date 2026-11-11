//! The pixel container handed from the camera to everything that looks at a frame.

use crate::error::{Error, Result};

/// A top-down, tightly packed BGRA image (4 bytes per pixel, row stride = `width * 4`).
/// The fourth byte is padding; nothing reads it as alpha.
#[derive(Clone, Debug, Default)]
pub struct BgraFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
    /// Capture time on the camera's clock, in seconds.
    pub timestamp: f64,
}

impl BgraFrame {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, data: vec![0; width * height * 4], timestamp: 0.0 }
    }

    /// Decodes an uncompressed 24- or 32-bit Windows bitmap file, the format
    /// every paint program can export. Used to run the tracker on a still picture.
    pub fn from_bmp(file: &[u8]) -> Result<Self> {
        let field = |at: usize, len: usize| -> Result<i64> {
            let bytes = file.get(at..at + len).ok_or_else(|| Error::new("the bitmap file is cut short"))?;
            let mut value = [0u8; 8];
            value[..len].copy_from_slice(bytes);
            Ok(i64::from_le_bytes(value))
        };
        if file.get(..2) != Some(b"BM".as_slice()) {
            return Err(Error::new("not a bitmap (.bmp) file"));
        }
        let pixels_at = field(10, 4)? as usize;
        let width = field(18, 4)? as i32;
        let signed_height = field(22, 4)? as i32;
        let (bits, compression) = (field(28, 2)?, field(30, 4)?);
        // Compression 3 (bit fields) is how 32-bit files are usually written; the default masks are BGRA.
        if !(bits == 24 || bits == 32) || !(compression == 0 || (bits == 32 && compression == 3)) {
            return Err(Error::new("only uncompressed 24- or 32-bit bitmaps are supported"));
        }
        if width <= 0 || signed_height == 0 || width > 16_384 || signed_height.unsigned_abs() > 16_384 {
            return Err(Error::new("the bitmap has an unusable size"));
        }
        let (width, height) = (width as usize, signed_height.unsigned_abs() as usize);
        let bytes_per_pixel = bits as usize / 8;
        let stride = (width * bytes_per_pixel + 3) & !3;
        let rows = pixels_at
            .checked_add(stride * height)
            .and_then(|end| file.get(pixels_at..end))
            .ok_or_else(|| Error::new("the bitmap file is cut short"))?;
        let mut frame = Self::new(width, height);
        for y in 0..height {
            // A positive height means the rows are stored bottom-up.
            let stored = if signed_height > 0 { height - 1 - y } else { y };
            let row = &rows[stored * stride..stored * stride + width * bytes_per_pixel];
            for (x, pixel) in row.chunks_exact(bytes_per_pixel).enumerate() {
                frame.data[(y * width + x) * 4..(y * width + x) * 4 + 3].copy_from_slice(&pixel[..3]);
            }
        }
        Ok(frame)
    }

    /// Makes the buffer exactly fit `width` x `height`, reusing the allocation.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.data.resize(width * height * 4, 0);
    }

    /// (blue, green, red) at integer pixel coordinates clamped to the image.
    #[inline]
    pub fn bgr_clamped(&self, x: i32, y: i32) -> [u8; 3] {
        let x = x.clamp(0, self.width as i32 - 1) as usize;
        let y = y.clamp(0, self.height as i32 - 1) as usize;
        let i = (y * self.width + x) * 4;
        [self.data[i], self.data[i + 1], self.data[i + 2]]
    }

    /// 8-bit luma (BT.601 weights) of every pixel, row by row.
    pub fn to_gray(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.width * self.height);
        for px in self.data.chunks_exact(4) {
            let luma = 29 * px[0] as u32 + 150 * px[1] as u32 + 77 * px[2] as u32;
            out.push((luma >> 8) as u8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bitmap file with the given header fields and pixel bytes.
    fn bmp(width: i32, height: i32, bits: u16, pixels: &[u8]) -> Vec<u8> {
        let mut file = vec![0u8; 54];
        file[..2].copy_from_slice(b"BM");
        file[10..14].copy_from_slice(&54u32.to_le_bytes());
        file[14..18].copy_from_slice(&40u32.to_le_bytes());
        file[18..22].copy_from_slice(&width.to_le_bytes());
        file[22..26].copy_from_slice(&height.to_le_bytes());
        file[26..28].copy_from_slice(&1u16.to_le_bytes());
        file[28..30].copy_from_slice(&bits.to_le_bytes());
        file.extend_from_slice(pixels);
        file
    }

    #[test]
    fn a_bottom_up_24_bit_bitmap_is_read_top_down_without_its_row_padding() {
        // 3 x 2 pixels, rows padded from 9 to 12 bytes; the first stored row is the bottom row of the image.
        let bottom = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0];
        let top = [11, 12, 13, 14, 15, 16, 17, 18, 19, 0, 0, 0];
        let frame = BgraFrame::from_bmp(&bmp(3, 2, 24, &[bottom, top].concat())).unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
        assert_eq!(frame.bgr_clamped(0, 0), [11, 12, 13]);
        assert_eq!(frame.bgr_clamped(2, 0), [17, 18, 19]);
        assert_eq!(frame.bgr_clamped(1, 1), [4, 5, 6]);
    }

    #[test]
    fn a_top_down_32_bit_bitmap_keeps_its_row_order() {
        let pixels = [1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255];
        let frame = BgraFrame::from_bmp(&bmp(2, -2, 32, &pixels)).unwrap();
        assert_eq!(frame.bgr_clamped(1, 0), [4, 5, 6]);
        assert_eq!(frame.bgr_clamped(0, 1), [7, 8, 9]);
    }

    #[test]
    fn broken_or_unsupported_bitmaps_are_refused() {
        assert!(BgraFrame::from_bmp(b"PNG").is_err());
        assert!(BgraFrame::from_bmp(&bmp(2, 2, 24, &[0; 8])).is_err(), "cut short");
        assert!(BgraFrame::from_bmp(&bmp(2, 2, 8, &[0; 8])).is_err(), "palette images");
        assert!(BgraFrame::from_bmp(&bmp(0, 2, 24, &[])).is_err());
        assert!(BgraFrame::from_bmp(&bmp(i32::MAX, i32::MAX, 24, &[])).is_err());
    }

    #[test]
    fn gray_uses_luma_weights_and_keeps_white_white() {
        let mut frame = BgraFrame::new(2, 1);
        frame.data.copy_from_slice(&[255, 255, 255, 0, 0, 0, 255, 0]);
        let mut gray = Vec::new();
        frame.to_gray(&mut gray);
        assert_eq!(gray, vec![255, 76]);
    }

    #[test]
    fn clamped_lookup_replicates_the_border() {
        let mut frame = BgraFrame::new(2, 2);
        frame.data[12..15].copy_from_slice(&[1, 2, 3]);
        assert_eq!(frame.bgr_clamped(9, 9), [1, 2, 3]);
        assert_eq!(frame.bgr_clamped(-5, -5), [0, 0, 0]);
    }
}
