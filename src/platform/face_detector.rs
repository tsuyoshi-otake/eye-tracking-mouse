//! Finds faces in a whole frame with the detector built into Windows.
//!
//! Contract: `largest_face` returns the biggest face box in frame pixels, or
//! `None`. It looks at a half-size grey copy, which is plenty for a face that
//! fills a useful part of a webcam image, and takes a few tens of milliseconds:
//! call it to find a face, not to follow one.

use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
use windows::Media::FaceAnalysis::FaceDetector as OsFaceDetector;

use crate::error::{Context, Error, Result};
use crate::image::BgraFrame;

use super::inference::buffer_from_bytes;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceRect {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

pub struct FaceDetector {
    detector: OsFaceDetector,
    gray: Vec<u8>,
    half: Vec<u8>,
}

impl FaceDetector {
    pub fn new() -> Result<FaceDetector> {
        if !OsFaceDetector::IsSupported().context("ask whether face detection is available")? {
            return Err(Error::new("this Windows installation has no face detector"));
        }
        let detector = OsFaceDetector::CreateAsync()?.join().context("create the face detector")?;
        Ok(FaceDetector { detector, gray: Vec::new(), half: Vec::new() })
    }

    pub fn largest_face(&mut self, frame: &BgraFrame) -> Result<Option<FaceRect>> {
        // Rows of a grey bitmap are padded to four bytes; a width that is a multiple of four avoids padding.
        let (width, height) = ((frame.width / 2) & !3, frame.height / 2);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        frame.to_gray(&mut self.gray);
        self.half.clear();
        for y in 0..height {
            let (upper, lower) = (2 * y * frame.width, (2 * y + 1) * frame.width);
            self.half.extend((0..width).map(|x| {
                let sum = u16::from(self.gray[upper + 2 * x])
                    + u16::from(self.gray[upper + 2 * x + 1])
                    + u16::from(self.gray[lower + 2 * x])
                    + u16::from(self.gray[lower + 2 * x + 1]);
                (sum / 4) as u8
            }));
        }
        let bitmap = SoftwareBitmap::CreateCopyFromBuffer(
            &buffer_from_bytes(&self.half)?,
            BitmapPixelFormat::Gray8,
            width as i32,
            height as i32,
        )
        .context("hand the frame to the face detector")?;
        let faces = self.detector.DetectFacesAsync(&bitmap)?.join().context("detect faces")?;
        let mut largest: Option<FaceRect> = None;
        for face in faces {
            let bounds = face.FaceBox()?;
            let rect = FaceRect {
                left: bounds.X as f32 * 2.0,
                top: bounds.Y as f32 * 2.0,
                width: bounds.Width as f32 * 2.0,
                height: bounds.Height as f32 * 2.0,
            };
            if largest.is_none_or(|best| rect.width * rect.height > best.width * best.height) {
                largest = Some(rect);
            }
        }
        Ok(largest)
    }
}
