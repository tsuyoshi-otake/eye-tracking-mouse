//! USB camera capture through Media Foundation's source reader.
//!
//! Contract: `Camera::open` picks the native format closest to the request and
//! asks the OS to convert it to RGB32, so callers always receive top-down BGRA
//! frames regardless of what the device produces (MJPG, YUY2, NV12...).
//! `read` blocks until the next frame. Any error means the device is gone or
//! busy; the caller owns the retry policy.

use std::ffi::c_void;
use std::time::Instant;

use windows::Win32::Media::MediaFoundation::{
    IMF2DBuffer, IMFActivate, IMFAttributes, IMFMediaSource, IMFMediaType, IMFSample,
    IMFSourceReader, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID, MF_E_NO_MORE_TYPES, MF_MT_FRAME_RATE,
    MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_SOURCE_READER_ALL_STREAMS,
    MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, MF_SOURCE_READER_FIRST_VIDEO_STREAM,
    MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED, MF_SOURCE_READERF_ENDOFSTREAM,
    MF_SOURCE_READERF_ERROR, MFCreateAttributes, MFCreateMediaType,
    MFCreateSourceReaderFromMediaSource, MFEnumDeviceSources, MFMediaType_Video,
    MFVideoFormat_MJPG, MFVideoFormat_RGB24, MFVideoFormat_RGB32,
};
use windows::Win32::System::Com::CoTaskMemFree;
use windows::core::{GUID, Interface, PWSTR};

use crate::error::{Context, Error, Result};
use crate::image::BgraFrame;
use crate::platform::runtime::MediaFoundation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CameraRequest {
    /// Position in `Camera::device_names()`.
    pub device_index: usize,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for CameraRequest {
    fn default() -> Self {
        Self { device_index: 0, width: 1280, height: 720, fps: 30 }
    }
}

pub struct Camera {
    // Released before the source is shut down; see `Drop`.
    reader: Option<IMFSourceReader>,
    source: IMFMediaSource,
    stream: u32,
    width: usize,
    height: usize,
    opened_at: Instant,
    description: String,
    // Declared last so Media Foundation outlives every MF object above.
    _media_foundation: MediaFoundation,
}

/// One format a device can produce natively.
struct NativeFormat {
    media_type: IMFMediaType,
    width: u32,
    height: u32,
    fps: f32,
    subtype: GUID,
}

impl Camera {
    /// Friendly names of the attached video capture devices, in device-index order.
    pub fn device_names() -> Result<Vec<String>> {
        let _mf = MediaFoundation::startup()?;
        let devices = enumerate_devices()?;
        devices.iter().map(friendly_name).collect()
    }

    pub fn open(request: &CameraRequest) -> Result<Camera> {
        let media_foundation = MediaFoundation::startup()?;
        let devices = enumerate_devices()?;
        if devices.is_empty() {
            return Err(Error::new("no camera is connected"));
        }
        let activate = devices.get(request.device_index).ok_or_else(|| {
            Error::new(format!(
                "camera index {} does not exist ({} camera(s) found)",
                request.device_index,
                devices.len()
            ))
        })?;
        let name = friendly_name(activate)?;
        let source: IMFMediaSource =
            unsafe { activate.ActivateObject() }.context("start the camera (is another app using it?)")?;

        let stream = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;
        let reader = unsafe {
            let attributes = new_attributes(1)?;
            // Lets the reader insert the OS decoder and colour converter we need for RGB32.
            attributes.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)?;
            let reader = MFCreateSourceReaderFromMediaSource(&source, &attributes)
                .context("create the camera reader")?;
            reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
            reader.SetStreamSelection(stream, true)?;
            reader
        };

        let native = pick_native_format(&reader, stream, request)?;
        let (width, height) = unsafe {
            reader
                .SetCurrentMediaType(stream, None, &native.media_type)
                .context("select the camera format")?;
            let output = MFCreateMediaType()?;
            output.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            output.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
            reader
                .SetCurrentMediaType(stream, None, &output)
                .context("convert camera frames to RGB32")?;
            frame_size(&reader.GetCurrentMediaType(stream)?)?
        };

        let description = format!(
            "{name}: {}x{} @ {:.0} fps ({})",
            native.width,
            native.height,
            native.fps,
            subtype_name(&native.subtype)
        );
        Ok(Camera {
            reader: Some(reader),
            source,
            stream,
            width: width as usize,
            height: height as usize,
            opened_at: Instant::now(),
            description,
            _media_foundation: media_foundation,
        })
    }

    /// Device name and the native format in use, for logs.
    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Blocks until the next frame and copies it into `frame`.
    /// Returns `Ok(false)` when the device signalled a gap instead of an image.
    pub fn read(&mut self, frame: &mut BgraFrame) -> Result<bool> {
        let reader = self.reader.as_ref().context("camera already closed")?;
        let mut flags = 0u32;
        let mut sample: Option<IMFSample> = None;
        unsafe { reader.ReadSample(self.stream, 0, None, Some(&mut flags), None, Some(&mut sample)) }
            .context("read a camera frame")?;
        let ended = (MF_SOURCE_READERF_ERROR.0 | MF_SOURCE_READERF_ENDOFSTREAM.0) as u32;
        if flags & ended != 0 {
            return Err(Error::new("the camera stream ended"));
        }
        if flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED.0 as u32 != 0 {
            let (w, h) = unsafe { frame_size(&reader.GetCurrentMediaType(self.stream)?)? };
            self.width = w as usize;
            self.height = h as usize;
        }
        let Some(sample) = sample else {
            return Ok(false);
        };
        frame.resize(self.width, self.height);
        frame.timestamp = self.opened_at.elapsed().as_secs_f64();
        unsafe { copy_sample(&sample, frame) }.context("copy a camera frame")?;
        Ok(true)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // Order matters: the reader must let go of the source before the source
        // is shut down, otherwise the device can stay claimed until process exit.
        self.reader = None;
        let _ = unsafe { self.source.Shutdown() };
    }
}

fn new_attributes(initial_size: u32) -> Result<IMFAttributes> {
    let mut attributes: Option<IMFAttributes> = None;
    unsafe { MFCreateAttributes(&mut attributes, initial_size) }?;
    attributes.context("Media Foundation returned no attribute store")
}

fn enumerate_devices() -> Result<Vec<IMFActivate>> {
    unsafe {
        let attributes = new_attributes(1)?;
        attributes.SetGUID(
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
            &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
        )?;
        let mut array: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attributes, &mut array, &mut count).context("list cameras")?;
        let mut devices = Vec::with_capacity(count as usize);
        if !array.is_null() {
            for i in 0..count as usize {
                // Move each interface out so it is released exactly once, by us.
                if let Some(device) = (*array.add(i)).take() {
                    devices.push(device);
                }
            }
            CoTaskMemFree(Some(array as *const c_void));
        }
        Ok(devices)
    }
}

fn friendly_name(device: &IMFActivate) -> Result<String> {
    unsafe {
        let mut text = PWSTR::null();
        let mut length = 0u32;
        device
            .GetAllocatedString(&MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &mut text, &mut length)
            .context("read the camera name")?;
        let name = String::from_utf16_lossy(std::slice::from_raw_parts(text.0, length as usize));
        CoTaskMemFree(Some(text.0 as *const c_void));
        Ok(name)
    }
}

fn pick_native_format(
    reader: &IMFSourceReader,
    stream: u32,
    request: &CameraRequest,
) -> Result<NativeFormat> {
    let mut best: Option<(FormatRank, NativeFormat)> = None;
    for index in 0.. {
        let media_type = match unsafe { reader.GetNativeMediaType(stream, index) } {
            Ok(media_type) => media_type,
            Err(e) if e.code() == MF_E_NO_MORE_TYPES => break,
            Err(e) => return Err(e).context("list camera formats"),
        };
        let Ok(format) = (unsafe { describe(media_type) }) else {
            continue; // A format we cannot describe is a format we cannot request.
        };
        let rank = rank_format(format.width, format.height, format.fps, is_compressed(&format), request);
        if best.as_ref().is_none_or(|(best_rank, _)| rank < *best_rank) {
            best = Some((rank, format));
        }
    }
    best.map(|(_, format)| format).context("the camera offers no usable video format")
}

/// Lower is better: size mismatch first, then missing frame rate, then decode cost.
type FormatRank = (u64, u32, u8);

fn rank_format(width: u32, height: u32, fps: f32, compressed: bool, request: &CameraRequest) -> FormatRank {
    let size_mismatch = u64::from(width.abs_diff(request.width)) + u64::from(height.abs_diff(request.height));
    // Frame rate beyond the request is worth nothing; falling short is penalised.
    let fps_shortfall = (request.fps as f32 - fps.min(request.fps as f32)).max(0.0);
    (size_mismatch, (fps_shortfall * 100.0) as u32, compressed as u8)
}

fn is_compressed(format: &NativeFormat) -> bool {
    format.subtype == MFVideoFormat_MJPG
}

unsafe fn describe(media_type: IMFMediaType) -> Result<NativeFormat> {
    unsafe {
        let subtype = media_type.GetGUID(&MF_MT_SUBTYPE)?;
        let (width, height) = frame_size(&media_type)?;
        let rate = media_type.GetUINT64(&MF_MT_FRAME_RATE)?;
        let (numerator, denominator) = ((rate >> 32) as u32, rate as u32);
        let fps = if denominator == 0 { 0.0 } else { numerator as f32 / denominator as f32 };
        Ok(NativeFormat { media_type, width, height, fps, subtype })
    }
}

unsafe fn frame_size(media_type: &IMFMediaType) -> Result<(u32, u32)> {
    let packed = unsafe { media_type.GetUINT64(&MF_MT_FRAME_SIZE) }.context("read the frame size")?;
    let (width, height) = ((packed >> 32) as u32, packed as u32);
    if width == 0 || height == 0 {
        return Err(Error::new("the camera reported an empty frame size"));
    }
    Ok((width, height))
}

fn subtype_name(subtype: &GUID) -> String {
    if *subtype == MFVideoFormat_RGB32 {
        return "RGB32".into();
    }
    if *subtype == MFVideoFormat_RGB24 {
        return "RGB24".into();
    }
    // Video subtypes carry their FOURCC in the first four bytes.
    let fourcc = subtype.data1.to_le_bytes();
    if fourcc.iter().all(|c| c.is_ascii_graphic()) {
        String::from_utf8_lossy(&fourcc).into_owned()
    } else {
        format!("{:08X}", subtype.data1)
    }
}

/// Copies an RGB32 sample into `frame` as top-down rows.
unsafe fn copy_sample(sample: &IMFSample, frame: &mut BgraFrame) -> Result<()> {
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer()?;
        // The 2D interface reports the real stride, including the sign that says
        // whether the image is stored bottom-up.
        let buffer_2d: IMF2DBuffer = buffer.cast().context("camera buffer has no 2D access")?;
        let mut scanline0: *mut u8 = std::ptr::null_mut();
        let mut pitch = 0i32;
        buffer_2d.Lock2D(&mut scanline0, &mut pitch)?;
        let row_bytes = frame.width * 4;
        let usable = !scanline0.is_null() && pitch.unsigned_abs() as usize >= row_bytes;
        if usable {
            for y in 0..frame.height {
                let source = scanline0.offset(y as isize * pitch as isize);
                let target = frame.data.as_mut_ptr().add(y * row_bytes);
                std::ptr::copy_nonoverlapping(source, target, row_bytes);
            }
        }
        buffer_2d.Unlock2D()?;
        if usable { Ok(()) } else { Err(Error::new("camera buffer is smaller than a frame")) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: CameraRequest = CameraRequest { device_index: 0, width: 1280, height: 720, fps: 30 };

    #[test]
    fn exact_size_beats_a_faster_wrong_size() {
        let exact = rank_format(1280, 720, 10.0, false, &REQUEST);
        let wrong = rank_format(640, 480, 30.0, false, &REQUEST);
        assert!(exact < wrong);
    }

    #[test]
    fn full_frame_rate_beats_avoiding_the_decoder() {
        // Typical UVC webcam: 720p is 30 fps only as MJPG, 10 fps uncompressed.
        let mjpg = rank_format(1280, 720, 30.0, true, &REQUEST);
        let yuy2 = rank_format(1280, 720, 10.0, false, &REQUEST);
        assert!(mjpg < yuy2);
    }

    #[test]
    fn uncompressed_wins_when_everything_else_ties() {
        let mjpg = rank_format(1280, 720, 30.0, true, &REQUEST);
        let yuy2 = rank_format(1280, 720, 30.0, false, &REQUEST);
        assert!(yuy2 < mjpg);
        // Extra frame rate beyond the request does not outrank the decode cost.
        assert!(yuy2 < rank_format(1280, 720, 60.0, true, &REQUEST));
    }

    #[test]
    fn subtype_names_show_the_fourcc() {
        assert_eq!(subtype_name(&MFVideoFormat_MJPG), "MJPG");
        assert_eq!(subtype_name(&MFVideoFormat_RGB32), "RGB32");
    }
}
