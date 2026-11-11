//! `--selftest <name>`: checks of the OS boundaries that unit tests cannot reach.
//!
//! Each check prints what it measured and ends with a single `PASS` or `FAIL`
//! line; the process exit code says the same thing.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetStockObject, HBRUSH, WHITE_BRUSH};
use windows::Win32::UI::WindowsAndMessaging::{
    DefWindowProcW, DestroyWindow, SW_SHOWNOACTIVATE, ShowWindow, WM_LBUTTONDOWN, WM_LBUTTONUP,
    WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_POPUP, WindowFromPoint,
};
use windows::core::w;

use crate::error::{Context, Error, Result};
use crate::geometry::{RectI, Vec2};
use crate::image::BgraFrame;
use crate::platform::camera::{Camera, CameraRequest};
use crate::platform::pointer::{self, Desktop};
use crate::platform::window;
use crate::vision::eyes;
use crate::vision::landmarks::{EYE_SPAN, FaceMesh, INPUT_LEN, INPUT_SIDE, LANDMARK_COUNT, xy};
use crate::vision::tracker::LandmarkModel;

use super::face::{self, WinMlLandmarks};

/// Runs the named check. `Ok(true)` means it passed. `image` replaces the
/// camera with a still picture where a check can use one.
pub fn run(name: &str, camera: &CameraRequest, image: Option<&str>) -> Result<bool> {
    let passed = match name {
        "camera" => camera_check(camera)?,
        "face" => face_check(camera, image)?,
        "model" => model_check()?,
        "pointer" => pointer_check()?,
        other => return Err(Error::new(format!("unknown selftest '{other}' (camera, face, model, pointer)"))),
    };
    println!("{}", if passed { "PASS" } else { "FAIL" });
    Ok(passed)
}

/// Captures for three seconds and checks frame rate, size and that the image is not blank.
fn camera_check(request: &CameraRequest) -> Result<bool> {
    const MIN_FPS: f64 = 20.0;
    const MIN_CONTRAST: f64 = 2.0;

    let mut camera = Camera::open(request)?;
    println!("camera: {}", camera.description());
    let mut frame = BgraFrame::default();
    // The first frames arrive late while exposure settles; do not count them.
    let warm_up = Instant::now();
    while warm_up.elapsed() < Duration::from_millis(1000) {
        camera.read(&mut frame)?;
    }
    let started = Instant::now();
    let mut frames = 0u32;
    while started.elapsed() < Duration::from_secs(3) {
        if camera.read(&mut frame)? {
            frames += 1;
        }
    }
    let fps = f64::from(frames) / started.elapsed().as_secs_f64();
    let mut gray = Vec::new();
    frame.to_gray(&mut gray);
    let (mean, deviation) = mean_and_deviation(&gray);
    println!("frames: {frames} in 3 s = {fps:.1} fps (need >= {MIN_FPS})");
    println!("size: {}x{} (requested {}x{})", frame.width, frame.height, request.width, request.height);
    println!("brightness: mean {mean:.1}, standard deviation {deviation:.1} (need > {MIN_CONTRAST})");
    let size_ok = frame.width == request.width as usize && frame.height == request.height as usize;
    Ok(fps >= MIN_FPS && size_ok && deviation > MIN_CONTRAST)
}

fn mean_and_deviation(values: &[u8]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let variance = values.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

/// Outputs of the original TensorFlow Lite model for the two test images:
/// 1434 landmark values followed by the face flag, little-endian `f32`.
const LCG_REFERENCE: &[u8] = include_bytes!("../../assets/reference/lcg_output.bin");
const RAMP_REFERENCE: &[u8] = include_bytes!("../../assets/reference/ramp_output.bin");

/// Builds a test image from a rule for the value at (row, column, channel), visited in that order.
fn test_image(mut value: impl FnMut(usize, usize, usize) -> f32) -> Vec<f32> {
    let mut image = vec![0.0; INPUT_LEN];
    for row in 0..INPUT_SIDE {
        for column in 0..INPUT_SIDE {
            for channel in 0..3 {
                image[(channel * INPUT_SIDE + row) * INPUT_SIDE + column] = value(row, column, channel);
            }
        }
    }
    image
}

/// Runs the embedded network through Windows ML on two fixed images and
/// compares with what the original model answered for them.
fn model_check() -> Result<bool> {
    const TOLERANCE: f32 = 0.05;
    let mut model = WinMlLandmarks::load()?;
    let mut state = 12345u32;
    let noise = test_image(|_, _, _| {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 8) as f32 / 16_777_216.0
    });
    let ramp = test_image(|row, column, channel| ((column * 3 + row * 5 + channel * 17) % 256) as f32 / 255.0);
    let mut landmarks = vec![0.0f32; LANDMARK_COUNT * 3];
    let mut passed = true;
    for (name, image, reference) in [("lcg", &noise, LCG_REFERENCE), ("ramp", &ramp, RAMP_REFERENCE)] {
        let expected: Vec<f32> =
            reference.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if expected.len() != landmarks.len() + 1 {
            return Err(Error::new("the embedded reference output has the wrong size"));
        }
        let flag = model.infer(image, &mut landmarks)?;
        let landmark_diff = landmarks.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let flag_diff = (flag - expected[landmarks.len()]).abs();
        println!("{name}: max abs diff landmarks {landmark_diff:.6}, face flag {flag_diff:.6} (need <= {TOLERANCE})");
        // A NaN anywhere fails both comparisons.
        passed &= landmarks.iter().all(|v| v.is_finite()) && landmark_diff <= TOLERANCE && flag_diff <= TOLERANCE;
    }
    const RUNS: u32 = 30;
    let started = Instant::now();
    for _ in 0..RUNS {
        model.infer(&ramp, &mut landmarks)?;
    }
    println!("inference: {:.1} ms per frame on the CPU", started.elapsed().as_secs_f64() * 1000.0 / f64::from(RUNS));
    Ok(passed)
}

enum Frames {
    Live(Camera),
    Still(BgraFrame),
}

impl Frames {
    fn next(&mut self, index: u32, frame: &mut BgraFrame) -> Result<bool> {
        match self {
            Frames::Live(camera) => camera.read(frame),
            Frames::Still(picture) => {
                frame.clone_from(picture);
                frame.timestamp = f64::from(index) / 30.0;
                Ok(true)
            }
        }
    }
}

/// Follows a face for 90 frames, from the camera or a picture, and checks the
/// landmark conventions the rest of the program relies on: which eye is which,
/// which way is down, and that a smaller z is nearer the camera.
fn face_check(request: &CameraRequest, image: Option<&str>) -> Result<bool> {
    const FRAMES: u32 = 90;
    // Landmarks outside the tables the tracker uses: nose tip, inner upper lip, face edge at each ear.
    const NOSE_TIP: usize = 1;
    const UPPER_LIP: usize = 13;
    const FACE_EDGES: (usize, usize) = (234, 454);

    let mut frames = match image {
        Some(path) => Frames::Still(BgraFrame::from_bmp(&std::fs::read(path).context("read the picture")?)?),
        None => Frames::Live(Camera::open(request)?),
    };
    let mut tracker = face::new_tracker()?;
    let mut frame = BgraFrame::default();
    let mut mesh = FaceMesh::default();
    let (mut seen, mut tracked) = (0u32, 0u32);
    let mut track_time = Duration::ZERO;
    let deadline = Instant::now() + Duration::from_secs(15);
    while seen < FRAMES && Instant::now() < deadline {
        if !frames.next(seen, &mut frame)? {
            continue;
        }
        seen += 1;
        let before = Instant::now();
        tracked += u32::from(tracker.track(&frame, &mut mesh)?);
        track_time += before.elapsed();
    }
    let per_frame = track_time.as_secs_f64() * 1000.0 / f64::from(seen.max(1));
    println!("tracked: {tracked} of {seen} frames, {per_frame:.1} ms per frame");
    if tracked == 0 {
        println!("no face was found: check the lighting and that the face is in the picture");
        return Ok(false);
    }
    let points = &mesh.points;
    let (right_eye, left_eye) = (xy(points, EYE_SPAN.0), xy(points, EYE_SPAN.1));
    let span = right_eye.distance(left_eye);
    let along = (left_eye - right_eye) * (1.0 / span);
    let down = Vec2::new(-along.y, along.x);
    let lips_below = (xy(points, UPPER_LIP) - (right_eye + left_eye) * 0.5).dot(down) / span;
    let nose_depth = ((points[FACE_EDGES.0][2] + points[FACE_EDGES.1][2]) * 0.5 - points[NOSE_TIP][2]) / span;
    println!("presence: {:.3} (need >= 0.9)", mesh.presence);
    println!("eye span: {span:.0} px, eye line tilted {:.1} degrees", along.y.atan2(along.x).to_degrees());
    println!("lips below the eye line by {lips_below:.2} eye spans (need > 0.3)");
    println!("nose tip in front of the face edges by {nose_depth:.2} eye spans (need > 0.3)");
    let eyes_ok = match eyes::measure(points) {
        Some(s) => {
            println!(
                "iris offset ({:+.3}, {:+.3}) eye widths, eye openness {:.2} and {:.2} (need 0.08..0.6)",
                s.iris.x, s.iris.y, s.openness[0], s.openness[1]
            );
            s.iris.x.abs() < 0.35 && s.iris.y.abs() < 0.35 && s.openness.iter().all(|o| (0.08..0.6).contains(o))
        }
        None => false,
    };
    Ok(tracked * 10 >= seen * 8 && mesh.presence >= 0.9 && lips_below > 0.3 && nose_depth > 0.3 && eyes_ok)
}

static TEST_WINDOW_DOWNS: AtomicU32 = AtomicU32::new(0);
static TEST_WINDOW_UPS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn test_window_proc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match message {
        WM_LBUTTONDOWN => {
            TEST_WINDOW_DOWNS.fetch_add(1, Ordering::SeqCst);
            LRESULT(0)
        }
        WM_LBUTTONUP => {
            TEST_WINDOW_UPS.fetch_add(1, Ordering::SeqCst);
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    }
}

/// Moves the real cursor across every monitor, clicks a window of our own, and
/// puts the cursor back. Keep hands off the mouse for the second this takes.
fn pointer_check() -> Result<bool> {
    let desktop = Desktop::query()?;
    let original = pointer::cursor_position().ok_or_else(|| Error::new("cannot read the cursor position"))?;
    let outcome = pointer_check_inner(&desktop);
    // Restore the cursor on every path, including failures above.
    let restored = pointer::move_to(&desktop, original.0, original.1).is_ok()
        && wait_for_cursor(original, Duration::from_millis(300));
    println!("cursor restored to {original:?}: {restored}");
    Ok(outcome? && restored)
}

fn pointer_check_inner(desktop: &Desktop) -> Result<bool> {
    println!("monitors: {:?}, primary {:?}", desktop.monitors(), desktop.primary());
    let mut targets = 0u32;
    let mut misses = 0u32;
    for monitor in desktop.monitors() {
        for fy in [0.0, 0.5, 1.0] {
            for fx in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let x = monitor.left + ((monitor.width() - 1) as f64 * fx).round() as i32;
                let y = monitor.top + ((monitor.height() - 1) as f64 * fy).round() as i32;
                pointer::move_to(desktop, x, y)?;
                targets += 1;
                if !wait_for_cursor((x, y), Duration::from_millis(300)) {
                    misses += 1;
                    println!("miss: asked ({x}, {y}), cursor at {:?}", pointer::cursor_position());
                }
            }
        }
    }
    println!("moves: {} of {targets} grid targets hit exactly", targets - misses);

    let clicks_ok = click_check(desktop)?;
    Ok(misses == 0 && clicks_ok)
}

fn wait_for_cursor(expected: (i32, i32), timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if pointer::cursor_position() == Some(expected) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn click_check(desktop: &Desktop) -> Result<bool> {
    const SIZE: i32 = 240;
    let class = w!("EyeTrackingMouseSelfTest");
    let background = HBRUSH(unsafe { GetStockObject(WHITE_BRUSH) }.0);
    window::register_class(class, Some(test_window_proc), Some(background))?;
    let primary = desktop.primary();
    let left = primary.left + (primary.width() - SIZE) / 2;
    let top = primary.top + (primary.height() - SIZE) / 2;
    let hwnd = window::create(
        class,
        w!("selftest"),
        WS_POPUP,
        WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        RectI::new(left, top, left + SIZE, top + SIZE),
    )?;
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
    }
    window::pump_until(Duration::from_millis(150), || false);

    let center = (left + SIZE / 2, top + SIZE / 2);
    let on_top = unsafe { WindowFromPoint(POINT { x: center.0, y: center.1 }) } == hwnd;
    let mut passed = false;
    if on_top {
        TEST_WINDOW_DOWNS.store(0, Ordering::SeqCst);
        TEST_WINDOW_UPS.store(0, Ordering::SeqCst);
        pointer::click_left_at(desktop, center.0, center.1)?;
        let received = || (TEST_WINDOW_DOWNS.load(Ordering::SeqCst), TEST_WINDOW_UPS.load(Ordering::SeqCst));
        window::pump_until(Duration::from_millis(1000), || received() == (1, 1));
        // Linger briefly so a duplicated click would be caught too.
        window::pump_until(Duration::from_millis(150), || false);
        let (downs, ups) = received();
        println!("click: test window received {downs} button-down and {ups} button-up (need exactly 1 and 1)");
        passed = (downs, ups) == (1, 1);
    } else {
        println!("click: another window covers the test window, so the click was not sent");
    }
    unsafe {
        let _ = DestroyWindow(hwnd);
    }
    window::pump_until(Duration::from_millis(50), || false);
    Ok(passed)
}
