//! Starts the real resident program and drives it from outside: a second
//! start, the messages the system sends it, the hotkeys, the calibration
//! page, the preview window, and quitting.
//!
//! Not part of the ordinary test run, because things appear on the desktop
//! for a few seconds (notifications, the preview window, and for a moment
//! the page that covers the screen the pointer is on) and Ctrl+Alt+P, C and Q
//! are taken meanwhile:
//!
//!     cargo test --test resident -- --ignored
//!
//! The program is given a camera that does not exist, so it never gets a
//! picture and never touches the pointer. The test ends the program it
//! started whichever way it goes, and never touches one it did not start.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use eye_tracking_mouse::app::tray::{Action, MAIN_CLASS, VIEW_CLASS};
use eye_tracking_mouse::platform::window::{self, Window};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    GetWindowThreadProcessId, PostMessageW, SMTO_ABORTIFHUNG, SendMessageTimeoutW, WM_CLOSE, WM_DISPLAYCHANGE,
    WM_HOTKEY, WM_KEYDOWN,
};

const PROGRAM: &str = env!("CARGO_BIN_EXE_eye-tracking-mouse");
/// A camera index no machine has.
const NO_SUCH_CAMERA: &str = "31";

fn program(arguments: &[&str]) -> Command {
    let mut command = Command::new(PROGRAM);
    command.args(arguments).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    command
}

/// The program under test and its folder. Both are gone when this is dropped, however the test ends.
struct Resident {
    child: Child,
    folder: PathBuf,
}

impl Drop for Resident {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.folder);
    }
}

fn wait_until(what: &str, seconds: u64, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !done() {
        assert!(Instant::now() < deadline, "waited {seconds} s in vain for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Waits for the process to end by itself and returns its exit code. It is killed when it does not.
fn exit_code(child: &mut Child, what: &str, seconds: u64) -> Option<i32> {
    let deadline = Instant::now() + Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            return status.code();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{what} did not end within {seconds} s");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn log(folder: &Path) -> String {
    std::fs::read_to_string(folder.join("log.txt")).unwrap_or_default()
}

/// Sends a message the way the system does: straight to the window, not through its queue.
fn send(window: Window, message: u32) {
    let answered = unsafe { SendMessageTimeoutW(window, message, WPARAM(0), LPARAM(0), SMTO_ABORTIFHUNG, 5000, None) };
    assert!(answered.0 != 0, "the program did not take message {message:#06x}");
}

/// Puts a message in the window's queue, the way a key press or a hotkey arrives.
fn post(window: Window, message: u32, wparam: usize) {
    unsafe { PostMessageW(Some(window), message, WPARAM(wparam), LPARAM(0)) }.unwrap();
}

/// The one window the program shows the person (the calibration page or the preview), if it is there.
fn view() -> Option<Window> {
    window::find(VIEW_CLASS)
}

#[test]
#[ignore = "starts the resident program: notifications and windows appear and three hotkeys are taken for a few seconds"]
fn the_resident_program_starts_obeys_and_quits_without_a_camera() {
    assert!(window::find(MAIN_CLASS).is_none(), "the program is already running; quit it before running this test");
    assert!(view().is_none(), "a window of the program is still open; close it before running this test");
    let folder = std::env::temp_dir().join(format!("etm-resident-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    let data = folder.to_str().unwrap();
    let child = program(&["--camera", NO_SUCH_CAMERA, "--data-dir", data]).spawn().unwrap();
    let mut resident = Resident { child, folder: folder.clone() };

    wait_until("the main window", 10, || window::find(MAIN_CLASS).is_some());
    let main = window::find(MAIN_CLASS).unwrap();
    let mut owner = 0u32;
    unsafe { GetWindowThreadProcessId(main, Some(&mut owner)) };
    assert_eq!(owner, resident.child.id(), "the window belongs to the program this test started");

    // A camera that is not there is waited for; it does not end the program.
    wait_until("the missing camera in the log", 30, || log(&folder).contains("再試行中"));
    assert!(resident.child.try_wait().unwrap().is_none(), "the program ended without a camera");
    assert!(folder.join("settings.txt").exists(), "the first start writes the settings template");

    // A second start ends at once and leaves the first one running.
    let mut second = program(&["--camera", NO_SUCH_CAMERA, "--data-dir", data]).spawn().unwrap();
    assert_eq!(exit_code(&mut second, "the second start", 10), Some(1));
    wait_until("the second start in the log", 5, || log(&folder).contains("すでに起動しています"));

    // What the system sends rather than posts must still reach the program's loop.
    send(main, WM_DISPLAYCHANGE);
    wait_until("the monitor change in the log", 5, || log(&folder).contains("the monitors changed"));
    send(main, window::registered_message("TaskbarCreated"));
    wait_until("the new taskbar in the log", 5, || log(&folder).contains("the taskbar was created anew"));

    // A hotkey, as the system delivers it: a message carrying the number of what it asks for.
    let press = |action: Action| post(main, WM_HOTKEY, action as usize);
    press(Action::Pause);
    wait_until("the pause in the log", 5, || {
        let text = log(&folder);
        text.contains("paused by the person") && text.contains("一時停止中")
    });
    press(Action::Pause);
    wait_until("the end of the pause in the log", 30, || {
        let text = log(&folder);
        text.contains("resumed by the person") && text.matches("再試行中").count() >= 2
    });

    // The menu's choice of pointing: taken at once, written into the settings file, and said to
    // need a calibration, which this folder does not have. Choosing again puts it back.
    let settings = || std::fs::read_to_string(folder.join("settings.txt")).unwrap();
    let first = settings();
    assert!(first.contains("
pointing = head
"), "the template starts with the head");
    press(Action::PointHead);
    press(Action::PointEye);
    wait_until("eye pointing in the log and the file", 5, || {
        let text = log(&folder);
        text.contains("pointing set to eye by the person")
            && text.contains("キャリブレーションが必要です")
            && settings().contains("
pointing = eye
")
    });
    press(Action::PointHybrid);
    wait_until("hybrid pointing in the log and the file", 5, || {
        log(&folder).contains("pointing set to hybrid by the person") && settings().contains("
pointing = hybrid
")
    });
    press(Action::PointHead);
    wait_until("head pointing in the log and the file", 5, || {
        log(&folder).contains("pointing set to head by the person") && settings().contains("
pointing = head
")
    });
    assert_eq!(settings(), first, "the file is as it was first written");

    // Without a camera a calibration cannot be done: the run is given a page, fails, says so,
    // and the page that covered the screen is gone again without anybody closing it.
    press(Action::Calibrate);
    wait_until("the failed calibration in the log", 15, || {
        let text = log(&folder);
        text.contains("calibration run 1 started") && text.contains("キャリブレーションできませんでした")
    });
    wait_until("the calibration page to go", 5, || view().is_none());
    assert!(!log(&folder).contains("cancelled by the person"), "nobody cancelled the run");

    // The preview comes and goes when asked for, and goes when the person closes it: with Escape ...
    press(Action::Preview);
    wait_until("the preview", 5, || view().is_some());
    post(view().unwrap(), WM_KEYDOWN, window::KEY_ESCAPE as usize);
    wait_until("the preview to go on Escape", 5, || view().is_none());
    // ... with its close button ...
    press(Action::Preview);
    wait_until("the preview a second time", 5, || view().is_some());
    post(view().unwrap(), WM_CLOSE, 0);
    wait_until("the preview to go when closed", 5, || view().is_none());
    // ... and by asking again.
    press(Action::Preview);
    wait_until("the preview a third time", 5, || view().is_some());
    press(Action::Preview);
    wait_until("the preview to go when asked again", 5, || view().is_none());
    wait_until("the three closings in the log", 5, || log(&folder).matches("preview closed").count() == 3);
    // This one is left open: quitting has to take it along.
    press(Action::Preview);
    wait_until("the preview a fourth time", 5, || view().is_some());

    // Quitting from outside waits until the program is gone.
    let mut quit = program(&["--quit"]).spawn().unwrap();
    assert_eq!(exit_code(&mut quit, "--quit", 10), Some(0));
    assert!(window::find(MAIN_CLASS).is_none(), "--quit came back before the program was gone");
    assert!(view().is_none(), "the preview outlived the program");
    assert_eq!(exit_code(&mut resident.child, "the program", 5), Some(0));
    let text = log(&folder);
    // Seen with `--nocapture`: everything the program wrote down while it was driven.
    println!("{text}");
    assert!(text.contains("asked to quit from outside"), "{text}");
    assert!(text.trim_end().ends_with("stopped"), "{text}");

    let mut again = program(&["--quit"]).spawn().unwrap();
    assert_eq!(exit_code(&mut again, "the second --quit", 10), Some(1), "nothing is left to quit");
}
