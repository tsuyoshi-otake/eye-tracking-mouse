//! Plain Win32 window plumbing shared by every window this app creates.

use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HINSTANCE, HWND};
use windows::Win32::Graphics::Gdi::HBRUSH;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DispatchMessageW, IDC_ARROW, LoadCursorW, MSG,
    PM_REMOVE, PeekMessageW, RegisterClassW, TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE,
    WM_QUIT, WNDCLASSW, WNDPROC,
};
use windows::core::PCWSTR;

use crate::error::{Context, Error, Result};
use crate::geometry::RectI;

pub fn instance() -> Result<HINSTANCE> {
    Ok(unsafe { GetModuleHandleW(None) }.context("get the module handle")?.into())
}

/// Registers a window class. `background` of `None` means the window paints every pixel itself.
pub fn register_class(name: PCWSTR, procedure: WNDPROC, background: Option<HBRUSH>) -> Result<()> {
    let class = WNDCLASSW {
        style: CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: procedure,
        hInstance: instance()?,
        hCursor: unsafe { LoadCursorW(None, IDC_ARROW) }.unwrap_or_default(),
        hbrBackground: background.unwrap_or_default(),
        lpszClassName: name,
        ..Default::default()
    };
    if unsafe { RegisterClassW(&class) } == 0 {
        return Err(Error::from(windows::core::Error::from_thread())).context("register a window class");
    }
    Ok(())
}

pub fn create(
    class: PCWSTR,
    title: PCWSTR,
    style: WINDOW_STYLE,
    ex_style: WINDOW_EX_STYLE,
    rect: RectI,
) -> Result<HWND> {
    unsafe {
        CreateWindowExW(
            ex_style,
            class,
            title,
            style,
            rect.left,
            rect.top,
            rect.width(),
            rect.height(),
            None,
            None,
            Some(instance()?),
            None,
        )
    }
    .context("create a window")
}

/// Dispatches this thread's pending window messages. Returns `false` once `WM_QUIT` arrived.
pub fn drain_messages() -> bool {
    let mut message = MSG::default();
    while unsafe { PeekMessageW(&mut message, None, 0, 0, PM_REMOVE) }.as_bool() {
        if message.message == WM_QUIT {
            return false;
        }
        unsafe {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    true
}

/// Keeps dispatching messages until `done` returns true or `timeout` passes.
/// Returns what `done` last said.
pub fn pump_until(timeout: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        drain_messages();
        if done() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}
