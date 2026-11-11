//! Plain Win32 window plumbing shared by every window this app creates.
//!
//! Two ways to have a window:
//!
//! - `register_class` + `create` with a window procedure of the caller's own
//!   (the self-test's click target).
//! - Reporting windows (`register_reporting_class` + `open`). Their procedure
//!   runs no program logic: the system may call it in the middle of anything
//!   (inside another call into the system, inside a menu), so it only notes
//!   what happened. The thread's loop takes the notes one at a time with
//!   `next_event` and acts on them with nothing else half-done.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{HBRUSH, ValidateRect};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, RegisterHotKey, UnregisterHotKey,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CS_HREDRAW, CS_VREDRAW, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, EndMenu, FindWindowW,
    GetClientRect, GetMessageW, IDC_ARROW, KillTimer, LoadCursorW, MB_ICONWARNING, MB_OK, MB_SETFOREGROUND, MSG,
    MessageBoxW, PM_REMOVE, PeekMessageW, PostMessageW,
    RegisterClassW, RegisterWindowMessageW, SW_HIDE, SW_SHOW, SW_SHOWNOACTIVATE, SetForegroundWindow, SetTimer,
    ShowWindow, TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP, WM_CLOSE, WM_DISPLAYCHANGE, WM_ERASEBKGND,
    WM_HOTKEY, WM_KEYDOWN, WM_NULL, WM_PAINT, WM_QUIT, WM_TIMER, WNDCLASSW, WNDPROC, WS_EX_TOPMOST,
    WS_OVERLAPPEDWINDOW, WS_POPUP,
};
use windows::core::PCWSTR;

use crate::error::{Context, Error, Result};
use crate::geometry::RectI;

/// The first message number a program may use for its own purposes.
pub const APP_MESSAGE: u32 = WM_APP;
/// The virtual-key code of the Escape key, as reported in `Event::KeyDown`.
pub const KEY_ESCAPE: u32 = 0x1B;

/// A window, as the system names it. Callers keep it and hand it back to the functions here, nothing else.
pub type Window = HWND;

/// A string as the system wants it: UTF-16 with a terminating zero.
pub fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

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

/// What a reporting window has to say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// The window's content must be drawn again.
    Paint(Window),
    /// The person asked for the window to go away (its close button, Alt+F4).
    Close(Window),
    /// A key went down while the window had the keyboard; the number is its virtual-key code.
    KeyDown(Window, u32),
    /// A combination registered with `register_hotkey` was pressed.
    Hotkey(i32),
    /// A timer started with `start_timer` is due.
    Timer(Window, usize),
    /// The monitors changed: a resolution, the arrangement, or one came or went.
    DisplayChanged,
    /// A message numbered `APP_MESSAGE` or above: the program's own and those from `registered_message`.
    Message { window: Window, message: u32, wparam: usize, lparam: isize },
}

thread_local! {
    static EVENTS: RefCell<VecDeque<Event>> = const { RefCell::new(VecDeque::new()) };
}

unsafe extern "system" fn report(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let event = match message {
        WM_PAINT => {
            // Drawing happens when the event is handled; until then the request must not keep coming.
            let _ = unsafe { ValidateRect(Some(hwnd), None) };
            Event::Paint(hwnd)
        }
        // Every pixel is drawn by the program; erasing first would only flicker.
        WM_ERASEBKGND => return LRESULT(1),
        WM_CLOSE => Event::Close(hwnd),
        WM_KEYDOWN => Event::KeyDown(hwnd, wparam.0 as u32),
        WM_HOTKEY => {
            // A menu of this thread would hold every event back until it closes; a hotkey must act now.
            let _ = unsafe { EndMenu() };
            Event::Hotkey(wparam.0 as i32)
        }
        WM_TIMER => Event::Timer(hwnd, wparam.0),
        WM_DISPLAYCHANGE => Event::DisplayChanged,
        number if number >= WM_APP => Event::Message { window: hwnd, message, wparam: wparam.0, lparam: lparam.0 },
        _ => return unsafe { DefWindowProcW(hwnd, message, wparam, lparam) },
    };
    let first = EVENTS.with(|events| {
        let mut events = events.borrow_mut();
        // A request to paint and a timer that is due say nothing new the second time, and they are
        // said again and again while nobody takes the notes (for as long as a menu is open, for one).
        let said = matches!(event, Event::Paint(_) | Event::Timer(..)) && events.contains(&event);
        if !said {
            events.push_back(event);
        }
        !said && events.len() == 1
    });
    if first {
        // A message sent straight to the window does not end the loop's wait for the next one; this does.
        let _ = unsafe { PostMessageW(Some(hwnd), WM_NULL, WPARAM(0), LPARAM(0)) };
    }
    LRESULT(0)
}

/// Registers a class for reporting windows.
pub fn register_reporting_class(name: &str) -> Result<()> {
    register_class(PCWSTR(wide(name).as_ptr()), Some(report), None)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Never shown: only there to receive messages.
    Hidden,
    /// An ordinary resizable window with a title bar.
    Framed,
    /// No border, above every other window. For covering a whole monitor.
    Covering,
}

/// Creates a window of a reporting class. `rect` is in desktop pixels; the window starts out hidden.
pub fn open(class: &str, title: &str, kind: Kind, rect: RectI) -> Result<Window> {
    let (style, ex_style) = match kind {
        Kind::Hidden => (WS_POPUP, WINDOW_EX_STYLE(0)),
        Kind::Framed => (WS_OVERLAPPEDWINDOW, WINDOW_EX_STYLE(0)),
        Kind::Covering => (WS_POPUP, WS_EX_TOPMOST),
    };
    create(PCWSTR(wide(class).as_ptr()), PCWSTR(wide(title).as_ptr()), style, ex_style, rect)
}

/// Waits for the next thing one of this thread's reporting windows has to
/// say. `None` when the thread was told to quit and nothing more will come.
pub fn next_event() -> Option<Event> {
    loop {
        if let Some(event) = EVENTS.with(|events| events.borrow_mut().pop_front()) {
            return Some(event);
        }
        let mut message = MSG::default();
        // Zero is WM_QUIT and -1 an error: either way there is nothing more to wait for.
        if unsafe { GetMessageW(&mut message, None, 0, 0) }.0 <= 0 {
            return None;
        }
        unsafe {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

/// The top-level window of class `class`, in whichever process, if there is one.
pub fn find(class: &str) -> Option<Window> {
    let found = unsafe { FindWindowW(PCWSTR(wide(class).as_ptr()), PCWSTR::null()) };
    found.ok().filter(|window| !window.is_invalid())
}

/// The number of a message every program that asks for `name` gets the same answer for.
pub fn registered_message(name: &str) -> u32 {
    unsafe { RegisterWindowMessageW(PCWSTR(wide(name).as_ptr())) }
}

/// Lets any thread or process tell a window something: `post` makes the
/// window report `Event::Message` with this message number.
#[derive(Clone, Copy, Debug)]
pub struct Poster {
    // Kept as a number so that the poster may cross threads; it is only ever handed back to the system.
    window: isize,
    message: u32,
}

impl Poster {
    pub fn new(window: Window, message: u32) -> Poster {
        Poster { window: window.0 as isize, message }
    }

    /// False when the window is gone or its queue is full.
    pub fn post(&self) -> bool {
        unsafe { PostMessageW(Some(HWND(self.window as *mut c_void)), self.message, WPARAM(0), LPARAM(0)) }.is_ok()
    }
}

/// Makes Ctrl+Alt+`key` (an upper-case letter) report `Event::Hotkey(id)` to
/// `window` wherever the keyboard focus is. False when the combination is taken.
pub fn register_hotkey(window: Window, id: i32, key: char) -> bool {
    unsafe { RegisterHotKey(Some(window), id, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, key as u32) }.is_ok()
}

pub fn unregister_hotkey(window: Window, id: i32) {
    let _ = unsafe { UnregisterHotKey(Some(window), id) };
}

/// Reports `Event::Timer(window, id)` every `milliseconds` until `stop_timer`.
pub fn start_timer(window: Window, id: usize, milliseconds: u32) -> bool {
    unsafe { SetTimer(Some(window), id, milliseconds, None) != 0 }
}

pub fn stop_timer(window: Window, id: usize) {
    let _ = unsafe { KillTimer(Some(window), id) };
}

/// Shows the window; with `focus` it also comes to the front and gets the keyboard.
pub fn show(window: Window, focus: bool) {
    unsafe {
        let _ = ShowWindow(window, if focus { SW_SHOW } else { SW_SHOWNOACTIVATE });
        if focus {
            let _ = SetForegroundWindow(window);
        }
    }
}

pub fn hide(window: Window) {
    let _ = unsafe { ShowWindow(window, SW_HIDE) };
}

pub fn destroy(window: Window) {
    let _ = unsafe { DestroyWindow(window) };
}

/// Tells the person something in a box with an OK button, and waits until they have dismissed it.
/// For when there is no window or icon of the program's own to say it with.
pub fn alert(title: &str, text: &str) {
    let (title, text) = (wide(title), wide(text));
    let style = MB_OK | MB_ICONWARNING | MB_SETFOREGROUND;
    let _ = unsafe { MessageBoxW(None, PCWSTR(text.as_ptr()), PCWSTR(title.as_ptr()), style) };
}

/// Width and height of the part of the window the program draws in.
pub fn client_size(window: Window) -> (i32, i32) {
    let mut rect = RECT::default();
    let _ = unsafe { GetClientRect(window, &mut rect) };
    (rect.right - rect.left, rect.bottom - rect.top)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::UI::WindowsAndMessaging::SendMessageW;

    #[test]
    fn a_reporting_window_notes_what_happens_in_order_and_a_reminder_only_once() {
        const CLASS: &str = "EtmWindowTest";
        register_reporting_class(CLASS).unwrap();
        let window = open(CLASS, "window test", Kind::Hidden, RectI::default()).unwrap();
        // Sent, not posted: the way the system calls the procedure in the middle of something else.
        let send = |message: u32, wparam: usize| {
            unsafe { SendMessageW(window, message, Some(WPARAM(wparam)), Some(LPARAM(7))) };
        };
        send(WM_TIMER, 1);
        send(APP_MESSAGE + 5, 3);
        send(WM_TIMER, 1);
        send(WM_TIMER, 2);
        send(WM_KEYDOWN, KEY_ESCAPE as usize);
        send(WM_KEYDOWN, KEY_ESCAPE as usize);
        send(WM_CLOSE, 0);
        let expected = [
            Event::Timer(window, 1),
            Event::Message { window, message: APP_MESSAGE + 5, wparam: 3, lparam: 7 },
            Event::Timer(window, 2),
            Event::KeyDown(window, KEY_ESCAPE),
            Event::KeyDown(window, KEY_ESCAPE),
            Event::Close(window),
        ];
        for event in expected {
            assert_eq!(next_event(), Some(event));
        }
        // Once a reminder was heard, the next one is noted again.
        send(WM_TIMER, 1);
        assert_eq!(next_event(), Some(Event::Timer(window, 1)));

        // A request to close leaves the window standing: whether it goes is for its owner to say.
        assert_eq!(find(CLASS), Some(window));
        destroy(window);
        assert_eq!(find(CLASS), None);
    }
}
