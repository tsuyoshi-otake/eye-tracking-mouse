//! The program's icon in the taskbar's notification area, with its balloons and its menu.
//!
//! Contract: the icon exists from `NotifyIcon::add` until the value is
//! dropped. Mouse actions on it arrive at the owning window as
//! `window::Event::Message` carrying the message number given to `add`;
//! `wants_menu` tells which of them ask for the menu.

use std::ffi::c_void;
use std::mem::size_of;

use windows::Win32::Foundation::{HWND, LPARAM, POINT, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateBitmap, CreateDIBSection, DIB_RGB_COLORS, DeleteObject,
};
use windows::Win32::UI::Shell::{
    NIF_ICON, NIF_INFO, NIF_MESSAGE, NIF_TIP, NIIF_INFO, NIIF_WARNING, NIM_ADD, NIM_DELETE, NIM_MODIFY,
    NOTIFY_ICON_DATA_FLAGS, NOTIFY_ICON_MESSAGE, NOTIFYICONDATAW, Shell_NotifyIconW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    AppendMenuW, CreateIconIndirect, CreatePopupMenu, DestroyIcon, DestroyMenu, GetCursorPos, GetSystemMetrics, HICON,
    ICONINFO, MF_CHECKED, MF_SEPARATOR, MF_STRING, PostMessageW, SM_CXSMICON, SetForegroundWindow, TPM_NONOTIFY,
    TPM_RETURNCMD, TPM_RIGHTBUTTON, TrackPopupMenu, WM_CONTEXTMENU, WM_LBUTTONUP, WM_NULL, WM_RBUTTONUP,
};
use windows::core::PCWSTR;

use crate::error::{Context, Error, Result};

use super::window::{Window, wide};

/// The only icon this program shows.
const ICON_ID: u32 = 1;

/// A square picture: `size * size` pixels, row by row from the top, each
/// 0xAARRGGBB with the colour already multiplied by the alpha.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IconImage {
    pub size: usize,
    pub pixels: Vec<u32>,
}

/// The side, in pixels, of the pictures the notification area shows.
pub fn icon_size() -> usize {
    unsafe { GetSystemMetrics(SM_CXSMICON) }.max(16) as usize
}

/// Whether a mouse action on the icon (the `lparam` of its message) asks for the menu.
pub fn wants_menu(lparam: isize) -> bool {
    matches!(lparam as u32, WM_RBUTTONUP | WM_LBUTTONUP | WM_CONTEXTMENU)
}

pub enum MenuItem<'a> {
    /// `id` must not be zero.
    Action { id: u32, label: &'a str, checked: bool },
    Separator,
}

fn make_icon(image: &IconImage) -> Result<HICON> {
    let side = image.size as i32;
    if image.size == 0 || image.pixels.len() != image.size * image.size {
        return Err(Error::new("the icon picture is not square"));
    }
    let header = BITMAPINFOHEADER {
        biSize: size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: side,
        biHeight: -side, // top-down
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB.0,
        ..Default::default()
    };
    // The mask is not looked at for a picture with an alpha channel, but there must be one; rows are 16-bit aligned.
    let mask_bits = vec![0u8; image.size.div_ceil(16) * 2 * image.size];
    unsafe {
        let mut bits: *mut c_void = std::ptr::null_mut();
        let info = BITMAPINFO { bmiHeader: header, ..Default::default() };
        let colour = CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0).context("create the icon picture")?;
        // SAFETY: the section was just made with `size * size` 32-bit pixels.
        std::ptr::copy_nonoverlapping(image.pixels.as_ptr(), bits.cast::<u32>(), image.pixels.len());
        let mask = CreateBitmap(side, side, 1, 1, Some(mask_bits.as_ptr().cast()));
        let icon = CreateIconIndirect(&ICONINFO { fIcon: true.into(), hbmMask: mask, hbmColor: colour, ..Default::default() });
        // The icon keeps copies of both.
        let _ = DeleteObject(colour.into());
        let _ = DeleteObject(mask.into());
        icon.context("create the icon")
    }
}

/// Copies as much of `text` as fits, never splitting a character, and ends it with a zero.
fn copy_text(target: &mut [u16], text: &str) {
    let mut used = 0;
    let mut units = [0u16; 2];
    for character in text.chars() {
        let encoded = character.encode_utf16(&mut units);
        if used + encoded.len() >= target.len() {
            break;
        }
        target[used..used + encoded.len()].copy_from_slice(encoded);
        used += encoded.len();
    }
    target[used] = 0;
}

pub struct NotifyIcon {
    window: HWND,
    message: u32,
    icon: HICON,
    tip: String,
}

impl NotifyIcon {
    /// Shows the icon with `tip` as the text under the mouse.
    pub fn add(window: Window, message: u32, image: &IconImage, tip: &str) -> Result<NotifyIcon> {
        let icon = NotifyIcon { window, message, icon: make_icon(image)?, tip: tip.to_string() };
        // Dropping `icon` on failure frees the picture again.
        if !icon.send(NIM_ADD, icon.data(NIF_MESSAGE | NIF_ICON | NIF_TIP)) {
            return Err(Error::new("the notification area did not accept the icon"));
        }
        Ok(icon)
    }

    fn data(&self, flags: NOTIFY_ICON_DATA_FLAGS) -> NOTIFYICONDATAW {
        let mut data = NOTIFYICONDATAW {
            cbSize: size_of::<NOTIFYICONDATAW>() as u32,
            hWnd: self.window,
            uID: ICON_ID,
            uFlags: flags,
            uCallbackMessage: self.message,
            hIcon: self.icon,
            ..Default::default()
        };
        copy_text(&mut data.szTip, &self.tip);
        data
    }

    fn send(&self, action: NOTIFY_ICON_MESSAGE, data: NOTIFYICONDATAW) -> bool {
        unsafe { Shell_NotifyIconW(action, &data) }.as_bool()
    }

    /// Changes the picture and the text under the mouse.
    pub fn set(&mut self, image: &IconImage, tip: &str) -> Result<()> {
        let old = std::mem::replace(&mut self.icon, make_icon(image)?);
        self.tip = tip.to_string();
        let shown = self.send(NIM_MODIFY, self.data(NIF_ICON | NIF_TIP));
        let _ = unsafe { DestroyIcon(old) };
        if shown { Ok(()) } else { Err(Error::new("the notification area did not take the new icon")) }
    }

    /// Puts the icon back after the taskbar was created anew (Explorer was restarted).
    pub fn restore(&self) -> bool {
        self.send(NIM_ADD, self.data(NIF_MESSAGE | NIF_ICON | NIF_TIP))
    }

    /// Shows a notification next to the icon. False when the system did not take it.
    pub fn balloon(&self, title: &str, text: &str, warning: bool) -> bool {
        let mut data = self.data(NIF_INFO);
        copy_text(&mut data.szInfoTitle, title);
        copy_text(&mut data.szInfo, text);
        data.dwInfoFlags = if warning { NIIF_WARNING } else { NIIF_INFO };
        self.send(NIM_MODIFY, data)
    }

    /// Shows a menu at the mouse and waits for the choice. `None` when it was dismissed.
    pub fn menu(&self, items: &[MenuItem]) -> Option<u32> {
        unsafe {
            let menu = CreatePopupMenu().ok()?;
            for item in items {
                let _ = match item {
                    MenuItem::Separator => AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null()),
                    MenuItem::Action { id, label, checked } => {
                        let label = wide(label);
                        let flags = if *checked { MF_STRING | MF_CHECKED } else { MF_STRING };
                        AppendMenuW(menu, flags, *id as usize, PCWSTR(label.as_ptr()))
                    }
                };
            }
            let mut at = POINT::default();
            let _ = GetCursorPos(&mut at);
            // A menu whose window is not in front does not close when the person clicks elsewhere.
            let _ = SetForegroundWindow(self.window);
            let flags = TPM_RETURNCMD | TPM_NONOTIFY | TPM_RIGHTBUTTON;
            let chosen = TrackPopupMenu(menu, flags, at.x, at.y, None, self.window, None).0;
            // Lets the system finish closing the menu (a long-standing quirk of notification-area menus).
            let _ = PostMessageW(Some(self.window), WM_NULL, WPARAM(0), LPARAM(0));
            let _ = DestroyMenu(menu);
            (chosen > 0).then_some(chosen as u32)
        }
    }
}

impl Drop for NotifyIcon {
    fn drop(&mut self) {
        self.send(NIM_DELETE, self.data(NOTIFY_ICON_DATA_FLAGS(0)));
        let _ = unsafe { DestroyIcon(self.icon) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_cut_to_fit_without_splitting_a_character() {
        let mut short = [0xFFFFu16; 4];
        copy_text(&mut short, "abcdef");
        assert_eq!(short, [b'a' as u16, b'b' as u16, b'c' as u16, 0]);

        // An emoji takes two units: with room for only one more, it is left out whole.
        let mut tight = [0xFFFFu16; 4];
        copy_text(&mut tight, "ab😀c");
        assert_eq!(tight[..3], [b'a' as u16, b'b' as u16, 0]);

        let mut roomy = [0xFFFFu16; 8];
        copy_text(&mut roomy, "視線");
        assert_eq!(roomy[..3], ['視' as u16, '線' as u16, 0]);

        let mut empty = [0xFFFFu16; 1];
        copy_text(&mut empty, "x");
        assert_eq!(empty, [0]);
    }

    #[test]
    fn the_menu_opens_on_either_mouse_button_and_the_keyboard() {
        assert!(wants_menu(WM_RBUTTONUP as isize) && wants_menu(WM_LBUTTONUP as isize));
        assert!(wants_menu(WM_CONTEXTMENU as isize));
        assert!(!wants_menu(0x0200), "moving the mouse over the icon is not a request");
    }

    #[test]
    fn a_picture_of_the_wrong_size_is_refused() {
        assert!(make_icon(&IconImage { size: 16, pixels: vec![0; 255] }).is_err());
        assert!(make_icon(&IconImage { size: 0, pixels: Vec::new() }).is_err());
    }

    #[test]
    fn a_square_picture_becomes_an_icon() {
        let icon = make_icon(&IconImage { size: 20, pixels: vec![0xFF20_8040; 400] }).unwrap();
        assert!(unsafe { DestroyIcon(icon) }.is_ok());
    }
}
