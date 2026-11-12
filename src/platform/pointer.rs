//! Mouse output and desktop geometry.
//!
//! Contract: positions are physical desktop pixels (the process is per-monitor
//! DPI aware). `move_to` and `click_left_at` are for pixels that lie on a
//! monitor; keeping a point on the screens is the caller's business (the
//! control layer does it with `geometry::clamp_to_union`).
//! A click is one atomic down+up pair: no code path can leave the button held.

use std::mem::size_of;

use windows::Win32::Foundation::{LPARAM, POINT, RECT};
use windows::Win32::Graphics::Gdi::{EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    INPUT, INPUT_0, INPUT_MOUSE, MOUSE_EVENT_FLAGS, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MOVE, MOUSEEVENTF_VIRTUALDESK, MOUSEINPUT, SendInput,
};
use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, MONITORINFOF_PRIMARY};
use windows::core::BOOL;

use crate::error::{Error, Result};
use crate::geometry::RectI;

/// Marks the events this app injects (visible to input hooks as `dwExtraInfo`).
const INJECTION_TAG: usize = 0x4554_4D31; // "ETM1"

/// The monitors as they were when `query` ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Desktop {
    monitors: Vec<RectI>,
    primary: RectI,
    bounds: RectI,
}

impl Desktop {
    pub fn query() -> Result<Desktop> {
        let mut found: Vec<(RectI, bool)> = Vec::new();
        let ok = unsafe {
            EnumDisplayMonitors(None, None, Some(collect_monitor), LPARAM(&raw mut found as isize))
        };
        if !ok.as_bool() || found.is_empty() {
            return Err(Error::new("could not list the monitors"));
        }
        Desktop::from_monitors(&found)
    }

    /// `monitors` pairs each rectangle with "is the primary monitor".
    pub fn from_monitors(monitors: &[(RectI, bool)]) -> Result<Desktop> {
        let rects: Vec<RectI> = monitors.iter().map(|(r, _)| *r).filter(|r| !r.is_empty()).collect();
        let first = *rects.first().ok_or_else(|| Error::new("no monitor has a usable size"))?;
        let primary = monitors.iter().find(|(r, primary)| *primary && !r.is_empty()).map_or(first, |(r, _)| *r);
        let bounds = rects.iter().fold(first, |acc, r| acc.union(r));
        Ok(Desktop { monitors: rects, primary, bounds })
    }

    pub fn monitors(&self) -> &[RectI] {
        &self.monitors
    }

    pub fn primary(&self) -> RectI {
        self.primary
    }

    /// The monitor the pixel is on, if it is on any.
    pub fn monitor_at(&self, x: i32, y: i32) -> Option<RectI> {
        self.monitors.iter().copied().find(|monitor| monitor.contains(x, y))
    }

    /// Bounding box of all monitors (the "virtual screen").
    pub fn bounds(&self) -> RectI {
        self.bounds
    }
}

unsafe extern "system" fn collect_monitor(monitor: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> BOOL {
    // SAFETY: `data` is the `&mut Vec` passed by `Desktop::query`, alive for the whole enumeration.
    let found = unsafe { &mut *(data.0 as *mut Vec<(RectI, bool)>) };
    let mut info = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
    if unsafe { GetMonitorInfoW(monitor, &mut info) }.as_bool() {
        let r = info.rcMonitor;
        found.push((RectI::new(r.left, r.top, r.right, r.bottom), info.dwFlags & MONITORINFOF_PRIMARY != 0));
    }
    true.into()
}

/// Where the cursor is now; `None` while the desktop is not reachable (lock screen, UAC prompt).
pub fn cursor_position() -> Option<(i32, i32)> {
    let mut point = POINT::default();
    unsafe { GetCursorPos(&mut point) }.ok().map(|()| (point.x, point.y))
}

/// Normalised 0..65535 coordinate that Windows maps back onto pixel `offset`
/// of an axis `extent` pixels long.
///
/// Windows turns a normalised value `n` into `n * extent / 65536` rounded down,
/// so aiming at the middle of the pixel's interval lands on it exactly.
fn absolute_coordinate(offset: i32, extent: i32) -> i32 {
    ((i64::from(offset) * 65536 + 32768) / i64::from(extent.max(1))) as i32
}

fn mouse_event(desktop: &Desktop, position: Option<(i32, i32)>, flags: MOUSE_EVENT_FLAGS) -> INPUT {
    let bounds = desktop.bounds();
    let (dx, dy, flags) = match position {
        Some((x, y)) => (
            absolute_coordinate(x - bounds.left, bounds.width()),
            absolute_coordinate(y - bounds.top, bounds.height()),
            flags | MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        ),
        None => (0, 0, flags),
    };
    INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT { dx, dy, mouseData: 0, dwFlags: flags, time: 0, dwExtraInfo: INJECTION_TAG },
        },
    }
}

fn send(events: &[INPUT]) -> Result<()> {
    let sent = unsafe { SendInput(events, size_of::<INPUT>() as i32) };
    if sent as usize == events.len() {
        Ok(())
    } else {
        // Typically UIPI: the foreground window belongs to an elevated process.
        Err(Error::new(format!("Windows accepted {sent} of {} input events", events.len())))
    }
}

/// Moves the cursor to an on-screen pixel.
pub fn move_to(desktop: &Desktop, x: i32, y: i32) -> Result<()> {
    send(&[mouse_event(desktop, Some((x, y)), MOUSE_EVENT_FLAGS(0))])
}

/// Moves to an on-screen pixel and left-clicks there, as one uninterruptible batch.
pub fn click_left_at(desktop: &Desktop, x: i32, y: i32) -> Result<()> {
    send(&[
        mouse_event(desktop, Some((x, y)), MOUSE_EVENT_FLAGS(0)),
        mouse_event(desktop, None, MOUSEEVENTF_LEFTDOWN),
        mouse_event(desktop, None, MOUSEEVENTF_LEFTUP),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The documented mapping Windows applies to a normalised coordinate.
    fn windows_maps_to(normalised: i32, extent: i32) -> i32 {
        ((i64::from(normalised) * i64::from(extent)) / 65536) as i32
    }

    #[test]
    fn every_pixel_of_common_desktop_sizes_round_trips() {
        for extent in [1, 2, 800, 1080, 1500, 1920, 1980, 2560, 3840, 5120, 7680] {
            for offset in 0..extent {
                let n = absolute_coordinate(offset, extent);
                assert!((0..65536).contains(&n), "normalised value {n} out of range");
                assert_eq!(windows_maps_to(n, extent), offset, "extent {extent}");
            }
        }
    }

    #[test]
    fn desktop_describes_a_two_monitor_layout() {
        let upper = RectI::new(0, 0, 2560, 1080);
        let lower = RectI::new(0, 1080, 1500, 1980);
        let desktop = Desktop::from_monitors(&[(lower, false), (upper, true)]).unwrap();
        assert_eq!(desktop.primary(), upper);
        assert_eq!(desktop.bounds(), RectI::new(0, 0, 2560, 1980));
        assert_eq!(desktop.monitors(), [lower, upper]);
        assert_eq!(desktop.monitor_at(1499, 1080), Some(lower));
        assert_eq!(desktop.monitor_at(1500, 1079), Some(upper));
        assert_eq!(desktop.monitor_at(1500, 1080), None, "the corner no monitor covers");
    }

    #[test]
    fn desktop_falls_back_to_the_first_monitor_without_a_primary_flag() {
        let only = RectI::new(-1920, 0, 0, 1080);
        let desktop = Desktop::from_monitors(&[(RectI::default(), true), (only, false)]).unwrap();
        assert_eq!(desktop.primary(), only);
        assert!(Desktop::from_monitors(&[(RectI::default(), true)]).is_err());
    }

    #[test]
    fn absolute_events_carry_the_virtual_desktop_flags_and_tag() {
        let desktop = Desktop::from_monitors(&[(RectI::new(-100, 0, 1820, 1080), true)]).unwrap();
        let event = mouse_event(&desktop, Some((-100, 1079)), MOUSE_EVENT_FLAGS(0));
        let mi = unsafe { event.Anonymous.mi };
        assert_eq!(windows_maps_to(mi.dx, 1920), 0);
        assert_eq!(windows_maps_to(mi.dy, 1080), 1079);
        assert_eq!(mi.dwFlags, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK);
        assert_eq!(mi.dwExtraInfo, INJECTION_TAG);
    }
}
