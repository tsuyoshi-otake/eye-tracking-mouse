//! Process- and thread-level runtime setup: COM, Media Foundation, DPI awareness.

use windows::Win32::Media::MediaFoundation::{MF_VERSION, MFSTARTUP_NOSOCKET, MFShutdown, MFStartup};
use windows::Win32::System::Com::{COINIT_MULTITHREADED, CoInitializeEx, CoUninitialize};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};

use crate::error::{Context, Result};

/// Keeps the calling thread in the COM multithreaded apartment while alive.
pub struct ComApartment(());

impl ComApartment {
    pub fn enter() -> Result<Self> {
        // S_FALSE ("already initialised") is a success and still needs a matching uninit.
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.ok().context("initialise COM")?;
        Ok(Self(()))
    }
}

impl Drop for ComApartment {
    fn drop(&mut self) {
        unsafe { CoUninitialize() };
    }
}

/// Keeps Media Foundation started while alive. Startup is reference counted by
/// the OS, so several guards may coexist.
pub struct MediaFoundation(());

impl MediaFoundation {
    pub fn startup() -> Result<Self> {
        unsafe { MFStartup(MF_VERSION, MFSTARTUP_NOSOCKET) }.context("start Media Foundation")?;
        Ok(Self(()))
    }
}

impl Drop for MediaFoundation {
    fn drop(&mut self) {
        // Nothing useful can be done if shutdown fails while tearing down.
        let _ = unsafe { MFShutdown() };
    }
}

/// Makes every coordinate this process sees a physical pixel. Without it, a
/// scaled monitor reports virtualised coordinates and the pointer would miss.
/// Must run before any window is created; a second call fails harmlessly.
pub fn enable_per_monitor_dpi() {
    let _ = unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
}
