//! Camera-driven mouse: pointing with the head, the eyes or both, and clicking with a long blink.
//!
//! Layering (dependencies point downward only):
//!
//! - `app`      wires everything to threads and the process lifetime.
//! - `platform` is the Win32 / WinRT boundary: camera, inference, input injection, windows.
//! - `control`  is pure decision logic: calibration, gaze mapping, pointer control, clicks.
//! - `vision`   is pure geometry on face landmarks: crop region, head pose, eye features.
//! - `geometry`, `image`, `error` are small shared building blocks.
//!
//! `vision` and `control` never touch the OS, so everything they decide is tested
//! without a camera, a face, or a desktop.
//!
//! Only what the program's entry point and the resident test use is public:
//! `app::{diag, selftest, tray}`, `platform::{camera, runtime, window}` and `error`.
//! Everything else stays inside the crate, so the compiler reports what nothing uses.

pub mod app;
mod control;
pub mod error;
mod geometry;
mod image;
pub mod platform;
mod vision;
