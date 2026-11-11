//! Camera-driven mouse: gaze jump, head fine-tuning, blink click.
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

pub mod app;
pub mod control;
pub mod error;
pub mod geometry;
pub mod image;
pub mod platform;
pub mod vision;
