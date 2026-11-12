//! The Win32 / WinRT boundary. Everything that talks to the OS lives here and
//! nothing in here decides what the pointer should do.

pub mod camera;
pub(crate) mod canvas;
pub(crate) mod face_detector;
pub(crate) mod inference;
pub(crate) mod notify_icon;
pub(crate) mod pointer;
pub mod runtime;
pub(crate) mod sound;
pub mod window;
