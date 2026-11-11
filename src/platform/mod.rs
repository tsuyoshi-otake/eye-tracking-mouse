//! The Win32 / WinRT boundary. Everything that talks to the OS lives here and
//! nothing in here decides what the pointer should do.

pub mod camera;
pub mod face_detector;
pub mod inference;
pub mod pointer;
pub mod runtime;
pub mod window;
