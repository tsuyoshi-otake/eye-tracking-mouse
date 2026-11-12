//! One error type for the whole app.
//!
//! Errors here are only ever shown to a person or written to the log, and every
//! recovery decision (retry the camera, re-detect the face) is taken by the caller
//! on "it failed" alone, so a message with context is all an error needs to carry.

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<windows::core::Error> for Error {
    fn from(e: windows::core::Error) -> Self {
        Self(format!("{} (0x{:08X})", e.message(), e.code().0 as u32))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self(e.to_string())
    }
}

/// Adds "what we were doing" to an error on its way up.
pub trait Context<T> {
    fn context(self, what: &str) -> Result<T>;
}

impl<T, E: Into<Error>> Context<T> for std::result::Result<T, E> {
    fn context(self, what: &str) -> Result<T> {
        self.map_err(|e| {
            let e: Error = e.into();
            Error(format!("{what}: {e}"))
        })
    }
}

impl<T> Context<T> for Option<T> {
    fn context(self, what: &str) -> Result<T> {
        self.ok_or_else(|| Error(what.to_string()))
    }
}
