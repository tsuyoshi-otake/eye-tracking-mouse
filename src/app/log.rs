//! A short plain-text record of what the resident program did, for when something does not work.
//!
//! The program has no console, so this file is the only place where the
//! reason for "it does nothing" can be looked up afterwards. It holds states
//! and messages only: no pictures, no positions, nothing that was typed.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::platform::runtime;

/// Lines written per run. A state that flickers must not fill the disk.
const MAX_LINES: u32 = 2000;

pub struct Log {
    file: Option<File>,
    lines_left: u32,
}

impl Log {
    /// Starts the record afresh at `path`. A file that cannot be written is done without.
    pub fn start(path: &Path) -> Log {
        Log { file: File::create(path).ok(), lines_left: MAX_LINES }
    }

    /// Adds a line, stamped with the seconds since the program started.
    pub fn line(&mut self, text: &str) {
        let Some(file) = &mut self.file else { return };
        let text = if self.lines_left == 0 { "(the log is full; nothing more is recorded)" } else { text };
        // A line break inside the text would look like a second entry.
        let written = writeln!(file, "{:9.3} {}", runtime::seconds(), text.replace('\n', " / "));
        if written.is_err() || self.lines_left == 0 {
            self.file = None;
        }
        self.lines_left = self.lines_left.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_stamped_kept_single_and_stop_at_the_limit() {
        let path = std::env::temp_dir().join(format!("etm-log-test-{}.txt", std::process::id()));
        let mut log = Log::start(&path);
        log.line("first");
        log.line("two\nlines");
        for _ in 0..MAX_LINES + 50 {
            log.line("again");
        }
        drop(log);
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), MAX_LINES as usize + 1, "the limit, then one line saying so");
        assert!(lines[0].ends_with(" first") && lines[0].split_whitespace().next().unwrap().parse::<f64>().is_ok());
        assert!(lines[1].ends_with(" two / lines"));
        assert!(lines[MAX_LINES as usize].contains("the log is full"));
    }

    #[test]
    fn a_log_that_cannot_be_written_is_done_without() {
        let nowhere = std::env::temp_dir().join(format!("etm-no-such-folder-{}", std::process::id())).join("log.txt");
        let mut log = Log::start(&nowhere);
        log.line("goes nowhere");
        assert!(!nowhere.exists());
    }
}
