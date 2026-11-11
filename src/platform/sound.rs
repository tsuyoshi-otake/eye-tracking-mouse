//! Short tones, for feedback that cannot be seen (the eyes are shut when a click is armed).

use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;

use windows::Win32::System::Diagnostics::Debug::Beep;

/// Plays tones on a thread of its own, because the system call lasts as long as the tone.
///
/// Contract: `play` never waits. At most one tone is kept waiting behind the
/// one that is sounding; any further one is dropped, so tones can never pile
/// up. Dropping the beeper lets the tones already asked for finish.
pub struct Beeper {
    requests: Option<SyncSender<(u32, u32)>>,
    thread: Option<JoinHandle<()>>,
}

impl Beeper {
    pub fn new() -> Beeper {
        let (requests, queue) = sync_channel::<(u32, u32)>(1);
        let thread = std::thread::Builder::new().name("beeper".to_string()).spawn(move || {
            // Ends when the beeper, and with it the sending end, is dropped.
            for (frequency, milliseconds) in queue {
                let _ = unsafe { Beep(frequency, milliseconds) };
            }
        });
        // Without the thread, requests go nowhere: the program works on in silence.
        Beeper { requests: Some(requests), thread: thread.ok() }
    }

    /// A tone of `frequency` hertz lasting `milliseconds`.
    pub fn play(&self, frequency: u32, milliseconds: u32) {
        if let Some(requests) = &self.requests {
            let _ = requests.try_send((frequency, milliseconds));
        }
    }
}

impl Default for Beeper {
    fn default() -> Self {
        Beeper::new()
    }
}

impl Drop for Beeper {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
