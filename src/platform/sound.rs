//! Spoken feedback for what cannot be seen (the eyes are shut when a click is armed).
//!
//! Each cue is a short Japanese phrase said by the speech synthesis that ships
//! with Windows. Where no Japanese voice is installed, a cue is a tone instead.

use std::sync::mpsc::{SyncSender, sync_channel};
use std::thread::JoinHandle;

use windows::Win32::Media::Speech::{
    ISpObjectTokenCategory, ISpVoice, SPCAT_VOICES, SPF_ASYNC, SPF_PURGEBEFORESPEAK, SpObjectTokenCategory, SpVoice,
};
use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance};
use windows::Win32::System::Diagnostics::Debug::Beep;
use windows::core::{HSTRING, PCWSTR, w};

use super::runtime::ComApartment;

/// What the person is told.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cue {
    /// The eyes have been shut long enough: opening them now clicks.
    Armed,
    Clicked,
    Cancelled,
}

impl Cue {
    /// The phrase. The armed one is said while the person waits with shut eyes
    /// and must leave time to open them before the click is dropped, so it is one word.
    fn phrase(self) -> &'static str {
        match self {
            Cue::Armed => "どうぞ",
            Cue::Clicked => "クリックしました",
            Cue::Cancelled => "取り消しました",
        }
    }

    /// Hertz and milliseconds of the tone that stands in for the phrase.
    fn tone(self) -> (u32, u32) {
        match self {
            Cue::Armed => (880, 60),
            Cue::Clicked => (1320, 60),
            Cue::Cancelled => (440, 90),
        }
    }
}

/// How much faster than the voice's normal speed the phrases are said (-10 to 10).
const RATE: i32 = 3;
/// Milliseconds the last phrase gets to finish when the announcer is dropped.
const LAST_WORDS: u32 = 2000;

/// Says cues on a thread of its own.
///
/// Contract: `say` never waits. A new cue cuts off the phrase that is being
/// said, so the person always hears the latest state and cues never pile up.
/// Dropping the announcer lets the last phrase finish.
pub struct Announcer {
    requests: Option<SyncSender<Cue>>,
    thread: Option<JoinHandle<()>>,
}

impl Announcer {
    pub fn new() -> Announcer {
        let (requests, queue) = sync_channel::<Cue>(1);
        let thread = std::thread::Builder::new().name("announcer".to_string()).spawn(move || {
            // The voice lives and dies inside the apartment of this thread.
            let apartment = ComApartment::enter().ok();
            let voice = apartment.as_ref().and_then(|_| japanese_voice());
            // Ends when the announcer, and with it the sending end, is dropped.
            for cue in queue {
                let spoken = voice.as_ref().is_some_and(|voice| {
                    let flags = (SPF_ASYNC.0 | SPF_PURGEBEFORESPEAK.0) as u32;
                    // SAFETY: the text outlives the call, which copies it before returning.
                    unsafe { voice.Speak(&HSTRING::from(cue.phrase()), flags, None) }.is_ok()
                });
                if !spoken {
                    let (frequency, milliseconds) = cue.tone();
                    let _ = unsafe { Beep(frequency, milliseconds) };
                }
            }
            if let Some(voice) = &voice {
                let _ = unsafe { voice.WaitUntilDone(LAST_WORDS) };
            }
        });
        // Without the thread, requests go nowhere: the program works on in silence.
        Announcer { requests: Some(requests), thread: thread.ok() }
    }

    pub fn say(&self, cue: Cue) {
        if let Some(requests) = &self.requests {
            let _ = requests.try_send(cue);
        }
    }
}

/// The system's speech synthesis with a Japanese voice selected, if there is one.
fn japanese_voice() -> Option<ISpVoice> {
    // SAFETY: plain COM calls on objects created here, inside this thread's apartment.
    unsafe {
        let category: ISpObjectTokenCategory = CoCreateInstance(&SpObjectTokenCategory, None, CLSCTX_ALL).ok()?;
        category.SetId(SPCAT_VOICES, false).ok()?;
        // 411 is the language number of Japanese, in hexadecimal as the registry writes it.
        let token = category.EnumTokens(w!("Language=411"), PCWSTR::null()).ok()?.Item(0).ok()?;
        let voice: ISpVoice = CoCreateInstance(&SpVoice, None, CLSCTX_ALL).ok()?;
        voice.SetVoice(&token).ok()?;
        let _ = voice.SetRate(RATE);
        Some(voice)
    }
}

impl Default for Announcer {
    fn default() -> Self {
        Announcer::new()
    }
}

impl Drop for Announcer {
    fn drop(&mut self) {
        self.requests = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Speaks aloud, so it runs only when asked for: `cargo test voice -- --ignored`.
    #[test]
    #[ignore = "speaks through the loudspeakers"]
    fn a_japanese_voice_is_found_and_says_every_cue() {
        let _apartment = ComApartment::enter().unwrap();
        let voice = japanese_voice().expect("a Japanese voice is installed");
        for cue in [Cue::Armed, Cue::Clicked, Cue::Cancelled] {
            // Without the asynchronous flag the call returns when the phrase has been said.
            unsafe { voice.Speak(&HSTRING::from(cue.phrase()), 0, None) }.unwrap();
        }
    }
}

