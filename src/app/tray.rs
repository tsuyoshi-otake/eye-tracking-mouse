//! The resident program: an icon in the notification area, its menu, the
//! hotkeys, and the worker that does the pointing.
//!
//! Everything here runs on the thread that owns the windows and handles one
//! event at a time (see `platform::window`), so nothing in it is shared.
//! The worker is only ever talked to through `Pipeline`.
//!
//! Contract with whoever starts the program:
//! - One instance per desktop session. A second start tells the first and ends.
//! - `run` returns only when the program has ended: asked to by the person
//!   (menu, hotkey), from outside (`quit_running`, a request to close), or
//!   because it could not start, which it says in a box first.
//! - After start-up nothing here fails: trouble is shown next to the icon and
//!   written to the log, and the program keeps running.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::error::{Context, Error, Result};
use crate::geometry::RectI;
use crate::platform::camera::CameraRequest;
use crate::platform::notify_icon::{self, MenuItem, NotifyIcon};
use crate::platform::window::{self, Event, Kind, Poster, Window};

use super::icon::{self, Look};
use super::log::Log;
use super::pipeline::{Pipeline, Setup, Status};
use super::storage::Storage;
use super::wording::{self, Message};

/// The class of the hidden main window. Other starts of the program find the running one by it.
pub const MAIN_CLASS: &str = "EyeTrackingMouseMain";
/// Messages to the main window: from the icon, from the worker, and from other starts of the program.
const FROM_ICON: u32 = window::APP_MESSAGE + 1;
const FROM_WORKER: u32 = window::APP_MESSAGE + 2;
const PLEASE_QUIT: u32 = window::APP_MESSAGE + 3;
const SECOND_START: u32 = window::APP_MESSAGE + 4;
/// The system broadcasts this when the taskbar is created anew and every icon has to be added again.
const TASKBAR_CREATED: &str = "TaskbarCreated";
/// How long `quit_running` waits for the running program to go away. Its own shutdown takes two seconds at most.
const QUIT_WAIT: Duration = Duration::from_secs(5);

/// What the person can ask for, from the menu or with a hotkey. The numbers identify menu items and hotkeys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Pause = 1,
    Settings,
    Quit,
}

impl Action {
    const ALL: [Action; 3] = [Action::Pause, Action::Settings, Action::Quit];

    fn from_id(id: u32) -> Option<Action> {
        Action::ALL.into_iter().find(|&action| action as u32 == id)
    }
}

/// Ctrl+Alt+letter for the actions that have a hotkey.
const HOTKEYS: [(Action, char); 2] = [(Action::Pause, 'P'), (Action::Quit, 'Q')];

#[derive(PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

pub struct Options {
    /// Overrides the camera chosen in the settings file.
    pub camera: Option<usize>,
    /// Where settings, calibration and log are kept, instead of the usual place.
    pub data_folder: Option<PathBuf>,
}

/// Runs the program until it is asked to end. `Ok(false)` when it was already running.
pub fn run(options: &Options) -> Result<bool> {
    if let Some(running) = window::find(MAIN_CLASS) {
        Poster::new(running, SECOND_START).post();
        return Ok(false);
    }
    let outcome = run_with_window(options);
    if let Err(error) = &outcome {
        // Started from the desktop there is no console, and without the icon nothing else to say it with.
        let message = wording::could_not_start(error.message());
        window::alert(message.title, &message.text);
    }
    outcome.map(|()| true)
}

fn run_with_window(options: &Options) -> Result<()> {
    // The window comes before anything slow, so that a second start finds it.
    window::register_reporting_class(MAIN_CLASS)?;
    let main = window::open(MAIN_CLASS, wording::APP_NAME, Kind::Hidden, RectI::default())?;
    let outcome = App::start(main, options).map(|mut app| {
        app.serve();
        app.shut_down();
    });
    window::destroy(main);
    outcome
}

/// Asks the running program to quit and waits until it has. `Ok(false)` when none was running.
pub fn quit_running() -> Result<bool> {
    let Some(running) = window::find(MAIN_CLASS) else { return Ok(false) };
    Poster::new(running, PLEASE_QUIT).post();
    let deadline = Instant::now() + QUIT_WAIT;
    while window::find(MAIN_CLASS).is_some() {
        if Instant::now() >= deadline {
            return Err(Error::new("the running program did not quit"));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    Ok(true)
}

fn tip(status: &Status) -> String {
    format!("{}: {}", wording::APP_NAME, wording::status(status))
}

struct App {
    window: Window,
    icon: NotifyIcon,
    pipeline: Pipeline,
    storage: Storage,
    log: Log,
    /// The worker's status as last shown.
    status: Status,
    taskbar_created: u32,
}

impl App {
    fn start(window: Window, options: &Options) -> Result<App> {
        let folder = match &options.data_folder {
            Some(folder) => std::path::absolute(folder).context("find the data folder")?,
            None => Storage::default_folder()?,
        };
        let storage = Storage::open(&folder)?;
        let mut log = Log::start(&storage.log_path());
        log.line(&format!("{} {} started; data in {}", wording::APP_NAME, env!("CARGO_PKG_VERSION"), folder.display()));
        let (settings, mut problems) = storage.load_settings();
        let status = Status::default();
        let picture = icon::picture(notify_icon::icon_size(), Look::of(&status));
        let icon = NotifyIcon::add(window, FROM_ICON, &picture, &tip(&status))?;
        for (action, key) in HOTKEYS {
            if !window::register_hotkey(window, action as i32, key) {
                problems.push(wording::hotkey_taken(key));
            }
        }
        let camera = CameraRequest { device_index: options.camera.unwrap_or(settings.camera), ..CameraRequest::default() };
        log.line(&format!("camera {}", camera.device_index));
        let poster = Poster::new(window, FROM_WORKER);
        let setup = Setup { camera, tuning: settings.tuning, storage: storage.clone() };
        let pipeline = Pipeline::start(setup, move || {
            poster.post();
        })?;
        let taskbar_created = window::registered_message(TASKBAR_CREATED);
        let mut app = App { window, icon, pipeline, storage, log, status, taskbar_created };
        if problems.is_empty() {
            app.show(&wording::started(app.storage.has_calibration()));
        } else {
            app.show(&wording::startup_problems(&problems));
            // The notification has room for the first three; the rest is only in the log.
            for problem in problems.iter().skip(3) {
                app.log.line(problem);
            }
        }
        Ok(app)
    }

    fn serve(&mut self) {
        while let Some(event) = window::next_event() {
            if self.handle(event) == Flow::Quit {
                break;
            }
        }
    }

    fn shut_down(self) {
        for (action, _) in HOTKEYS {
            window::unregister_hotkey(self.window, action as i32);
        }
        let App { pipeline, icon, mut log, .. } = self;
        pipeline.stop();
        drop(icon);
        log.line("stopped");
    }

    fn handle(&mut self, event: Event) -> Flow {
        match event {
            Event::Hotkey(id) => return self.perform(Action::from_id(id as u32)),
            Event::Close(window) if window == self.window => {
                self.log.line("asked to close");
                return Flow::Quit;
            }
            Event::Message { window, message, lparam, .. } if window == self.window => match message {
                FROM_ICON if notify_icon::wants_menu(lparam) => {
                    let chosen = self.icon.menu(&self.menu());
                    return self.perform(chosen.and_then(Action::from_id));
                }
                FROM_WORKER => self.news(),
                PLEASE_QUIT => {
                    self.log.line("asked to quit from outside");
                    return Flow::Quit;
                }
                SECOND_START => self.show(&wording::already_running()),
                other if other == self.taskbar_created => {
                    let restored = self.icon.restore();
                    self.log.line(&format!("the taskbar was created anew; icon restored: {restored}"));
                }
                _ => {}
            },
            Event::DisplayChanged => {
                self.log.line("the monitors changed");
                self.pipeline.display_changed();
            }
            _ => {}
        }
        Flow::Continue
    }

    fn menu(&self) -> Vec<MenuItem<'static>> {
        let pause = if self.pipeline.is_paused() { wording::MENU_RESUME } else { wording::MENU_PAUSE };
        vec![
            MenuItem::Action { id: Action::Pause as u32, label: pause, checked: false },
            MenuItem::Separator,
            MenuItem::Action { id: Action::Settings as u32, label: wording::MENU_SETTINGS, checked: false },
            MenuItem::Separator,
            MenuItem::Action { id: Action::Quit as u32, label: wording::MENU_QUIT, checked: false },
        ]
    }

    fn perform(&mut self, action: Option<Action>) -> Flow {
        match action {
            Some(Action::Pause) => {
                let paused = !self.pipeline.is_paused();
                self.pipeline.set_paused(paused);
                self.log.line(if paused { "paused by the person" } else { "resumed by the person" });
            }
            Some(Action::Settings) => self.open_settings(),
            Some(Action::Quit) => {
                self.log.line("quit by the person");
                return Flow::Quit;
            }
            None => {}
        }
        Flow::Continue
    }

    /// Opens the settings file in Notepad, which is on every machine and leaves the file as plain text.
    fn open_settings(&mut self) {
        let path = self.storage.settings_path();
        if let Err(error) = std::process::Command::new("notepad.exe").arg(&path).spawn() {
            self.show(&wording::settings_not_opened(&path.display().to_string(), &error.to_string()));
        }
    }

    /// Takes in what the worker has to tell.
    fn news(&mut self) {
        let news = self.pipeline.take();
        for notice in &news.notices {
            self.show(&wording::notice(notice));
        }
        if news.status == self.status {
            return;
        }
        // The kind of status, as opposed to its details (which activity, which reason).
        let turned = std::mem::discriminant(&news.status) != std::mem::discriminant(&self.status);
        self.status = news.status;
        // While pointing, the activity changes all the time; that is shown but not recorded.
        if turned || !matches!(self.status, Status::Pointing(_)) {
            self.log.line(&wording::status(&self.status));
        }
        let picture = icon::picture(notify_icon::icon_size(), Look::of(&self.status));
        if let Err(error) = self.icon.set(&picture, &tip(&self.status)) {
            self.log.line(error.message());
        }
        // Trouble is announced when it begins, not each time its wording changes.
        if let Some(message) = wording::trouble(&self.status).filter(|_| turned) {
            self.show(&message);
        }
    }

    /// Tells the person something, and keeps a record of it.
    fn show(&mut self, message: &Message) {
        self.log.line(&format!("{}: {}", message.title, message.text));
        self.icon.balloon(message.title, &message.text, message.warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_action_is_found_again_by_its_number_and_zero_means_none() {
        for action in Action::ALL {
            assert_eq!(Action::from_id(action as u32), Some(action));
        }
        // Zero is what a dismissed menu answers, and no menu item or hotkey may share a number.
        assert_eq!(Action::from_id(0), None);
        assert_eq!(Action::from_id(Action::ALL.len() as u32 + 1), None);
        assert!(HOTKEYS.iter().all(|(action, key)| Action::ALL.contains(action) && key.is_ascii_uppercase()));
    }
}
