//! The resident program: an icon in the notification area, its menu, the
//! hotkeys, the calibration page, the camera preview, and the worker that
//! does the pointing.
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
//! - The worker calibrates only while the calibration page is open: when the
//!   page goes, for whatever reason, its run ends with it.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use crate::control::controller::PointingMode;
use crate::error::{Context, Error, Result};
use crate::geometry::RectI;
use crate::platform::camera::CameraRequest;
use crate::platform::notify_icon::{self, MenuItem, NotifyIcon};
use crate::platform::pointer::{self, Desktop};
use crate::platform::runtime;
use crate::platform::window::{self, Event, Kind, Poster, Window};

use super::calibration_screen::CalibrationScreen;
use super::icon::{self, Look};
use super::log::Log;
use super::pipeline::{CalibrationProblem, Notice, Pipeline, Routine, Setup, Status};
use super::preview::PreviewWindow;
use super::settings;
use super::storage::Storage;
use super::wording::{self, Message};

/// The class of the hidden main window. Other starts of the program find the running one by it.
pub const MAIN_CLASS: &str = "EyeTrackingMouseMain";
/// The class of the windows the person sees: the calibration page and the preview. It is not
/// the main window's, so that looking for the running program never finds one of them.
pub const VIEW_CLASS: &str = "EyeTrackingMouseView";
/// Messages to the main window: from the icon, from the worker, and from other starts of the program.
const FROM_ICON: u32 = window::APP_MESSAGE + 1;
const FROM_WORKER: u32 = window::APP_MESSAGE + 2;
const PLEASE_QUIT: u32 = window::APP_MESSAGE + 3;
const SECOND_START: u32 = window::APP_MESSAGE + 4;
/// The system broadcasts this when the taskbar is created anew and every icon has to be added again.
const TASKBAR_CREATED: &str = "TaskbarCreated";
/// How long `quit_running` waits for the running program to go away. Its own shutdown takes two seconds at most.
const QUIT_WAIT: Duration = Duration::from_secs(5);
/// Why a calibration run is given up when the camera has gone quiet behind the page.
const CAMERA_SILENT: &str = "the camera sends no pictures";

/// What the person can ask for, from the menu or with a hotkey. The numbers identify the menu
/// items and the hotkeys; a hotkey reaches the main window as `WM_HOTKEY` carrying its number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Pause = 1,
    Calibrate,
    /// Check the calibration there is and add what the check finds to it.
    Check,
    /// Fit the head pointer's speeds to how far the person turns their head.
    HeadCalibrate,
    Preview,
    /// What moves the cursor: the head, the eyes, or the eyes with the head correcting.
    PointHead,
    PointEye,
    PointHybrid,
    Settings,
    Quit,
}

impl Action {
    const ALL: [Action; 10] = [
        Action::Pause,
        Action::Calibrate,
        Action::Check,
        Action::HeadCalibrate,
        Action::Preview,
        Action::PointHead,
        Action::PointEye,
        Action::PointHybrid,
        Action::Settings,
        Action::Quit,
    ];

    fn from_id(id: u32) -> Option<Action> {
        Action::ALL.into_iter().find(|&action| action as u32 == id)
    }
}

/// Ctrl+Alt+letter for the actions that have a hotkey. The texts in `wording` name these letters.
const HOTKEYS: [(Action, char); 5] =
    [(Action::Pause, 'P'), (Action::Calibrate, 'C'), (Action::Check, 'T'), (Action::HeadCalibrate, 'H'), (Action::Quit, 'Q')];

/// The action that starts a run of `routine`.
fn starter(routine: Routine) -> Action {
    match routine {
        Routine::Calibration => Action::Calibrate,
        Routine::Check => Action::Check,
        Routine::Head => Action::HeadCalibrate,
    }
}

/// The letter of the hotkey that starts (and ends) a run of `routine`.
fn run_key(routine: Routine) -> char {
    let action = starter(routine);
    HOTKEYS.iter().find(|(with, _)| *with == action).map_or('?', |&(_, key)| key)
}

const ROUTINES: [Routine; 3] = [Routine::Calibration, Routine::Check, Routine::Head];

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
    let started = window::register_reporting_class(VIEW_CLASS).and_then(|()| App::start(main, options));
    let outcome = started.map(|mut app| {
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

/// Whether the camera's pictures are coming in while the worker is in this state.
fn is_watching(status: &Status) -> bool {
    matches!(status, Status::Pointing(_) | Status::Calibrating)
}

/// The screen a calibration run is for: the one the pointer is on, or the main one while the pointer cannot be found.
fn screen_under_pointer() -> Result<RectI> {
    let desktop = Desktop::query()?;
    let under = pointer::cursor_position().and_then(|(x, y)| desktop.monitor_at(x, y));
    Ok(under.unwrap_or(desktop.primary()))
}

/// A calibration run, by the number the worker knows it by, and the page that shows it.
struct Run {
    number: u32,
    routine: Routine,
    page: CalibrationScreen,
}

struct App {
    window: Window,
    icon: NotifyIcon,
    pipeline: Pipeline,
    storage: Storage,
    log: Log,
    /// What moves the cursor, as the worker was last told. The settings file says the same.
    pointing: PointingMode,
    /// The worker's status as last shown.
    status: Status,
    taskbar_created: u32,
    calibration: Option<Run>,
    preview: Option<PreviewWindow>,
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
        let pointing = settings.tuning.pointing;
        let (record_calibration, record_pointing) = (settings.record_calibration, settings.record_pointing);
        let setup = Setup { camera, tuning: settings.tuning, storage: storage.clone(), record_calibration, record_pointing };
        let pipeline = Pipeline::start(setup, move || {
            poster.post();
        })?;
        let taskbar_created = window::registered_message(TASKBAR_CREATED);
        let mut app =
            App { window, icon, pipeline, storage, log, pointing, status, taskbar_created, calibration: None, preview: None };
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

    fn shut_down(mut self) {
        for (action, _) in HOTKEYS {
            window::unregister_hotkey(self.window, action as i32);
        }
        // The windows go before the worker is asked to stop: a run ends with its page.
        self.close_calibration();
        self.close_preview();
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
            Event::Close(window) | Event::KeyDown(window, window::KEY_ESCAPE) => self.dismiss(window),
            Event::Paint(window) => self.paint(window),
            Event::Timer(window, _) if self.shows_page(window) => self.watch_camera(),
            Event::Message { window, message, lparam, .. } if window == self.window => match message {
                FROM_ICON if notify_icon::wants_menu(lparam) => {
                    let running = self.calibration.as_ref().map(|run| run.routine);
                    let runs = ROUTINES.map(|routine| wording::menu_run(routine, run_key(routine), running));
                    let chosen = self.icon.menu(&self.menu(&runs));
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

    /// The icon's menu. `runs` are the labels of the entries that start the runs, in the order of `ROUTINES`.
    fn menu<'a>(&self, runs: &'a [String; 3]) -> Vec<MenuItem<'a>> {
        let pause = if self.pipeline.is_paused() { wording::MENU_RESUME } else { wording::MENU_PAUSE };
        let item = |action: Action, label, checked| MenuItem::Action { id: action as u32, label, checked };
        let [calibrate, check, head] = runs;
        vec![
            item(Action::Pause, pause, false),
            item(Action::Calibrate, calibrate, false),
            item(Action::Check, check, false),
            item(Action::HeadCalibrate, head, false),
            item(Action::Preview, wording::MENU_PREVIEW, self.preview.is_some()),
            MenuItem::Separator,
            item(Action::PointHead, wording::MENU_POINT_HEAD, self.pointing == PointingMode::Head),
            item(Action::PointEye, wording::MENU_POINT_EYE, self.pointing == PointingMode::Eye),
            item(Action::PointHybrid, wording::MENU_POINT_HYBRID, self.pointing == PointingMode::Hybrid),
            MenuItem::Separator,
            item(Action::Settings, wording::MENU_SETTINGS, false),
            MenuItem::Separator,
            item(Action::Quit, wording::MENU_QUIT, false),
        ]
    }

    fn perform(&mut self, action: Option<Action>) -> Flow {
        match action {
            Some(Action::Pause) => {
                let paused = !self.pipeline.is_paused();
                self.pipeline.set_paused(paused);
                self.log.line(if paused { "paused by the person" } else { "resumed by the person" });
            }
            // Whichever run is going, any of their keys ends it: the page covers the screen.
            Some(Action::Calibrate | Action::Check | Action::HeadCalibrate) if self.calibration.is_some() => {
                self.cancel_calibration();
            }
            Some(Action::Calibrate) => self.start_run(Routine::Calibration),
            Some(Action::Check) => self.start_run(Routine::Check),
            Some(Action::HeadCalibrate) => self.start_run(Routine::Head),
            Some(Action::Preview) if self.preview.is_some() => self.close_preview(),
            Some(Action::Preview) => self.open_preview(),
            Some(Action::PointHead) => self.choose_pointing(PointingMode::Head),
            Some(Action::PointEye) => self.choose_pointing(PointingMode::Eye),
            Some(Action::PointHybrid) => self.choose_pointing(PointingMode::Hybrid),
            Some(Action::Settings) => self.open_settings(),
            Some(Action::Quit) => {
                self.log.line("quit by the person");
                return Flow::Quit;
            }
            None => {}
        }
        Flow::Continue
    }

    /// Hands pointing to what the person chose: at once for the worker, and in the settings
    /// file for the next start. A choice that could not be written down still holds until the
    /// program ends, and the person is told that it will not last. Choosing what is already
    /// in use changes nothing.
    fn choose_pointing(&mut self, pointing: PointingMode) {
        if self.pointing == pointing {
            return;
        }
        self.pointing = pointing;
        self.pipeline.set_pointing(pointing);
        self.log.line(&format!("pointing set to {} by the person", settings::pointing_word(pointing)));
        match self.storage.save_pointing(pointing) {
            Err(error) => self.show(&wording::pointing_not_saved(error.message())),
            Ok(()) if pointing != PointingMode::Head && !self.storage.has_calibration() => {
                self.show(&wording::eye_pointing_needs_calibration());
            }
            Ok(()) => {}
        }
    }

    /// Opens the settings file in Notepad, which is on every machine and leaves the file as plain text.
    fn open_settings(&mut self) {
        let path = self.storage.settings_path();
        if let Err(error) = std::process::Command::new("notepad.exe").arg(&path).spawn() {
            self.show(&wording::settings_not_opened(&path.display().to_string(), &error.to_string()));
        }
    }

    fn shows_page(&self, window: Window) -> bool {
        self.calibration.as_ref().is_some_and(|run| run.page.window() == window)
    }

    /// Opens the calibration page on the screen the pointer is on and starts a run for that screen.
    fn start_run(&mut self, routine: Routine) {
        if matches!(self.status, Status::Stopped(_)) {
            // Without the worker there is nothing to calibrate with; the person is told again why it is gone.
            if let Some(message) = wording::trouble(&self.status) {
                self.show(&message);
            }
            return;
        }
        let (frames, now) = (self.pipeline.frames(), runtime::seconds());
        let hint = wording::how_to_cancel(run_key(routine));
        let opened = screen_under_pointer()
            .and_then(|area| CalibrationScreen::open(VIEW_CLASS, area, hint, frames, now).map(|page| (area, page)));
        match opened {
            Ok((area, page)) => {
                let number = self.pipeline.start_run(routine, area);
                let (left, top, width, height) = (area.left, area.top, area.width(), area.height());
                self.log.line(&format!("{routine:?} run {number} started for the screen at {left},{top} ({width}x{height})"));
                self.calibration = Some(Run { number, routine, page });
            }
            Err(error) => self.show(&wording::window_not_opened(error.message())),
        }
    }

    /// Takes the calibration page away, which ends its run if that has not ended by itself.
    /// Returns the number of the run, if there was a page.
    fn close_calibration(&mut self) -> Option<u32> {
        let run = self.calibration.take()?;
        self.pipeline.cancel_calibration(run.number);
        Some(run.number)
    }

    fn cancel_calibration(&mut self) {
        if let Some(number) = self.close_calibration() {
            self.log.line(&format!("calibration run {number} cancelled by the person"));
        }
    }

    /// Asked a few times a second while the calibration page is open: a camera that has gone
    /// quiet must not leave the person in front of a page that covers their screen.
    fn watch_camera(&mut self) {
        let (frames, now) = (self.pipeline.frames(), runtime::seconds());
        let silent = self.calibration.as_mut().and_then(|run| run.page.camera_is_silent(frames, now).then_some(run.routine));
        if let Some(routine) = silent {
            self.close_calibration();
            self.show(&wording::calibration_failed(routine, &CalibrationProblem::Camera(CAMERA_SILENT.to_string())));
        }
    }

    fn open_preview(&mut self) {
        let opened = Desktop::query().and_then(|desktop| PreviewWindow::open(VIEW_CLASS, desktop.primary()));
        match opened {
            Ok(mut preview) => {
                preview.describe(wording::status(&self.status), is_watching(&self.status));
                self.preview = Some(preview);
                self.pipeline.set_preview(true);
                self.log.line("preview opened");
            }
            Err(error) => self.show(&wording::window_not_opened(error.message())),
        }
    }

    fn close_preview(&mut self) {
        if self.preview.take().is_some() {
            self.pipeline.set_preview(false);
            self.log.line("preview closed");
        }
    }

    /// The person wants one of the program's windows gone.
    fn dismiss(&mut self, window: Window) {
        if self.shows_page(window) {
            self.cancel_calibration();
        } else if self.preview.as_ref().is_some_and(|preview| preview.window() == window) {
            self.close_preview();
        }
    }

    fn paint(&mut self, window: Window) {
        if let Some(run) = self.calibration.as_mut().filter(|run| run.page.window() == window) {
            run.page.draw();
        } else if let Some(preview) = self.preview.as_mut().filter(|preview| preview.window() == window) {
            preview.draw();
        }
    }

    /// Takes in what the worker has to tell.
    fn news(&mut self) {
        let news = self.pipeline.take();
        let shown = self.calibration.as_ref().map(|run| run.number);
        for notice in &news.notices {
            self.show(&wording::notice(notice));
            let ended = match notice {
                Notice::CalibrationDone { run, .. }
                | Notice::HeadCalibrated { run, .. }
                | Notice::CalibrationFailed { run, .. } => Some(*run),
                _ => None,
            };
            // The run on the page is over, so the page has nothing more to show.
            if ended.is_some() && ended == shown {
                self.close_calibration();
            }
        }
        match (&mut self.calibration, news.view) {
            // The view of an earlier run may still be around; this page is not for it.
            (Some(run), Some((number, view))) if run.number == number => run.page.show(view),
            _ => {}
        }
        let changed = news.status != self.status;
        if changed {
            self.turn(news.status);
        }
        if matches!(self.status, Status::Stopped(_)) {
            // The worker is gone, and the run with it.
            self.close_calibration();
        }
        if let Some(preview) = &mut self.preview {
            let watching = is_watching(&self.status);
            if changed {
                preview.describe(wording::status(&self.status), watching);
            }
            // During a break no picture is taken, so that none from before it shows up in it.
            let fresh = watching && self.pipeline.take_preview(preview.picture());
            if changed || fresh {
                preview.draw();
            }
        }
    }

    /// Shows the status the worker has turned to: on the icon, in the log, and next to the icon when it means trouble.
    fn turn(&mut self, status: Status) {
        // The kind of status, as opposed to its details (which activity, which reason).
        let turned = std::mem::discriminant(&status) != std::mem::discriminant(&self.status);
        self.status = status;
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
        let keys: std::collections::HashSet<char> = HOTKEYS.iter().map(|&(_, key)| key).collect();
        assert_eq!(keys.len(), HOTKEYS.len(), "no two actions share a key");
    }

    #[test]
    fn every_run_has_its_own_key_and_the_texts_name_the_right_one() {
        // The notices and the page name these letters in so many words.
        assert_eq!(ROUTINES.map(run_key), ['C', 'T', 'H']);
        assert_eq!(ROUTINES.map(starter), [Action::Calibrate, Action::Check, Action::HeadCalibrate]);
        let idle = ROUTINES.map(|routine| wording::menu_run(routine, run_key(routine), None));
        assert_eq!(idle, ["視線のキャリブレーション\tCtrl+Alt+C", "追加補正テスト\tCtrl+Alt+T", "頭のキャリブレーション\tCtrl+Alt+H"]);
        // While the head is calibrated, every entry ends that run, each with its own key.
        let busy = ROUTINES.map(|routine| wording::menu_run(routine, run_key(routine), Some(Routine::Head)));
        assert!(busy.iter().all(|label| label.starts_with("頭のキャリブレーションを中止\t")), "{busy:?}");
        assert_eq!(wording::how_to_cancel(run_key(Routine::Head)), "Esc または Ctrl+Alt+H で中止");
    }

    #[test]
    fn the_camera_delivers_only_while_pointing_or_calibrating() {
        use crate::control::controller::Activity;
        assert!(is_watching(&Status::Pointing(Activity::NoFace)) && is_watching(&Status::Calibrating));
        for idle in [Status::Starting, Status::Paused, Status::Waiting(String::new()), Status::Stopped(String::new())] {
            assert!(!is_watching(&idle), "{idle:?}");
        }
    }
}
