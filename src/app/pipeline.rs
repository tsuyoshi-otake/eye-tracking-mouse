//! The worker thread: camera in, pointer out.
//!
//! Contract with the thread that owns it:
//! - `Pipeline::start` returns at once; everything slow happens on the worker.
//! - When there is something new to see, the worker calls `notify`, once,
//!   and not again until the owner has looked with `take`.
//! - `set_paused(true)` stops pointer output immediately, even in the middle
//!   of a frame, and the worker lets go of the camera at its next step.
//! - A camera or tracker that fails is tried again after a growing wait, for
//!   as long as the program runs. `Status::Waiting` says what is wrong meanwhile.
//! - Every calibration run has a number, handed out by `start_calibration`.
//!   While the run is under way, `News::view` carries that number. The run
//!   ends with exactly one `Notice::CalibrationDone`, `Notice::HeadCalibrated` or
//!   `Notice::CalibrationFailed` carrying it, or without a word when
//!   `cancel_calibration` was called for it, when a newer run was asked for,
//!   or when the worker itself ended (`Status::Stopped`).
//! - `stop` returns within `STOP_WAIT` whatever the worker is doing. A worker
//!   stuck in a camera read is left behind and ends with the process.
//! - `frames` counts the camera frames taken in, so the owner can tell a
//!   worker that is busy from one whose camera has gone quiet.
//! - While `set_preview(true)` holds, the worker keeps one picture of what
//!   the camera sees ready for `take_preview`, and makes the next one only
//!   after that one was taken. After `set_preview(false)` none is kept.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::control::calibration::{Failure, View};
use crate::control::controller::{Activity, PointingMode, Sound, Tuning};
use crate::control::head_pointer::HeadGain;
use crate::error::{Context, Error, Result};
use crate::geometry::RectI;
use crate::image::BgraFrame;
use crate::platform::camera::{Camera, CameraRequest};
use crate::platform::pointer::{self, Desktop};
use crate::platform::runtime::{self, ComApartment};
use crate::platform::sound::{Announcer, Cue};
use crate::vision::landmarks::FaceMesh;

use super::backoff::Backoff;
use super::engine::{Engine, GazeProblem, Input, Mode, Outcome, Output, PointerAction, gaze_problem};
use super::face::{self, Tracker};
use super::preview::{self, Picture};
use super::record::{CalibrationRecord, PointingRecord};
use super::storage::Storage;

/// How long `stop` waits for the worker.
const STOP_WAIT: Duration = Duration::from_secs(2);
/// A camera that hands over this many empty reads in a row has failed.
const MAX_GAPS: u32 = 200;
/// The rest after an empty read, so that a run of them cannot become a busy loop.
const GAP_PAUSE: Duration = Duration::from_millis(10);
/// Notices kept for an owner that is not looking; older ones are dropped.
const MAX_NOTICES: usize = 16;
/// Tones (hertz, milliseconds) for a click that is armed and one that was given up.

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Status {
    /// Getting ready: loading the model, opening the camera.
    #[default]
    Starting,
    Paused,
    /// Not working for now and trying again by itself; the text says what went wrong.
    Waiting(String),
    Pointing(Activity),
    Calibrating,
    /// The worker has ended for good; the text says why (empty when it was asked to).
    Stopped(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum CalibrationProblem {
    /// The routine itself gave up.
    Routine(Failure),
    /// The camera or the tracker could not be used; the text says why.
    Camera(String),
    DisplayChanged,
    Paused,
    /// A check was asked for, and there is no calibration for this screen and camera to check.
    NotCalibrated,
}

/// What a run does: make a calibration of the gaze, check the one there is and add to it,
/// or fit the head pointer to how far the person turns their head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routine {
    Calibration,
    Check,
    Head,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Notice {
    /// Gaze calibration run `run` (`Calibration` or `Check`) succeeded. `saved` tells whether it will still be there after a restart.
    CalibrationDone { run: u32, routine: Routine, error_px: f32, saved: std::result::Result<(), String> },
    /// Head calibration run `run` succeeded and its speeds are in use. `saved` tells whether
    /// the settings file has them for the next start.
    HeadCalibrated { run: u32, gain: HeadGain, saved: std::result::Result<(), String> },
    /// Calibration run `run` of `routine` ended without a result.
    CalibrationFailed { run: u32, routine: Routine, problem: CalibrationProblem },
    /// The calibration cannot be used for pointing with the eyes as things are; pointing with the head goes on.
    GazeUnavailable(GazeProblem),
    /// The saved calibration could not be read; the text says why.
    CalibrationUnreadable(String),
}

/// What the owner finds when it looks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct News {
    pub status: Status,
    /// While a calibration run is under way: its number and what to show.
    pub view: Option<(u32, View)>,
    /// What happened since the last look, oldest first.
    pub notices: Vec<Notice>,
}

pub struct Setup {
    pub camera: CameraRequest,
    pub tuning: Tuning,
    pub storage: Storage,
    /// Keep a record of every calibration run, however it ends.
    pub record_calibration: bool,
    /// Keep a record of pointing.
    pub record_pointing: bool,
}

enum Command {
    /// The shared flags changed.
    Wake,
    StartCalibration { run: u32, area: RectI, routine: Routine },
    CancelCalibration(u32),
    DisplayChanged,
    SetPointing(PointingMode),
    Quit,
}

/// The picture for the preview window, on its way from the worker to the owner.
#[derive(Default)]
struct Preview {
    picture: Picture,
    /// True from when the worker made the picture until the owner takes it.
    fresh: bool,
}

struct Shared {
    /// Written only by the owner.
    paused: AtomicBool,
    /// Written only by the owner.
    preview_wanted: AtomicBool,
    /// Camera frames taken in so far. Written only by the worker.
    frames: AtomicU64,
    /// True from a `notify` until the owner looks.
    news_waiting: AtomicBool,
    board: Mutex<News>,
    preview: Mutex<Preview>,
    notify: Box<dyn Fn() + Send + Sync>,
}

impl Shared {
    fn new(notify: impl Fn() + Send + Sync + 'static) -> Shared {
        Shared {
            paused: AtomicBool::new(false),
            preview_wanted: AtomicBool::new(false),
            frames: AtomicU64::new(0),
            news_waiting: AtomicBool::new(false),
            board: Mutex::new(News::default()),
            preview: Mutex::new(Preview::default()),
            notify: Box::new(notify),
        }
    }

    fn board(&self) -> MutexGuard<'_, News> {
        // The board holds plain values, so it is still good after a panic elsewhere.
        self.board.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn announce(&self) {
        if !self.news_waiting.swap(true, Ordering::SeqCst) {
            (self.notify)();
        }
    }

    fn set(&self, status: Status, view: Option<(u32, View)>) {
        let changed = {
            let mut board = self.board();
            let changed = board.status != status || board.view != view;
            board.status = status;
            board.view = view;
            changed
        };
        if changed {
            self.announce();
        }
    }

    fn notice(&self, notice: Notice) {
        {
            let mut board = self.board();
            if board.notices.len() >= MAX_NOTICES {
                board.notices.remove(0);
            }
            board.notices.push(notice);
        }
        self.announce();
    }

    fn preview(&self) -> MutexGuard<'_, Preview> {
        // A picture is good or simply replaced, whatever happened while it was held.
        self.preview.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Makes a picture of the frame for the preview, if one is wanted and the last one has been looked at.
    fn offer_preview(&self, frame: &BgraFrame, mesh: Option<&FaceMesh>) {
        {
            let mut preview = self.preview();
            // Asked while the picture is held, so that none is made after `set_preview(false)` has cleared up.
            if preview.fresh || !self.preview_wanted.load(Ordering::SeqCst) {
                return;
            }
            preview::compose(frame, mesh, &mut preview.picture);
            preview.fresh = true;
        }
        self.announce();
    }

    fn set_preview(&self, wanted: bool) {
        self.preview_wanted.store(wanted, Ordering::SeqCst);
        if !wanted {
            *self.preview() = Preview::default();
        }
    }

    fn take_preview(&self, picture: &mut Picture) -> bool {
        let mut preview = self.preview();
        if !preview.fresh {
            return false;
        }
        // The worker draws its next picture into the buffer handed back here.
        std::mem::swap(&mut preview.picture, picture);
        preview.fresh = false;
        true
    }
}

pub struct Pipeline {
    shared: Arc<Shared>,
    commands: Sender<Command>,
    worker: JoinHandle<()>,
    /// The number of the calibration run asked for last.
    last_run: Cell<u32>,
}

impl Pipeline {
    /// `notify` is called on the worker thread and must only pass the word on.
    pub fn start(setup: Setup, notify: impl Fn() + Send + Sync + 'static) -> Result<Pipeline> {
        let shared = Arc::new(Shared::new(notify));
        let (commands, inbox) = channel();
        let for_worker = shared.clone();
        let worker = std::thread::Builder::new()
            .name("pipeline".to_string())
            .spawn(move || run(&for_worker, inbox, setup))
            .context("start the camera thread")?;
        Ok(Pipeline { shared, commands, worker, last_run: Cell::new(0) })
    }

    fn send(&self, command: Command) {
        // A worker that has ended has no use for commands.
        let _ = self.commands.send(command);
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::SeqCst)
    }

    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::SeqCst);
        self.send(Command::Wake);
    }

    /// Starts a run of `routine` on the screen `area` (desktop pixels), ending a pause if there is
    /// one and taking the place of a run that is still going. Returns the number of the new run.
    pub fn start_run(&self, routine: Routine, area: RectI) -> u32 {
        let run = self.last_run.get().wrapping_add(1);
        self.last_run.set(run);
        self.shared.paused.store(false, Ordering::SeqCst);
        self.send(Command::StartCalibration { run, area, routine });
        run
    }

    /// Ends run `run` without a result and without a notice. Nothing happens when it is over already.
    pub fn cancel_calibration(&self, run: u32) {
        self.send(Command::CancelCalibration(run));
    }

    /// Tells the worker that the monitors are not what they were.
    pub fn display_changed(&self) {
        self.send(Command::DisplayChanged);
    }

    /// Changes what moves the cursor, from the next frame on. The setup's choice holds until then.
    pub fn set_pointing(&self, pointing: PointingMode) {
        self.send(Command::SetPointing(pointing));
    }

    /// The current state, and the notices since the last call.
    pub fn take(&self) -> News {
        // Cleared before reading, so that anything written after the read announces itself again.
        self.shared.news_waiting.store(false, Ordering::SeqCst);
        let mut board = self.shared.board();
        News { status: board.status.clone(), view: board.view, notices: std::mem::take(&mut board.notices) }
    }

    /// How many camera frames the worker has taken in so far.
    pub fn frames(&self) -> u64 {
        self.shared.frames.load(Ordering::SeqCst)
    }

    /// Asks for pictures of what the camera sees, or says that no more are needed.
    /// A picture that was waiting to be taken is thrown away then.
    pub fn set_preview(&self, wanted: bool) {
        self.shared.set_preview(wanted);
    }

    /// Exchanges `picture` for a newer one, if there is one. False when `picture` is left as it was.
    pub fn take_preview(&self, picture: &mut Picture) -> bool {
        self.shared.take_preview(picture)
    }

    pub fn stop(self) {
        self.send(Command::Quit);
        let deadline = Instant::now() + STOP_WAIT;
        while !self.worker.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if self.worker.is_finished() {
            let _ = self.worker.join();
        }
    }
}

/// Tells the owner that the worker is gone, however it went (a panic included).
struct Farewell<'a> {
    shared: &'a Shared,
    reason: String,
}

impl Drop for Farewell<'_> {
    fn drop(&mut self) {
        self.shared.set(Status::Stopped(std::mem::take(&mut self.reason)), None);
    }
}

fn run(shared: &Shared, inbox: Receiver<Command>, setup: Setup) {
    let mut farewell = Farewell { shared, reason: "the camera thread ended unexpectedly".to_string() };
    farewell.reason = match work(shared, inbox, setup) {
        Ok(()) => String::new(),
        Err(error) => error.message().to_string(),
    };
}

fn work(shared: &Shared, inbox: Receiver<Command>, setup: Setup) -> Result<()> {
    // Declared first, so that it outlives the tracker and the camera that need it.
    let _com = ComApartment::enter()?;
    let mut tracker = face::new_tracker()?;
    Worker::new(shared, inbox, setup).run(&mut tracker);
    Ok(())
}

#[derive(PartialEq, Eq)]
enum Flow {
    Continue,
    Quit,
}

enum Wait {
    No,
    UntilCommand,
    Seconds(f64),
}

struct Worker<'a> {
    shared: &'a Shared,
    inbox: Receiver<Command>,
    request: CameraRequest,
    storage: Storage,
    engine: Engine,
    desktop: Option<Desktop>,
    /// When the monitors were last asked for, while they are not known.
    desktop_asked: f64,
    camera: Option<Camera>,
    backoff: Backoff,
    /// The camera is not opened before this time.
    retry_at: f64,
    gaps: u32,
    /// A run waiting for the camera to deliver: its number, its screen and what it is to do.
    pending_calibration: Option<(u32, RectI, Routine)>,
    /// What the run the engine is working on does.
    routine: Routine,
    /// The number of the run the engine is working on: there exactly while the engine is
    /// calibrating, and never together with a run that is waiting.
    run: Option<u32>,
    record_wanted: bool,
    /// The record of the run the engine is working on, when one is wanted.
    record: Option<CalibrationRecord>,
    /// The record of pointing since it was last put away, when one is wanted.
    pointing_record: Option<PointingRecord>,
    /// What the owner was last told about the gaze.
    gaze_problem: Option<GazeProblem>,
    announcer: Announcer,
    frame: BgraFrame,
    mesh: FaceMesh,
}

impl<'a> Worker<'a> {
    fn new(shared: &'a Shared, inbox: Receiver<Command>, setup: Setup) -> Worker<'a> {
        let calibration = setup.storage.load_calibration().unwrap_or_else(|error| {
            shared.notice(Notice::CalibrationUnreadable(error.message().to_string()));
            None
        });
        Worker {
            shared,
            inbox,
            request: setup.camera,
            storage: setup.storage,
            engine: Engine::new(setup.tuning, calibration),
            desktop: Desktop::query().ok(),
            desktop_asked: 0.0,
            camera: None,
            backoff: Backoff::new((runtime::seconds() * 1e6) as u32 ^ std::process::id()),
            retry_at: 0.0,
            gaps: 0,
            pending_calibration: None,
            routine: Routine::Calibration,
            run: None,
            record_wanted: setup.record_calibration,
            record: None,
            pointing_record: setup.record_pointing.then(PointingRecord::default),
            gaze_problem: None,
            announcer: Announcer::new(),
            frame: BgraFrame::default(),
            mesh: FaceMesh::default(),
        }
    }

    /// Works until the owner asks it to stop or goes away.
    fn run(&mut self, tracker: &mut Tracker) {
        loop {
            let flow = if self.shared.paused.load(Ordering::SeqCst) {
                self.keep_pointing_record();
                self.close(CalibrationProblem::Paused);
                self.shared.set(Status::Paused, None);
                let flow = self.listen(Wait::UntilCommand);
                if !self.shared.paused.load(Ordering::SeqCst) {
                    self.shared.set(Status::Starting, None);
                }
                flow
            } else if self.camera.is_none() {
                let seconds = self.retry_at - runtime::seconds();
                if seconds > 0.0 {
                    self.listen(Wait::Seconds(seconds))
                } else {
                    self.open();
                    Flow::Continue
                }
            } else {
                let flow = self.listen(Wait::No);
                if flow == Flow::Continue && !self.shared.paused.load(Ordering::SeqCst) {
                    self.one_frame(tracker);
                }
                flow
            };
            if flow == Flow::Quit {
                self.keep_pointing_record();
                return;
            }
        }
    }

    /// Puts the record of pointing away in a file and begins a new one. Nothing happens without frames in it.
    fn keep_pointing_record(&mut self) {
        if let Some(record) = self.pointing_record.as_mut().filter(|record| !record.is_empty()) {
            // The record is an aid for whoever asked for it; pointing goes on whether or not it could be kept.
            let _ = self.storage.save_pointing_record(&std::mem::take(record).finish());
        }
    }

    /// Takes in what the owner asked for, waiting for the first command as told.
    fn listen(&mut self, wait: Wait) -> Flow {
        // `Err(true)` means the owner is gone: nobody is left to work for.
        let mut next = match wait {
            Wait::No => self.inbox.try_recv().map_err(|error| error == TryRecvError::Disconnected),
            Wait::UntilCommand => self.inbox.recv().map_err(|_| true),
            Wait::Seconds(seconds) => self
                .inbox
                .recv_timeout(Duration::from_secs_f64(seconds))
                .map_err(|error| error == RecvTimeoutError::Disconnected),
        };
        loop {
            match next {
                Ok(command) => {
                    if self.obey(command) == Flow::Quit {
                        return Flow::Quit;
                    }
                }
                Err(gone) => return if gone { Flow::Quit } else { Flow::Continue },
            }
            next = self.inbox.try_recv().map_err(|error| error == TryRecvError::Disconnected);
        }
    }

    fn obey(&mut self, command: Command) -> Flow {
        match command {
            Command::Quit => return Flow::Quit,
            Command::Wake => self.retry_now(),
            Command::StartCalibration { run, area, routine } => {
                // A run that is still going makes way without a word, like one that was cancelled.
                self.abandon_run();
                self.pending_calibration = Some((run, area, routine));
                self.retry_now();
            }
            Command::CancelCalibration(run) => {
                self.pending_calibration.take_if(|(waiting, ..)| *waiting == run);
                if self.run == Some(run) {
                    self.abandon_run();
                }
            }
            Command::DisplayChanged => {
                self.desktop = Desktop::query().ok();
                self.end_calibration(CalibrationProblem::DisplayChanged);
            }
            Command::SetPointing(pointing) => self.engine.set_pointing(pointing),
        }
        Flow::Continue
    }

    /// Something the person did deserves an immediate try, whatever wait was in force.
    fn retry_now(&mut self) {
        self.retry_at = 0.0;
        self.backoff.reset();
    }

    /// Hands a run that was waiting for the camera to the engine, now that there is a frame.
    fn begin_waiting_run(&mut self, frame_size: (u32, u32), time: f64) {
        if let Some((run, area, routine)) = self.pending_calibration.take() {
            let started = match routine {
                Routine::Calibration => {
                    self.engine.start_calibration(area, frame_size, time);
                    true
                }
                Routine::Check => self.engine.start_check(area, frame_size, time),
                Routine::Head => {
                    self.engine.start_head_calibration(area, time);
                    true
                }
            };
            if !started {
                self.shared.notice(Notice::CalibrationFailed { run, routine, problem: CalibrationProblem::NotCalibrated });
                return;
            }
            self.routine = routine;
            self.run = Some(run);
            self.record = self.record_wanted.then(|| CalibrationRecord::begin(area, frame_size));
        }
    }

    /// Takes the run the engine is working on away from it. Returns its number and what it did, if there was one.
    fn abandon_run(&mut self) -> Option<(u32, Routine)> {
        self.engine.cancel_calibration();
        // A run that was broken off is the one somebody will want to look into: its record is kept too.
        if let Some(record) = self.record.take() {
            let _ = self.storage.save_calibration_record(&record.finish("abandoned"));
        }
        self.run.take().map(|run| (run, self.routine))
    }

    /// Ends a calibration run that is under way or waiting to start, telling the owner why.
    fn end_calibration(&mut self, problem: CalibrationProblem) {
        let waiting = self.pending_calibration.take().map(|(run, _, routine)| (run, routine));
        if let Some((run, routine)) = self.abandon_run().or(waiting) {
            self.shared.notice(Notice::CalibrationFailed { run, routine, problem });
        }
    }

    /// Lets go of the camera and of everything that depended on its frames coming in.
    fn close(&mut self, problem: CalibrationProblem) {
        self.camera = None;
        self.engine.interrupt();
        self.end_calibration(problem);
    }

    fn open(&mut self) {
        match Camera::open(&self.request) {
            Ok(camera) => {
                self.camera = Some(camera);
                self.gaps = 0;
                self.engine.interrupt();
            }
            Err(error) => self.failed(&error),
        }
    }

    /// The camera or the tracker failed: start over after a wait.
    fn failed(&mut self, error: &Error) {
        self.close(CalibrationProblem::Camera(error.message().to_string()));
        self.retry_at = runtime::seconds() + self.backoff.next_wait();
        self.shared.set(Status::Waiting(error.message().to_string()), None);
    }

    fn one_frame(&mut self, tracker: &mut Tracker) {
        let Some(camera) = self.camera.as_mut() else { return };
        match camera.read(&mut self.frame) {
            Ok(true) => {}
            Ok(false) => {
                self.gaps += 1;
                if self.gaps > MAX_GAPS {
                    self.failed(&Error::new("the camera delivers no pictures"));
                } else {
                    std::thread::sleep(GAP_PAUSE);
                }
                return;
            }
            Err(error) => return self.failed(&error),
        }
        self.gaps = 0;
        self.backoff.reset();
        self.shared.frames.fetch_add(1, Ordering::SeqCst);
        let found = match tracker.track(&self.frame, &mut self.mesh) {
            Ok(found) => found,
            Err(error) => return self.failed(&error),
        };
        self.shared.offer_preview(&self.frame, found.then_some(&self.mesh));
        let time = self.frame.timestamp;
        let frame_size = (self.frame.width as u32, self.frame.height as u32);
        if self.desktop.is_none() && time - self.desktop_asked >= 1.0 {
            self.desktop_asked = time;
            self.desktop = Desktop::query().ok();
        }
        self.begin_waiting_run(frame_size, time);
        let screens = self.desktop.as_ref().map_or(&[][..], Desktop::monitors);
        let cursor = pointer::cursor_position();
        let output = self.engine.frame(&Input { time, mesh: found.then_some(&self.mesh), frame_size, cursor, screens });
        if let Some(record) = &mut self.pointing_record {
            record.add(time, self.engine.pointing(), cursor, &output);
        }
        let gaze_used = self.engine.uses_gaze();
        let problem = self.engine.calibration().filter(|_| gaze_used).and_then(|c| gaze_problem(c, frame_size, screens));
        let refused = self.act(&output);
        if let (Mode::Calibrating(view), Some(record), true) = (&output.mode, &mut self.record, found) {
            record.add(view, &self.mesh);
        }
        self.report(&output, problem, refused);
        if self.pointing_record.as_ref().is_some_and(PointingRecord::is_full) {
            self.keep_pointing_record();
        }
    }

    /// Carries out what the engine answered. True when the system refused the pointer action.
    fn act(&self, output: &Output) -> bool {
        match output.sound {
            Some(Sound::Armed) => self.announcer.say(Cue::Armed),
            Some(Sound::Clicked) => self.announcer.say(Cue::Clicked),
            Some(Sound::Cancelled) => self.announcer.say(Cue::Cancelled),
            None => {}
        }
        let (Some(action), Some(desktop)) = (output.pointer, &self.desktop) else { return false };
        // The owner may have paused while this frame was being worked on.
        if self.shared.paused.load(Ordering::SeqCst) {
            return false;
        }
        let sent = match action {
            PointerAction::Move(x, y) => pointer::move_to(desktop, x, y),
            PointerAction::Click(x, y) => pointer::click_left_at(desktop, x, y),
        };
        // Nothing more to do about a refusal here: the engine sees a cursor that did not follow and holds off.
        sent.is_err()
    }

    fn report(&mut self, output: &Output, problem: Option<GazeProblem>, refused: bool) {
        // The engine ends only a run it was working on, so the number is there whenever an outcome is.
        if let (Some(outcome), Some(run)) = (output.finished, self.run) {
            self.run = None;
            if let Some(record) = self.record.take() {
                let ending = match outcome {
                    Ok(Outcome::Gaze { error_px }) => format!("done {error_px}"),
                    Ok(Outcome::Head(gain)) => format!("head {} {} {}", gain.slow, gain.fast, gain.steadiness),
                    Err(failure) => format!("failed {failure:?}"),
                };
                // The record is an aid for whoever asked for it; the run's own outcome is what the person is told.
                let _ = self.storage.save_calibration_record(&record.finish(&ending));
            }
            let routine = self.routine;
            let kept = |saved: Result<()>| saved.map_err(|error| error.message().to_string());
            self.shared.notice(match outcome {
                Ok(Outcome::Gaze { error_px }) => {
                    let saved = self.engine.calibration().map_or(Ok(()), |c| self.storage.save_calibration(c));
                    Notice::CalibrationDone { run, routine, error_px, saved: kept(saved) }
                }
                Ok(Outcome::Head(gain)) => Notice::HeadCalibrated { run, gain, saved: kept(self.storage.save_head_gain(&gain)) },
                Err(failure) => Notice::CalibrationFailed { run, routine, problem: CalibrationProblem::Routine(failure) },
            });
        }
        if problem != self.gaze_problem {
            self.gaze_problem = problem;
            if let Some(problem) = problem {
                self.shared.notice(Notice::GazeUnavailable(problem));
            }
        }
        let (status, view) = match output.mode {
            Mode::Calibrating(view) => (Status::Calibrating, self.run.map(|run| (run, view))),
            Mode::Pointing(_) if refused => (Status::Pointing(Activity::Blocked), None),
            Mode::Pointing(activity) => (Status::Pointing(activity), None),
        };
        self.shared.set(status, view);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::calibration::Phase;
    use std::sync::atomic::AtomicUsize;

    fn start(run: u32, area: RectI) -> Command {
        Command::StartCalibration { run, area, routine: Routine::Calibration }
    }

    /// A camera index no machine has, so the worker never gets a frame and never touches the pointer.
    const NO_SUCH_CAMERA: usize = 31;

    struct Scratch(std::path::PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Looks until `wanted` says yes, collecting the notices seen on the way.
    fn wait_for(pipeline: &Pipeline, notices: &mut Vec<Notice>, wanted: impl Fn(&News, &[Notice]) -> bool) -> News {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let news = pipeline.take();
            notices.extend(news.notices.iter().cloned());
            if wanted(&news, notices) {
                return news;
            }
            assert!(Instant::now() < deadline, "still {:?} with {notices:?}", news.status);
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A folder of the test's own, and what a worker needs to start: no camera, the usual tuning.
    fn setup(name: &str) -> (Scratch, Setup) {
        let scratch = Scratch(std::env::temp_dir().join(format!("etm-pipeline-{name}-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&scratch.0);
        let setup = Setup {
            camera: CameraRequest { device_index: NO_SUCH_CAMERA, ..CameraRequest::default() },
            tuning: Tuning::default(),
            storage: Storage::open(&scratch.0).unwrap(),
            record_calibration: false,
            record_pointing: false,
        };
        (scratch, setup)
    }

    #[test]
    fn a_run_ends_with_one_notice_of_its_own_number_or_silently_when_cancelled_or_replaced() {
        let (_scratch, setup) = setup("runs");
        let shared = Shared::new(|| {});
        let (_commands, inbox) = channel();
        let mut worker = Worker::new(&shared, inbox, setup);
        let area = RectI::new(0, 0, 1920, 1080);
        let notices = || std::mem::take(&mut shared.board().notices);
        let failed =
            |run| vec![Notice::CalibrationFailed { run, routine: Routine::Calibration, problem: CalibrationProblem::DisplayChanged }];

        // While a run waits for the camera, only its own number cancels it.
        worker.obey(start(1, area));
        worker.obey(Command::CancelCalibration(2));
        worker.obey(Command::DisplayChanged);
        assert_eq!(notices(), failed(1));
        worker.obey(Command::DisplayChanged);
        assert!(notices().is_empty(), "a run ends once");

        // A check without a calibration to check ends at once, saying so, and leaves no run behind.
        worker.obey(Command::StartCalibration { run: 20, area, routine: Routine::Check });
        worker.begin_waiting_run((1280, 720), 0.5);
        let problem = CalibrationProblem::NotCalibrated;
        assert_eq!(notices(), [Notice::CalibrationFailed { run: 20, routine: Routine::Check, problem }]);
        assert_eq!((worker.run, worker.engine.is_calibrating()), (None, false));

        worker.obey(start(2, area));
        worker.obey(Command::CancelCalibration(2));
        worker.obey(Command::DisplayChanged);
        assert!(notices().is_empty(), "a cancelled run ends without a word");

        // A newer run takes the place of one that waits ...
        worker.obey(start(3, area));
        worker.obey(start(4, area));
        worker.begin_waiting_run((1280, 720), 0.0);
        assert!(worker.engine.is_calibrating() && worker.run == Some(4));
        worker.obey(Command::CancelCalibration(3));
        assert!(worker.engine.is_calibrating(), "the number of another run does not end this one");
        // ... and of one the engine is working on.
        worker.obey(start(5, area));
        assert!(!worker.engine.is_calibrating() && worker.run.is_none());
        worker.begin_waiting_run((1280, 720), 1.0);
        worker.obey(Command::DisplayChanged);
        assert_eq!(notices(), failed(5), "runs 3 and 4 made way without a word");
        assert!(!worker.engine.is_calibrating());

        worker.obey(start(6, area));
        worker.begin_waiting_run((1280, 720), 2.0);
        worker.obey(Command::CancelCalibration(6));
        assert!(!worker.engine.is_calibrating());
        worker.obey(Command::DisplayChanged);
        assert!(notices().is_empty());

        // What the engine reports is passed on under the run's number: its view, then its ending.
        worker.obey(start(7, area));
        worker.begin_waiting_run((1280, 720), 3.0);
        let seen = || {
            let board = shared.board();
            (board.status.clone(), board.view)
        };
        let view = View {
            phase: Phase::Rest,
            target: (960, 540),
            dwell: 0.0,
            progress: 0.0,
            face_visible: true,
            answer: None,
            score: None,
        };
        let quiet = |mode, finished| Output { trace: None, pointer: None, sound: None, mode, finished };
        worker.report(&quiet(Mode::Calibrating(view), None), None, false);
        assert_eq!(seen(), (Status::Calibrating, Some((7, view))));
        let ended = quiet(Mode::Pointing(Activity::Tracking), Some(Err(Failure::FaceLost)));
        worker.report(&ended, None, false);
        let problem = CalibrationProblem::Routine(Failure::FaceLost);
        assert_eq!(notices(), [Notice::CalibrationFailed { run: 7, routine: Routine::Calibration, problem }]);
        assert_eq!(seen(), (Status::Pointing(Activity::Tracking), None));
        assert_eq!(worker.run, None);
    }

    #[test]
    fn a_head_calibration_run_ends_with_its_speeds_in_the_settings_file_or_says_what_it_was() {
        let (scratch, setup) = setup("head");
        let shared = Shared::new(|| {});
        let (_commands, inbox) = channel();
        let mut worker = Worker::new(&shared, inbox, setup);
        let area = RectI::new(0, 0, 2560, 1080);
        let notices = || std::mem::take(&mut shared.board().notices);
        let quiet = |finished| Output { trace: None, pointer: None, sound: None, mode: Mode::Pointing(Activity::Tracking), finished };

        worker.obey(Command::StartCalibration { run: 1, area, routine: Routine::Head });
        worker.begin_waiting_run((1280, 720), 0.0);
        assert!(worker.engine.is_calibrating() && worker.run == Some(1));
        let gain = HeadGain { slow: 1270.0, fast: 3810.0, steadiness: 3.0 };
        worker.report(&quiet(Some(Ok(Outcome::Head(gain)))), None, false);
        assert_eq!(notices(), [Notice::HeadCalibrated { run: 1, gain, saved: Ok(()) }]);
        let (settings, warnings) = worker.storage.load_settings();
        assert_eq!((settings.tuning.head_gain, warnings), (gain, Vec::<String>::new()), "the next start points with it");

        // A head run that fails, or that the display ends, says it was the head's.
        worker.obey(Command::StartCalibration { run: 2, area, routine: Routine::Head });
        worker.begin_waiting_run((1280, 720), 1.0);
        worker.report(&quiet(Some(Err(Failure::HeadNotMoved))), None, false);
        let problem = CalibrationProblem::Routine(Failure::HeadNotMoved);
        assert_eq!(notices(), [Notice::CalibrationFailed { run: 2, routine: Routine::Head, problem }]);
        worker.obey(Command::StartCalibration { run: 3, area, routine: Routine::Head });
        worker.obey(Command::DisplayChanged);
        let problem = CalibrationProblem::DisplayChanged;
        assert_eq!(notices(), [Notice::CalibrationFailed { run: 3, routine: Routine::Head, problem: problem.clone() }], "while it waits");
        worker.obey(Command::StartCalibration { run: 4, area, routine: Routine::Head });
        worker.begin_waiting_run((1280, 720), 2.0);
        worker.obey(Command::DisplayChanged);
        assert_eq!(notices(), [Notice::CalibrationFailed { run: 4, routine: Routine::Head, problem }], "under way");
        drop(scratch);
    }

    #[test]
    fn a_picture_for_the_preview_is_made_only_when_wanted_and_after_the_last_was_taken() {
        let announced = Arc::new(AtomicUsize::new(0));
        let counter = announced.clone();
        let shared = Shared::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
        });
        let frame = |shade: u8| {
            let mut frame = BgraFrame::new(8, 4);
            frame.data.fill(shade);
            frame
        };
        let mut picture = Picture::default();

        shared.offer_preview(&frame(1), None);
        assert!(!shared.take_preview(&mut picture), "nobody asked for a picture");
        assert_eq!(announced.load(Ordering::SeqCst), 0);

        shared.set_preview(true);
        shared.offer_preview(&frame(1), None);
        assert_eq!(announced.load(Ordering::SeqCst), 1, "the owner is told that there is a picture");
        shared.offer_preview(&frame(2), None);
        assert!(shared.take_preview(&mut picture));
        assert_eq!((picture.width, picture.height, picture.pixels[0]), (8, 4, 0x0001_0101), "the first one waited");
        assert!(!shared.take_preview(&mut picture), "a picture is handed over once");

        shared.offer_preview(&frame(3), None);
        shared.set_preview(false);
        assert!(!shared.take_preview(&mut picture), "what was waiting is thrown away");
        shared.offer_preview(&frame(4), None);
        assert!(!shared.take_preview(&mut picture));
        assert_eq!(picture.pixels[0], 0x0001_0101, "the owner's picture is left as it was");
        assert_eq!(shared.preview().picture, Picture::default(), "and none is kept");
    }

    #[test]
    fn a_missing_camera_is_waited_for_and_pause_calibration_and_stop_still_work() {
        let (_scratch, setup) = setup("camera");
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let pipeline = Pipeline::start(setup, move || {
            counter.fetch_add(1, Ordering::SeqCst);
        })
        .unwrap();
        let mut notices = Vec::new();

        let news = wait_for(&pipeline, &mut notices, |news, _| matches!(news.status, Status::Waiting(_)));
        assert!(matches!(&news.status, Status::Waiting(reason) if reason.contains("camera")), "{:?}", news.status);
        assert!(calls.load(Ordering::SeqCst) >= 1, "the owner was told that there is news");

        // A calibration cannot start without a camera, and says so exactly once.
        let run = pipeline.start_run(Routine::Calibration, RectI::new(0, 0, 1920, 1080));
        wait_for(&pipeline, &mut notices, |_, notices| !notices.is_empty());
        std::thread::sleep(Duration::from_millis(300));
        wait_for(&pipeline, &mut notices, |_, _| true);
        assert!(
            matches!(
                notices.as_slice(),
                [Notice::CalibrationFailed { run: failed, routine: Routine::Calibration, problem: CalibrationProblem::Camera(_) }]
                    if *failed == run
            ),
            "{notices:?}"
        );
        assert_ne!(pipeline.start_run(Routine::Calibration, RectI::new(0, 0, 1920, 1080)), run, "every run has a number of its own");
        wait_for(&pipeline, &mut notices, |_, notices| notices.len() == 2);
        notices.pop();

        pipeline.set_paused(true);
        assert!(pipeline.is_paused());
        wait_for(&pipeline, &mut notices, |news, _| news.status == Status::Paused);
        pipeline.set_paused(false);
        wait_for(&pipeline, &mut notices, |news, _| matches!(news.status, Status::Waiting(_)));
        assert_eq!(notices.len(), 1, "pausing with no calibration under way says nothing: {notices:?}");

        let asked = Instant::now();
        pipeline.stop();
        assert!(asked.elapsed() < STOP_WAIT, "the worker ended by itself in {:?}", asked.elapsed());
    }
}
