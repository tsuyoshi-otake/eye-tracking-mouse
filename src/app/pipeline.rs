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
//! - A calibration asked for with `start_calibration` ends with exactly one
//!   `Notice::CalibrationDone` or `Notice::CalibrationFailed`, unless
//!   `cancel_calibration` was called.
//! - `stop` returns within `STOP_WAIT` whatever the worker is doing. A worker
//!   stuck in a camera read is left behind and ends with the process.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::control::calibration::{Failure, View};
use crate::control::controller::{Activity, Sound, Tuning};
use crate::error::{Context, Error, Result};
use crate::geometry::RectI;
use crate::image::BgraFrame;
use crate::platform::camera::{Camera, CameraRequest};
use crate::platform::pointer::{self, Desktop};
use crate::platform::runtime::{self, ComApartment};
use crate::platform::sound::Beeper;
use crate::vision::landmarks::FaceMesh;

use super::backoff::Backoff;
use super::engine::{Engine, GazeProblem, Input, Mode, Output, PointerAction, gaze_problem};
use super::face::{self, Tracker};
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
const ARMED_TONE: (u32, u32) = (880, 60);
const CANCELLED_TONE: (u32, u32) = (440, 90);

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
}

#[derive(Clone, Debug, PartialEq)]
pub enum Notice {
    /// A calibration run succeeded. `saved` tells whether it will still be there after a restart.
    CalibrationDone { error_px: f32, saved: std::result::Result<(), String> },
    CalibrationFailed(CalibrationProblem),
    /// The calibration cannot be used for gaze jumps as things are; pointing with the head goes on.
    GazeUnavailable(GazeProblem),
    /// The saved calibration could not be read; the text says why.
    CalibrationUnreadable(String),
}

/// What the owner finds when it looks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct News {
    pub status: Status,
    /// What to show while a calibration run is under way.
    pub view: Option<View>,
    /// What happened since the last look, oldest first.
    pub notices: Vec<Notice>,
}

pub struct Setup {
    pub camera: CameraRequest,
    pub tuning: Tuning,
    pub storage: Storage,
}

enum Command {
    /// The shared flags changed.
    Wake,
    StartCalibration(RectI),
    CancelCalibration,
    DisplayChanged,
    Quit,
}

struct Shared {
    /// Written only by the owner.
    paused: AtomicBool,
    /// True from a `notify` until the owner looks.
    news_waiting: AtomicBool,
    board: Mutex<News>,
    notify: Box<dyn Fn() + Send + Sync>,
}

impl Shared {
    fn board(&self) -> MutexGuard<'_, News> {
        // The board holds plain values, so it is still good after a panic elsewhere.
        self.board.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn announce(&self) {
        if !self.news_waiting.swap(true, Ordering::SeqCst) {
            (self.notify)();
        }
    }

    fn set(&self, status: Status, view: Option<View>) {
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
}

pub struct Pipeline {
    shared: Arc<Shared>,
    commands: Sender<Command>,
    worker: JoinHandle<()>,
}

impl Pipeline {
    /// `notify` is called on the worker thread and must only pass the word on.
    pub fn start(setup: Setup, notify: impl Fn() + Send + Sync + 'static) -> Result<Pipeline> {
        let shared = Arc::new(Shared {
            paused: AtomicBool::new(false),
            news_waiting: AtomicBool::new(false),
            board: Mutex::new(News::default()),
            notify: Box::new(notify),
        });
        let (commands, inbox) = channel();
        let for_worker = shared.clone();
        let worker = std::thread::Builder::new()
            .name("pipeline".to_string())
            .spawn(move || run(&for_worker, inbox, setup))
            .context("start the camera thread")?;
        Ok(Pipeline { shared, commands, worker })
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

    /// Starts a calibration run on the screen `area` (desktop pixels), ending a pause if there is one.
    pub fn start_calibration(&self, area: RectI) {
        self.shared.paused.store(false, Ordering::SeqCst);
        self.send(Command::StartCalibration(area));
    }

    pub fn cancel_calibration(&self) {
        self.send(Command::CancelCalibration);
    }

    /// Tells the worker that the monitors are not what they were.
    pub fn display_changed(&self) {
        self.send(Command::DisplayChanged);
    }

    /// The current state, and the notices since the last call.
    pub fn take(&self) -> News {
        // Cleared before reading, so that anything written after the read announces itself again.
        self.shared.news_waiting.store(false, Ordering::SeqCst);
        let mut board = self.shared.board();
        News { status: board.status.clone(), view: board.view, notices: std::mem::take(&mut board.notices) }
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
    gaze_jump: bool,
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
    /// A calibration run waiting for the camera to deliver.
    pending_calibration: Option<RectI>,
    /// What the owner was last told about the gaze.
    gaze_problem: Option<GazeProblem>,
    beeper: Beeper,
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
            gaze_jump: setup.tuning.gaze_jump,
            storage: setup.storage,
            engine: Engine::new(setup.tuning, calibration),
            desktop: Desktop::query().ok(),
            desktop_asked: 0.0,
            camera: None,
            backoff: Backoff::new((runtime::seconds() * 1e6) as u32 ^ std::process::id()),
            retry_at: 0.0,
            gaps: 0,
            pending_calibration: None,
            gaze_problem: None,
            beeper: Beeper::new(),
            frame: BgraFrame::default(),
            mesh: FaceMesh::default(),
        }
    }

    /// Works until the owner asks it to stop or goes away.
    fn run(&mut self, tracker: &mut Tracker) {
        loop {
            let flow = if self.shared.paused.load(Ordering::SeqCst) {
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
                return;
            }
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
            Command::StartCalibration(area) => {
                self.pending_calibration = Some(area);
                self.retry_now();
            }
            Command::CancelCalibration => {
                self.pending_calibration = None;
                self.engine.cancel_calibration();
            }
            Command::DisplayChanged => {
                self.desktop = Desktop::query().ok();
                self.end_calibration(CalibrationProblem::DisplayChanged);
            }
        }
        Flow::Continue
    }

    /// Something the person did deserves an immediate try, whatever wait was in force.
    fn retry_now(&mut self) {
        self.retry_at = 0.0;
        self.backoff.reset();
    }

    /// Ends a calibration run that is under way or waiting to start, telling the owner why.
    fn end_calibration(&mut self, problem: CalibrationProblem) {
        let waiting = self.pending_calibration.take().is_some();
        if self.engine.cancel_calibration() || waiting {
            self.shared.notice(Notice::CalibrationFailed(problem));
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
        let found = match tracker.track(&self.frame, &mut self.mesh) {
            Ok(found) => found,
            Err(error) => return self.failed(&error),
        };
        let time = self.frame.timestamp;
        let frame_size = (self.frame.width as u32, self.frame.height as u32);
        if self.desktop.is_none() && time - self.desktop_asked >= 1.0 {
            self.desktop_asked = time;
            self.desktop = Desktop::query().ok();
        }
        if let Some(area) = self.pending_calibration.take() {
            self.engine.start_calibration(area, frame_size, time);
        }
        let screens = self.desktop.as_ref().map_or(&[][..], Desktop::monitors);
        let output = self.engine.frame(&Input {
            time,
            mesh: found.then_some(&self.mesh),
            frame_size,
            cursor: pointer::cursor_position(),
            screens,
        });
        let problem =
            self.engine.calibration().filter(|_| self.gaze_jump).and_then(|c| gaze_problem(c, frame_size, screens));
        let refused = self.act(&output);
        self.report(&output, problem, refused);
    }

    /// Carries out what the engine answered. True when the system refused the pointer action.
    fn act(&self, output: &Output) -> bool {
        match output.sound {
            Some(Sound::Armed) => self.beeper.play(ARMED_TONE.0, ARMED_TONE.1),
            Some(Sound::Cancelled) => self.beeper.play(CANCELLED_TONE.0, CANCELLED_TONE.1),
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
        if let Some(outcome) = output.finished {
            self.shared.notice(match outcome {
                Ok(error_px) => {
                    let saved = self.engine.calibration().map_or(Ok(()), |c| self.storage.save_calibration(c));
                    Notice::CalibrationDone { error_px, saved: saved.map_err(|error| error.message().to_string()) }
                }
                Err(failure) => Notice::CalibrationFailed(CalibrationProblem::Routine(failure)),
            });
        }
        if problem != self.gaze_problem {
            self.gaze_problem = problem;
            if let Some(problem) = problem {
                self.shared.notice(Notice::GazeUnavailable(problem));
            }
        }
        let (status, view) = match output.mode {
            Mode::Calibrating(view) => (Status::Calibrating, Some(view)),
            Mode::Pointing(_) if refused => (Status::Pointing(Activity::Blocked), None),
            Mode::Pointing(activity) => (Status::Pointing(activity), None),
        };
        self.shared.set(status, view);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

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

    #[test]
    fn a_missing_camera_is_waited_for_and_pause_calibration_and_stop_still_work() {
        let scratch = Scratch(std::env::temp_dir().join(format!("etm-pipeline-test-{}", std::process::id())));
        let _ = std::fs::remove_dir_all(&scratch.0);
        let setup = Setup {
            camera: CameraRequest { device_index: NO_SUCH_CAMERA, ..CameraRequest::default() },
            tuning: Tuning::default(),
            storage: Storage::open(&scratch.0).unwrap(),
        };
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
        pipeline.start_calibration(RectI::new(0, 0, 1920, 1080));
        wait_for(&pipeline, &mut notices, |_, notices| !notices.is_empty());
        std::thread::sleep(Duration::from_millis(300));
        wait_for(&pipeline, &mut notices, |_, _| true);
        assert!(
            matches!(notices.as_slice(), [Notice::CalibrationFailed(CalibrationProblem::Camera(_))]),
            "{notices:?}"
        );

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
