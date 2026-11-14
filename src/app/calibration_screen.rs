//! The page that covers a screen during a calibration run and shows the person where to look.
//!
//! `scene` decides what is on the page and `Watch` when the camera has gone
//! quiet for too long (both pure); `CalibrationScreen` owns the window.
//!
//! Contract: the page is there from `open` until the value is dropped. It
//! never ends a run by itself; its owner asks `camera_is_silent` on every
//! `Event::Timer` of the page's window and decides.

use crate::control::calibration::View;
use crate::error::Result;
use crate::geometry::RectI;
use crate::platform::canvas::{Colour, Shape, Surface};
use crate::platform::window::{self, Kind, Window};

use super::wording;

const BACKGROUND: Colour = (18, 18, 22);
const TEXT: Colour = (236, 236, 236);
const HINT: Colour = (150, 150, 156);
const WARNING: Colour = (242, 169, 0);
const RING: Colour = (45, 127, 249);
const DOT: Colour = (255, 255, 255);
const TRACK: Colour = (58, 58, 66);
const DONE: Colour = (46, 184, 92);

/// Sizes of the place to look at, in hundredths of the screen's shorter side:
/// the dot, the ring around it when a place appears and when it is done, and the ring's thickness.
const DOT_RADIUS: f32 = 0.8;
const RING_START: f32 = 5.0;
const RING_END: f32 = 1.6;
const RING_WIDTH: f32 = 0.5;
/// The model's answer for a place of the check, in the same hundredths.
const ANSWER_RADIUS: f32 = 1.2;
/// How far down the page the lines of text are, as shares of its height. The
/// rows of the grid are at 5, 27.5, 50, 72.5 and 95 percent, so these lie
/// between them. (The places of the check go anywhere and are drawn on top.)
const INSTRUCTION_AT: f32 = 0.3875;
const WARNING_AT: f32 = 0.6125;
const HINT_AT: f32 = 0.8375;
/// The bar that shows how far the run has got: its left and right ends and its
/// top and bottom, as shares of the page. Between the two upper rows of the grid.
const BAR: (f32, f32, f32, f32) = (0.3, 0.7, 0.155, 0.167);

/// Seconds without a camera frame after which the run is given up: while the
/// camera may still be starting, and once it has delivered.
const START_PATIENCE: f64 = 10.0;
const FRAME_PATIENCE: f64 = 3.0;
/// How often the owner is reminded to ask, in milliseconds.
const WATCH_INTERVAL: u32 = 250;
const WATCH_TIMER: usize = 1;

/// Everything on the page for the screen `area` (desktop pixels). `None` while the run has not begun.
/// `score` is the line shown in place of a spot to look at once the view carries the check's outcome.
pub fn scene<'a>(view: Option<&View>, score: &'a str, area: RectI) -> Vec<Shape<'a>> {
    let (width, height) = (area.width(), area.height());
    let unit = width.min(height) as f32 / 100.0;
    // Small enough for the longest line to fit the width, and for a line to fit between two rows of the grid.
    let letters = (width / 40).min(height / 24).max(12);
    let line = |text: &'a str, share: f32, colour: Colour| {
        let middle = (height as f32 * share) as i32;
        Shape::Text { text, area: RectI::new(0, middle - letters, width, middle + letters), height: letters, colour }
    };
    let mut shapes = Vec::new();
    match view {
        None => shapes.push(line(wording::CALIBRATION_PREPARING, INSTRUCTION_AT, TEXT)),
        Some(view) => {
            let said = if view.score.is_some() { score } else { wording::instruction(view.phase) };
            shapes.push(line(said, INSTRUCTION_AT, TEXT));
            if !view.face_visible && view.score.is_none() {
                shapes.push(line(wording::FACE_NOT_VISIBLE, WARNING_AT, WARNING));
            }
            let across = |share: f32| (width as f32 * share) as i32;
            let down = |share: f32| (height as f32 * share) as i32;
            let bar = RectI::new(across(BAR.0), down(BAR.2), across(BAR.1), down(BAR.3));
            let done = bar.left + (bar.width() as f32 * view.progress.clamp(0.0, 1.0)) as i32;
            shapes.push(Shape::Box { area: bar, colour: TRACK });
            shapes.push(Shape::Box { area: RectI::new(bar.left, bar.top, done, bar.bottom), colour: DONE });
            let on_page = |(x, y): (i32, i32)| (x - area.left, y - area.top);
            let radius = |hundredths: f32| (unit * hundredths).round() as i32;
            if view.score.is_none() {
                // The ring closes in on the dot while the place is being recorded.
                let centre = on_page(view.target);
                let ring = RING_START - (RING_START - RING_END) * view.dwell.clamp(0.0, 1.0);
                shapes.push(Shape::Disc { centre, radius: radius(ring), colour: RING });
                shapes.push(Shape::Disc { centre, radius: radius(ring - RING_WIDTH), colour: BACKGROUND });
                shapes.push(Shape::Disc { centre, radius: radius(DOT_RADIUS).max(2), colour: DOT });
            }
            if let Some((answer, hit)) = view.answer {
                let colour = if hit { DONE } else { WARNING };
                shapes.push(Shape::Disc { centre: on_page(answer), radius: radius(ANSWER_RADIUS), colour });
            }
        }
    }
    shapes.push(line(wording::CALIBRATION_HOW_TO_CANCEL, HINT_AT, HINT));
    shapes
}

/// Tells a camera that is busy from one that has stopped delivering, by the
/// worker's count of frames.
#[derive(Clone, Copy, Debug)]
struct Watch {
    frames: u64,
    /// When the count last moved, or when the watch began.
    moved: f64,
    /// Whether the count has moved at all since the watch began.
    delivering: bool,
}

impl Watch {
    fn new(frames: u64, now: f64) -> Watch {
        Watch { frames, moved: now, delivering: false }
    }

    /// Takes note of the count at time `now` (seconds). True when it has stood still for too long.
    fn is_silent(&mut self, frames: u64, now: f64) -> bool {
        if frames != self.frames {
            *self = Watch { frames, moved: now, delivering: true };
        }
        now - self.moved > if self.delivering { FRAME_PATIENCE } else { START_PATIENCE }
    }
}

pub struct CalibrationScreen {
    window: Window,
    surface: Surface,
    area: RectI,
    view: Option<View>,
    watch: Watch,
}

impl CalibrationScreen {
    /// Covers the screen `area` (desktop pixels) and takes the keyboard.
    /// `frames` is the worker's count of camera frames and `now` the time in seconds.
    pub fn open(class: &str, area: RectI, frames: u64, now: f64) -> Result<CalibrationScreen> {
        let window = window::open(class, wording::CALIBRATION_TITLE, Kind::Covering, area)?;
        // From here on the window goes with the value, also when something below fails.
        let mut screen =
            CalibrationScreen { window, surface: Surface::new(window), area, view: None, watch: Watch::new(frames, now) };
        window::start_timer(window, WATCH_TIMER, WATCH_INTERVAL);
        window::show(window, true);
        screen.draw();
        Ok(screen)
    }

    pub fn window(&self) -> Window {
        self.window
    }

    /// Shows the run as it stands.
    pub fn show(&mut self, view: View) {
        if self.view != Some(view) {
            self.view = Some(view);
            self.draw();
        }
    }

    pub fn draw(&mut self) {
        let score = self.view.and_then(|view| view.score).map(wording::check_score).unwrap_or_default();
        self.surface.paint(BACKGROUND, &scene(self.view.as_ref(), &score, self.area));
    }

    /// Takes note of the worker's count of camera frames at time `now`
    /// (seconds). True when no frame has arrived for too long.
    pub fn camera_is_silent(&mut self, frames: u64, now: f64) -> bool {
        self.watch.is_silent(frames, now)
    }
}

impl Drop for CalibrationScreen {
    fn drop(&mut self) {
        // The timer goes with the window.
        window::destroy(self.window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::calibration::{Phase, Score};

    /// A second monitor to the right of the main one.
    const AREA: RectI = RectI::new(1920, 0, 3840, 1080);

    fn view(phase: Phase, target: (i32, i32), dwell: f32, progress: f32) -> View {
        View { phase, target, dwell, progress, face_visible: true, answer: None, score: None }
    }

    /// The page for `view` on the second monitor, with no score to tell.
    fn scene(view: Option<&View>, area: RectI) -> Vec<Shape<'static>> {
        super::scene(view, "", area)
    }

    fn texts<'a>(shapes: &[Shape<'a>]) -> Vec<&'a str> {
        let text = |shape: &Shape<'a>| match *shape {
            Shape::Text { text, .. } => Some(text),
            _ => None,
        };
        shapes.iter().filter_map(text).collect()
    }

    fn discs(shapes: &[Shape]) -> Vec<((i32, i32), i32, Colour)> {
        let disc = |shape: &Shape| match *shape {
            Shape::Disc { centre, radius, colour } => Some((centre, radius, colour)),
            _ => None,
        };
        shapes.iter().filter_map(disc).collect()
    }

    #[test]
    fn the_place_to_look_at_is_drawn_where_it_is_on_that_screen() {
        // A place given in desktop pixels lands at the same spot of the page, whichever monitor the page covers.
        let shapes = scene(Some(&view(Phase::Places, (1920 + 153, 86), 0.0, 0.2)), AREA);
        let discs = discs(&shapes);
        assert_eq!(discs.len(), 3);
        assert!(discs.iter().all(|&(centre, _, _)| centre == (153, 86)));
        // Ring, the hole in it, then the dot on top; 1080 pixels make the unit 10.8.
        assert_eq!(discs[0], ((153, 86), 54, RING));
        assert_eq!(discs[1], ((153, 86), 49, BACKGROUND));
        assert_eq!(discs[2], ((153, 86), 9, DOT));
    }

    #[test]
    fn the_ring_closes_in_while_a_place_is_recorded_and_never_beyond() {
        let ring = |dwell: f32| discs(&scene(Some(&view(Phase::Places, (2000, 500), dwell, 0.5)), AREA))[0].1;
        assert!(ring(0.0) > ring(0.5) && ring(0.5) > ring(1.0));
        assert_eq!(ring(1.0), 17);
        assert_eq!(ring(7.0), ring(1.0));
        assert_eq!(ring(-3.0), ring(0.0));
        assert_eq!(ring(f32::NAN), ring(f32::NAN), "a dwell that is not a number still draws");
    }

    #[test]
    fn each_part_of_the_run_says_what_to_do_and_how_to_stop() {
        for phase in [Phase::Rest, Phase::Places, Phase::Sweep, Phase::Check] {
            let shapes = scene(Some(&view(phase, (2880, 540), 0.3, 0.4)), AREA);
            assert_eq!(texts(&shapes), [wording::instruction(phase), wording::CALIBRATION_HOW_TO_CANCEL]);
        }
        let hidden = View { face_visible: false, ..view(Phase::Places, (2880, 540), 0.3, 0.4) };
        assert_eq!(
            texts(&scene(Some(&hidden), AREA)),
            [wording::instruction(Phase::Places), wording::FACE_NOT_VISIBLE, wording::CALIBRATION_HOW_TO_CANCEL]
        );
        // Before the camera delivers there is no place to look at yet, only how to get out.
        let waiting = scene(None, AREA);
        assert_eq!(texts(&waiting), [wording::CALIBRATION_PREPARING, wording::CALIBRATION_HOW_TO_CANCEL]);
        assert_eq!(waiting.len(), 2);
    }

    #[test]
    fn the_text_stays_clear_of_every_place_and_the_bar_shows_the_progress() {
        let shapes = scene(Some(&View { face_visible: false, ..view(Phase::Places, (2880, 540), 0.0, 0.25) }), AREA);
        // The rows of the grid, with the largest ring around them.
        let rows = [0.05f32, 0.275, 0.5, 0.725, 0.95].map(|share| (share * 1079.0).round() as i32).map(|y| (y - 54, y + 54));
        for shape in &shapes {
            if let Shape::Text { area, height, .. } = *shape {
                assert_eq!((area.left, area.right), (0, 1920), "centred across the page");
                assert!(height >= 12 && area.height() == 2 * height);
                assert!(rows.iter().all(|&(top, bottom)| area.bottom < top || area.top > bottom), "{area:?} covers a place");
            }
        }
        let boxes: Vec<RectI> = shapes
            .iter()
            .filter_map(|shape| match *shape {
                Shape::Box { area, .. } => Some(area),
                _ => None,
            })
            .collect();
        let [track, done] = boxes[..] else { panic!("{boxes:?}") };
        assert_eq!((track.left, track.right), (576, 1344));
        assert_eq!((done.left, done.top, done.bottom), (track.left, track.top, track.bottom));
        assert_eq!(done.width(), track.width() / 4);
        assert!(track.top > rows[0].1 && track.bottom < rows[1].0, "between the two upper rows");

        // A small upright screen still gets letters that can be read.
        let narrow = scene(Some(&view(Phase::Rest, (300, 400), 0.0, 0.0)), RectI::new(0, 0, 600, 800));
        assert!(narrow.iter().all(|shape| !matches!(*shape, Shape::Text { height, .. } if height < 12)));
    }

    #[test]
    fn the_check_shows_the_models_answer_by_its_place_and_ends_with_the_score_alone() {
        let place = view(Phase::Check, (1920 + 700, 400), 1.0, 0.8);
        let answered = |answer, hit| discs(&super::scene(Some(&View { answer: Some((answer, hit)), ..place }), "", AREA));
        // The place stays on the page and the answer is drawn over it, where the model put it on this screen.
        let hit = answered((1920 + 720, 410), true);
        assert_eq!(hit.len(), 4);
        assert_eq!(hit[2], ((700, 400), 9, DOT));
        assert_eq!(hit[3], ((720, 410), 13, DONE));
        assert_eq!(answered((1920 + 900, 100), false)[3], ((900, 100), 13, WARNING));
        // An answer beyond the screen's edge is no reason not to draw.
        assert_eq!(answered((1900, -40), false)[3].0, (-20, -40));

        let score = Score { hits: 9, places: 12, mean_px: 71.4 };
        let line = wording::check_score(score);
        // The face no longer matters once the score is up, and there is nothing left to look at.
        let ended = View { score: Some(score), face_visible: false, ..place };
        let shapes = super::scene(Some(&ended), &line, AREA);
        assert_eq!(texts(&shapes), [line.as_str(), wording::CALIBRATION_HOW_TO_CANCEL]);
        assert!(discs(&shapes).is_empty());
        assert!(line.contains("9 / 12") && line.contains("71ピクセル"), "{line}");
    }

    #[test]
    fn a_camera_gets_time_to_start_and_less_once_it_has_delivered() {
        let mut watch = Watch::new(100, 50.0);
        assert!(!watch.is_silent(100, 50.0 + START_PATIENCE - 0.1), "still starting");
        assert!(watch.is_silent(100, 50.0 + START_PATIENCE + 0.1), "it never started");

        let mut watch = Watch::new(100, 50.0);
        assert!(!watch.is_silent(101, 58.0), "the first frame");
        assert!(!watch.is_silent(101, 58.0 + FRAME_PATIENCE - 0.1));
        assert!(watch.is_silent(101, 58.0 + FRAME_PATIENCE + 0.1), "it stopped");
        assert!(!watch.is_silent(102, 70.0), "and delivers again");
        // A count that starts over (the worker was started anew) is a sign of life like any other.
        assert!(!watch.is_silent(0, 72.9));
        assert!(watch.is_silent(0, 76.0));
    }
}
