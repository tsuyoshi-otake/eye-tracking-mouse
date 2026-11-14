//! The calibration routine and the result it leaves behind.
//!
//! The person is shown one place to look at after another while the camera
//! measures their face. Stages, each leading to the next or to a failure:
//!
//! ```text
//! Rest --> Places (a grid over the screen) --> Sweep --> Check --> Done
//!   |          |                                  |
//!   +----------+----------------------------------+--> Failed
//! ```
//!
//! - Rest: looking at the middle, holding still. Captures the person's face
//!   shape and where their head normally is.
//! - Places: looking at each place in turn.
//! - Sweep: still looking at the middle while turning the head. This is what
//!   lets the model tell a turn of the head from a change of gaze.
//!
//! - Check: the model is made by now. The person looks at a dozen places it
//!   was not fitted to, and after each one the page shows where the model put
//!   the gaze. How far off it was on average is the accuracy that is reported.
//!   A check that cannot be completed does not cost the calibration.
//!
//! No stage waits forever: a face that stays away fails the routine, and so
//! does a stage that drags on.

use std::collections::HashMap;
use std::str::FromStr;

use crate::error::{Error, Result};
use crate::geometry::{RectI, Vec2};
use crate::vision::landmarks::{FaceMesh, RIGID};
use crate::vision::pose::{ReferenceBuilder, ReferenceShape};
use crate::vision::{self, FaceMeasurement};

use super::gaze::{FEATURES, GazeModel, Neutral, Sample, held_out_error};

/// Seconds given to find the first place and settle before anything is recorded.
const REST_SETTLE: f64 = 1.5;
/// Frames averaged into the face shape.
const REST_FRAMES: usize = 30;
/// Seconds the eyes are given to reach a place that has just appeared.
const TRAVEL: f64 = 1.0;
/// Frames recorded at each place.
const PLACE_FRAMES: usize = 24;
/// Shortest sweep, in seconds.
const SWEEP_SECONDS: f64 = 4.0;
/// How far the head must turn during the sweep, sideways and vertically, as
/// the change of its facing vector (0.2 is about 12 degrees from end to end).
const SWEEP_TURN: Vec2 = Vec2::new(0.20, 0.12);
/// Sweep frames with the head turned further than this from straight ahead (as the facing
/// vector, about 12 degrees) stay out of the fit: nobody points with the head turned that far,
/// and fitting them costs accuracy where the head usually is.
const SWEEP_USED: f32 = 0.2;
/// At most this many sweep frames go into the fit, so they do not outweigh the places.
const SWEEP_FRAMES: usize = 150;
/// A stage that has taken this many seconds has failed.
const STAGE_LIMIT: f64 = 25.0;
/// So has a face that stayed out of view this long.
const FACE_LIMIT: f64 = 5.0;
/// An eye narrower than this share of its resting size is blinking: the frame is not recorded.
const BLINK_SHARE: f32 = 0.45;
/// Resting lid gap, as a share of the eye's width, below which the eyes were not open during the rest stage.
const MIN_OPEN_SIZE: f32 = 0.1;
/// A frame belongs to a place while its eye signals are within this many typical
/// deviations of the place's middle value...
const SETTLED_SPREADS: f32 = 4.5;
/// ...where a typical deviation is taken to be at least this share of the eye's
/// width (a resting eye measured 0.002 to 0.004).
const SETTLED_FLOOR: f32 = 0.003;
/// A result whose held-out error exceeds this share of the screen's shorter side is refused.
const MAX_ERROR_SHARE: f32 = 0.25;

/// The places looked at form a grid, given as shares of the screen's width and height.
/// The outer ones are as near the edge as a place can be and still be seen whole: the model
/// is only as good at the edge as the places it was shown there.
const COLUMNS: [f32; 5] = [0.03, 0.265, 0.5, 0.735, 0.97];
const ROWS: [f32; 5] = [0.05, 0.275, 0.5, 0.725, 0.95];
/// How many places there are: every crossing of the grid.
const PLACES: usize = COLUMNS.len() * ROWS.len();

/// The places of the check, as shares of the screen's width and height: none of them on the
/// grid, some by the edge, and in an order that makes the eyes jump about.
const CHECKS: [(f32, f32); 12] = [
    (0.38, 0.16),
    (0.9, 0.86),
    (0.15, 0.62),
    (0.62, 0.84),
    (0.1, 0.14),
    (0.97, 0.39),
    (0.38, 0.95),
    (0.15, 0.39),
    (0.9, 0.14),
    (0.62, 0.39),
    (0.1, 0.86),
    (0.8, 0.62),
];
/// Frames measured at each place of the check.
const CHECK_FRAMES: usize = 24;
/// Seconds the model's answer for a place of the check stays on the page.
const CHECK_SHOWN: f64 = 0.6;
/// Seconds the score stays on the page before the run ends.
const SCORE_SHOWN: f64 = 4.0;
/// A place of the check is hit when the model's answer is within this share of the screen's shorter side.
const HIT_SHARE: f32 = 0.08;

/// Where place number `place` is, as shares of the screen's width and height. The first is
/// the middle, where the eyes rest before and after; the others follow row by row from the
/// top, back and forth, so the eyes never have far to go.
fn place_share(place: usize) -> (f32, f32) {
    let middle = (COLUMNS.len() / 2, ROWS.len() / 2);
    let crossing = |step: usize| {
        let (row, along) = (step / COLUMNS.len(), step % COLUMNS.len());
        (if row % 2 == 0 { along } else { COLUMNS.len() - 1 - along }, row)
    };
    let (column, row) = match place {
        0 => middle,
        _ => (0..PLACES).map(crossing).filter(|&at| at != middle).nth(place - 1).unwrap_or(middle),
    };
    (COLUMNS[column], ROWS[row])
}

const HEADER: &str = "eye-tracking-mouse calibration 2";

/// Everything learned about one person at one desk.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// The person's face shape, which head pose is measured against.
    pub reference: ReferenceShape,
    pub gaze: GazeModel,
    /// The screen that was looked at, in desktop pixels. Gaze estimates mean nothing elsewhere.
    pub area: RectI,
    /// How far off the gaze estimate typically is at a place it was not fitted to, in pixels.
    pub error_px: f32,
}

impl Calibration {
    /// A few lines of `name = numbers` under a version header.
    pub fn to_text(&self) -> String {
        fn join<T: ToString>(values: impl IntoIterator<Item = T>) -> String {
            values.into_iter().map(|value| value.to_string()).collect::<Vec<_>>().join(" ")
        }
        let neutral = self.gaze.neutral();
        format!(
            "{HEADER}\narea = {}\nerror_px = {}\ncamera = {}\nneutral = {}\nweights_x = {}\nweights_y = {}\nreference = {}\n",
            join([self.area.left, self.area.top, self.area.right, self.area.bottom]),
            self.error_px,
            join([neutral.frame_size.0, neutral.frame_size.1]),
            join([neutral.position.x, neutral.position.y, neutral.scale]),
            join(self.gaze.weights()[0]),
            join(self.gaze.weights()[1]),
            join(self.reference.points().iter().flatten().copied()),
        )
    }

    pub fn parse(text: &str) -> Result<Calibration> {
        let mut lines = text.lines().map(str::trim).filter(|line| !line.is_empty());
        if lines.next() != Some(HEADER) {
            return Err(Error::new("not a calibration this version can read"));
        }
        let mut fields = HashMap::new();
        for line in lines {
            let (name, values) =
                line.split_once('=').ok_or_else(|| Error::new(format!("the line '{line}' is not 'name = numbers'")))?;
            if fields.insert(name.trim(), values.trim()).is_some() {
                return Err(Error::new(format!("'{}' appears twice", name.trim())));
            }
        }
        fn numbers<T: FromStr>(fields: &HashMap<&str, &str>, name: &str, count: usize) -> Result<Vec<T>> {
            let text = fields.get(name).ok_or_else(|| Error::new(format!("'{name}' is missing")))?;
            let values: Vec<T> = text
                .split_whitespace()
                .map(str::parse)
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| Error::new(format!("'{name}' holds something that is not a number")))?;
            if values.len() != count {
                return Err(Error::new(format!("'{name}' needs {count} numbers but has {}", values.len())));
            }
            Ok(values)
        }
        let area = numbers::<i32>(&fields, "area", 4)?;
        let area = RectI::new(area[0], area[1], area[2], area[3]);
        let error_px = numbers::<f32>(&fields, "error_px", 1)?[0];
        let camera = numbers::<u32>(&fields, "camera", 2)?;
        let neutral = numbers::<f32>(&fields, "neutral", 3)?;
        let neutral =
            Neutral { frame_size: (camera[0], camera[1]), position: Vec2::new(neutral[0], neutral[1]), scale: neutral[2] };
        let mut weights = [[0.0f64; FEATURES]; 2];
        weights[0].copy_from_slice(&numbers::<f64>(&fields, "weights_x", FEATURES)?);
        weights[1].copy_from_slice(&numbers::<f64>(&fields, "weights_y", FEATURES)?);
        let reference = numbers::<f32>(&fields, "reference", RIGID.len() * 3)?;
        let reference = reference.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        let unusable = |what: &str| Error::new(format!("the stored {what} cannot be used"));
        if area.is_empty() || !(error_px.is_finite() && error_px >= 0.0) {
            return Err(unusable("screen area or error"));
        }
        Ok(Calibration {
            reference: ReferenceShape::from_points(reference).ok_or_else(|| unusable("face shape"))?,
            gaze: GazeModel::from_parts(neutral, weights).ok_or_else(|| unusable("gaze model"))?,
            area,
            error_px,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Rest,
    Places,
    Sweep,
    Check,
}

/// How the check went.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    /// Places where the model's answer was close enough to count.
    pub hits: u32,
    pub places: u32,
    /// How far the model's answer was from the place, on average, in pixels.
    pub mean_px: f32,
}

/// What to show the person right now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub phase: Phase,
    /// The place to look at, in desktop pixels.
    pub target: (i32, i32),
    /// How far the current place (or the sweep) has got, 0 to 1.
    pub dwell: f32,
    /// How far the whole routine has got, 0 to 1.
    pub progress: f32,
    /// False while the camera does not see the face.
    pub face_visible: bool,
    /// During the check, once a place has been measured: where the model put the gaze, in
    /// desktop pixels, and whether that is a hit.
    pub answer: Option<((i32, i32), bool)>,
    /// The outcome of the check, shown at its end.
    pub score: Option<Score>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Failure {
    /// The face stayed out of the camera's view.
    FaceLost,
    /// Too few usable frames: eyes shut, or a face the tracker cannot measure.
    TooSlow,
    /// The head did not turn far enough during the sweep.
    HeadNotMoved,
    /// No face shape or gaze model could be made from what was recorded.
    Unusable,
    /// The model disagrees with itself by this much: the places were probably not looked at.
    Inaccurate { error_px: f32 },
}

pub enum Step {
    Running(CalibrationSession),
    Done(Calibration),
    Failed(Failure),
}

/// What the rest stage establishes and every later frame is measured against.
struct Basis {
    reference: ReferenceShape,
    neutral: Neutral,
    /// Resting lid gap of the right and the left eye.
    open_size: [f32; 2],
}

impl Basis {
    fn from_rest(meshes: &[FaceMesh], frame_size: (u32, u32)) -> Option<(Basis, Vec<FaceMeasurement>)> {
        let mut builder = ReferenceBuilder::default();
        for mesh in meshes {
            builder.add(&mesh.points);
        }
        let reference = builder.build()?;
        let faces: Vec<FaceMeasurement> = meshes.iter().filter_map(|mesh| vision::measure(&reference, mesh)).collect();
        if faces.len() * 2 < meshes.len() {
            return None;
        }
        let share = 1.0 / faces.len() as f32;
        let position = faces.iter().fold(Vec2::ZERO, |sum, face| sum + face.pose.position * share);
        let scale = faces.iter().map(|face| face.pose.scale).sum::<f32>() * share;
        let open_size = [0, 1].map(|eye| {
            let mut sizes: Vec<f32> = faces.iter().map(|face| face.eyes.openness[eye]).collect();
            sizes.sort_by(f32::total_cmp);
            sizes[sizes.len() / 2]
        });
        // Without a resting size there is no telling a blink from an open eye.
        if open_size.iter().any(|&size| size < MIN_OPEN_SIZE) {
            return None;
        }
        Some((Basis { reference, neutral: Neutral { frame_size, position, scale }, open_size }, faces))
    }

    /// The frame's measurement, unless the face cannot be measured or the eyes are blinking.
    fn measure(&self, mesh: &FaceMesh) -> Option<FaceMeasurement> {
        let face = vision::measure(&self.reference, mesh)?;
        self.eyes_open(&face).then_some(face)
    }

    fn eyes_open(&self, face: &FaceMeasurement) -> bool {
        (0..2).all(|eye| face.eyes.openness[eye] >= BLINK_SHARE * self.open_size[eye])
    }
}

enum Stage {
    Rest { meshes: Vec<FaceMesh> },
    Places { basis: Basis, index: usize, recorded: usize },
    /// `low` and `high` bound the facing vectors seen so far.
    Sweep { basis: Basis, low: Vec2, high: Vec2, frames: Vec<Sample> },
    /// `estimates` are the model's answers for place `index` so far, `misses` how far off the
    /// places before it were, and `shown` the answer on the page with the time it appeared.
    Check {
        basis: Basis,
        calibration: Box<Calibration>,
        index: usize,
        estimates: Vec<Vec2>,
        misses: Vec<f32>,
        shown: Option<(Vec2, f64)>,
    },
    Score { calibration: Box<Calibration>, score: Score },
}

/// What the check found, from how far off each of its places was. `None` when no place was measured.
fn score(area: RectI, misses: &[f32]) -> Option<Score> {
    let reach = HIT_SHARE * area.width().min(area.height()) as f32;
    (!misses.is_empty()).then(|| Score {
        hits: misses.iter().filter(|&&miss| miss <= reach).count() as u32,
        places: misses.len() as u32,
        mean_px: misses.iter().sum::<f32>() / misses.len() as f32,
    })
}

/// The calibration with what the check measured as its accuracy, when it measured anything:
/// that says more than the model's own estimate from the places it was fitted to.
fn checked(mut calibration: Calibration, score: Option<Score>) -> Calibration {
    if let Some(score) = score {
        calibration.error_px = score.mean_px;
    }
    calibration
}

/// Where place number `place` is on the screen `area`, in desktop pixels.
fn target(area: RectI, place: usize) -> (i32, i32) {
    let (across, down) = place_share(place);
    (
        area.left + (across * (area.width() - 1) as f32).round() as i32,
        area.top + (down * (area.height() - 1) as f32).round() as i32,
    )
}

/// Where place number `index` of the check is on the screen `area`, in desktop pixels.
fn check_point(area: RectI, index: usize) -> Vec2 {
    let (across, down) = CHECKS[index];
    Vec2::new(
        area.left as f32 + (across * (area.width() - 1) as f32).round(),
        area.top as f32 + (down * (area.height() - 1) as f32).round(),
    )
}

fn target_point(area: RectI, place: usize) -> Vec2 {
    let (x, y) = target(area, place);
    Vec2::new(x as f32, y as f32)
}

pub struct CalibrationSession {
    area: RectI,
    frame_size: (u32, u32),
    stage: Stage,
    stage_started: f64,
    face_seen: f64,
    face_visible: bool,
    samples: Vec<Sample>,
}

impl CalibrationSession {
    /// `area` is the screen the places are shown on, in desktop pixels;
    /// `frame_size` the size of the camera image the meshes come from.
    pub fn new(area: RectI, frame_size: (u32, u32), time: f64) -> CalibrationSession {
        CalibrationSession {
            area,
            frame_size,
            stage: Stage::Rest { meshes: Vec::new() },
            stage_started: time,
            face_seen: time,
            face_visible: false,
            samples: Vec::new(),
        }
    }

    pub fn view(&self, time: f64) -> View {
        let elapsed = time - self.stage_started;
        let steps = (PLACES + 1 + CHECKS.len()) as f32;
        let plain = |phase, target, dwell: f32, finished: usize| View {
            phase,
            target,
            dwell,
            progress: (finished as f32 + dwell) / steps,
            face_visible: self.face_visible,
            answer: None,
            score: None,
        };
        let pixel = |point: Vec2| (point.x.round() as i32, point.y.round() as i32);
        let (phase, place, finished, dwell) = match &self.stage {
            Stage::Rest { meshes } => {
                let settled = (elapsed / REST_SETTLE).clamp(0.0, 1.0) as f32;
                (Phase::Rest, 0, 0, 0.5 * settled + 0.5 * meshes.len() as f32 / REST_FRAMES as f32)
            }
            Stage::Places { index, recorded, .. } => (Phase::Places, *index, *index, *recorded as f32 / PLACE_FRAMES as f32),
            Stage::Sweep { low, high, .. } => {
                let turned = ((high.x - low.x) / SWEEP_TURN.x).min((high.y - low.y) / SWEEP_TURN.y);
                let waited = ((elapsed - TRAVEL) / SWEEP_SECONDS) as f32;
                (Phase::Sweep, 0, PLACES, turned.min(waited).clamp(0.0, 1.0))
            }
            Stage::Check { index, estimates, shown, .. } => {
                let place = check_point(self.area, *index);
                let reach = HIT_SHARE * self.area.width().min(self.area.height()) as f32;
                let dwell = estimates.len() as f32 / CHECK_FRAMES as f32;
                return View {
                    answer: shown.map(|(answer, _)| (pixel(answer), answer.distance(place) <= reach)),
                    ..plain(Phase::Check, pixel(place), dwell.min(1.0), PLACES + 1 + index)
                };
            }
            Stage::Score { score, .. } => {
                let last = PLACES + CHECKS.len();
                return View { score: Some(*score), ..plain(Phase::Check, target(self.area, 0), 1.0, last) };
            }
        };
        plain(phase, target(self.area, place), dwell, finished)
    }

    /// Feeds one camera frame; `mesh` is `None` when no face was found in it.
    pub fn step(mut self, time: f64, mesh: Option<&FaceMesh>) -> Step {
        self.face_visible = mesh.is_some();
        let stuck = time - self.stage_started >= STAGE_LIMIT;
        let gone = mesh.is_none() && time - self.face_seen >= FACE_LIMIT;
        match self.stage {
            // The score needs no face in front of the camera to be read.
            Stage::Score { calibration, score } if time - self.stage_started >= SCORE_SHOWN => {
                return Step::Done(checked(*calibration, Some(score)));
            }
            Stage::Score { .. } => return Step::Running(self),
            // The model is made; a check that cannot go on ends with the places it got to.
            Stage::Check { calibration, misses, .. } if stuck || gone => {
                return Step::Done(checked(*calibration, score(self.area, &misses)));
            }
            Stage::Sweep { .. } if stuck => return Step::Failed(Failure::HeadNotMoved),
            _ if stuck => return Step::Failed(Failure::TooSlow),
            _ if gone => return Step::Failed(Failure::FaceLost),
            _ => {}
        }
        let Some(mesh) = mesh else { return Step::Running(self) };
        self.face_seen = time;
        let waiting = time - self.stage_started < if matches!(self.stage, Stage::Rest { .. }) { REST_SETTLE } else { TRAVEL };
        if waiting {
            return Step::Running(self);
        }
        let area = self.area;
        match self.stage {
            Stage::Rest { mut meshes } => {
                meshes.push(mesh.clone());
                if meshes.len() < REST_FRAMES {
                    self.stage = Stage::Rest { meshes };
                    return Step::Running(self);
                }
                let Some((basis, faces)) = Basis::from_rest(&meshes, self.frame_size) else {
                    return Step::Failed(Failure::Unusable);
                };
                let target = target_point(area, 0);
                self.samples.extend(
                    faces.into_iter().filter(|face| basis.eyes_open(face)).map(|face| Sample { face, target, place: Some(0) }),
                );
                self.stage = Stage::Places { basis, index: 1, recorded: 0 };
                self.stage_started = time;
            }
            Stage::Places { basis, index, mut recorded } => {
                if let Some(face) = basis.measure(mesh) {
                    self.samples.push(Sample { face, target: target_point(area, index), place: Some(index) });
                    recorded += 1;
                }
                if recorded < PLACE_FRAMES {
                    self.stage = Stage::Places { basis, index, recorded };
                } else if index + 1 < PLACES {
                    self.stage = Stage::Places { basis, index: index + 1, recorded: 0 };
                    self.stage_started = time;
                } else {
                    let (low, high) = (Vec2::new(f32::INFINITY, f32::INFINITY), Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY));
                    self.stage = Stage::Sweep { basis, low, high, frames: Vec::new() };
                    self.stage_started = time;
                }
            }
            Stage::Sweep { basis, mut low, mut high, mut frames } => {
                if let Some(face) = basis.measure(mesh) {
                    let facing = face.pose.forward();
                    low = Vec2::new(low.x.min(facing.x), low.y.min(facing.y));
                    high = Vec2::new(high.x.max(facing.x), high.y.max(facing.y));
                    frames.push(Sample { face, target: target_point(area, 0), place: None });
                }
                let turned = high.x - low.x >= SWEEP_TURN.x && high.y - low.y >= SWEEP_TURN.y;
                if turned && time - self.stage_started >= TRAVEL + SWEEP_SECONDS {
                    let calibration = match fit(area, std::mem::take(&mut self.samples), &basis, frames) {
                        Ok(calibration) => Box::new(calibration),
                        Err(failure) => return Step::Failed(failure),
                    };
                    let (estimates, misses) = (Vec::new(), Vec::new());
                    self.stage = Stage::Check { basis, calibration, index: 0, estimates, misses, shown: None };
                    self.stage_started = time;
                } else {
                    self.stage = Stage::Sweep { basis, low, high, frames };
                }
            }
            Stage::Check { basis, calibration, mut index, mut estimates, mut misses, mut shown } => {
                match shown {
                    // The answer has been looked at: on to the next place, or to the score.
                    Some((_, since)) if time - since >= CHECK_SHOWN => {
                        (index, shown) = (index + 1, None);
                        estimates.clear();
                        self.stage_started = time;
                    }
                    Some(_) => {}
                    None => {
                        estimates.extend(basis.measure(mesh).map(|face| calibration.gaze.predict(&face)));
                        if estimates.len() >= CHECK_FRAMES {
                            let share = 1.0 / estimates.len() as f32;
                            let answer = estimates.iter().fold(Vec2::ZERO, |sum, &one| sum + one * share);
                            misses.push(answer.distance(check_point(area, index)));
                            shown = Some((answer, time));
                        }
                    }
                }
                self.stage = match score(area, &misses).filter(|_| index == CHECKS.len()) {
                    Some(score) => Stage::Score { calibration, score },
                    None => Stage::Check { basis, calibration, index, estimates, misses, shown },
                };
            }
            // Ends by the clock, above.
            Stage::Score { .. } => {}
        }
        Step::Running(self)
    }

}

/// The middle value of `values`, which must not be empty.
fn median(mut values: Vec<f32>) -> f32 {
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

/// The frames of each place in which the eyes rested on it: a frame whose eye
/// signals are far from that place's usual ones was recorded while the eyes
/// were still on their way, or half-way through a blink, and is left out.
fn settled(samples: Vec<Sample>) -> Vec<Sample> {
    let signals = |sample: &Sample| [sample.face.eyes.iris.x, sample.face.eyes.iris.y, sample.face.eyes.lid];
    let mut bounds: HashMap<Option<usize>, [(f32, f32); 3]> = HashMap::new();
    for place in samples.iter().map(|sample| sample.place) {
        bounds.entry(place).or_insert_with(|| {
            let inside: Vec<[f32; 3]> = samples.iter().filter(|sample| sample.place == place).map(signals).collect();
            [0, 1, 2].map(|signal| {
                let middle = median(inside.iter().map(|values| values[signal]).collect());
                let spread = median(inside.iter().map(|values| (values[signal] - middle).abs()).collect());
                (middle, SETTLED_SPREADS * spread.max(SETTLED_FLOOR))
            })
        });
    }
    samples
        .into_iter()
        .filter(|sample| {
            let limits = &bounds[&sample.place];
            signals(sample).iter().zip(limits).all(|(value, (middle, reach))| (value - middle).abs() <= *reach)
        })
        .collect()
}

/// Fits the model to everything recorded and decides whether it is good enough to go on with.
fn fit(area: RectI, samples: Vec<Sample>, basis: &Basis, sweep: Vec<Sample>) -> std::result::Result<Calibration, Failure> {
    let mut samples = settled(samples);
    let sweep: Vec<Sample> = sweep.into_iter().filter(|sample| sample.face.pose.forward().length() <= SWEEP_USED).collect();
    let stride = sweep.len().div_ceil(SWEEP_FRAMES).max(1);
    samples.extend(sweep.into_iter().step_by(stride));
    let fitted = GazeModel::fit(basis.neutral, &samples).zip(held_out_error(basis.neutral, &samples));
    let Some((gaze, error_px)) = fitted else {
        return Err(Failure::Unusable);
    };
    if error_px > MAX_ERROR_SHARE * area.width().min(area.height()) as f32 {
        return Err(Failure::Inaccurate { error_px });
    }
    Ok(Calibration { reference: basis.reference.clone(), gaze, area, error_px })
}

#[cfg(test)]
pub(crate) mod simulation {
    //! A simulated person going through the routine.

    use super::*;
    use crate::control::gaze::simulation::{FRAME_SIZE, Person, Posture, SCREEN};

    #[derive(Clone, Copy)]
    pub struct Behaviour {
        /// Looks at the place shown; otherwise keeps staring at the middle of the screen.
        pub attentive: bool,
        /// Turns the head during the sweep.
        pub sweeps: bool,
        /// Blinks for five frames out of every this many.
        pub blink_every: Option<u32>,
        /// Keeps the eyes shut from this time on.
        pub shut_from: f64,
        /// Out of the camera's view between these times.
        pub away: (f64, f64),
    }

    impl Default for Behaviour {
        fn default() -> Self {
            Behaviour { attentive: true, sweeps: true, blink_every: None, shut_from: f64::INFINITY, away: (0.0, 0.0) }
        }
    }

    /// How the head is held while turning it for the sweep, `time` seconds in.
    pub fn sweeping(time: f64) -> Posture {
        let swing = time as f32;
        Posture {
            place: [0.2 * (2.2 * swing).sin(), 1.2, 6.7],
            yaw: 0.14 * (2.2 * swing).sin(),
            pitch: 0.08 * (1.7 * swing + 1.0).sin(),
            roll: 0.03 * (0.9 * swing).sin(),
        }
    }

    /// How the head is held while only the eyes go to `gaze`: it follows them a little, as it naturally does.
    pub fn following(gaze: Vec2) -> Posture {
        let middle = Vec2::new((SCREEN.left + SCREEN.right) as f32 * 0.5, (SCREEN.top + SCREEN.bottom) as f32 * 0.5);
        Posture { yaw: -(gaze.x - middle.x) * 6e-5, pitch: (gaze.y - middle.y) * 6e-5, ..Posture::default() }
    }

    /// Runs the routine at 30 frames a second until it ends, showing every
    /// view to `watch`. Returns the outcome and the time it took.
    pub fn run(person: &mut Person, behaviour: &Behaviour, mut watch: impl FnMut(&View)) -> (Step, f64) {
        let middle = Vec2::new((SCREEN.left + SCREEN.right) as f32 * 0.5, (SCREEN.top + SCREEN.bottom) as f32 * 0.5);
        let mut session = CalibrationSession::new(SCREEN, FRAME_SIZE, 0.0);
        let mut gaze = middle;
        let mut shown = (session.view(0.0).target, 0.0);
        let mut frame = 0u32;
        loop {
            let time = f64::from(frame) / 30.0;
            assert!(time < 300.0, "the routine must end");
            let view = session.view(time);
            watch(&view);
            if view.target != shown.0 {
                shown = (view.target, time);
            }
            // The eyes take a third of a second to land on a place that has just appeared.
            if behaviour.attentive && time - shown.1 >= 0.3 {
                gaze = Vec2::new(view.target.0 as f32, view.target.1 as f32);
            }
            let posture =
                if view.phase == Phase::Sweep && behaviour.sweeps { sweeping(time) } else { following(gaze) };
            let blinking = behaviour.blink_every.is_some_and(|period| frame % period < 5) || time >= behaviour.shut_from;
            let mesh = person.look(time, &posture, gaze, if blinking { 0.0 } else { 1.0 });
            let in_view = !(behaviour.away.0..behaviour.away.1).contains(&time);
            match session.step(time, in_view.then_some(&mesh)) {
                Step::Running(next) => session = next,
                finished => return (finished, time),
            }
            frame += 1;
        }
    }

    /// The calibration of a person who does everything right, with realistic landmark noise.
    pub fn calibrated() -> Calibration {
        match run(&mut Person::new(1.0), &Behaviour::default(), |_| ()).0 {
            Step::Done(calibration) => calibration,
            _ => panic!("the routine should succeed"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::simulation::{Behaviour, calibrated, run};
    use super::*;
    use crate::control::gaze::simulation::{Person, Posture, SCREEN};

    fn failure(behaviour: &Behaviour) -> (Failure, f64) {
        match run(&mut Person::new(1.0), behaviour, |_| ()) {
            (Step::Failed(failure), time) => (failure, time),
            _ => panic!("the routine should fail"),
        }
    }

    #[test]
    fn a_person_who_follows_the_routine_gets_a_model_that_finds_new_places() {
        let mut person = Person::new(1.0);
        let (outcome, seconds) = run(&mut person, &Behaviour::default(), |_| ());
        let Step::Done(calibration) = outcome else { panic!("the routine should succeed") };
        assert!((75.0..95.0).contains(&seconds), "took {seconds} s");
        assert!((5.0..90.0).contains(&calibration.error_px), "missed the places of the check by {} px", calibration.error_px);
        assert_eq!(calibration.area, SCREEN);
        // A place and a head position that were never shown.
        let posture = Posture { yaw: 0.06, place: [0.15, 1.2, 6.7], ..Posture::default() };
        let target = Vec2::new(1750.0, 380.0);
        let share = 1.0 / 6.0;
        let estimate = (0..6).fold(Vec2::ZERO, |sum, frame| {
            let mesh = person.look(100.0 + f64::from(frame) / 30.0, &posture, target, 1.0);
            let face = vision::measure(&calibration.reference, &mesh).unwrap();
            sum + calibration.gaze.predict(&face) * share
        });
        assert!(estimate.distance(target) < 90.0, "estimate {estimate:?} for {target:?}");
    }

    #[test]
    fn the_middle_is_shown_first_then_the_grid_row_by_row_back_and_forth_then_the_middle_again() {
        let mut targets: Vec<(Phase, (i32, i32))> = Vec::new();
        let mut progress = Vec::new();
        run(&mut Person::new(0.0), &Behaviour::default(), |view| {
            if targets.last() != Some(&(view.phase, view.target)) {
                targets.push((view.phase, view.target));
            }
            progress.push(view.progress);
            assert!((0.0..=1.0).contains(&view.dwell) && (0.0..=1.0).contains(&view.progress), "{view:?}");
        });
        // Five columns and five rows; the middle crossing is not shown a second time among them.
        let (columns, rows) = ([77, 678, 1280, 1881, 2482], [54, 297, 540, 782, 1025]);
        let mut expected = vec![(Phase::Rest, (1280, 540))];
        for (index, y) in rows.into_iter().enumerate() {
            let mut row: Vec<(i32, i32)> = columns.iter().map(|&x| (x, y)).filter(|&place| place != (1280, 540)).collect();
            if index % 2 == 1 {
                row.reverse();
            }
            expected.extend(row.into_iter().map(|place| (Phase::Places, place)));
        }
        assert_eq!(expected.len(), PLACES);
        expected.push((Phase::Sweep, (1280, 540)));
        // Then the check, at places the grid does not have, and the middle again under the score.
        let grid = expected.clone();
        let checks = [(972, 173), (2303, 928), (384, 669), (1587, 906), (256, 151), (2482, 421)]
            .into_iter()
            .chain([(972, 1025), (384, 421), (2303, 151), (1587, 421), (256, 928), (2047, 669)]);
        expected.extend(checks.map(|place| (Phase::Check, place)));
        assert!(expected[grid.len()..].iter().all(|check| grid.iter().all(|place| place.1 != check.1)));
        expected.push((Phase::Check, (1280, 540)));
        assert_eq!(targets, expected);
        assert!(progress.windows(2).all(|pair| pair[1] >= pair[0]), "progress never goes back");
        assert!(progress[0] < 0.01 && *progress.last().unwrap() > 0.95);
    }

    #[test]
    fn the_check_shows_where_the_model_put_each_place_and_its_score_becomes_the_accuracy() {
        /// A place of the check, the answer shown for it, and whether the page called it a hit.
        type Shown = ((i32, i32), (i32, i32), bool);
        let mut answers: Vec<Shown> = Vec::new();
        let mut scores = Vec::new();
        let (outcome, _) = run(&mut Person::new(1.0), &Behaviour::default(), |view| {
            assert!(view.phase == Phase::Check || (view.answer.is_none() && view.score.is_none()), "{view:?}");
            if let Some((answer, hit)) = view.answer {
                assert!(view.score.is_none() && view.dwell == 1.0, "{view:?}");
                if answers.last().map(|last| last.0) != Some(view.target) {
                    answers.push((view.target, answer, hit));
                }
            }
            scores.extend(view.score);
        });
        let Step::Done(calibration) = outcome else { panic!("the routine should succeed") };
        assert_eq!(answers.len(), CHECKS.len(), "one answer for every place of the check");
        // What the page showed is what was counted: a hit is an answer within 8 percent of the screen's height.
        let miss = |&(place, answer, _): &Shown| {
            Vec2::new(place.0 as f32, place.1 as f32).distance(Vec2::new(answer.0 as f32, answer.1 as f32))
        };
        assert!(answers.iter().all(|shown| shown.2 == (miss(shown) <= 0.08 * 1080.0)), "{answers:?}");
        let score = scores[0];
        assert!(scores.iter().all(|&one| one == score) && scores.len() > 100, "the score stays up for four seconds");
        assert_eq!(score.places, 12);
        assert_eq!(score.hits, answers.iter().filter(|shown| shown.2).count() as u32);
        assert!(score.hits >= 10, "{score:?}");
        let mean = answers.iter().map(miss).sum::<f32>() / 12.0;
        assert!((score.mean_px - mean).abs() < 1.0, "{score:?} against {mean}");
        assert_eq!(calibration.error_px, score.mean_px);
    }

    #[test]
    fn a_check_that_cannot_be_completed_still_gives_the_calibration() {
        // The eyes shut, or the person gone, once the model is made: what was checked until then counts.
        let fitted_at = {
            let mut began = None;
            let mut time = 0.0;
            run(&mut Person::new(1.0), &Behaviour::default(), |view| {
                time += 1.0 / 30.0;
                began = began.or((view.phase == Phase::Check).then_some(time));
            });
            began.unwrap()
        };
        let whole = calibrated();
        for behaviour in [
            Behaviour { shut_from: fitted_at + 5.0, ..Behaviour::default() },
            Behaviour { away: (fitted_at + 5.0, 900.0), ..Behaviour::default() },
        ] {
            let mut scores = 0;
            let (outcome, seconds) = run(&mut Person::new(1.0), &behaviour, |view| scores += u32::from(view.score.is_some()));
            let Step::Done(calibration) = outcome else { panic!("the model was made before the trouble began") };
            assert_eq!(calibration.gaze, whole.gaze, "the same model as from a whole run");
            assert!(calibration.error_px > 0.0 && calibration.error_px != whole.error_px);
            assert!(seconds < fitted_at + 5.0 + STAGE_LIMIT + 1.0, "ended at {seconds} s");
            assert_eq!(scores, 0, "no score is shown for a check that was broken off");
        }
        // Gone before any place of the check was measured: the model's own estimate is what there is.
        let (outcome, _) = run(&mut Person::new(1.0), &Behaviour { away: (fitted_at, 900.0), ..Behaviour::default() }, |_| ());
        let Step::Done(calibration) = outcome else { panic!("the model was made before the person left") };
        assert!(calibration.error_px > 0.0 && calibration.gaze == whole.gaze);
    }

    #[test]
    fn staring_at_the_middle_instead_of_the_places_is_refused() {
        let (failure, _) = failure(&Behaviour { attentive: false, ..Behaviour::default() });
        assert!(matches!(failure, Failure::Inaccurate { error_px } if error_px > 270.0), "{failure:?}");
    }

    #[test]
    fn a_head_that_stays_still_in_the_sweep_fails_at_the_stage_limit() {
        let (failure, seconds) = failure(&Behaviour { sweeps: false, ..Behaviour::default() });
        assert_eq!(failure, Failure::HeadNotMoved);
        assert!((65.0..75.0).contains(&seconds), "gave up after {seconds} s");
    }

    #[test]
    fn a_short_absence_is_forgiven_and_a_long_one_fails_after_five_seconds() {
        let (outcome, _) = run(&mut Person::new(1.0), &Behaviour { away: (6.0, 9.0), ..Behaviour::default() }, |_| ());
        assert!(matches!(outcome, Step::Done(_)));
        let (failure, seconds) = failure(&Behaviour { away: (6.0, 60.0), ..Behaviour::default() });
        assert_eq!(failure, Failure::FaceLost);
        assert!((10.9..11.2).contains(&seconds), "gave up at {seconds} s");
    }

    #[test]
    fn the_view_says_when_the_face_is_out_of_sight() {
        let mut seen = Vec::new();
        run(&mut Person::new(0.0), &Behaviour { away: (6.0, 8.0), ..Behaviour::default() }, |view| seen.push(view.face_visible));
        // Frame 0 is shown before any frame was fed; frames 180..240 were fed without a face.
        assert!(seen[1..181].iter().all(|&visible| visible) && seen[181..241].iter().all(|&visible| !visible) && seen[241]);
    }

    #[test]
    fn frames_recorded_before_the_eyes_had_arrived_or_during_a_blink_are_left_out_of_their_place() {
        use crate::vision::eyes::EyeSignals;
        use crate::vision::pose::HeadPose;
        let sample = |place: usize, iris_x: f32, lid: f32| Sample {
            face: FaceMeasurement {
                time: 0.0,
                pose: HeadPose { yaw: 0.0, pitch: 0.0, roll: 0.0, position: Vec2::new(640.0, 360.0), scale: 250.0 },
                eyes: EyeSignals { iris: Vec2::new(iris_x, -0.03), lid, openness: [0.3; 2] },
            },
            target: Vec2::ZERO,
            place: Some(place),
        };
        // Measured on a person: resting eyes scatter by about 0.004 of the eye's width, eyes
        // still on their way are 0.03 and more off, and a blink drags the lid by 0.03 to 0.1.
        let mut samples = Vec::new();
        for frame in 0..24 {
            let wobble = (frame % 5) as f32 * 0.002 - 0.004;
            // Place 1: the first five frames come from the place before it.
            samples.push(sample(1, if frame < 5 { 0.02 } else { -0.105 + wobble }, -0.13 + wobble * 0.5));
            // Place 2: three frames in the middle are half-way through a blink.
            samples.push(sample(2, 0.09 + wobble, if (10..13).contains(&frame) { -0.08 } else { -0.17 + wobble * 0.5 }));
            // Place 3: nothing wrong with it.
            samples.push(sample(3, wobble, -0.15 + wobble * 0.5));
        }
        let kept = settled(samples);
        let count = |place: usize| kept.iter().filter(|sample| sample.place == Some(place)).count();
        assert_eq!((count(1), count(2), count(3)), (19, 21, 24));
        assert!(kept.iter().all(|sample| sample.face.eyes.iris.x != 0.02 && sample.face.eyes.lid != -0.08));
    }

    #[test]
    fn blinks_are_left_out_and_eyes_kept_shut_fail_the_routine() {
        let (outcome, _) = run(&mut Person::new(1.0), &Behaviour { blink_every: Some(45), ..Behaviour::default() }, |_| ());
        let Step::Done(calibration) = outcome else { panic!("blinking is normal") };
        assert!(calibration.error_px < 90.0, "held-out error {} px", calibration.error_px);
        // Shut during the rest stage: there is no open eye to compare later frames with.
        assert_eq!(failure(&Behaviour { shut_from: 0.0, ..Behaviour::default() }).0, Failure::Unusable);
        // Shut while the second place is shown: nothing more is recorded until the stage limit.
        let (failure, seconds) = failure(&Behaviour { shut_from: 5.0, ..Behaviour::default() });
        assert_eq!(failure, Failure::TooSlow);
        assert!((28.0..31.0).contains(&seconds), "gave up at {seconds} s");
    }

    #[test]
    fn a_calibration_survives_being_written_and_read_back() {
        let calibration = calibrated();
        let text = calibration.to_text();
        assert_eq!(Calibration::parse(&text).unwrap(), calibration);
        assert_eq!(Calibration::parse(&text.replace('\n', "\r\n")).unwrap(), calibration);
    }

    #[test]
    fn damaged_or_foreign_files_are_refused_with_a_reason() {
        let text = calibrated().to_text();
        let reason = |text: &str| Calibration::parse(text).unwrap_err().to_string();
        assert!(reason("").contains("this version"));
        assert!(reason(&text.replace("calibration 2", "calibration 3")).contains("this version"));
        assert!(reason(&text.replace("error_px", "error")).contains("'error_px' is missing"));
        assert!(reason(&format!("{text}area = 0 0 1 1\n")).contains("'area' appears twice"));
        assert!(reason(&text.replace("weights_x = ", "weights_x = 1 ")).contains("needs 11 numbers but has 12"));
        assert!(reason(&text.replace("neutral = ", "neutral = x")).contains("not a number"));
        assert!(reason(&format!("{text}stray line\n")).contains("'name = numbers'"));
        let line = |name: &str| text.lines().find(|line| line.starts_with(name)).unwrap().to_string();
        let first_weight = line("weights_y").split_whitespace().nth(2).unwrap().to_string();
        let not_finite = line("weights_y").replacen(&first_weight, "NaN", 1);
        assert!(reason(&text.replace(&line("weights_y"), &not_finite)).contains("gaze model"));
        assert!(reason(&text.replace(&line("area"), "area = 0 0 0 1080")).contains("screen area"));
        assert!(reason(&text.replace(&line("camera"), "camera = 0 720")).contains("gaze model"));
        let flat = format!("reference = {}", vec!["0"; RIGID.len() * 3].join(" "));
        assert!(reason(&text.replace(&line("reference"), &flat)).contains("face shape"));
    }
}
