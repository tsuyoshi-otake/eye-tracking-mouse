//! Where on the screen the person is looking.
//!
//! The screen point is the eye's place in space plus the gaze direction times
//! the distance to the screen. Each part is close to linear in something the
//! camera measures: the head's place in the image, how far the head is turned,
//! and where the iris sits in the eye. So the estimate is a linear map from
//! those measurements to pixels, fitted to frames recorded while the person
//! looked at known places.
//!
//! Everything angular is multiplied by the distance relative to calibration
//! (from the size of the face in the image), so leaning in or back keeps the
//! estimate roughly right without needing calibration data at several distances.

use crate::geometry::{Vec2, solve_spd};
use crate::vision::FaceMeasurement;

/// Number of inputs to the linear map, including the constant.
pub const FEATURES: usize = 10;

/// Typical ranges of the measurements. Dividing by them makes every input of
/// order one, so the regularisation treats them alike.
const IRIS_UNIT: f64 = 0.1;
const TURN_UNIT: f64 = 0.25;
const SHIFT_UNIT: f64 = 1.0;
const DISTANCE_UNIT: f64 = 0.3;

/// Strength of the pull toward "this input has no effect", per sample. It
/// settles inputs that barely varied during calibration instead of letting
/// them fit noise.
const RIDGE: f64 = 2e-3;
/// The fit needs at least this many samples per input.
const SAMPLES_PER_FEATURE: usize = 4;
/// Share of samples, the worst-fitting ones, left out of the final fit (blinks, glances away).
const TRIM: f64 = 0.1;
/// Accuracy is only estimated from at least this many separate places.
const MIN_PLACES: usize = 4;

/// How the person sat during calibration, as the camera saw it. Head movement
/// is measured from here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neutral {
    /// Width and height of the camera image, in pixels.
    pub frame_size: (u32, u32),
    /// Head position in the image, in pixels.
    pub position: Vec2,
    /// Image pixels per eye span.
    pub scale: f32,
}

impl Neutral {
    /// Where a head seen at `position` with size `scale` is, sideways and
    /// vertically from the camera's axis, in eye spans. Dividing by the size
    /// undoes the camera's perspective, so moving straight toward the camera
    /// does not count as moving sideways.
    fn place(&self, position: Vec2, scale: f32) -> Vec2 {
        let axis = Vec2::new(self.frame_size.0 as f32, self.frame_size.1 as f32) * 0.5;
        (position - axis) * (1.0 / scale)
    }
}

/// One frame recorded while the person looked at a known place.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub face: FaceMeasurement,
    /// Desktop pixels.
    pub target: Vec2,
    /// The calibration place this frame belongs to, for estimating accuracy;
    /// `None` for frames that should take part in every fit.
    pub place: Option<usize>,
}

fn features(face: &FaceMeasurement, neutral: &Neutral) -> [f64; FEATURES] {
    let pose = &face.pose;
    // Above 1 when the head is farther away than it was at calibration.
    let distance = f64::from(neutral.scale / pose.scale);
    // The eye signals are measured along the eye line; turn them into the camera's axes.
    let (sin, cos) = pose.roll.sin_cos();
    let iris = face.eyes.iris;
    let iris = Vec2::new(iris.x * cos - iris.y * sin, iris.x * sin + iris.y * cos);
    let forward = pose.forward();
    let shift = neutral.place(pose.position, pose.scale) - neutral.place(neutral.position, neutral.scale);
    [
        1.0,
        distance * f64::from(iris.x) / IRIS_UNIT,
        distance * f64::from(iris.y) / IRIS_UNIT,
        distance * f64::from(face.eyes.lid) / IRIS_UNIT,
        distance * f64::from(forward.x) / TURN_UNIT,
        distance * f64::from(forward.y) / TURN_UNIT,
        f64::from(pose.roll) / TURN_UNIT,
        f64::from(shift.x) / SHIFT_UNIT,
        f64::from(shift.y) / SHIFT_UNIT,
        (distance - 1.0) / DISTANCE_UNIT,
    ]
}

type Weights = [[f64; FEATURES]; 2];

fn apply(weights: &Weights, row: &[f64; FEATURES]) -> Vec2 {
    let dot = |w: &[f64; FEATURES]| w.iter().zip(row).map(|(w, f)| w * f).sum::<f64>() as f32;
    Vec2::new(dot(&weights[0]), dot(&weights[1]))
}

/// Ridge regression of the targets on the rows; the constant is not penalised.
fn solve(rows: &[([f64; FEATURES], Vec2)]) -> Option<Weights> {
    let mut normal = [0.0f64; FEATURES * FEATURES];
    let mut rhs = [[0.0f64; FEATURES]; 2];
    for (row, target) in rows {
        for i in 0..FEATURES {
            for j in 0..FEATURES {
                normal[i * FEATURES + j] += row[i] * row[j];
            }
            rhs[0][i] += row[i] * f64::from(target.x);
            rhs[1][i] += row[i] * f64::from(target.y);
        }
    }
    for i in 1..FEATURES {
        normal[i * FEATURES + i] += RIDGE * rows.len() as f64;
    }
    let mut weights = [[0.0f64; FEATURES]; 2];
    for (axis, rhs) in rhs.iter().enumerate() {
        weights[axis].copy_from_slice(&solve_spd(&normal, rhs, FEATURES)?);
    }
    Some(weights)
}

/// Fits, drops the samples that fit worst, and fits again.
fn fit_rows(mut rows: Vec<([f64; FEATURES], Vec2)>) -> Option<Weights> {
    if rows.len() < FEATURES * SAMPLES_PER_FEATURE || rows.iter().any(|(row, target)| !(row.iter().all(|f| f.is_finite()) && target.is_finite())) {
        return None;
    }
    let first = solve(&rows)?;
    let keep = rows.len() - (rows.len() as f64 * TRIM) as usize;
    let miss = |entry: &([f64; FEATURES], Vec2)| apply(&first, &entry.0).distance(entry.1);
    rows.sort_by(|a, b| miss(a).total_cmp(&miss(b)));
    rows.truncate(keep);
    solve(&rows)
}

#[derive(Clone, Debug, PartialEq)]
pub struct GazeModel {
    neutral: Neutral,
    /// For screen x and screen y: pixels per unit of each input.
    weights: Weights,
}

impl GazeModel {
    /// `None` when the samples cannot support a fit.
    pub fn fit(neutral: Neutral, samples: &[Sample]) -> Option<GazeModel> {
        let rows = samples.iter().map(|s| (features(&s.face, &neutral), s.target)).collect();
        GazeModel::from_parts(neutral, fit_rows(rows)?)
    }

    /// Rebuilds a stored model; `None` if the numbers cannot be one.
    pub fn from_parts(neutral: Neutral, weights: Weights) -> Option<GazeModel> {
        let usable = neutral.frame_size.0 > 0
            && neutral.frame_size.1 > 0
            && neutral.position.is_finite()
            && neutral.scale.is_finite()
            && neutral.scale > 0.0
            && weights.iter().flatten().all(|w| w.is_finite());
        usable.then_some(GazeModel { neutral, weights })
    }

    pub fn neutral(&self) -> Neutral {
        self.neutral
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// The estimated gaze point in desktop pixels. Not confined to any screen.
    pub fn predict(&self, face: &FaceMeasurement) -> Vec2 {
        apply(&self.weights, &features(face, &self.neutral))
    }
}

/// How far off the model is at a place it was not fitted to, in pixels: each
/// place in turn is left out of the fit and predicted from the others (root
/// mean square over the places). `None` when there are too few places or a fit fails.
pub fn held_out_error(neutral: Neutral, samples: &[Sample]) -> Option<f32> {
    let mut places: Vec<usize> = samples.iter().filter_map(|s| s.place).collect();
    places.sort_unstable();
    places.dedup();
    if places.len() < MIN_PLACES {
        return None;
    }
    let mut squared = 0.0f32;
    for &place in &places {
        let (inside, outside): (Vec<&Sample>, Vec<&Sample>) = samples.iter().partition(|s| s.place == Some(place));
        let weights = fit_rows(outside.iter().map(|s| (features(&s.face, &neutral), s.target)).collect())?;
        let share = 1.0 / inside.len() as f32;
        let mean = |pick: &dyn Fn(&Sample) -> Vec2| inside.iter().fold(Vec2::ZERO, |sum, s| sum + pick(s) * share);
        let miss = mean(&|s| apply(&weights, &features(&s.face, &neutral))).distance(mean(&|s| s.target));
        squared += miss * miss / places.len() as f32;
    }
    Some(squared.sqrt())
}

#[cfg(test)]
pub(crate) mod simulation {
    //! A person at a desk for tests: a synthetic head in front of a camera
    //! that sits on top of a screen, looking at chosen pixels.

    use crate::geometry::{RectI, Vec2};
    use crate::vision::landmarks::{FaceMesh, Point3};
    use crate::vision::synthetic::{SyntheticHead, View};

    /// A 2560x1080 screen, 80 cm wide, with the camera on its top edge.
    pub const SCREEN: RectI = RectI::new(0, 0, 2560, 1080);
    pub const FRAME_SIZE: (u32, u32) = (1280, 720);
    /// Eye spans (9 cm) per screen pixel.
    const SPANS_PER_PIXEL: f32 = 80.0 / 2560.0 / 9.0;
    /// Camera focal length in image pixels (a 48 degree wide webcam at 1280 pixels).
    const FOCAL: f32 = 1440.0;

    /// Where the head is and how it is turned.
    #[derive(Clone, Copy, Debug)]
    pub struct Posture {
        /// Head position in front of the camera, in eye spans: x toward the
        /// image's right, y down, z away from the camera.
        pub place: [f32; 3],
        pub yaw: f32,
        pub pitch: f32,
        pub roll: f32,
    }

    impl Default for Posture {
        fn default() -> Self {
            // 60 cm from the screen, eyes a little below the camera.
            Posture { place: [0.0, 1.2, 6.7], yaw: 0.0, pitch: 0.0, roll: 0.0 }
        }
    }

    pub struct Person {
        head: SyntheticHead,
        noise: u32,
        /// Landmark noise amplitude in image pixels.
        pub jitter: f32,
    }

    impl Person {
        pub fn new(jitter: f32) -> Person {
            Person { head: SyntheticHead::new(), noise: 2024, jitter }
        }

        /// The mesh the tracker would report while the person looks at desktop pixel `target`.
        pub fn look(&mut self, time: f64, posture: &Posture, target: Vec2, openness: f32) -> FaceMesh {
            let [x, y, z] = posture.place;
            let view = View {
                yaw: posture.yaw,
                pitch: posture.pitch,
                roll: posture.roll,
                scale: FOCAL / z,
                position: Vec2::new(FRAME_SIZE.0 as f32 * 0.5 + FOCAL * x / z, FRAME_SIZE.1 as f32 * 0.5 + FOCAL * y / z),
            };
            // The screen faces the person, so its right-hand side is toward the image's left.
            let middle = (SCREEN.left + SCREEN.right) as f32 * 0.5;
            let on_screen = [-(target.x - middle) * SPANS_PER_PIXEL, (target.y - SCREEN.top as f32) * SPANS_PER_PIXEL, 0.0];
            let mut points = self.head.render(&view, Some([on_screen[0] - x, on_screen[1] - y, -z]), openness);
            for point in &mut points {
                self.shake(point);
            }
            FaceMesh { points, presence: 1.0, time }
        }

        fn shake(&mut self, point: &mut Point3) {
            for value in point.iter_mut() {
                self.noise = self.noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *value += ((self.noise >> 8) as f32 / 8_388_608.0 - 1.0) * self.jitter;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::simulation::{FRAME_SIZE, Person, Posture};
    use super::*;
    use crate::vision;
    use crate::vision::pose::{ReferenceBuilder, ReferenceShape};

    const CENTRE: Vec2 = Vec2::new(1280.0, 540.0);

    struct Bench {
        person: Person,
        reference: ReferenceShape,
        neutral: Neutral,
        frame: u32,
    }

    impl Bench {
        fn new(jitter: f32) -> Bench {
            let mut person = Person::new(0.0);
            let mut builder = ReferenceBuilder::default();
            let resting = person.look(0.0, &Posture::default(), CENTRE, 1.0);
            builder.add(&resting.points);
            let reference = builder.build().unwrap();
            let pose = vision::measure(&reference, &resting).unwrap().pose;
            person.jitter = jitter;
            Bench { person, reference, neutral: Neutral { frame_size: FRAME_SIZE, position: pose.position, scale: pose.scale }, frame: 0 }
        }

        fn measure(&mut self, posture: &Posture, target: Vec2) -> FaceMeasurement {
            self.frame += 1;
            let mesh = self.person.look(f64::from(self.frame) / 30.0, posture, target, 1.0);
            vision::measure(&self.reference, &mesh).unwrap()
        }

        /// Nine places looked at with the head following a little, then the
        /// middle looked at while the head turns and shifts.
        fn calibration_samples(&mut self) -> Vec<Sample> {
            let mut samples = Vec::new();
            for place in 0..9 {
                let target = Vec2::new(205.0 + (place % 3) as f32 * 1075.0, 86.0 + (place / 3) as f32 * 454.0);
                let follow = Posture {
                    yaw: -(target.x - CENTRE.x) * 6e-5,
                    pitch: (target.y - CENTRE.y) * 6e-5,
                    ..Posture::default()
                };
                for _ in 0..24 {
                    samples.push(Sample { face: self.measure(&follow, target), target, place: Some(place) });
                }
            }
            for step in 0..150 {
                let angle = step as f32 * 0.12;
                let posture = Posture {
                    place: [0.25 * angle.cos(), 1.2 + 0.15 * (angle * 0.7).sin(), 6.7],
                    yaw: 0.14 * angle.sin(),
                    pitch: 0.08 * (angle * 1.3).cos(),
                    roll: 0.05 * (angle * 0.5).sin(),
                };
                samples.push(Sample { face: self.measure(&posture, CENTRE), target: CENTRE, place: None });
            }
            samples
        }

        /// Mean miss, in pixels, over places and postures that were not calibrated.
        fn mean_miss(&mut self, model: &GazeModel, postures: &[Posture]) -> f32 {
            let mut total = 0.0;
            let mut count = 0;
            for posture in postures {
                for target in [Vec2::new(500.0, 250.0), Vec2::new(1900.0, 300.0), Vec2::new(1400.0, 900.0), Vec2::new(300.0, 800.0), Vec2::new(2300.0, 700.0)] {
                    let share = 1.0 / 6.0;
                    let estimate = (0..6).fold(Vec2::ZERO, |sum, _| sum + model.predict(&self.measure(posture, target)) * share);
                    total += estimate.distance(target);
                    count += 1;
                }
            }
            total / count as f32
        }
    }

    #[test]
    fn a_calibrated_model_finds_places_it_was_not_shown() {
        let mut bench = Bench::new(0.0);
        let samples = bench.calibration_samples();
        let model = GazeModel::fit(bench.neutral, &samples).unwrap();
        let miss = bench.mean_miss(&model, &[Posture::default()]);
        // What is left is the straight-line model against the screen's flat geometry: it
        // overshoots between the middle and the edges and falls short in the corners.
        assert!(miss < 50.0, "mean miss {miss} px on a 2560 px wide screen");
    }

    #[test]
    fn turning_and_shifting_the_head_while_looking_at_one_place_barely_moves_the_estimate() {
        let mut bench = Bench::new(0.0);
        let samples = bench.calibration_samples();
        let model = GazeModel::fit(bench.neutral, &samples).unwrap();
        let moved = [
            Posture { yaw: 0.1, ..Posture::default() },
            Posture { pitch: -0.06, yaw: -0.08, ..Posture::default() },
            Posture { place: [0.3, 1.1, 6.7], ..Posture::default() },
            Posture { roll: 0.12, ..Posture::default() },
        ];
        let miss = bench.mean_miss(&model, &moved);
        assert!(miss < 55.0, "mean miss {miss} px with the head moved");
        // Without the head terms the same head turn would read as a gaze shift of several hundred pixels.
        let turned = model.predict(&bench.measure(&Posture { yaw: 0.1, ..Posture::default() }, CENTRE));
        let straight = model.predict(&bench.measure(&Posture::default(), CENTRE));
        assert!(turned.distance(straight) < 30.0, "{turned:?} vs {straight:?}");
    }

    #[test]
    fn leaning_toward_the_screen_is_compensated_by_the_face_size() {
        let mut bench = Bench::new(0.0);
        let samples = bench.calibration_samples();
        let model = GazeModel::fit(bench.neutral, &samples).unwrap();
        let miss = bench.mean_miss(&model, &[Posture { place: [0.0, 1.2, 5.6], ..Posture::default() }]);
        assert!(miss < 85.0, "mean miss {miss} px from 50 cm after calibrating at 60 cm");
    }

    #[test]
    fn landmark_noise_costs_tens_of_pixels_not_hundreds() {
        let mut bench = Bench::new(1.0);
        let samples = bench.calibration_samples();
        let model = GazeModel::fit(bench.neutral, &samples).unwrap();
        let miss = bench.mean_miss(&model, &[Posture::default(), Posture { yaw: 0.08, ..Posture::default() }]);
        assert!(miss < 60.0, "mean miss {miss} px with one pixel of landmark noise");
        let error = held_out_error(bench.neutral, &samples).unwrap();
        assert!((20.0..80.0).contains(&error), "held-out error {error} px");
    }

    #[test]
    fn the_held_out_error_exposes_a_calibration_where_the_person_did_not_look() {
        let mut bench = Bench::new(0.0);
        let mut samples = bench.calibration_samples();
        let honest = held_out_error(bench.neutral, &samples).unwrap();
        // The person stared at the middle the whole time while the targets moved.
        for sample in &mut samples {
            sample.face = bench.measure(&Posture::default(), CENTRE);
        }
        let careless = held_out_error(bench.neutral, &samples).unwrap();
        assert!(honest < 60.0 && careless > 400.0, "honest {honest} careless {careless}");
    }

    #[test]
    fn a_few_wild_samples_do_not_bend_the_fit() {
        let mut bench = Bench::new(0.0);
        let mut samples = bench.calibration_samples();
        let clean = GazeModel::fit(bench.neutral, &samples).unwrap();
        // Five percent of the frames were recorded while the person glanced at a corner.
        for sample in samples.iter_mut().step_by(20) {
            sample.face = bench.measure(&Posture::default(), Vec2::new(2500.0, 1000.0));
        }
        let spoiled = GazeModel::fit(bench.neutral, &samples).unwrap();
        let (clean_miss, spoiled_miss) = (bench.mean_miss(&clean, &[Posture::default()]), bench.mean_miss(&spoiled, &[Posture::default()]));
        assert!(spoiled_miss < clean_miss + 15.0, "clean {clean_miss} spoiled {spoiled_miss}");
    }

    #[test]
    fn too_little_or_broken_data_gives_no_model() {
        let mut bench = Bench::new(0.0);
        let samples = bench.calibration_samples();
        assert!(GazeModel::fit(bench.neutral, &samples[..30]).is_none());
        assert!(held_out_error(bench.neutral, &samples[..60]).is_none(), "only three places");
        let mut broken = samples.clone();
        broken[5].target = Vec2::new(f32::NAN, 0.0);
        assert!(GazeModel::fit(bench.neutral, &broken).is_none());
        let model = GazeModel::fit(bench.neutral, &samples).unwrap();
        assert_eq!(GazeModel::from_parts(model.neutral(), *model.weights()), Some(model.clone()));
        assert!(GazeModel::from_parts(Neutral { scale: 0.0, ..model.neutral() }, *model.weights()).is_none());
        assert!(GazeModel::from_parts(Neutral { frame_size: (0, 720), ..model.neutral() }, *model.weights()).is_none());
        let mut weights = *model.weights();
        weights[1][3] = f64::INFINITY;
        assert!(GazeModel::from_parts(model.neutral(), weights).is_none());
    }
}
