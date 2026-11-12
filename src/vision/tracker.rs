//! Follows one face from frame to frame.
//!
//! Contract: `track` returns `true` and fills the mesh when the frame shows a
//! face. While a face is tracked, each frame's crop comes from the previous
//! frame's landmarks and the (slow) face locator is not called. When the
//! model stops seeing a face the tracker searches again, at most once per
//! `SEARCH_INTERVAL`, so an empty chair does not keep the detector busy.

use crate::error::Result;
use crate::image::BgraFrame;

use super::landmarks::{FaceMesh, INPUT_LEN, LANDMARK_COUNT, PRESENCE_THRESHOLD};
use super::roi::Roi;

/// Seconds between searches while no face is in view.
const SEARCH_INTERVAL: f64 = 0.2;

/// An upright box around a face, in image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

/// Finds the most prominent face in a whole frame.
pub trait FaceLocator {
    fn locate(&mut self, frame: &BgraFrame) -> Result<Option<FaceBox>>;
}

/// The landmark network: `image` is a crop as `Roi::extract` produces it,
/// `landmarks` receives x, y, z per point in crop pixels. Returns the
/// face-presence logit.
pub trait LandmarkModel {
    fn infer(&mut self, image: &[f32], landmarks: &mut [f32]) -> Result<f32>;
}

pub struct FaceTracker<M, L> {
    model: M,
    locator: L,
    roi: Option<Roi>,
    next_search: f64,
    image: Vec<f32>,
    raw: Vec<f32>,
}

impl<M: LandmarkModel, L: FaceLocator> FaceTracker<M, L> {
    pub fn new(model: M, locator: L) -> Self {
        FaceTracker {
            model,
            locator,
            roi: None,
            next_search: f64::NEG_INFINITY,
            image: vec![0.0; INPUT_LEN],
            raw: vec![0.0; LANDMARK_COUNT * 3],
        }
    }

    pub fn track(&mut self, frame: &BgraFrame, mesh: &mut FaceMesh) -> Result<bool> {
        let tracked = match self.roi {
            Some(roi) => self.read(frame, roi, mesh)?,
            None => false,
        };
        if tracked || self.acquire(frame, mesh)? {
            return Ok(true);
        }
        self.roi = None;
        Ok(false)
    }

    /// Searches the whole frame, if a search is due.
    fn acquire(&mut self, frame: &BgraFrame, mesh: &mut FaceMesh) -> Result<bool> {
        if frame.timestamp < self.next_search {
            return Ok(false);
        }
        self.next_search = frame.timestamp + SEARCH_INTERVAL;
        let Some(found) = self.locator.locate(frame)? else {
            return Ok(false);
        };
        let first = Roi::from_box(found.left, found.top, found.width, found.height);
        if !first.is_usable(frame.width, frame.height) || !self.read(frame, first, mesh)? {
            return Ok(false);
        }
        // The detector's box is only approximate. Read once more from the
        // landmark-based region so the first result is as good as later ones.
        match self.roi {
            Some(refined) => self.read(frame, refined, mesh),
            None => Ok(false),
        }
    }

    /// Runs the model on one region. On success the mesh is filled and the
    /// region for the next frame is stored.
    fn read(&mut self, frame: &BgraFrame, roi: Roi, mesh: &mut FaceMesh) -> Result<bool> {
        self.roi = None;
        roi.extract(frame, &mut self.image);
        let logit = self.model.infer(&self.image, &mut self.raw)?;
        let presence = 1.0 / (1.0 + (-logit).exp());
        if presence.is_nan() || presence < PRESENCE_THRESHOLD {
            return Ok(false);
        }
        roi.project(&self.raw, &mut mesh.points);
        mesh.presence = presence;
        mesh.time = frame.timestamp;
        let next = Roi::from_mesh(&mesh.points);
        if !next.is_usable(frame.width, frame.height) {
            return Ok(false);
        }
        self.roi = Some(next);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use crate::vision::landmarks::EYE_SPAN;

    /// Reports the same crop-space face every time, with a switchable presence logit.
    struct FakeModel {
        logit: Rc<Cell<f32>>,
        runs: Rc<Cell<u32>>,
    }

    impl LandmarkModel for FakeModel {
        fn infer(&mut self, _image: &[f32], landmarks: &mut [f32]) -> Result<f32> {
            self.runs.set(self.runs.get() + 1);
            // A face centred in the crop and filling its middle two thirds, eyes
            // level: exactly what a correctly placed region shows, so tracking
            // this answer leaves the region where it is.
            let pitch = 256.0 / 1.5 / 21.0;
            for (index, point) in landmarks.chunks_exact_mut(3).enumerate() {
                let (column, row) = ((index % 22) as f32, (index / 22) as f32);
                point.copy_from_slice(&[128.0 + (column - 10.5) * pitch, 128.0 + (row - 10.5) * pitch, 0.0]);
            }
            landmarks[EYE_SPAN.0 * 3..EYE_SPAN.0 * 3 + 2].copy_from_slice(&[90.0, 110.0]);
            landmarks[EYE_SPAN.1 * 3..EYE_SPAN.1 * 3 + 2].copy_from_slice(&[166.0, 110.0]);
            Ok(self.logit.get())
        }
    }

    struct FakeLocator {
        answer: Rc<Cell<Option<FaceBox>>>,
        calls: Rc<Cell<u32>>,
    }

    impl FaceLocator for FakeLocator {
        fn locate(&mut self, _frame: &BgraFrame) -> Result<Option<FaceBox>> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.answer.get())
        }
    }

    struct Rig {
        tracker: FaceTracker<FakeModel, FakeLocator>,
        logit: Rc<Cell<f32>>,
        runs: Rc<Cell<u32>>,
        answer: Rc<Cell<Option<FaceBox>>>,
        calls: Rc<Cell<u32>>,
        frame: BgraFrame,
        mesh: FaceMesh,
    }

    impl Rig {
        fn new() -> Rig {
            let logit = Rc::new(Cell::new(5.0));
            let runs = Rc::new(Cell::new(0));
            let answer = Rc::new(Cell::new(Some(FaceBox { left: 540.0, top: 260.0, width: 200.0, height: 200.0 })));
            let calls = Rc::new(Cell::new(0));
            let tracker = FaceTracker::new(
                FakeModel { logit: logit.clone(), runs: runs.clone() },
                FakeLocator { answer: answer.clone(), calls: calls.clone() },
            );
            Rig { tracker, logit, runs, answer, calls, frame: BgraFrame::new(1280, 720), mesh: FaceMesh::default() }
        }

        fn step(&mut self, time: f64) -> bool {
            self.frame.timestamp = time;
            self.tracker.track(&self.frame, &mut self.mesh).unwrap()
        }
    }

    #[test]
    fn it_finds_a_face_with_the_locator_and_then_tracks_without_it() {
        let mut rig = Rig::new();
        assert!(rig.step(0.0));
        assert_eq!((rig.calls.get(), rig.runs.get()), (1, 2), "one search, one first read and one refinement");
        for frame in 1..=5 {
            assert!(rig.step(f64::from(frame) / 30.0));
        }
        assert_eq!((rig.calls.get(), rig.runs.get()), (1, 7));
        assert_eq!(rig.mesh.points.len(), LANDMARK_COUNT);
        assert!(rig.mesh.presence > 0.99);
        assert!((rig.mesh.time - 5.0 / 30.0).abs() < 1e-9);
    }

    #[test]
    fn landmarks_come_back_in_image_pixels() {
        let mut rig = Rig::new();
        assert!(rig.step(0.0));
        // The fake face is centred in the crop, so tracking settles on the box centre.
        for frame in 1..=20 {
            assert!(rig.step(f64::from(frame) / 30.0));
        }
        let (low, high) = rig.mesh.points.iter().fold((f32::MAX, f32::MIN), |(low, high), p| (low.min(p[0]), high.max(p[0])));
        assert!(((low + high) * 0.5 - 640.0).abs() < 2.0, "centre x {}", (low + high) * 0.5);
        assert!(low > 400.0 && high < 880.0);
    }

    #[test]
    fn losing_the_face_starts_a_new_search_which_is_rate_limited() {
        let mut rig = Rig::new();
        assert!(rig.step(0.0));
        rig.logit.set(-5.0);
        rig.answer.set(None);
        // The frame that loses the face may search at once only if a search is due.
        assert!(!rig.step(0.05));
        assert_eq!(rig.calls.get(), 1, "the last search was 0.05 s ago");
        for frame in 0..30 {
            assert!(!rig.step(0.1 + f64::from(frame) / 30.0));
        }
        // One second without a face: a search every 0.2 s, not one per frame.
        assert!((4..=6).contains(&(rig.calls.get() - 1)), "{} searches", rig.calls.get() - 1);
    }

    #[test]
    fn a_face_that_returns_is_picked_up_again() {
        let mut rig = Rig::new();
        assert!(rig.step(0.0));
        rig.logit.set(-5.0);
        assert!(!rig.step(1.0));
        rig.logit.set(5.0);
        assert!(rig.step(2.0));
        assert!(rig.calls.get() >= 2);
    }

    #[test]
    fn a_located_box_that_the_model_rejects_is_not_a_face() {
        let mut rig = Rig::new();
        rig.logit.set(-3.0);
        assert!(!rig.step(0.0));
        assert_eq!((rig.calls.get(), rig.runs.get()), (1, 1));
    }
}
