//! Connects the face tracker to the Windows services that feed it.

use crate::error::Result;
use crate::image::BgraFrame;
use crate::platform::face_detector::FaceDetector;
use crate::platform::inference::Network;
use crate::vision::landmarks::{INPUT_NAME, INPUT_SHAPE, MODEL_ONNX, OUTPUT_FLAG, OUTPUT_LANDMARKS};
use crate::vision::tracker::{FaceBox, FaceLocator, FaceTracker, LandmarkModel};

pub struct WinMlLandmarks(Network);

impl WinMlLandmarks {
    pub fn load() -> Result<WinMlLandmarks> {
        Network::load(MODEL_ONNX, (INPUT_NAME, &INPUT_SHAPE), &[OUTPUT_LANDMARKS, OUTPUT_FLAG]).map(WinMlLandmarks)
    }
}

impl LandmarkModel for WinMlLandmarks {
    fn infer(&mut self, image: &[f32], landmarks: &mut [f32]) -> Result<f32> {
        let mut flag = [0.0f32];
        self.0.run(image, &mut [landmarks, &mut flag])?;
        Ok(flag[0])
    }
}

pub struct OsFaceLocator(FaceDetector);

impl FaceLocator for OsFaceLocator {
    fn locate(&mut self, frame: &BgraFrame) -> Result<Option<FaceBox>> {
        let found = self.0.largest_face(frame)?;
        Ok(found.map(|r| FaceBox { left: r.left, top: r.top, width: r.width, height: r.height }))
    }
}

pub type Tracker = FaceTracker<WinMlLandmarks, OsFaceLocator>;

pub fn new_tracker() -> Result<Tracker> {
    Ok(FaceTracker::new(WinMlLandmarks::load()?, OsFaceLocator(FaceDetector::new()?)))
}
