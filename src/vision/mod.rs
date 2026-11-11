//! Turns camera frames into measurements of one face. Pure computation: the
//! landmark network and the face detector are supplied by the caller.

pub mod eyes;
pub mod landmarks;
pub mod pose;
pub mod roi;
#[cfg(test)]
pub(crate) mod synthetic;
pub mod tracker;

use eyes::EyeSignals;
use landmarks::FaceMesh;
use pose::{HeadPose, ReferenceShape};

/// Everything the pointer logic needs to know about the face in one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceMeasurement {
    /// Seconds, on the camera's clock.
    pub time: f64,
    pub pose: HeadPose,
    pub eyes: EyeSignals,
}

/// `None` when the mesh is too distorted to measure.
pub fn measure(reference: &ReferenceShape, mesh: &FaceMesh) -> Option<FaceMeasurement> {
    Some(FaceMeasurement {
        time: mesh.time,
        pose: pose::estimate(reference, &mesh.points)?,
        eyes: eyes::measure(&mesh.points)?,
    })
}
