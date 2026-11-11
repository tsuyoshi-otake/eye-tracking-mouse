//! `--diag <seconds>`: prints what the tracker measures, once per second.
//!
//! Numbers only; no image leaves the process. Sit as you would while working
//! and hold still for the first second so the reference face shape is clean.

use std::time::Instant;

use crate::error::Result;
use crate::image::BgraFrame;
use crate::platform::camera::{Camera, CameraRequest};
use crate::platform::face_detector::FaceDetector;
use crate::vision;
use crate::vision::landmarks::FaceMesh;
use crate::vision::pose::{ReferenceBuilder, ReferenceShape};

use super::face;

/// Frames averaged into the reference face shape.
const REFERENCE_FRAMES: usize = 20;

#[derive(Default)]
struct Second {
    frames: u32,
    faces: u32,
    track_seconds: f64,
    last: Option<(f32, vision::FaceMeasurement)>,
}

pub fn run(request: &CameraRequest, seconds: u32) -> Result<()> {
    let mut camera = Camera::open(request)?;
    println!("camera: {}", camera.description());
    let mut tracker = face::new_tracker()?;
    // Asked directly when nothing is tracked, to tell "nobody in view" from "a face the model rejects".
    let mut detector = FaceDetector::new()?;
    let mut frame = BgraFrame::default();
    let mut mesh = FaceMesh::default();
    let mut builder = ReferenceBuilder::default();
    let mut reference: Option<ReferenceShape> = None;
    let mut second = Second::default();
    let mut total_faces = 0u32;
    let started = Instant::now();
    let mut next_report = 1.0;
    while started.elapsed().as_secs_f64() < f64::from(seconds) {
        if !camera.read(&mut frame)? {
            continue;
        }
        second.frames += 1;
        let before = Instant::now();
        let found = tracker.track(&frame, &mut mesh)?;
        second.track_seconds += before.elapsed().as_secs_f64();
        if found {
            second.faces += 1;
            total_faces += 1;
            match &reference {
                Some(shape) => second.last = vision::measure(shape, &mesh).map(|m| (mesh.presence, m)).or(second.last),
                None => {
                    builder.add(&mesh.points);
                    if builder.frames() >= REFERENCE_FRAMES {
                        reference = builder.build();
                    }
                }
            }
        }
        if started.elapsed().as_secs_f64() >= next_report {
            report(next_report, &second);
            if second.faces == 0 {
                let mut gray = Vec::new();
                frame.to_gray(&mut gray);
                let brightness = gray.iter().map(|&v| f64::from(v)).sum::<f64>() / gray.len().max(1) as f64;
                match detector.largest_face(&frame)? {
                    Some(r) => println!(
                        "      detector sees a face at ({:.0}, {:.0}) size {:.0}x{:.0}; mean brightness {brightness:.0}",
                        r.left, r.top, r.width, r.height
                    ),
                    None => println!("      detector sees no face; mean brightness {brightness:.0} of 255"),
                }
            }
            second = Second::default();
            next_report += 1.0;
        }
    }
    println!("faces tracked in total: {total_faces} frames");
    Ok(())
}

fn report(at: f64, second: &Second) {
    let per_frame = if second.frames > 0 { second.track_seconds / f64::from(second.frames) * 1000.0 } else { 0.0 };
    print!("t={at:>2.0}s frames={:>2} faces={:>2} track={per_frame:.1}ms", second.frames, second.faces);
    match &second.last {
        Some((presence, m)) => println!(
            " presence={presence:.2} yaw={:+.1} pitch={:+.1} roll={:+.1} deg span={:.0}px iris=({:+.3},{:+.3}) lid={:+.3} open=({:.2},{:.2})",
            m.pose.yaw.to_degrees(),
            m.pose.pitch.to_degrees(),
            m.pose.roll.to_degrees(),
            m.pose.scale,
            m.eyes.iris.x,
            m.eyes.iris.y,
            m.eyes.lid,
            m.eyes.openness[0],
            m.eyes.openness[1],
        ),
        None => println!(" (no measurement yet)"),
    }
}
