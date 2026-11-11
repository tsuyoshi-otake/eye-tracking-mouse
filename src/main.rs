// A tray application: no console window when started from Explorer.
#![windows_subsystem = "windows"]

use std::process::ExitCode;

use eye_tracking_mouse::app::{diag, selftest};
use eye_tracking_mouse::error::Error;
use eye_tracking_mouse::platform::camera::{Camera, CameraRequest};
use eye_tracking_mouse::platform::runtime;

const USAGE: &str = "\
eye-tracking-mouse [options]

  (no options)          run in the notification area
  --camera <index>      use this camera (see --list-cameras), default 0
  --list-cameras        print the attached cameras and exit
  --selftest <name>     run one check and exit: camera, face, model, pointer
  --image <file.bmp>    with --selftest face: use this picture instead of the camera
  --diag <seconds>      print what the face tracker measures, once per second
  --help                print this text
";

enum Mode {
    Run,
    ListCameras,
    SelfTest(String),
    Diag(u32),
    Help,
}

struct Options {
    mode: Mode,
    camera: CameraRequest,
    image: Option<String>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Options, Error> {
    let mut options = Options { mode: Mode::Run, camera: CameraRequest::default(), image: None };
    let mut args = args;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| Error::new(format!("{name} needs a value")));
        match arg.as_str() {
            "--camera" => {
                let text = value("--camera")?;
                options.camera.device_index =
                    text.parse().map_err(|_| Error::new(format!("'{text}' is not a camera index")))?;
            }
            "--list-cameras" => options.mode = Mode::ListCameras,
            "--selftest" => options.mode = Mode::SelfTest(value("--selftest")?),
            "--image" => options.image = Some(value("--image")?),
            "--diag" => {
                let text = value("--diag")?;
                let seconds = text.parse().map_err(|_| Error::new(format!("'{text}' is not a number of seconds")))?;
                options.mode = Mode::Diag(seconds);
            }
            "--help" | "-h" | "/?" => options.mode = Mode::Help,
            other => return Err(Error::new(format!("unknown option '{other}'"))),
        }
    }
    Ok(options)
}

fn run(options: &Options) -> Result<bool, Error> {
    runtime::enable_per_monitor_dpi();
    let _com = runtime::ComApartment::enter()?;
    match &options.mode {
        Mode::Help => print!("{USAGE}"),
        Mode::ListCameras => {
            for (index, name) in Camera::device_names()?.iter().enumerate() {
                println!("{index}: {name}");
            }
        }
        Mode::SelfTest(name) => return selftest::run(name, &options.camera, options.image.as_deref()),
        Mode::Diag(seconds) => diag::run(&options.camera, *seconds)?,
        Mode::Run => return Err(Error::new("the tray application is not built yet")),
    }
    Ok(true)
}

fn main() -> ExitCode {
    let outcome = parse_args(std::env::args().skip(1)).and_then(|options| run(&options));
    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(2)
        }
    }
}
