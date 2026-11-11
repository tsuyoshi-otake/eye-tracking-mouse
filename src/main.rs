// A tray application: no console window when started from Explorer.
#![windows_subsystem = "windows"]

use std::path::PathBuf;
use std::process::ExitCode;

use eye_tracking_mouse::app::{diag, selftest, tray};
use eye_tracking_mouse::error::Error;
use eye_tracking_mouse::platform::camera::{Camera, CameraRequest};
use eye_tracking_mouse::platform::runtime;

const USAGE: &str = "\
eye-tracking-mouse [options]

  (no options)          run in the notification area
  --camera <index>      use this camera (see --list-cameras) instead of the one in the settings file
  --data-dir <folder>   keep the settings, the calibration and the log in this folder
  --quit                ask the running program to quit and wait until it has
  --list-cameras        print the attached cameras and exit
  --selftest <name>     run one check and exit: camera, face, model, pointer
  --image <file.bmp>    with --selftest face: use this picture instead of the camera
  --diag <seconds>      print what the face tracker measures, once per second
  --help                print this text

Exit code: 0 done; 1 not done (a check failed, already running, nothing to quit); 2 an error.
";

enum Mode {
    Run,
    Quit,
    ListCameras,
    SelfTest(String),
    Diag(u32),
    Help,
}

struct Options {
    mode: Mode,
    /// `None` when the command line names no camera.
    camera: Option<usize>,
    data_folder: Option<PathBuf>,
    image: Option<String>,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Options, Error> {
    let mut options = Options { mode: Mode::Run, camera: None, data_folder: None, image: None };
    let mut args = args;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().ok_or_else(|| Error::new(format!("{name} needs a value")));
        match arg.as_str() {
            "--camera" => {
                let text = value("--camera")?;
                options.camera = Some(text.parse().map_err(|_| Error::new(format!("'{text}' is not a camera index")))?);
            }
            "--data-dir" => options.data_folder = Some(PathBuf::from(value("--data-dir")?)),
            "--quit" => options.mode = Mode::Quit,
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
    // The resident program does its camera work on a thread of its own, which enters COM there.
    let _com = match options.mode {
        Mode::ListCameras | Mode::SelfTest(_) | Mode::Diag(_) => Some(runtime::ComApartment::enter()?),
        Mode::Run | Mode::Quit | Mode::Help => None,
    };
    let camera = CameraRequest { device_index: options.camera.unwrap_or(0), ..CameraRequest::default() };
    match &options.mode {
        Mode::Help => print!("{USAGE}"),
        Mode::Run => {
            let options = tray::Options { camera: options.camera, data_folder: options.data_folder.clone() };
            let started = tray::run(&options)?;
            if !started {
                eprintln!("eye-tracking-mouse is already running");
            }
            return Ok(started);
        }
        Mode::Quit => {
            let quit = tray::quit_running()?;
            if !quit {
                eprintln!("eye-tracking-mouse is not running");
            }
            return Ok(quit);
        }
        Mode::ListCameras => {
            for (index, name) in Camera::device_names()?.iter().enumerate() {
                println!("{index}: {name}");
            }
        }
        Mode::SelfTest(name) => return selftest::run(name, &camera, options.image.as_deref()),
        Mode::Diag(seconds) => diag::run(&camera, *seconds)?,
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
