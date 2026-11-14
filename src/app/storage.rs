//! Where the settings, the calibration and the log are kept between runs.
//!
//! One folder holds everything: `settings.txt` is written once as a commented
//! template and from then on belongs to the person, except for the line a
//! choice in the menu rewrites (`save_pointing`); `calibration.txt` is
//! replaced after each successful calibration. A file is replaced in one
//! step, so a crash never leaves half of one.

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::control::calibration::Calibration;
use crate::control::controller::PointingMode;
use crate::error::{Context, Error, Result};

use super::settings::{self, Settings};

const FOLDER_NAME: &str = "eye-tracking-mouse";
const SETTINGS_FILE: &str = "settings.txt";
const CALIBRATION_FILE: &str = "calibration.txt";
const LOG_FILE: &str = "log.txt";

#[derive(Clone, Debug)]
pub struct Storage {
    folder: PathBuf,
}

impl Storage {
    /// The folder used when the person names none: under their roaming application data.
    pub fn default_folder() -> Result<PathBuf> {
        let base = std::env::var_os("APPDATA").filter(|value| !value.is_empty());
        Ok(PathBuf::from(base.context("the APPDATA folder is not known")?).join(FOLDER_NAME))
    }

    /// Uses `folder`, creating it if needed.
    pub fn open(folder: &Path) -> Result<Storage> {
        fs::create_dir_all(folder).context("create the settings folder")?;
        Ok(Storage { folder: folder.to_path_buf() })
    }

    pub fn settings_path(&self) -> PathBuf {
        self.folder.join(SETTINGS_FILE)
    }

    pub fn log_path(&self) -> PathBuf {
        self.folder.join(LOG_FILE)
    }

    /// Reads the settings, writing the template first when there is no file yet.
    /// Never fails: what went wrong comes back as warnings for the person, next to usable settings.
    pub fn load_settings(&self) -> (Settings, Vec<String>) {
        let path = self.settings_path();
        match fs::read_to_string(&path) {
            Ok(text) => settings::parse(&text),
            Err(error) if error.kind() == ErrorKind::NotFound => {
                let written = fs::write(&path, settings::template());
                let warnings = written.err().map(|error| format!("設定ファイルを作成できません: {error}"));
                (Settings::default(), warnings.into_iter().collect())
            }
            Err(error) => (Settings::default(), vec![format!("設定ファイルを読めません: {error}")]),
        }
    }

    /// Records the pointing mode in the settings file; the person's other lines and comments
    /// stay as they are. A file that cannot be read is left alone and the error says so.
    pub fn save_pointing(&self, mode: PointingMode) -> Result<()> {
        let path = self.settings_path();
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => settings::template(),
            Err(error) => return Err(Error::from(error)).context("read the settings"),
        };
        self.replace(SETTINGS_FILE, &settings::with_pointing(&text, mode)).context("write the settings")
    }

    /// Puts `text` in the place of the file `name`, in one step.
    fn replace(&self, name: &str, text: &str) -> Result<()> {
        let fresh = self.folder.join(format!("{name}.new"));
        fs::write(&fresh, text)?;
        fs::rename(&fresh, self.folder.join(name))?;
        Ok(())
    }

    /// Whether a calibration has been saved. It may still turn out to be unreadable.
    pub fn has_calibration(&self) -> bool {
        self.folder.join(CALIBRATION_FILE).exists()
    }

    /// `Ok(None)` when no calibration has been saved yet.
    pub fn load_calibration(&self) -> Result<Option<Calibration>> {
        match fs::read_to_string(self.folder.join(CALIBRATION_FILE)) {
            Ok(text) => Calibration::parse(&text).map(Some).context("read the saved calibration"),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(Error::from(error)).context("read the saved calibration"),
        }
    }

    pub fn save_calibration(&self, calibration: &Calibration) -> Result<()> {
        self.replace(CALIBRATION_FILE, &calibration.to_text()).context("write the calibration")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::calibration::simulation::calibrated;

    /// A folder of this test's own, removed again when the test ends, whichever way.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Scratch {
            let folder = std::env::temp_dir().join(format!("etm-storage-test-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&folder);
            Scratch(folder)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn the_first_start_writes_the_settings_template_and_later_starts_read_it() {
        let scratch = Scratch::new("settings");
        let storage = Storage::open(&scratch.0.join("made").join("on demand")).unwrap();
        assert_eq!(storage.load_settings(), (Settings::default(), Vec::new()));
        assert_eq!(fs::read_to_string(storage.settings_path()).unwrap(), settings::template());

        fs::write(storage.settings_path(), "camera = 3\nwhatever\n").unwrap();
        let (loaded, warnings) = storage.load_settings();
        assert_eq!(loaded.camera, 3);
        assert_eq!(warnings.len(), 1);
        assert_eq!(fs::read_to_string(storage.settings_path()).unwrap(), "camera = 3\nwhatever\n", "never rewritten");
    }

    #[test]
    fn a_saved_pointing_mode_is_read_at_the_next_start_and_the_rest_of_the_file_is_kept() {
        let scratch = Scratch::new("pointing");
        let storage = Storage::open(&scratch.0).unwrap();
        // Before the first start there is no file: the template is written with the choice in it.
        storage.save_pointing(PointingMode::Eye).unwrap();
        let (loaded, warnings) = storage.load_settings();
        assert_eq!((loaded.tuning.pointing, warnings), (PointingMode::Eye, Vec::new()));

        fs::write(storage.settings_path(), "camera = 3 # mine\nhead_steadiness = 4\n").unwrap();
        storage.save_pointing(PointingMode::Eye).unwrap();
        let text = fs::read_to_string(storage.settings_path()).unwrap();
        assert_eq!(text, "camera = 3 # mine\nhead_steadiness = 4\npointing = eye\n");
        storage.save_pointing(PointingMode::Head).unwrap();
        let (loaded, warnings) = storage.load_settings();
        assert_eq!((loaded.camera, loaded.tuning.pointing, warnings), (3, PointingMode::Head, Vec::new()));
        let left: Vec<_> = fs::read_dir(&scratch.0).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(left, [SETTINGS_FILE], "no temporary file stays behind");

        // A file that is not text is not overwritten.
        fs::write(storage.settings_path(), [0xFF, 0xFE, 0x00, 0xD8]).unwrap();
        let error = storage.save_pointing(PointingMode::Eye).unwrap_err();
        assert!(error.message().starts_with("read the settings: "), "{error}");
        assert_eq!(fs::read(storage.settings_path()).unwrap(), [0xFF, 0xFE, 0x00, 0xD8]);
    }

    #[test]
    fn settings_that_cannot_be_read_fall_back_to_the_defaults_with_a_warning() {
        let scratch = Scratch::new("unreadable");
        let storage = Storage::open(&scratch.0).unwrap();
        fs::write(storage.settings_path(), [0xFF, 0xFE, 0x00, 0xD8]).unwrap();
        let (loaded, warnings) = storage.load_settings();
        assert_eq!(loaded, Settings::default());
        assert_eq!(warnings.len(), 1, "{warnings:?}");
    }

    #[test]
    fn a_calibration_survives_being_saved_and_loaded_and_replaces_the_one_before() {
        let scratch = Scratch::new("calibration");
        let storage = Storage::open(&scratch.0).unwrap();
        assert_eq!(storage.load_calibration().unwrap(), None);
        assert!(!storage.has_calibration());

        let first = calibrated();
        storage.save_calibration(&first).unwrap();
        assert!(storage.has_calibration());
        // The text form rounds nothing, so the loaded value predicts exactly what the saved one did.
        assert_eq!(storage.load_calibration().unwrap(), Some(first.clone()));

        let mut second = first;
        second.error_px += 1.0;
        storage.save_calibration(&second).unwrap();
        assert_eq!(storage.load_calibration().unwrap(), Some(second));
        let left: Vec<_> = fs::read_dir(&scratch.0).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(left, [CALIBRATION_FILE], "no temporary file stays behind");
    }

    #[test]
    fn a_damaged_calibration_file_is_an_error_not_a_missing_calibration() {
        let scratch = Scratch::new("damaged");
        let storage = Storage::open(&scratch.0).unwrap();
        fs::write(scratch.0.join(CALIBRATION_FILE), "eye-tracking-mouse calibration 1\narea = 0 0\n").unwrap();
        let error = storage.load_calibration().unwrap_err();
        assert!(error.message().starts_with("read the saved calibration: "), "{error}");
    }
}
