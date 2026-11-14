//! What the program says to the person, in one place. Everything here is Japanese.

use crate::control::calibration::{Failure, Phase, Score};
use crate::control::controller::Activity;

use super::engine::GazeProblem;
use super::pipeline::{CalibrationProblem, Notice, Routine, Status};

pub const APP_NAME: &str = "eye-tracking-mouse";

/// The longest title and text a notification shows, in UTF-16 units.
#[cfg(test)]
const NOTIFICATION_LIMITS: (usize, usize) = (63, 255);

/// A notification: a short title, a sentence or two, and whether it reports a problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub title: &'static str,
    pub text: String,
    pub warning: bool,
}

/// One line saying what the program is doing, for the icon's tooltip and the log.
pub fn status(status: &Status) -> String {
    match status {
        Status::Starting => "準備中".to_string(),
        Status::Paused => "一時停止中".to_string(),
        Status::Waiting(reason) => format!("再試行中（{reason}）"),
        Status::Pointing(Activity::Tracking) => "動作中".to_string(),
        Status::Pointing(Activity::NoFace) => "顔が見つかりません".to_string(),
        Status::Pointing(Activity::Yielded) => "マウス操作を優先中".to_string(),
        Status::Pointing(Activity::Blocked) => "カーソルを動かせません".to_string(),
        Status::Calibrating => "キャリブレーション中".to_string(),
        Status::Stopped(reason) => format!("停止しました（{reason}）"),
    }
}

/// Shown once when the program has started.
pub fn started(has_calibration: bool) -> Message {
    let gaze = if has_calibration { "" } else { "\n視線で動かすには Ctrl+Alt+C でキャリブレーションします。" };
    Message {
        title: "eye-tracking-mouse を開始しました",
        text: format!("一時停止 Ctrl+Alt+P / 終了 Ctrl+Alt+Q。マウスを動かすと自動で待機します。{gaze}"),
        warning: false,
    }
}

/// Shown in a box when the program cannot start at all: there is no icon yet to say it with.
pub fn could_not_start(reason: &str) -> Message {
    Message { title: "eye-tracking-mouse を開始できません", text: format!("起動を中止しました。（{reason}）"), warning: true }
}

/// The menu of the icon. What follows the tab is shown at the right edge, as the key to press instead.
pub const MENU_PAUSE: &str = "一時停止\tCtrl+Alt+P";
pub const MENU_RESUME: &str = "再開\tCtrl+Alt+P";
pub const MENU_CALIBRATE: &str = "キャリブレーション\tCtrl+Alt+C";
pub const MENU_CALIBRATE_STOP: &str = "キャリブレーションを中止\tCtrl+Alt+C";
pub const MENU_CHECK: &str = "追加補正テスト\tCtrl+Alt+T";
pub const MENU_CHECK_STOP: &str = "追加補正テストを中止\tCtrl+Alt+T";
/// The other entry's key while a run is going: it ends that run too.
pub const MENU_CALIBRATE_STOP_SHORT: &str = "キャリブレーションを中止\tCtrl+Alt+T";
pub const MENU_CHECK_STOP_SHORT: &str = "追加補正テストを中止\tCtrl+Alt+C";
pub const MENU_PREVIEW: &str = "カメラのプレビュー";
/// What moves the cursor; the one in use is checked.
pub const MENU_POINT_HEAD: &str = "頭だけで動かす";
pub const MENU_POINT_EYE: &str = "視線で動かす（eye tracking）";
pub const MENU_POINT_HYBRID: &str = "視線で動かし、頭で微調整（ハイブリッド）";
pub const MENU_SETTINGS: &str = "設定ファイルを開く（再起動で反映）";
pub const MENU_QUIT: &str = "終了\tCtrl+Alt+Q";

pub fn already_running() -> Message {
    Message {
        title: "すでに起動しています",
        text: "通知領域のアイコンから操作できます（一時停止 Ctrl+Alt+P / 終了 Ctrl+Alt+Q）。".to_string(),
        warning: false,
    }
}

/// Problems found while starting: lines of the settings file, hotkeys that are taken.
pub fn startup_problems(problems: &[String]) -> Message {
    let shown = problems.iter().take(3).cloned().collect::<Vec<_>>().join("\n");
    let more = if problems.len() > 3 { format!("\nほか {} 件（log.txt に記録）", problems.len() - 3) } else { String::new() };
    Message { title: "設定を確認してください", text: format!("{shown}{more}"), warning: true }
}

pub fn hotkey_taken(key: char) -> String {
    format!("Ctrl+Alt+{key} は他のアプリが使用中のため登録できません（メニューからは操作できます）")
}

pub fn settings_not_opened(path: &str, reason: &str) -> Message {
    Message { title: "設定ファイルを開けません", text: format!("{path}（{reason}）"), warning: true }
}

/// Shown when pointing with the eyes is chosen and there is no calibration for it to work with.
pub fn eye_pointing_needs_calibration() -> Message {
    Message {
        title: "キャリブレーションが必要です",
        text: "視線でカーソルを動かすには Ctrl+Alt+C でキャリブレーションします。それまでは頭の動きで動かします。".to_string(),
        warning: false,
    }
}

/// Shown when the choice of pointing made in the menu holds only until the program ends.
pub fn pointing_not_saved(reason: &str) -> Message {
    Message {
        title: "設定を保存できません",
        text: format!("切り替えは今回の起動中だけ有効です。（{reason}）"),
        warning: true,
    }
}

/// Shown when the worker stops pointing for a reason the person should know about.
pub fn trouble(status: &Status) -> Option<Message> {
    match status {
        Status::Waiting(reason) => Some(Message {
            title: "カメラを使えません",
            text: format!("自動で再試行します。カメラの接続と、他のアプリが使っていないかを確認してください。（{reason}）"),
            warning: true,
        }),
        Status::Stopped(reason) if !reason.is_empty() => Some(Message {
            title: "動作を停止しました",
            text: format!("アプリを終了して起動し直してください。（{reason}）"),
            warning: true,
        }),
        _ => None,
    }
}

const RETRY: &str = "Ctrl+Alt+C でやり直せます。";
/// After a run that failed: which key starts which run again.
const RUN_AGAIN: &str = "キャリブレーションは Ctrl+Alt+C、追加補正テストは Ctrl+Alt+T でやり直せます。";

/// Shown when a calibration run ended without a result.
pub fn calibration_failed(problem: &CalibrationProblem) -> Message {
    let text = match problem {
        CalibrationProblem::Routine(Failure::FaceLost) => "顔がカメラに映っていませんでした。".to_string(),
        CalibrationProblem::Routine(Failure::TooSlow) => {
            "目を開けた顔を十分に測れませんでした。部屋の明るさとカメラの向きを確認してください。".to_string()
        }
        CalibrationProblem::Routine(Failure::HeadNotMoved) => {
            "最後のステップで頭の動きが足りませんでした。中央の点を見たまま、頭を上下左右にゆっくり回します。".to_string()
        }
        CalibrationProblem::Routine(Failure::Unusable) => "測定結果から視線のモデルを作れませんでした。".to_string(),
        CalibrationProblem::Routine(Failure::Inaccurate { error_px }) => {
            format!("誤差が大きすぎました（約{error_px:.0}ピクセル）。頭は動かさず、目だけで点を追います。")
        }
        CalibrationProblem::Camera(reason) => format!("カメラを使えません。（{reason}）"),
        CalibrationProblem::DisplayChanged => "途中で画面の構成が変わりました。".to_string(),
        CalibrationProblem::Paused => "一時停止したため中止しました。".to_string(),
        CalibrationProblem::NotCalibrated => {
            return Message {
                title: "追加補正テストはまだできません",
                text: "この画面とカメラのキャリブレーションがありません。先に Ctrl+Alt+C でキャリブレーションします。".to_string(),
                warning: true,
            };
        }
    };
    Message { title: "キャリブレーションできませんでした", text: format!("{text}{RUN_AGAIN}"), warning: true }
}

pub fn notice(notice: &Notice) -> Message {
    match notice {
        Notice::CalibrationDone { routine: Routine::Calibration, error_px, saved: Ok(()), .. } => Message {
            title: "キャリブレーション完了",
            text: format!(
                "視線が使えます（誤差の目安 約{error_px:.0}ピクセル）。Ctrl+Alt+T の追加補正テストで、やるたびに精度を上げられます。"
            ),
            warning: false,
        },
        Notice::CalibrationDone { routine: Routine::Check, error_px, saved: Ok(()), .. } => Message {
            title: "追加補正しました",
            text: format!(
                "補正前のずれは平均 約{error_px:.0}ピクセルでした。この結果を補正に加えました。Ctrl+Alt+T でもう一度やると、さらに補正されます。"
            ),
            warning: false,
        },
        Notice::CalibrationDone { saved: Err(reason), .. } => Message {
            title: "完了（保存に失敗）",
            text: format!("今回の結果は終了するまで使えますが、ファイルに保存できなかったので次回の起動時には残りません。（{reason}）"),
            warning: true,
        },
        Notice::CalibrationFailed { problem, .. } => calibration_failed(problem),
        Notice::GazeUnavailable(problem) => {
            let cause = match problem {
                GazeProblem::ScreenChanged => "キャリブレーションした画面が見つかりません。",
                GazeProblem::CameraChanged => "カメラの解像度がキャリブレーション時と違います。",
            };
            Message {
                title: "視線での操作は休止中です",
                text: format!("{cause}頭での操作とクリックはそのまま使えます。{RETRY}"),
                warning: true,
            }
        }
        Notice::CalibrationUnreadable(reason) => Message {
            title: "保存済みのキャリブレーションを読めません",
            text: format!("頭での操作とクリックは使えます。{RETRY}（{reason}）"),
            warning: true,
        },
    }
}

/// What to do during each part of a calibration run.
pub fn instruction(phase: Phase) -> &'static str {
    match phase {
        Phase::Rest => "中央の点を見つめて、頭を動かさずにお待ちください",
        Phase::Places => "頭は動かさず、目だけで点を追ってください",
        Phase::Sweep => "中央の点を見つめたまま、頭を上下左右にゆっくり回してください",
        Phase::Check => "追加補正テスト：出てくる点を目で追ってください",
    }
}

/// The line that ends the check: how many of its places the model hit and how far off it was.
pub fn check_score(score: Score) -> String {
    format!("命中 {} / {}　　平均のずれ 約{:.0}ピクセル", score.hits, score.places, score.mean_px)
}

/// Shown when one of the program's own windows cannot be opened.
pub fn window_not_opened(reason: &str) -> Message {
    Message { title: "画面を開けません", text: format!("もう一度お試しください。（{reason}）"), warning: true }
}

pub const CALIBRATION_TITLE: &str = "eye-tracking-mouse キャリブレーション";
pub const PREVIEW_TITLE: &str = "eye-tracking-mouse プレビュー";
pub const FACE_NOT_VISIBLE: &str = "顔がカメラに映っていません";
pub const CALIBRATION_PREPARING: &str = "カメラを準備しています…";
pub const CALIBRATION_HOW_TO_CANCEL: &str = "Esc または Ctrl+Alt+C で中止";

#[cfg(test)]
mod tests {
    use super::*;

    fn fits(message: &Message) {
        let (title, text) = (message.title.encode_utf16().count(), message.text.encode_utf16().count());
        assert!(title > 0 && title <= NOTIFICATION_LIMITS.0, "title of {message:?} takes {title} units");
        assert!(text > 0 && text <= NOTIFICATION_LIMITS.1, "text of {message:?} takes {text} units");
    }

    /// As long as the longest reason the system is likely to give.
    fn reason() -> String {
        "camera index 31 does not exist (2 camera(s) found): read a camera frame (0x80070005)".to_string()
    }

    #[test]
    fn every_status_has_a_line_of_its_own() {
        let statuses = [
            Status::Starting,
            Status::Paused,
            Status::Waiting(reason()),
            Status::Pointing(Activity::Tracking),
            Status::Pointing(Activity::NoFace),
            Status::Pointing(Activity::Yielded),
            Status::Pointing(Activity::Blocked),
            Status::Calibrating,
            Status::Stopped(reason()),
        ];
        let lines: std::collections::HashSet<String> = statuses.iter().map(status).collect();
        assert_eq!(lines.len(), statuses.len());
        assert!(lines.iter().all(|line| !line.is_empty() && !line.contains('\n')));
    }

    #[test]
    fn every_notification_fits_what_the_system_shows() {
        let problems = [
            CalibrationProblem::Routine(Failure::FaceLost),
            CalibrationProblem::Routine(Failure::TooSlow),
            CalibrationProblem::Routine(Failure::HeadNotMoved),
            CalibrationProblem::Routine(Failure::Unusable),
            CalibrationProblem::NotCalibrated,
            CalibrationProblem::Routine(Failure::Inaccurate { error_px: 1234.5 }),
            CalibrationProblem::Camera(reason()),
            CalibrationProblem::DisplayChanged,
            CalibrationProblem::Paused,
        ];
        let mut notices: Vec<Notice> =
            problems.into_iter().map(|problem| Notice::CalibrationFailed { run: 1, problem }).collect();
        notices.push(Notice::CalibrationDone { run: 1, routine: Routine::Calibration, error_px: 48.6, saved: Ok(()) });
        notices.push(Notice::CalibrationDone { run: 1, routine: Routine::Check, error_px: 48.6, saved: Ok(()) });
        notices.push(Notice::CalibrationDone { run: 1, routine: Routine::Calibration, error_px: 48.6, saved: Err(reason()) });
        notices.push(Notice::GazeUnavailable(GazeProblem::ScreenChanged));
        notices.push(Notice::GazeUnavailable(GazeProblem::CameraChanged));
        notices.push(Notice::CalibrationUnreadable(reason()));
        let texts: std::collections::HashSet<String> = notices.iter().map(|n| notice(n).text).collect();
        assert_eq!(texts.len(), notices.len(), "each says something of its own");
        notices.iter().map(notice).for_each(|message| fits(&message));

        fits(&started(true));
        fits(&started(false));
        fits(&already_running());
        fits(&trouble(&Status::Waiting(reason())).unwrap());
        fits(&trouble(&Status::Stopped(reason())).unwrap());
        let many: Vec<String> = ('A'..='F').map(hotkey_taken).collect();
        fits(&startup_problems(&many));
        fits(&startup_problems(&many[..1]));
        fits(&window_not_opened(&reason()));
        fits(&eye_pointing_needs_calibration());
        fits(&pointing_not_saved(&reason()));
    }

    #[test]
    fn only_real_trouble_is_announced() {
        assert_eq!(trouble(&Status::Stopped(String::new())), None, "stopping because it was asked to");
        for quiet in [Status::Starting, Status::Paused, Status::Calibrating, Status::Pointing(Activity::NoFace)] {
            assert_eq!(trouble(&quiet), None);
        }
        let done = notice(&Notice::CalibrationDone { run: 1, routine: Routine::Calibration, error_px: 40.0, saved: Ok(()) });
        assert!(done.text.contains("約40ピクセル") && !done.warning);
    }
}
