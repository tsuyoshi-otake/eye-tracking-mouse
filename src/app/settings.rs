//! What the person can adjust, read once at startup from a small text file.
//!
//! Contract: `parse` never fails. A line it cannot use is reported as a
//! warning and that setting keeps its default; a number outside its range is
//! pulled to the nearest allowed value, also with a warning. Warnings are
//! shown to the person, so they are written in Japanese like the file itself.

use crate::control::controller::{PointingMode, Tuning};
use crate::control::head_pointer::HeadGain;

/// The highest camera index accepted (a position in `--list-cameras`).
const MAX_CAMERA: usize = 31;
/// A still-shut pair of eyes cancels the click at least this many seconds after it was armed.
const LIMIT_MARGIN: f32 = 0.3;
/// An eye counts as open again this far above the "shut" share of its open size.
const OPEN_MARGIN: f32 = 0.1;
/// Settings an older version wrote into the file and this one no longer has.
/// Their lines are passed over without a warning, since the person did nothing wrong.
/// `gaze_jump` switched the jump to where the person looked in head mode, which is head-only now.
const RETIRED: [&str; 1] = ["gaze_jump"];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Settings {
    /// Position of the camera in `Camera::device_names()`.
    pub camera: usize,
    pub tuning: Tuning,
    /// Keep a record of every calibration run (see `record`).
    pub record_calibration: bool,
    /// Keep a record of pointing (see `record`).
    pub record_pointing: bool,
}

/// One setting's place in `Settings`, with the values it accepts.
enum Slot<'a> {
    Camera(&'a mut usize),
    Switch(&'a mut bool),
    Number(&'a mut f32, f32, f32),
    /// `head` or `eye`.
    Pointing(&'a mut PointingMode),
}

fn slot<'a>(settings: &'a mut Settings, name: &str) -> Option<Slot<'a>> {
    let tuning = &mut settings.tuning;
    Some(match name {
        "camera" => Slot::Camera(&mut settings.camera),
        "calibration_record" => Slot::Switch(&mut settings.record_calibration),
        "pointing_record" => Slot::Switch(&mut settings.record_pointing),
        "pointing" => Slot::Pointing(&mut tuning.pointing),
        "blink_click" => Slot::Switch(&mut tuning.blink_click),
        "head_speed_slow" => Slot::Number(&mut tuning.head_gain.slow, HeadGain::SLOW.0, HeadGain::SLOW.1),
        "head_speed_fast" => Slot::Number(&mut tuning.head_gain.fast, HeadGain::FAST.0, HeadGain::FAST.1),
        "head_steadiness" => {
            Slot::Number(&mut tuning.head_gain.steadiness, HeadGain::STEADINESS.0, HeadGain::STEADINESS.1)
        }
        "eye_steadiness" => Slot::Number(&mut tuning.eye_steadiness, 0.5, 8.0),
        "hybrid_head_speed" => Slot::Number(&mut tuning.hybrid_speed, 20.0, 5000.0),
        "jump_distance" => Slot::Number(&mut tuning.jump_distance, 120.0, 5000.0),
        "blink_hold" => Slot::Number(&mut tuning.blink.hold, 0.15, 3.0),
        "blink_limit" => Slot::Number(&mut tuning.blink.limit, 0.45, 10.0),
        "blink_closed" => Slot::Number(&mut tuning.blink.closed_ratio, 0.2, 0.8),
        _ => return None,
    })
}

fn switch(value: &str) -> Option<bool> {
    match value.to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" => Some(true),
        "off" | "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

/// Reads the settings file's text. Returns the settings and what was wrong with the text.
pub fn parse(text: &str) -> (Settings, Vec<String>) {
    let mut settings = Settings::default();
    let mut warnings = Vec::new();
    // Some editors put a byte-order mark in front of the first line.
    for (index, raw) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = raw.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        let place = format!("{}行目", index + 1);
        let Some((name, value)) = line.split_once('=') else {
            warnings.push(format!("{place}: 「{line}」は「名前 = 値」の形ではありません"));
            continue;
        };
        let (name, value) = (name.trim(), value.trim());
        if RETIRED.contains(&name) {
            continue;
        }
        match slot(&mut settings, name) {
            None => warnings.push(format!("{place}: 「{name}」という設定はありません")),
            Some(Slot::Switch(target)) => match switch(value) {
                Some(on) => *target = on,
                None => warnings.push(format!("{place}: {name} は on か off で指定してください")),
            },
            Some(Slot::Pointing(target)) => match value.to_ascii_lowercase().as_str() {
                "head" => *target = PointingMode::Head,
                "eye" => *target = PointingMode::Eye,
                "hybrid" => *target = PointingMode::Hybrid,
                _ => warnings.push(format!("{place}: {name} は head か eye か hybrid で指定してください")),
            },
            Some(Slot::Number(target, low, high)) => match value.parse::<f32>() {
                Ok(number) if number.is_finite() => {
                    *target = number.clamp(low, high);
                    if *target != number {
                        warnings.push(format!("{place}: {name} は {low}〜{high} の範囲なので {} にしました", *target));
                    }
                }
                _ => warnings.push(format!("{place}: {name} の値「{value}」は数値ではありません")),
            },
            Some(Slot::Camera(target)) => match value.parse::<usize>() {
                Ok(index) => {
                    *target = index.min(MAX_CAMERA);
                    if index > MAX_CAMERA {
                        warnings.push(format!("{place}: {name} は 0〜{MAX_CAMERA} の範囲なので {MAX_CAMERA} にしました"));
                    }
                }
                Err(_) => warnings.push(format!("{place}: {name} の値「{value}」は 0 以上の整数ではありません")),
            },
        }
    }
    // Rules that tie two settings together hold whatever order the lines came in.
    let tuning = &mut settings.tuning;
    if tuning.head_gain.fast < tuning.head_gain.slow {
        tuning.head_gain.fast = tuning.head_gain.slow;
        warnings.push("head_speed_fast が head_speed_slow より小さいので、同じ値にしました".to_string());
    }
    let shortest = tuning.blink.hold + LIMIT_MARGIN;
    if tuning.blink.limit < shortest {
        tuning.blink.limit = shortest;
        warnings.push(format!("blink_limit が短すぎるので {shortest} 秒にしました（blink_hold + {LIMIT_MARGIN} 秒以上）"));
    }
    tuning.blink.open_ratio = tuning.blink.closed_ratio + OPEN_MARGIN;
    (settings, warnings)
}

/// How a pointing mode is written in the file.
pub fn pointing_word(mode: PointingMode) -> &'static str {
    match mode {
        PointingMode::Head => "head",
        PointingMode::Eye => "eye",
        PointingMode::Hybrid => "hybrid",
    }
}

/// The settings file's `text` with the pointing mode set to `mode`, as `with_setting` does it.
pub fn with_pointing(text: &str, mode: PointingMode) -> String {
    with_setting(text, "pointing", pointing_word(mode))
}

/// The settings file's `text` with the head pointer's speeds and steadiness set to `gain`,
/// as `with_setting` does it. The speeds are written as whole pixels.
pub fn with_head_gain(text: &str, gain: &HeadGain) -> String {
    let text = with_setting(text, "head_speed_slow", &format!("{:.0}", gain.slow));
    let text = with_setting(&text, "head_speed_fast", &format!("{:.0}", gain.fast));
    with_setting(&text, "head_steadiness", &gain.steadiness.to_string())
}

/// The settings file's `text` with the setting `name` set to `word`. Every line
/// that sets it gets the new value and keeps its comment; a text without such
/// a line gets one at the end. Nothing else in the text changes.
fn with_setting(text: &str, name: &str, word: &str) -> String {
    let mut changed = String::with_capacity(text.len() + 32);
    let mut found = false;
    // A byte-order mark in front of the first line is not part of it, as in `parse`.
    let lines = text.trim_start_matches('\u{feff}');
    changed.push_str(&text[..text.len() - lines.len()]);
    for raw in lines.split_inclusive('\n') {
        let body = raw.trim_end_matches(['\r', '\n']);
        let (setting, comment) = body.split_at(body.find('#').unwrap_or(body.len()));
        let name_at = setting.len() - setting.trim_start().len();
        if setting[name_at..].split_once('=').is_some_and(|(written, _)| written.trim() == name) {
            found = true;
            let gap = if comment.is_empty() { "" } else { " " };
            changed.push_str(&format!("{}{name} = {word}{gap}{comment}{}", &setting[..name_at], &raw[body.len()..]));
        } else {
            changed.push_str(raw);
        }
    }
    if !found {
        let ending = if text.contains("\r\n") { "\r\n" } else { "\n" };
        if !changed.is_empty() && !changed.ends_with('\n') {
            changed.push_str(ending);
        }
        changed.push_str(&format!("{name} = {word}{ending}"));
    }
    changed
}

/// The file written when there is none yet: every setting at its default, explained.
pub fn template() -> String {
    let Settings { camera, tuning, record_calibration, record_pointing } = Settings::default();
    let on_off = |on: bool| if on { "on" } else { "off" };
    format!(
        "\
# eye-tracking-mouse の設定
# 「名前 = 値」を1行に1つ書きます。# から行末まではコメントです。
# 変更は、アプリを終了して起動し直すと反映されます。

# 使うカメラの番号（eye-tracking-mouse --list-cameras で確認できます）
camera = {camera}

# カーソルを動かすもの: head / eye / hybrid
# head: 頭の動きだけで動かす。視線は使いません（既定）。
# eye: 視線がそのままカーソルの位置になる、連続追従の eye tracking。
# hybrid: 視線がカーソルを運び、頭の小さな動きでそこから微調整する。
#         別の場所（jump_distance 以上離れた所）を見ると微調整は消えます。
# eye と hybrid はキャリブレーション済みの画面でのみ動き、未実施なら頭の動きだけで動かします。
pointing = {pointing}

# 長いまばたきで左クリック: on / off
blink_click = {blink_click}

# 頭の動きに対するカーソルの速さ（大きいほど速い）
# ゆっくり動かしたとき（50〜5000）
head_speed_slow = {slow}
# すばやく動かしたとき（head_speed_slow 以上、20000 まで）
head_speed_fast = {fast}
# カーソルの安定度（0.5〜8、大きいほど止まっているときに震えず、動き出しは少し遅くなる）
head_steadiness = {steadiness}

# pointing = eye / hybrid のときの視線カーソルの安定度（0.5〜8）
# 大きいほど見つめている場所でカーソルが震えず、視線を移したときの追従は少し遅くなります。
eye_steadiness = {eye_steadiness}

# pointing = hybrid のときの、頭の動きに対するカーソルの速さ（20〜5000、大きいほど速い）
# 微調整用なので head_speed_slow より小さくし、すばやく動かしても速くなりません。
hybrid_head_speed = {hybrid_speed}

# pointing = hybrid のとき、頭での微調整が消える距離（ピクセル、120〜5000）
# 視線がカーソルをこの距離だけ運ぶと、別の場所を見たとみなして微調整を消します。
jump_distance = {jump}

# クリックになるまで目を閉じておく時間（秒、0.15〜3.0）
blink_hold = {hold}
# 閉じたままこの時間を過ぎるとクリックを取り消します（秒、blink_hold + {LIMIT_MARGIN} 以上、10 まで）
blink_limit = {limit}
# 目を「閉じた」とみなす開き具合（ふだんの開きに対する割合、0.2〜0.8）
blink_closed = {closed}

# キャリブレーションの記録（精度を調べるためのもの）: on / off
# on にすると、キャリブレーションのたびに calibration-record-<時刻>.txt をこのフォルダーに残します。
# 中身は各フレームの顔の点の座標（数値のみ、1回で数MB）で、画像は含みません。不要になったら削除してください。
calibration_record = {record}

# カーソルの動きの記録（動きがおかしいときの原因を調べるためのもの）: on / off
# on にすると、動作中の毎フレームの数値（頭の向き、目の信号、視線の推定位置、カーソルへの指示）を
# pointing-record-<時刻>.txt に残します。一時停止・終了のとき、または10分ごとに書き出します。
# 画像は含みません。不要になったら削除してください。
pointing_record = {pointing_record}
",
        pointing = pointing_word(tuning.pointing),
        blink_click = on_off(tuning.blink_click),
        slow = tuning.head_gain.slow,
        fast = tuning.head_gain.fast,
        steadiness = tuning.head_gain.steadiness,
        eye_steadiness = tuning.eye_steadiness,
        hybrid_speed = tuning.hybrid_speed,
        jump = tuning.jump_distance,
        hold = tuning.blink.hold,
        limit = tuning.blink.limit,
        closed = tuning.blink.closed_ratio,
        record = on_off(record_calibration),
        pointing_record = on_off(record_pointing),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_reads_back_as_the_defaults_without_complaint() {
        let (settings, warnings) = parse(&template());
        assert_eq!(settings, Settings::default());
        assert_eq!(warnings, Vec::<String>::new());
        // Every setting is offered in the template.
        let names = ["camera", "calibration_record", "pointing_record", "pointing", "blink_click", "head_speed_slow", "head_speed_fast", "head_steadiness", "eye_steadiness", "hybrid_head_speed"];
        for name in names.into_iter().chain(["jump_distance", "blink_hold", "blink_limit", "blink_closed"]) {
            assert!(slot(&mut Settings::default(), name).is_some(), "{name}");
            assert!(template().contains(&format!("\n{name} = ")), "{name}");
        }
        for name in RETIRED {
            assert!(slot(&mut Settings::default(), name).is_none() && !template().contains(name), "{name}");
        }
    }

    #[test]
    fn a_setting_an_older_version_wrote_is_passed_over_without_a_warning() {
        // A file written by v0.2.0 still says `gaze_jump = on`; the person did nothing wrong.
        for text in ["gaze_jump = on\nblink_click = off\n", "  gaze_jump=OFF # old\nblink_click = off\n", "gaze_jump = maybe\nblink_click = off\n"] {
            let (settings, warnings) = parse(text);
            assert_eq!(warnings, Vec::<String>::new(), "{text}");
            assert!(!settings.tuning.blink_click, "the lines after it still count: {text}");
        }
    }

    #[test]
    fn values_are_read_past_comments_spacing_and_a_byte_order_mark() {
        let text = "\u{feff}camera=2 # the second camera\r\n\r\n  blink_click = no\nhead_speed_slow = 400\n\
                    head_speed_fast=1500.5\njump_distance = 300\nblink_hold = 0.6\nblink_limit = 2\nblink_closed = 0.4\n";
        let (settings, warnings) = parse(text);
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(settings.camera, 2);
        let tuning = settings.tuning;
        assert!(!tuning.blink_click);
        assert_eq!((tuning.head_gain.slow, tuning.head_gain.fast), (400.0, 1500.5));
        assert_eq!(tuning.jump_distance, 300.0);
        assert_eq!((tuning.blink.hold, tuning.blink.limit, tuning.blink.closed_ratio), (0.6, 2.0, 0.4));
        assert!((tuning.blink.open_ratio - 0.5).abs() < 1e-6, "the open share follows the shut share");
    }

    #[test]
    fn the_pointing_mode_accepts_its_three_words_and_rejects_anything_else() {
        let (settings, warnings) = parse("pointing = Hybrid\n");
        assert_eq!((settings.tuning.pointing, warnings.len()), (PointingMode::Hybrid, 0));
        let (settings, warnings) = parse("pointing = eye\neye_steadiness = 3\n");
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(settings.tuning.pointing, PointingMode::Eye);
        assert_eq!(settings.tuning.eye_steadiness, 3.0);
        let (settings, warnings) = parse("pointing = HEAD\n");
        assert_eq!((settings.tuning.pointing, warnings.len()), (PointingMode::Head, 0));
        let (settings, warnings) = parse("pointing = gaze\n");
        assert_eq!((settings.tuning.pointing, warnings.len()), (PointingMode::Head, 1));
        assert!(warnings[0].contains("head か eye か hybrid"), "{}", warnings[0]);
    }

    #[test]
    fn setting_the_pointing_mode_in_a_text_changes_that_line_and_nothing_else() {
        // The template's own line, among everything else it says.
        let template = template();
        let eye = with_pointing(&template, PointingMode::Eye);
        assert_eq!(eye, template.replace("\npointing = head\n", "\npointing = eye\n"));
        assert_eq!(with_pointing(&eye, PointingMode::Head), template, "and back again");

        // A person's own file: spacing, a comment on the line, Windows line ends, a mark in front.
        let own = "\u{feff}  pointing=head   # what moves the cursor\r\ncamera = 2\r\n# pointing = eye would be the other\r\n";
        let changed = with_pointing(own, PointingMode::Eye);
        assert_eq!(changed, "\u{feff}  pointing = eye # what moves the cursor\r\ncamera = 2\r\n# pointing = eye would be the other\r\n");

        // A file from before the setting existed gets the line at its end, in its own line ends.
        assert_eq!(with_pointing("camera = 1", PointingMode::Eye), "camera = 1\npointing = eye\n");
        assert_eq!(with_pointing("camera = 1\r\n", PointingMode::Eye), "camera = 1\r\npointing = eye\r\n");
        assert_eq!(with_pointing("", PointingMode::Head), "pointing = head\n");

        // Whatever the text was, reading the result gives the mode asked for and the rest as before.
        // A mark anywhere but at the very start makes a line `parse` cannot use: it is left
        // alone and the setting goes on a line of its own.
        let marked = "camera = 1\n\u{feff}pointing = head\n";
        assert_eq!(with_pointing(marked, PointingMode::Eye), format!("{marked}pointing = eye\n"));
        let texts = [template.as_str(), "camera = 1", "pointing = eye\npointing = head\ncamera = 3\n", "pointing = nonsense\n"];
        for text in texts.into_iter().chain([marked, "  \u{feff}pointing = head\n", "\u{feff}\u{feff}pointing = head"]) {
            for mode in [PointingMode::Head, PointingMode::Eye, PointingMode::Hybrid] {
                let (before, _) = parse(text);
                let (after, warnings) = parse(&with_pointing(text, mode));
                assert_eq!(after.tuning.pointing, mode, "{text:?}");
                assert_eq!(after.camera, before.camera, "{text:?}");
                // The line with a misplaced mark is still there, and still reported.
                let unusable = text.contains("\n\u{feff}") || text.starts_with(' ');
                assert_eq!(warnings.len(), usize::from(unusable), "{text:?}: {warnings:?}");
            }
        }
    }

    #[test]
    fn the_head_speeds_a_calibration_fits_replace_their_lines_and_read_back() {
        let gain = HeadGain { slow: 1_272.894_7, fast: 3_818.683_8, steadiness: 1.5 };
        let template = template();
        let fitted = with_head_gain(&template, &gain);
        let expected = template
            .replace("\nhead_speed_slow = 700\n", "\nhead_speed_slow = 1273\n")
            .replace("\nhead_speed_fast = 2200\n", "\nhead_speed_fast = 3819\n")
            .replace("\nhead_steadiness = 2\n", "\nhead_steadiness = 1.5\n");
        assert_ne!(expected, template, "the template has the three lines");
        assert_eq!(fitted, expected, "the three lines change and nothing else");
        let (settings, warnings) = parse(&fitted);
        assert_eq!(warnings, Vec::<String>::new());
        assert_eq!(settings.tuning.head_gain, HeadGain { slow: 1273.0, fast: 3819.0, steadiness: 1.5 });

        // A person's own file keeps its comments and its other lines; missing lines go at the end.
        let own = "camera = 1\r\nhead_speed_fast=900 # mine\r\n";
        assert_eq!(
            with_head_gain(own, &gain),
            "camera = 1\r\nhead_speed_fast = 3819 # mine\r\nhead_speed_slow = 1273\r\nhead_steadiness = 1.5\r\n"
        );
        // A setting whose name only begins like another is not that setting.
        assert_eq!(with_setting("head_speed = 5\n", "head_speed_slow", "9"), "head_speed = 5\nhead_speed_slow = 9\n");
    }

    #[test]
    fn an_empty_file_means_the_defaults() {
        assert_eq!(parse(""), (Settings::default(), Vec::new()));
        assert_eq!(parse("# nothing but a comment\n\n"), (Settings::default(), Vec::new()));
    }

    #[test]
    fn a_line_that_cannot_be_used_is_reported_and_leaves_the_default() {
        let defaults = Settings::default();
        for (text, mentions) in [
            ("head_speed_slow = fast", "head_speed_slow"),
            ("head_speed_slow = NaN", "head_speed_slow"),
            ("blink_click = maybe", "blink_click"),
            ("camera = -1", "camera"),
            ("camera = 1.5", "camera"),
            ("mouse_speed = 3", "mouse_speed"),
            ("just some words", "just some words"),
        ] {
            let (settings, warnings) = parse(&format!("blink_hold = 0.5\n{text}\n"));
            assert_eq!(warnings.len(), 1, "{text}: {warnings:?}");
            assert!(warnings[0].starts_with("2行目") && warnings[0].contains(mentions), "{}", warnings[0]);
            assert_eq!(settings.camera, defaults.camera, "{text}");
            assert_eq!(settings.tuning.head_gain, defaults.tuning.head_gain, "{text}");
            assert_eq!(settings.tuning.blink_click, defaults.tuning.blink_click, "{text}");
            assert_eq!(settings.tuning.blink.hold, 0.5, "the usable line still counts");
        }
    }

    #[test]
    fn numbers_outside_their_range_are_pulled_in_and_reported() {
        let (settings, warnings) = parse(
            "camera = 99\nhead_speed_slow = 1\njump_distance = 1e9\nblink_hold = 0\nblink_closed = 0.95\nhead_speed_fast = inf\n",
        );
        assert_eq!(warnings.len(), 6, "{warnings:?}");
        assert_eq!(settings.camera, 31);
        assert_eq!(settings.tuning.head_gain.slow, 50.0);
        assert_eq!(settings.tuning.jump_distance, 5000.0);
        assert_eq!(settings.tuning.blink.hold, 0.15);
        assert_eq!(settings.tuning.blink.closed_ratio, 0.8);
        assert!((settings.tuning.blink.open_ratio - 0.9).abs() < 1e-6, "below the eye's full size at the largest shut share");
        assert_eq!(settings.tuning.head_gain.fast, Settings::default().tuning.head_gain.fast, "infinity is not a number");
    }

    #[test]
    fn settings_that_depend_on_each_other_are_made_consistent_in_either_order() {
        for text in ["head_speed_fast = 300\nhead_speed_slow = 900\n", "head_speed_slow = 900\nhead_speed_fast = 300\n"] {
            let (settings, warnings) = parse(text);
            assert_eq!((settings.tuning.head_gain.slow, settings.tuning.head_gain.fast), (900.0, 900.0));
            assert_eq!(warnings.len(), 1, "{warnings:?}");
        }
        for text in ["blink_limit = 1\nblink_hold = 2\n", "blink_hold = 2\nblink_limit = 1\n"] {
            let (settings, warnings) = parse(text);
            assert_eq!((settings.tuning.blink.hold, settings.tuning.blink.limit), (2.0, 2.3));
            assert_eq!(warnings.len(), 1, "{warnings:?}");
        }
    }
}
