# eye-tracking-mouse

Camera-driven mouse for Windows. Look at a spot and the cursor jumps there, small
head movements nudge it the rest of the way, and a deliberate long blink clicks.

Native Rust on Win32: one executable, no GUI, imaging or ML framework. The only
direct dependency is Microsoft's `windows` crate. Capture is Media Foundation,
the face landmark model runs on the Windows ML that ships with Windows, input is
`SendInput`, and drawing is GDI.

## Status

Everything that can be checked without a person in front of the camera is checked
automatically (see [Checks](#checks)). Not exercised yet: the live loop with a real
face. Expect to adjust the speeds and the blink times in the settings file, and
treat these as untested until you have tried them:

- a calibration run from start to finish, and cancelling one that is under way
- pausing in the middle of a calibration run
- the preview with real camera pictures
- how pointing and clicking feel

## Requirements

- Windows 11, x64. Developed and tested on build 26300. An older Windows whose
  built-in Windows ML cannot load an ONNX opset 13 model will not work;
  `--selftest model` tells.
- A USB camera that delivers 1280x720.
- Rust 1.85 or newer with the MSVC toolchain, to build.

## Build

```
cargo build --release
```

The program is `target\release\eye-tracking-mouse.exe`. The model is embedded in it;
nothing else needs to be installed or copied.

## Run

Start `eye-tracking-mouse.exe`. It has no window: an eye-shaped icon appears in the
notification area (look under the `^` if it is not visible) and a notification
says that it has started. The icon's colour and its tooltip show what the program
is doing.

| What you do | What happens |
|---|---|
| Turn or shift your head a little | The cursor moves. A quick movement travels farther than a slow one. |
| Close both eyes for about 0.4 s, until the voice says "どうぞ", then open them | Left click where the cursor was when the eyes closed; the voice says "クリックしました". |
| Keep the eyes closed for more than 1.5 s | The click is dropped; the voice says "取り消しました". |
| Look far down, so that the eyes narrow as if shut | "どうぞ" and then "取り消しました", no click; the cursor moves again about 3 s later. Blink clicks do not work while looking that far down. |

The phrases are spoken by the Japanese voice that ships with Windows; without one, three tones (high, higher, low) stand in for them.
| Rest your gaze on a place far from the cursor (after calibration) | The cursor jumps there. |
| Move the real mouse | The program lets go at once and takes over again 1.5 s after you stop. |

Pointing with the head and clicking work straight away. The gaze jump needs a
[calibration](#calibration) first.

### Hotkeys and menu

| Key | Menu item | |
|---|---|---|
| Ctrl+Alt+P | 一時停止 / 再開 | Pause and resume. While paused the camera is released. |
| Ctrl+Alt+C | キャリブレーション / キャリブレーションを中止 | Start a calibration run, or cancel the one under way. |
| | カメラのプレビュー | Show or hide what the camera sees, with the points found on your face. |
| | 設定ファイルを開く | Open the settings file in Notepad. |
| Ctrl+Alt+Q | 終了 | Quit. |

Right-click or left-click the icon for the menu. If another program already uses
one of the key combinations, a notification says so and the menu item still works.

## Calibration

A run takes about half a minute.

1. Sit the way you will sit when you use the program, and put the mouse pointer on
   the screen you want to calibrate. The run is for the screen the pointer is on.
2. Press Ctrl+Alt+C. A dark page covers that screen and tells you what to do.
   (This also ends a pause.)
3. Look at the dot in the middle, eyes open, head still, until it moves on.
4. Follow the dot with your eyes only, head still. It visits the middle and eight
   places around the edge; the ring around it closes while a place is recorded.
5. Look at the dot in the middle again and slowly turn your head left and right,
   up and down, until the bar at the bottom is full.
6. The page closes by itself. A notification says that the calibration is done and
   how far off it typically is in pixels, or why it did not work.

Esc or Ctrl+Alt+C cancels a run. A run also ends by itself, saying why, when the
face is out of view for 5 seconds, a step takes longer than 25 seconds, the camera
stops delivering, the monitors change, or you pause.

The result is kept in `calibration.txt` and used again at the next start.
Calibrate again when you move the camera, the monitor or your seat.

## Stopping

- Quit: Ctrl+Alt+Q, 終了 in the icon's menu, or `eye-tracking-mouse.exe --quit`.
- Get the cursor back at once without quitting: move the real mouse, or press
  Ctrl+Alt+P to pause.

## Settings and files

Everything is kept in `%APPDATA%\eye-tracking-mouse\` (or the folder given with
`--data-dir`):

| File | |
|---|---|
| `settings.txt` | Written with every setting at its default and an explanation the first time the program starts. Read at start only: quit and start again after changing it. |
| `calibration.txt` | The last successful calibration: numbers describing your face shape and how your gaze maps to the screen. |
| `log.txt` | What the program did since it was last started. Text only. |

`settings.txt` sets the camera number, switches the gaze jump and the blink click
on or off, and tunes the cursor speed for slow and quick head movements
(`head_speed_slow`, `head_speed_fast`), the shortest gaze jump (`jump_distance`), and
the blink click (`blink_hold`, `blink_limit`, `blink_closed`).

## Command line

```
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
```

Exit code: 0 done; 1 not done (a check failed, already running, nothing to quit);
2 an error.

The program is built without a console, so PowerShell neither waits for it nor
shows what it prints unless the output is piped:

```
.\target\release\eye-tracking-mouse.exe --list-cameras | Out-Host
```

Only one instance runs per desktop session; a second start tells the first one
and ends.

## Checks

```
cargo test                                # unit and simulation tests; no camera, face or desktop needed
cargo clippy --all-targets -- -D warnings
cargo test --test resident -- --ignored   # starts the real program for a second, with a camera that does not exist
```

The last one shows notifications, the preview window and, for a moment, the
calibration page; it never touches the pointer.

Checks of the machine it runs on (add `| Out-Host` in PowerShell):

| Command | What it checks |
|---|---|
| `--selftest model` | The embedded model loads on this Windows and reproduces the reference outputs. |
| `--selftest camera` | The camera delivers 1280x720 pictures fast enough. Turns the camera on for a few seconds. |
| `--selftest pointer` | Cursor moves and a click land exactly where asked. Moves the real cursor for about a second and clicks a window of its own. |
| `--selftest face` | A face is found and measured, from the camera or from `--image`. Prints numbers only. |
| `--diag <seconds>` | Prints what the tracker measures once per second, for adjusting the settings. |

## Known limits

- The gaze estimate is coarse: tens of pixels at best, more toward the corners.
  That is why the gaze only brings the cursor near and the head does the fine work.
- The gaze jump works on the calibrated screen only. The head moves the cursor
  across every screen.
- Left click only: no right click, drag or scroll.
- Windows of programs that run as administrator refuse the injected input while
  this program runs without those rights. The tooltip then says カーソルを動かせません.
- The hotkeys are fixed. On keyboard layouts where AltGr is Ctrl+Alt, characters
  typed with AltGr+P, C or Q are taken while the program runs.
- While the icon's menu is open, the program reacts to nothing but the hotkeys.
- Settings are read at start only.
- Starting at logon is not set up, and has not been tried.

## Privacy

Camera pictures are processed in memory and then dropped. None is written to disk
and none leaves the machine: the program contains no network code. The preview
only shows pictures on the screen. `calibration.txt` holds numbers about your face
and stays in the folder above.

## Source layout

```
src/app       the program: tray icon, worker thread, calibration page, preview, settings, wording
src/platform  the Win32 / WinRT boundary: camera, inference, pointer, windows, drawing, sound
src/control   pure decisions: calibration routine, gaze model, head pointing, blink click
src/vision    pure geometry on face landmarks: crop, head pose, eye features
tools/model   how the model was converted from TFLite to ONNX and compared
```

Dependencies point downward only, and `control` and `vision` never touch the
system, so what they decide is tested without a camera, a face or a desktop.

## Notices

The face landmark model is MediaPipe Face Mesh V2 by Google, Apache License 2.0.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
