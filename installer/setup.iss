; eye-tracking-mouse installer (Inno Setup 6.6+). Built by
; scripts\build-installer.ps1, which passes AppVersion from Cargo.toml and
; checks that the payload exists.
;
; Modelled on disk-manage's installer\setup.iss. What differs, and why:
; - Per-user install (PrivilegesRequired=lowest): the program is a tray
;   resident that injects input as the signing-in user; nothing it does
;   needs or benefits from admin rights, so Setup does not ask for them.
;   With {autopf} this lands in %LOCALAPPDATA%\Programs. The Program Files
;   confinement checks disk-manage carries exist only because its GUI runs
;   elevated and launches sibling tools -- not the case here.
; - One file of payload: the model is embedded in the exe (include_bytes!),
;   so the installer ships the exe, the icon, and the legal/readme texts.
; - Upgrades: the resident exe keeps its own image locked while running, so
;   CloseApplications asks the Restart Manager to shut it down first. The
;   user's files under %APPDATA%\eye-tracking-mouse are never touched by
;   install or uninstall.

; `WizardStyle=modern dynamic` (follows the Windows light/dark theme)
#if Ver < EncodeVer(6, 6, 0)
  #error Inno Setup 6.6 or later is required
#endif
#ifndef AppVersion
  #error AppVersion is not defined: build with scripts\build-installer.ps1
#endif
#ifndef PayloadDir
  #define PayloadDir "..\target\release"
#endif

[Languages]
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Messages]
ja.WelcomeLabel1=eye-tracking-mouse のセットアップ
ja.WelcomeLabel2=カメラでマウスを操作する eye-tracking-mouse をインストールします。%n%n視線でカーソルをジャンプさせ、頭の動きで微調整し、まばたきでクリックします。
ja.FinishedHeadingLabel=eye-tracking-mouse の準備ができました
ja.FinishedLabel=eye-tracking-mouse のインストールが完了しました。スタートメニューから起動できます。起動すると通知領域にアイコンが表示されます。
ja.FinishedLabelNoIcons=eye-tracking-mouse のインストールが完了しました。

en.WelcomeLabel1=Set up eye-tracking-mouse
en.WelcomeLabel2=This installs eye-tracking-mouse: a camera-driven mouse for Windows.%n%nLook at a spot and the cursor jumps there, small head movements nudge it the rest of the way, and a deliberate long blink clicks.
en.FinishedHeadingLabel=eye-tracking-mouse is ready
en.FinishedLabel=eye-tracking-mouse was installed. Start it from the Start menu; its icon appears in the notification area.
en.FinishedLabelNoIcons=eye-tracking-mouse was installed.

[CustomMessages]
ja.AppDescription=カメラでマウスを操作 (常駐)
ja.LaunchApp=eye-tracking-mouse を起動する
en.AppDescription=Camera-driven mouse (notification area)
en.LaunchApp=Start eye-tracking-mouse

[Setup]
; AppId keys upgrade detection and the Add/Remove Programs entry. Generated
; once; changing it makes the next version install side by side.
AppId={{1164F72D-1C86-4862-A25B-B0082BF269E3}
AppName=eye-tracking-mouse
AppVersion={#AppVersion}
AppVerName=eye-tracking-mouse {#AppVersion}
AppPublisher=tsuyoshi-otake
AppPublisherURL=https://github.com/tsuyoshi-otake/eye-tracking-mouse
AppSupportURL=https://github.com/tsuyoshi-otake/eye-tracking-mouse/issues
AppUpdatesURL=https://github.com/tsuyoshi-otake/eye-tracking-mouse/releases
VersionInfoVersion={#AppVersion}
VersionInfoProductVersion={#AppVersion}
VersionInfoDescription=eye-tracking-mouse Setup
LanguageDetectionMethod=uilanguage
ShowLanguageDialog=no
LicenseFile=..\LICENSES\Apache-2.0.txt
; lowest: per-user install, no elevation. {autopf} resolves to
; %LOCALAPPDATA%\Programs in this mode.
DefaultDirName={autopf}\eye-tracking-mouse
DefaultGroupName=eye-tracking-mouse
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 11's first public build: the in-box Windows ML must be able to load
; an ONNX opset 13 model (README "Requirements").
MinVersion=10.0.22000
OutputDir=out
OutputBaseFilename=eye-tracking-mouse-v{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\icon.ico
UninstallDisplayName=eye-tracking-mouse
WizardStyle=modern dynamic
DisableWelcomePage=no
; the resident exe is closed through the Restart Manager on upgrade
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#PayloadDir}\eye-tracking-mouse.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; DestName: "README-ja.md"; Flags: ignoreversion
Source: "..\LICENSES\Apache-2.0.txt"; DestDir: "{app}"; DestName: "LICENSE-Apache-2.0.txt"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\eye-tracking-mouse"; Filename: "{app}\eye-tracking-mouse.exe"; IconFilename: "{app}\icon.ico"; Comment: "{cm:AppDescription}"
Name: "{autodesktop}\eye-tracking-mouse"; Filename: "{app}\eye-tracking-mouse.exe"; IconFilename: "{app}\icon.ico"; Comment: "{cm:AppDescription}"; Tasks: desktopicon

[Run]
; The tray resident runs unelevated; runasoriginaluser also keeps it that way
; when Setup itself was started with admin rights.
Filename: "{app}\eye-tracking-mouse.exe"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent runasoriginaluser
