# Compile installer\setup.iss into installer\out\eye-tracking-mouse-vX.Y.Z-setup.exe.
# Usage (after the release build):
#   pwsh -File scripts\build-installer.ps1 [-IsccPath <ISCC.exe>] [-PayloadDir <dir>]
# The version comes from Cargo.toml, so the installer can never carry a
# different version than the binary it ships.
[CmdletBinding()]
param(
  [string]$IsccPath,
  [string]$PayloadDir
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $PayloadDir) { $PayloadDir = Join-Path $root 'target\release' }
$PayloadDir = [IO.Path]::GetFullPath($PayloadDir)

# `\r?$`: the file is stored with LF but a Windows checkout may hand us CRLF
$m = [regex]::Match([IO.File]::ReadAllText((Join-Path $root 'Cargo.toml')), '(?m)^version = "([0-9]+\.[0-9]+\.[0-9]+)"\r?$')
if (-not $m.Success) { throw "no version in Cargo.toml" }
$version = $m.Groups[1].Value

$exe = Join-Path $PayloadDir 'eye-tracking-mouse.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "payload missing: $exe (cargo build --release first)" }

function Resolve-Iscc {
  if ($IsccPath) {
    if (-not (Test-Path -LiteralPath $IsccPath -PathType Leaf)) { throw "ISCC is missing: $IsccPath" }
    return [IO.Path]::GetFullPath($IsccPath)
  }
  $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $candidates = @(
    $(if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe' }),
    $(if (${env:ProgramFiles(x86)}) { Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe' }),
    $(if ($env:ProgramFiles) { Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe' })
  ) | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) }
  if ($candidates) { return @($candidates)[0] }
  throw 'Inno Setup 6 compiler (ISCC.exe) was not found (winget install JRSoftware.InnoSetup)'
}
# setup.iss itself refuses an Inno Setup older than 6.6 (ISCC.exe has no
# version resource to check here)
$iscc = Resolve-Iscc

$out = Join-Path $root "installer\out\eye-tracking-mouse-v$version-setup.exe"
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out }
& $iscc /Q "/DAppVersion=$version" "/DPayloadDir=$PayloadDir" (Join-Path $root 'installer\setup.iss')
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }
if (-not (Test-Path -LiteralPath $out -PathType Leaf)) { throw "ISCC reported success but $out is missing" }

# Get-FileHash is missing on some stripped-down PowerShell installs; hash via .NET
$sha = [Security.Cryptography.SHA256]::Create()
$stream = [IO.File]::OpenRead($out)
try { $hash = ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
finally { $stream.Dispose(); $sha.Dispose() }
Write-Host "$iscc -> $out"
Write-Host "sha256 $hash  $((Get-Item -LiteralPath $out).Length) bytes"
