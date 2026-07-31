param(
    [switch]$Cuda,
    [switch]$SkipTests,
    [string]$BootstrapCheckpoint = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows\select_windows_preset.ps1")
$Backend = if ($Cuda) { "cuda" } else { "optimized_cpu" }
$Preset = Get-RlfWindowsPreset $Backend

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw @"
CMake was not found on PATH.
Install CMake 3.25 or newer, enable its 'Add CMake to PATH' option, reopen
PowerShell, and run this command again.
"@
}

$CMakeVersionText = (cmake --version | Select-Object -First 1) -replace '^cmake version ', ''
$CMakeVersion = [version]$CMakeVersionText
if ($Preset.StartsWith("windows-vs2026-") -and $CMakeVersion -lt [version]"4.2.0") {
    throw @"
Visual Studio 2026 was found, but CMake $CMakeVersion is too old for its
'Visual Studio 18 2026' generator. Install CMake 4.2 or newer, ensure the new
cmake.exe is first on PATH, reopen PowerShell, and run the command again.
"@
}

Write-Host "Selected Windows preset: $Preset"
Write-Host "Configuring $Preset"
cmake --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Building $Preset"
cmake --build --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipTests) {
    Write-Host "Running tests"
    ctest --preset $Preset
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Executable = Join-Path $PSScriptRoot "..\build\$Preset\Release\solstice.exe"
$Executable = [System.IO.Path]::GetFullPath($Executable)
Write-Host "Solstice executable: $Executable"

if ($BootstrapCheckpoint -ne "") {
    $Checkpoint = [System.IO.Path]::GetFullPath($BootstrapCheckpoint)
    Write-Host "Creating bootstrap checkpoint: $Checkpoint"
    & $Executable bootstrap --checkpoint $Checkpoint
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
