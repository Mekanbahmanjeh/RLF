param(
    [switch]$Cuda,
    [switch]$SkipTests,
    [string]$BootstrapCheckpoint = ""
)

$ErrorActionPreference = "Stop"
$Preset = if ($Cuda) { "windows-msvc-cuda-release" } else { "windows-msvc-release" }

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
