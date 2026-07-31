param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [switch]$Reset,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

. (Join-Path $PSScriptRoot "select_windows_preset.ps1")
$Preset = Get-RlfWindowsPreset $Backend

$CandidatePaths = @()
if ($Backend -eq "cuda") {
    $CandidatePaths += (Join-Path $Root "build\windows-msvc-cuda-release\Release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-msvc-cuda-release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-cuda-release\Release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-cuda-release\solstice.exe")
} else {
    $CandidatePaths += (Join-Path $Root "build\codex-windows-release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-msvc-release\Release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-msvc-release\solstice.exe")
}

$Executable = $CandidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Executable) {
    $Executable = Join-Path $Root "build\$Preset\Release\solstice.exe"
}

if ($Checkpoint -eq "") {
    $Checkpoint = Join-Path $Root "models\preview_videogen_6gb.rlfsp"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)
$DataDirectory = Join-Path $Root "demo_data\video_sequences"

if (-not (Test-Path $Executable)) {
    if ($SkipBuild) {
        throw "Missing $Executable and -SkipBuild was selected."
    }
    $BuildArguments = @("-SkipTests")
    if ($Backend -eq "cuda") {
        $BuildArguments += "-Cuda"
    }
    Write-Host "Building target preset for $Backend ($Preset)..."
    & (Join-Path $Root "BUILD_WINDOWS.bat") @BuildArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Using Solstice binary: $Executable"

# Generate video dataset
$VidScript = Join-Path $Root "scripts\download_and_build_video_dataset.py"
python $VidScript --output-dir $DataDirectory
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

[System.IO.Directory]::CreateDirectory(
    [System.IO.Path]::GetDirectoryName($Checkpoint)
) | Out-Null

if (Test-Path $Checkpoint) {
    if ($Reset) {
        Write-Host "Reset requested. Removing existing checkpoint: $Checkpoint"
        Remove-Item -Force $Checkpoint
    }
}

Write-Host "Training VideoGen learned motion prototype sequences..."
$ManifestFile = Join-Path $DataDirectory "videos.tsv"

$Common = @(
    "--checkpoint", $Checkpoint,
    "--profile", "preview-6g",
    "--backend", $Backend,
    "--enforce-profile"
)

# First train text/blank tokenizer if new checkpoint
if (-not (Test-Path $Checkpoint)) {
    & $Executable train-text @Common --blank --input (Join-Path $Root "demo_data\preview_conversation\corpus.txt")
}

& $Executable train-video @Common --manifest $ManifestFile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Verifying VideoGen checkpoint..."
& $Executable verify @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "VideoGen Training Complete: $Checkpoint"
Write-Host "Generate motion video with:"
Write-Host "  .\GENERATE_VIDEO_6GB_WINDOWS.bat -Prompt 'A red ball rolling across a wooden floor' -Output output_video"
