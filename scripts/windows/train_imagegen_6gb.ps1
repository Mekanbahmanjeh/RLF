param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [switch]$DownloadHF,
    [int]$MaxSamples = 1000,
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
    $Checkpoint = Join-Path $Root "models\preview_imagegen_6gb.rlfimg"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)
$DataDirectory = Join-Path $Root "demo_data\imagegen_pairs"

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

# Generate/fetch image dataset
$ImgHFScript = Join-Path $Root "scripts\download_and_build_hf_image_dataset.py"
python $ImgHFScript --output-dir $DataDirectory --max-samples $MaxSamples
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

if (-not (Test-Path $Checkpoint)) {
    Write-Host "Bootstrapping image generation checkpoint: $Checkpoint"
    & $Executable imagegen-bootstrap --checkpoint $Checkpoint --profile preview-6g --backend $Backend
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Training ImageGen prompt-image manifest..."
$ManifestFile = Join-Path $DataDirectory "manifest.tsv"
& $Executable imagegen-train-manifest `
    --checkpoint $Checkpoint `
    --profile preview-6g `
    --backend $Backend `
    --architecture resonant-fabric `
    --manifest $ManifestFile

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Verifying ImageGen checkpoint..."
& $Executable imagegen-verify --checkpoint $Checkpoint --profile preview-6g --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "ImageGen Training Complete: $Checkpoint"
Write-Host "Generate an image with:"
Write-Host "  .\GENERATE_IMAGE_6GB_WINDOWS.bat -Prompt 'A chair in a white room' -Output chair.png"
