param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [string]$Prompt = "A red ball rolling across a wooden floor",
    [string]$Output = "output_video",
    [int]$Frames = 16
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
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Root $Output))

if (-not (Test-Path $Executable)) {
    throw "Missing $Executable. Run TRAIN_VIDEOGEN_6GB_WINDOWS.bat first."
}
if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint $Checkpoint. Run TRAIN_VIDEOGEN_6GB_WINDOWS.bat first."
}

if (Test-Path $OutputPath) {
    Remove-Item -Recurse -Force $OutputPath
}

Write-Host "Rendering video motion frames for prompt: '$Prompt'..."
& $Executable generate-video `
    --checkpoint $Checkpoint `
    --profile preview-6g `
    --backend $Backend `
    --prompt $Prompt `
    --frames $Frames `
    --output $OutputPath

if ($LASTEXITCODE -eq 0) {
    # Convert generated frame PPM files into an animated GIF for easy viewing in Windows Photos
    $GifPath = $OutputPath + ".gif"
    python -c "
import glob, os
from PIL import Image

frame_files = sorted(glob.glob(os.path.join(r'$OutputPath', 'frames', '*.ppm')))
if not frame_files:
    frame_files = sorted(glob.glob(os.path.join(r'$OutputPath', '*.ppm')))

if frame_files:
    images = [Image.open(f) for f in frame_files]
    images[0].save(r'$GifPath', save_all=True, append_images=images[1:], duration=100, loop=0)
    print(f'[+] Created animated video GIF: $GifPath')
" 2>$null

    Write-Host ""
    Write-Host "Successfully generated video frames in: $OutputPath"
    if (Test-Path $GifPath) {
        Write-Host "Play animated video GIF: $GifPath"
    }
} else {
    exit $LASTEXITCODE
}
