param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [string]$Prompt = "A chair in a white room sitting on wooden floor",
    [string]$Output = "output_generated.ppm",
    [int]$Width = 256,
    [int]$Height = 256
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
$OutputPath = [System.IO.Path]::GetFullPath((Join-Path $Root $Output))

if (-not (Test-Path $Executable)) {
    throw "Missing $Executable. Run TRAIN_IMAGEGEN_6GB_WINDOWS.bat first."
}
if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint $Checkpoint. Run TRAIN_IMAGEGEN_6GB_WINDOWS.bat first."
}

if (Test-Path $OutputPath) {
    Remove-Item -Force $OutputPath
}

Write-Host "Generating image for prompt: '$Prompt'..."
& $Executable imagegen-generate `
    --checkpoint $Checkpoint `
    --profile preview-6g `
    --backend $Backend `
    --architecture resonant-fabric `
    --prompt $Prompt `
    --output $OutputPath `
    --width $Width `
    --height $Height

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Successfully generated image: $OutputPath"
} else {
    exit $LASTEXITCODE
}
