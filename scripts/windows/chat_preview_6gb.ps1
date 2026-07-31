param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [int]$MaximumResponseTokens = 96
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
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-release\Release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-release\solstice.exe")
}

$Executable = $CandidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Executable) {
    $Executable = Join-Path $Root "build\$Preset\Release\solstice.exe"
}

if ($Checkpoint -eq "") {
    $Checkpoint = Join-Path $Root "models\preview_conversation_6gb.rlfsp"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)

if (-not (Test-Path $Executable)) {
    throw "Missing $Executable. Run TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend $Backend first."
}
if (-not (Test-Path $Checkpoint)) {
    throw "Missing $Checkpoint. Run TRAIN_PREVIEW_6GB_WINDOWS.bat -Backend $Backend first."
}
if ($MaximumResponseTokens -le 0) {
    throw "MaximumResponseTokens must be positive."
}

& $Executable chat `
    --checkpoint $Checkpoint `
    --profile preview-6g `
    --backend $Backend `
    --enforce-profile `
    --no-tools `
    --max-tokens $MaximumResponseTokens
exit $LASTEXITCODE
