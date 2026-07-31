param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [int]$MaxTokens = 512
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
    $Checkpoint = Join-Path $Root "models\frontier_24g_master.rlfsp"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)

if (-not (Test-Path $Checkpoint)) {
    throw "Missing checkpoint: $Checkpoint. Run TRAIN_FRONTIER_24G_24H_WINDOWS.bat first."
}

Write-Host "================================================================="
Write-Host " Launching Interactive Chat — RLF Frontier 24G ($Backend)       "
Write-Host "================================================================="

& $Executable chat `
    --checkpoint $Checkpoint `
    --profile frontier-24g `
    --backend $Backend `
    --enforce-profile `
    --no-tools `
    --max-tokens $MaxTokens
