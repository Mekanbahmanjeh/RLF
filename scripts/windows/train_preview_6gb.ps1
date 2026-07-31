param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$AssistantName = "Aurora",
    [string]$OwnerName = "RLF Explorer",
    [string]$Checkpoint = "",
    [switch]$DownloadHF,
    [int]$MaxSamples = 10000,
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
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-release\Release\solstice.exe")
    $CandidatePaths += (Join-Path $Root "build\windows-vs2026-release\solstice.exe")
}

$Executable = $CandidatePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Executable) {
    $Executable = Join-Path $Root "build\$Preset\Release\solstice.exe"
}

$DataDirectory = Join-Path $Root "demo_data\preview_conversation"
if ($Checkpoint -eq "") {
    $Checkpoint = Join-Path $Root "models\preview_conversation_6gb.rlfsp"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)

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

& (Join-Path $PSScriptRoot "create_preview_conversation_data.ps1") `
    -OutputDirectory $DataDirectory `
    -AssistantName $AssistantName `
    -OwnerName $OwnerName

if ($DownloadHF) {
    Write-Host "Downloading and processing Hugging Face CoT reasoning datasets..."
    $HFScript = Join-Path $Root "scripts\download_and_build_hf_cot_dataset.py"
    python $HFScript --output-dir $DataDirectory --max-samples $MaxSamples
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}


$global:LASTEXITCODE = 0

[System.IO.Directory]::CreateDirectory(
    [System.IO.Path]::GetDirectoryName($Checkpoint)
) | Out-Null
if (Test-Path $Checkpoint) {
    if ($Reset) {
        Write-Host "Reset requested. Removing existing checkpoint: $Checkpoint"
        Remove-Item -Force $Checkpoint
    } else {
        Write-Host "Overwriting existing checkpoint: $Checkpoint"
        Remove-Item -Force $Checkpoint
    }
}

$Common = @(
    "--checkpoint", $Checkpoint,
    "--profile", "preview-6g",
    "--backend", $Backend,
    "--enforce-profile"
)

Write-Host "Checking preview profile and backend"
& $Executable profile-info --profile preview-6g --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Executable device-info --profile preview-6g --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training blank tokenizer and text fabric"
& $Executable train-text @Common --blank --input (Join-Path $DataDirectory "corpus.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training conversation episodes"
& $Executable train-dialogue @Common --manifest (Join-Path $DataDirectory "dialogues.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training instruction demonstrations with CoT rationales (<think> blocks)"
& $Executable train-instructions @Common --manifest (Join-Path $DataDirectory "instructions.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training preferences"
& $Executable train-preferences @Common --manifest (Join-Path $DataDirectory "preferences.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Training explicit facts"
& $Executable train-facts @Common --manifest (Join-Path $DataDirectory "facts.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Verifying checkpoint"
& $Executable verify-checkpoint @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Executable inspect-checkpoint @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Training complete: $Checkpoint"
Write-Host "Start conversation on your 6GB VRAM GPU with:"
Write-Host "  CHAT_PREVIEW_6GB_WINDOWS.bat -Backend $Backend"
