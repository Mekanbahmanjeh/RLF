param(
    [ValidateSet("cuda", "optimized_cpu", "scalar_cpu")]
    [string]$Backend = "cuda",
    [string]$Checkpoint = "",
    [int]$MaxSamples = 500000,
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
    $Checkpoint = Join-Path $Root "models\frontier_24g_master.rlfsp"
}
$Checkpoint = [System.IO.Path]::GetFullPath($Checkpoint)
$DataDirectory = Join-Path $Root "demo_data\frontier_24h"

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
Write-Host "Preparing 500,000 Row CoT Reasoning Dataset..."

# Ingest/download dataset
$DSFetchScript = Join-Path $Root "scripts\download_and_build_24h_frontier_dataset.py"
python $DSFetchScript --output-dir $DataDirectory --max-samples $MaxSamples
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Also merge base dialogue rows into dialogues.tsv
& (Join-Path $PSScriptRoot "create_preview_conversation_data.ps1") `
    -OutputDirectory $DataDirectory `
    -AssistantName "Aurora" `
    -OwnerName "RLF Explorer"

[System.IO.Directory]::CreateDirectory(
    [System.IO.Path]::GetDirectoryName($Checkpoint)
) | Out-Null

if (Test-Path $Checkpoint) {
    if ($Reset) {
        Write-Host "Reset requested. Removing existing checkpoint: $Checkpoint"
        Remove-Item -Force $Checkpoint
    }
}

$Common = @(
    "--checkpoint", $Checkpoint,
    "--profile", "frontier-24g",
    "--backend", $Backend,
    "--enforce-profile"
)

Write-Host "================================================================="
Write-Host " Starting RLF 24-Hour Frontier 24G Training Campaign ($Backend) "
Write-Host " Profile: frontier-24g (65,536 Vocab, 2M Episodes, 20M Contexts) "
Write-Host "================================================================="

& $Executable profile-info --profile frontier-24g --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Executable device-info --profile frontier-24g --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 1: Training 65,536 BPE Vocabulary & Text Fabric..."
& $Executable train-text @Common --blank --input (Join-Path $DataDirectory "corpus.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 2: Training Dialogue Attractor Episodes..."
& $Executable train-dialogue @Common --manifest (Join-Path $DataDirectory "dialogues.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 3: Training 500,000 CoT Instruction Demonstrations (<think> blocks)..."
& $Executable train-instructions @Common --manifest (Join-Path $DataDirectory "instructions.tsv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Step 4: Verifying Master 24G Checkpoint..."
& $Executable verify @Common
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "================================================================="
Write-Host " 24-Hour Frontier 24G Training Campaign Complete!              "
Write-Host " Master Checkpoint: $Checkpoint                                "
Write-Host " Start Chat with:                                              "
Write-Host "   .\CHAT_FRONTIER_24G_WINDOWS.bat -Backend cuda              "
Write-Host "================================================================="
