function Get-RlfVisualStudioMajor {
    if (-not [string]::IsNullOrWhiteSpace($env:VSCMD_VER)) {
        $DeveloperShellMajor = [int]($env:VSCMD_VER.Split(".")[0])
        if ($DeveloperShellMajor -in @(18, 17)) {
            return $DeveloperShellMajor
        }
    }

    $VsWhereCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"),
        (Join-Path $env:ProgramFiles "Microsoft Visual Studio\Installer\vswhere.exe")
    ) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
    $VsWhere = $VsWhereCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($VsWhere) {
        foreach ($Major in @(18, 17)) {
            $InstallationPath = & $VsWhere `
                -latest `
                -products * `
                -version "[$Major.0,$($Major + 1).0)" `
                -property installationPath
            if (-not [string]::IsNullOrWhiteSpace($InstallationPath)) {
                $DeveloperCommand = Join-Path $InstallationPath "Common7\Tools\VsDevCmd.bat"
                if (Test-Path $DeveloperCommand) {
                    return $Major
                }
            }
        }
    }

    $VisualStudio2026Root = Join-Path $env:ProgramFiles "Microsoft Visual Studio\18"
    if (Test-Path $VisualStudio2026Root) {
        $VisualStudio2026DeveloperCommand = Get-ChildItem `
            -Path $VisualStudio2026Root `
            -Filter VsDevCmd.bat `
            -File `
            -Recurse `
            -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($VisualStudio2026DeveloperCommand) {
            return 18
        }
    }

    throw @"
No supported Visual Studio C++ toolchain was found.
Install Visual Studio 2026 or 2022 with the 'Desktop development with C++'
workload, an x64/x86 MSVC toolset, a Windows SDK, and CMake tools.

As a direct workaround, open 'Developer PowerShell for Visual Studio 2026'
from the Start menu and run the training command there.
"@
}

function Get-RlfWindowsPreset([string]$Backend) {
    $VisualStudioMajor = Get-RlfVisualStudioMajor
    $Suffix = if ($Backend -eq "cuda") { "cuda-release" } else { "release" }
    if ($VisualStudioMajor -eq 18) {
        return "windows-vs2026-$Suffix"
    }
    return "windows-msvc-$Suffix"
}
