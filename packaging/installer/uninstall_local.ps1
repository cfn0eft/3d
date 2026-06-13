# PoseLab Studio local uninstaller. Removes the self-contained install
# directory and the Start Menu / desktop shortcuts created by
# install_local.ps1. With -Cache it also deletes downloaded model caches.
#
# Usage (either one):
#   - double-click packaging\installer\Uninstall-PoseLabStudio.cmd
#   - powershell -ExecutionPolicy Bypass -File packaging\installer\uninstall_local.ps1 [-Cache] [-Yes]
#
# ASCII only on purpose (Windows PowerShell 5.1 misreads non-ASCII files on
# non-English locales). Keep this file ASCII + CRLF.

[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'PoseLab Studio'),
    [switch]$Cache,  # also remove downloaded model caches (~\.cache\poselab, ~\.cache\torch)
    [switch]$Yes     # do not prompt for confirmation
)

$ErrorActionPreference = 'Continue'

$RULE = '  ' + ('-' * 52)
function Write-Banner {
    Write-Host ''
    Write-Host '  PoseLab Studio' -ForegroundColor Cyan
    Write-Host '  Uninstaller' -ForegroundColor DarkGray
    Write-Host $RULE -ForegroundColor DarkGray
}
function Write-Step($m) {
    Write-Host ''
    Write-Host '  > ' -ForegroundColor Cyan -NoNewline
    Write-Host $m -ForegroundColor White
}
function Remove-IfPresent($path, $label) {
    if (Test-Path $path) {
        try {
            Remove-Item -Recurse -Force $path -ErrorAction Stop
            Write-Host "    removed: $label" -ForegroundColor Gray
        } catch {
            Write-Host "    (could not remove $label : $_)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "    not found: $label" -ForegroundColor DarkGray
    }
}

Write-Banner

$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'PoseLab Studio.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) 'PoseLab Studio.lnk'

Write-Host ''
Write-Host '  The following will be removed:' -ForegroundColor White
Write-Host "    - $InstallDir" -ForegroundColor Gray
Write-Host "    - $startMenu" -ForegroundColor Gray
Write-Host "    - $desktop" -ForegroundColor Gray
if ($Cache) {
    Write-Host "    - $env:USERPROFILE\.cache\poselab  (MediaPipe models)" -ForegroundColor Gray
    Write-Host "    - $env:USERPROFILE\.cache\torch    (MMPose / PyTorch weights, shared with other PyTorch apps)" -ForegroundColor Yellow
}

if (-not $Yes) {
    $answer = Read-Host '  Continue? [y/N]'
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host '  Cancelled.' -ForegroundColor Yellow
        exit 0
    }
}

Write-Step 'Removing the application'
Remove-IfPresent $InstallDir 'install directory'

Write-Step 'Removing shortcuts'
Remove-IfPresent $startMenu 'Start Menu shortcut'
Remove-IfPresent $desktop 'desktop shortcut'

if ($Cache) {
    Write-Step 'Removing downloaded model caches'
    Remove-IfPresent (Join-Path $env:USERPROFILE '.cache\poselab') 'poselab model cache'
    Remove-IfPresent (Join-Path $env:USERPROFILE '.cache\torch') 'torch / MMPose weight cache'
}

Write-Host ''
Write-Host $RULE -ForegroundColor DarkGray
Write-Host '  Uninstall complete' -ForegroundColor Green
if (-not $Cache) {
    Write-Host '    Tip: re-run with -Cache to also delete downloaded model weights.' -ForegroundColor DarkGray
}
Write-Host '    The cloned source folder (if any) can be deleted manually.' -ForegroundColor DarkGray
Write-Host $RULE -ForegroundColor DarkGray
exit 0
