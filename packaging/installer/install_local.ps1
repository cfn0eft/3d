# PoseLab Studio local installer (uses only signed cmd / powershell / Python /
# uv). No unsigned .exe of ours is executed, so Smart App Control does not
# block it. Installs are run with uv for clean, uv-style progress.
#
# Usage (either one):
#   - double-click packaging\installer\Install-PoseLabStudio.cmd
#   - powershell -ExecutionPolicy Bypass -File packaging\installer\install_local.ps1
#
# ASCII only on purpose (Windows PowerShell 5.1 misreads non-ASCII files on
# non-English locales). Keep this file ASCII + CRLF.

[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'PoseLab Studio'),
    [switch]$Cpu,   # force CPU PyTorch
    [switch]$Gpu    # force CUDA PyTorch
)

# Not 'Stop': on Windows PowerShell 5.1, Stop turns any native-command stderr
# write (uv/pip progress, "py: no runtime", ...) into a terminating error. We
# check $LASTEXITCODE explicitly and use -ErrorAction Stop on key cmdlets.
$ErrorActionPreference = 'Continue'

$RULE = '  ' + ('-' * 52)
function Write-Banner {
    Write-Host ''
    Write-Host '  PoseLab Studio' -ForegroundColor Cyan
    Write-Host '  Local installer' -ForegroundColor DarkGray
    Write-Host $RULE -ForegroundColor DarkGray
}
function Write-Step($m) {
    Write-Host ''
    Write-Host '  > ' -ForegroundColor Cyan -NoNewline
    Write-Host $m -ForegroundColor White
}

# Repo root (this script lives in packaging/installer/)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

Write-Banner

New-Item -ItemType Directory -Force -Path $InstallDir -ErrorAction Stop | Out-Null
$venv = Join-Path $InstallDir 'env'
$python = Join-Path $venv 'Scripts\python.exe'

# --- Fetch uv (the package manager that drives the install + progress UI) ---
# uv is a single signed-by-Astral binary; downloading and running it is fine
# under Smart App Control. Keep everything under $InstallDir for clean removal.
$uv = Join-Path $InstallDir 'uv.exe'
if (-not (Test-Path $uv)) {
    Write-Step 'Fetching uv (package manager)'
    $zip = Join-Path $InstallDir 'uv.zip'
    $url = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
    try {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -ErrorAction Stop
        Expand-Archive -Path $zip -DestinationPath $InstallDir -Force -ErrorAction Stop
        Remove-Item $zip -ErrorAction SilentlyContinue
    } catch {
        throw "Failed to download uv: $_"
    }
}
if (-not (Test-Path $uv)) { throw "uv.exe not found after download: $uv" }

# uv targets the environment via VIRTUAL_ENV (an env var, so a path with spaces
# like "...\PoseLab Studio\env" is passed safely - never as a CLI argument).
$env:VIRTUAL_ENV = $venv
$env:UV_CACHE_DIR = Join-Path $InstallDir 'uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'

function Invoke-Uv([string[]]$UvArgs) {
    & $uv @UvArgs
    if ($LASTEXITCODE -ne 0) { throw ("uv failed: " + ($UvArgs -join ' ')) }
}

# --- Private Python 3.11 + venv (uv provisions Python; no system install) ---
# mmcv ships Windows wheels only up to 3.11, so pin 3.11.
if (-not (Test-Path $python)) {
    Write-Step 'Creating the private environment (Python 3.11)'
    Invoke-Uv @('venv', '--seed', '--python', '3.11', $venv)
}

# uv's seeded setuptools lacks pkg_resources, which mmengine imports at runtime.
# Install a real setuptools (<81 still ships pkg_resources) + wheel (needed for
# chumpy's --no-build-isolation build).
Write-Step 'Preparing build tools'
Invoke-Uv @('pip', 'install', 'setuptools<81', 'wheel')

# --- GPU detection ---
$useGpu = $false
if ($Gpu) { $useGpu = $true }
elseif (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) { try { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $useGpu = $true } } catch {} }
}

# --- PyTorch (numpy pinned inline so no --constraint file / no spaced args) ---
if ($useGpu) {
    Write-Step 'Installing CUDA 11.8 PyTorch (NVIDIA GPU detected, ~2GB)'
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
    $mmFind = 'https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html'
} else {
    Write-Step 'Installing CPU PyTorch (no GPU detected, lightweight)'
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
    $mmFind = 'https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html'
}
Invoke-Uv @('pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2', 'numpy<2', '--index-url', $torchIndex)

# --- chumpy (mmpose dependency; broken sdist needs no build isolation) ---
Write-Step 'Installing mmpose dependency (chumpy)'
Invoke-Uv @('pip', 'install', 'scipy', 'numpy<2')
Invoke-Uv @('pip', 'install', 'chumpy==0.70', '--no-build-isolation')

# --- OpenMMLab (mmcv wheel matched to torch via --find-links) ---
Write-Step 'Installing OpenMMLab (mmengine / mmcv / mmdet / mmpose)'
Invoke-Uv @('pip', 'install', 'mmengine', 'mmcv==2.1.0', 'mmdet==3.2.0', 'mmpose==1.3.2',
    'numpy<2', '--find-links', $mmFind)

# --- poselab itself (install "." from the repo to avoid a spaced path arg) ---
Write-Step 'Installing poselab'
Push-Location $RepoRoot
try { Invoke-Uv @('pip', 'install', '.', 'numpy<2') } finally { Pop-Location }

# --- Launcher + sanity check ---
$launcher = Join-Path $InstallDir 'PoseLab Studio.cmd'
Set-Content -Path $launcher -Encoding Ascii -ErrorAction Stop -Value @"
@echo off
"$python" -m poselab.studio %*
"@

Write-Step 'Verifying the installation'
& $python -c "import poselab; from poselab.studio import build_app_js; from poselab.studio.server import gpu_info; from mmpose.apis.inferencers import Pose3DInferencer; print('PoseLab Studio env OK:', poselab.__version__)" 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Post-install verification failed' }

# --- Shortcuts (Start Menu / desktop) ---
# Copy the app icon into the install dir so the shortcut points at a stable
# path that goes away with the rest of the install on removal.
$icon = Join-Path $InstallDir 'poselab.ico'
$iconSrc = Join-Path $RepoRoot 'packaging\installer\poselab.ico'
if (Test-Path $iconSrc) {
    Copy-Item $iconSrc $icon -Force -ErrorAction SilentlyContinue
} else {
    $icon = $null
}

function New-Shortcut($linkPath, $target, $iconPath) {
    $sh = New-Object -ComObject WScript.Shell
    $sc = $sh.CreateShortcut($linkPath)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = (Split-Path $target)
    $sc.Description = 'PoseLab Studio'
    if ($iconPath -and (Test-Path $iconPath)) { $sc.IconLocation = "$iconPath,0" }
    $sc.Save()
}
Write-Step 'Creating shortcuts'
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'PoseLab Studio.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) 'PoseLab Studio.lnk'
try { New-Shortcut $startMenu $launcher $icon; New-Shortcut $desktop $launcher $icon } catch {
    Write-Host "  (skipped shortcut creation: $_)" -ForegroundColor Yellow
}

Write-Host ''
Write-Host $RULE -ForegroundColor DarkGray
Write-Host '  Installation complete' -ForegroundColor Green
Write-Host "    Launch from the Start Menu / desktop: 'PoseLab Studio'" -ForegroundColor Gray
Write-Host "    or run: $launcher" -ForegroundColor DarkGray
if (-not $useGpu) {
    Write-Host '    Note: no GPU detected - running on CPU. Re-run on an NVIDIA GPU for CUDA.' -ForegroundColor Yellow
}
Write-Host $RULE -ForegroundColor DarkGray
exit 0
