# PoseLab Studio local installer (uses only signed cmd / powershell / Python).
#
# No unsigned .exe is executed, so Windows Smart App Control does not block it.
# Run it from the repository root: it builds a private venv and installs the
# GPU (or CPU) PyTorch + mmpose stack + poselab, then creates Start Menu and
# desktop shortcuts.
#
# Usage (either one):
#   - double-click packaging\installer\Install-PoseLabStudio.cmd
#   - powershell -ExecutionPolicy Bypass -File packaging\installer\install_local.ps1
#
# ASCII only on purpose: Windows PowerShell 5.1 misreads UTF-8 (no BOM) files on
# non-English locales, which corrupts the script. Keep this file ASCII + CRLF.
#
# Requires Python 3.11 (mmcv has no Windows wheel for 3.12). Installs it via
# winget if missing.

[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'PoseLab Studio'),
    [switch]$Cpu,   # force CPU PyTorch
    [switch]$Gpu    # force CUDA PyTorch
)

# Note: do NOT use $ErrorActionPreference='Stop'. On Windows PowerShell 5.1
# that turns any native-command stderr write (e.g. "py: No suitable Python
# runtime", or pip warnings) into a terminating error. We check $LASTEXITCODE
# explicitly instead, and use -ErrorAction Stop on the few critical cmdlets.
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
function Write-Note($m) { Write-Host "    $m" -ForegroundColor DarkGray }

# Repo root (this script lives in packaging/installer/)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

Write-Banner

# --- Find Python 3.11 (install via winget if missing) ---
function Resolve-Py311 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $p = (& py -3.11 -c "import sys;print(sys.executable)" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $p = (& python -c "import sys;print(sys.executable) if sys.version_info[:2]==(3,11) else None" 2>$null)
        if ($p -and $p.Trim() -and $p.Trim() -ne 'None') { return $p.Trim() }
    }
    return $null
}

Write-Step 'Checking for Python 3.11'
$py311 = Resolve-Py311
if (-not $py311) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Step 'Python 3.11 not found - installing it via winget (this can take a minute)'
        winget install -e --id Python.Python.3.11 --silent `
            --accept-package-agreements --accept-source-agreements 2>&1 | Out-Host
        $py311 = Resolve-Py311
    }
}
if (-not $py311) {
    throw 'Python 3.11 not found. Install it (winget install -e --id Python.Python.3.11) and re-run.'
}
Write-Host "  Using Python: $py311"

# --- Create the venv ---
New-Item -ItemType Directory -Force -Path $InstallDir -ErrorAction Stop | Out-Null
$venv = Join-Path $InstallDir 'env'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Step 'Creating the private virtual environment'
    & $py311 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment' }
}

# Force numpy<2 for every install (torch 2.1.x is built against NumPy 1.x)
$constraints = Join-Path $InstallDir 'constraints.txt'
Set-Content -Path $constraints -Value 'numpy<2' -Encoding Ascii -ErrorAction Stop
$env:PIP_CONSTRAINT = $constraints

function Invoke-Pip([string[]]$PipArgs) {
    & $python -m pip install --no-input @PipArgs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw ("pip install failed: " + ($PipArgs -join ' ')) }
}

# Upgrade pip / setuptools / wheel. wheel is required for chumpy's
# --no-build-isolation build (bdist_wheel); python -m venv does not seed it.
Invoke-Pip @('-U', 'pip', 'setuptools', 'wheel')

# --- GPU detection ---
$useGpu = $false
if ($Gpu) { $useGpu = $true }
elseif (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) { try { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $useGpu = $true } } catch {} }
}

# --- PyTorch ---
if ($useGpu) {
    Write-Step 'NVIDIA GPU detected - installing CUDA 11.8 PyTorch (about 2GB)'
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
} else {
    Write-Step 'No GPU detected - installing CPU PyTorch (lightweight)'
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
}
Invoke-Pip @('torch==2.1.2', 'torchvision==0.16.2', '--index-url', $torchIndex)

# --- chumpy (mmpose dependency, broken sdist needs no build isolation) ---
Write-Step 'Installing mmpose dependency (chumpy)'
Invoke-Pip @('numpy<2', 'scipy')
Invoke-Pip @('chumpy==0.70', '--no-build-isolation')

# --- OpenMMLab (mim picks the mmcv matching torch CUDA/CPU) ---
Write-Step 'Installing OpenMMLab (mmengine / mmcv / mmdet / mmpose)'
Invoke-Pip @('-U', 'openmim')
& $python -m mim install mmengine 'mmcv==2.1.0' 'mmdet==3.2.0' 'mmpose==1.3.2' 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'mim install failed' }

# --- poselab itself (from the repo) ---
Write-Step 'Installing poselab'
Invoke-Pip @($RepoRoot)

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
function New-Shortcut($linkPath, $target) {
    $sh = New-Object -ComObject WScript.Shell
    $sc = $sh.CreateShortcut($linkPath)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = (Split-Path $target)
    $sc.Description = 'PoseLab Studio'
    $sc.Save()
}
Write-Step 'Creating shortcuts'
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'PoseLab Studio.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) 'PoseLab Studio.lnk'
try { New-Shortcut $startMenu $launcher; New-Shortcut $desktop $launcher } catch {
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
