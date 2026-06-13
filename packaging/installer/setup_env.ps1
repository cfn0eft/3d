# Setup script for the PoseLab Studio online installer.
#
# Called at install time by the small Inno Setup installer; it provisions a
# private environment on the user's machine:
#   1. uv creates a private Python 3.11 + venv (system Python untouched)
#   2. PyTorch (CUDA 11.8 when an NVIDIA GPU is found, CPU otherwise)
#   3. OpenMMLab (mmengine / mmcv / mmdet / mmpose) via mim
#   4. poselab (bundled wheel)
#   5. import check (mmpose 3D inferencer + GUI builder)
#
# CI runs it with -Cpu to validate the whole recipe end to end.
#
# ASCII only on purpose: Windows PowerShell 5.1 misreads UTF-8 (no BOM) files on
# non-English locales. Keep this file ASCII + CRLF.
#
# Example:
#   pwsh -ExecutionPolicy Bypass -File setup_env.ps1 -InstallDir C:\PoseLabStudio

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [string]$UvExe = (Join-Path $PSScriptRoot 'uv.exe'),
    [string]$Wheelhouse = (Join-Path $PSScriptRoot 'wheels'),
    [switch]$Cpu,   # force CPU PyTorch (ignore GPU autodetection)
    [switch]$Gpu    # force CUDA PyTorch
)

# Note: not 'Stop' on purpose. Windows PowerShell 5.1 (used by the Inno
# installer) turns native-command stderr writes into terminating errors under
# Stop; we check $LASTEXITCODE explicitly instead.
$ErrorActionPreference = 'Continue'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

New-Item -ItemType Directory -Force -Path $InstallDir -ErrorAction Stop | Out-Null
$envDir = Join-Path $InstallDir 'env'
$python = Join-Path $envDir 'Scripts\python.exe'

# Keep the private Python and caches under InstallDir (so uninstall is clean)
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'
$env:UV_CACHE_DIR = Join-Path $InstallDir 'uv-cache'

# Force numpy<2 everywhere (torch 2.1.x is built against NumPy 1.x)
$constraints = Join-Path $InstallDir 'constraints.txt'
Set-Content -Path $constraints -Value 'numpy<2' -Encoding Ascii -ErrorAction Stop
$env:PIP_CONSTRAINT = $constraints   # applies to pip / mim

if (-not (Test-Path $UvExe)) { throw "uv.exe not found: $UvExe" }

# --- GPU detection (overridable with -Cpu/-Gpu) ---
$useGpu = $false
if ($Gpu) {
    $useGpu = $true
} elseif (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $useGpu = $true } } catch {}
    }
}

# --- 1. Private Python 3.11 + venv (with pip) ---
Write-Step 'Provisioning private Python 3.11 and a virtual environment'
& $UvExe venv --seed --python 3.11 $envDir 2>&1 | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment' }

function Invoke-UvPip([string[]]$PipArgs) {
    & $UvExe pip install --python $python @PipArgs --constraint $constraints 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw ("uv pip install failed: " + ($PipArgs -join ' ')) }
}
function Invoke-Py([string[]]$PyArgs) {
    & $python @PyArgs 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) { throw ("python failed: " + ($PyArgs -join ' ')) }
}

# --- 2. PyTorch (large; use uv's fast, resumable download) ---
if ($useGpu) {
    Write-Step 'NVIDIA GPU detected - installing CUDA 11.8 PyTorch (about 2GB)'
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
} else {
    Write-Step 'No GPU detected - installing CPU PyTorch (lightweight, ~200MB)'
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
}
Invoke-UvPip @('torch==2.1.2', 'torchvision==0.16.2', '--index-url', $torchIndex)

# --- 3. chumpy (mmpose dependency; broken sdist needs no build isolation) ---
Write-Step 'Installing mmpose dependency (chumpy)'
Invoke-Py @('-m', 'pip', 'install', '--no-input', 'scipy')
Invoke-Py @('-m', 'pip', 'install', '--no-input', 'chumpy==0.70', '--no-build-isolation')

# --- 4. OpenMMLab (mim picks the mmcv matching torch CUDA/CPU) ---
Write-Step 'Installing OpenMMLab (mmengine / mmcv / mmdet / mmpose)'
Invoke-Py @('-m', 'pip', 'install', '--no-input', '-U', 'openmim')
Invoke-Py @('-m', 'mim', 'install', 'mmengine', 'mmcv==2.1.0', 'mmdet==3.2.0', 'mmpose==1.3.2')

# --- 5. poselab (bundled wheel) ---
Write-Step 'Installing poselab'
$wheel = Get-ChildItem -Path $Wheelhouse -Filter 'poselab_toolkit-*.whl' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $wheel) { throw "poselab wheel not found in $Wheelhouse" }
Invoke-UvPip @($wheel.FullName)

# --- 6. Sanity check ---
Write-Step 'Verifying the setup'
Invoke-Py @('-c', 'import poselab; from poselab.studio import build_app_js; from poselab.studio.server import gpu_info; from mmpose.apis.inferencers import Pose3DInferencer; print("PoseLab Studio env OK:", poselab.__version__)')

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green
if (-not $useGpu) {
    Write-Host '  (No GPU detected: running on CPU. Re-run on an NVIDIA GPU machine for CUDA.)' -ForegroundColor Yellow
}
exit 0
