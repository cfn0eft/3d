# PoseLab Studio オンラインインストーラーのセットアップスクリプト。
#
# 小さな配布インストーラー (Inno Setup) からインストール時に呼ばれ、
# ユーザーのマシン上に専用環境を構築する:
#   1. uv で専用 Python 3.11 と venv を用意 (システムには触れない)
#   2. PyTorch を導入 (NVIDIA GPU 検出時は CUDA 11.8 版、無ければ CPU 版)
#   3. OpenMMLab (mmengine / mmcv / mmdet / mmpose) を mim で導入
#   4. poselab (同梱 wheel) を導入
#   5. 動作確認 (mmpose の 3D 推論クラスと GUI 生成を import)
#
# CI からは -Cpu を付けてレシピ全体を検証する (tests とは別に実機相当で確認)。
#
# 単体実行例:
#   pwsh -ExecutionPolicy Bypass -File setup_env.ps1 -InstallDir C:\PoseLabStudio

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [string]$UvExe = (Join-Path $PSScriptRoot 'uv.exe'),
    [string]$Wheelhouse = (Join-Path $PSScriptRoot 'wheels'),
    [switch]$Cpu,   # CPU 版 PyTorch を強制 (GPU 自動検出を無視)
    [switch]$Gpu    # CUDA 版 PyTorch を強制
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$envDir = Join-Path $InstallDir 'env'
$python = Join-Path $envDir 'Scripts\python.exe'

# 専用 Python とキャッシュを InstallDir 配下に置く (アンインストールで一掃できる)
$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'
$env:UV_CACHE_DIR = Join-Path $InstallDir 'uv-cache'

# numpy<2 を全インストールに強制 (torch 2.1 系は NumPy 1.x ビルド)
$constraints = Join-Path $InstallDir 'constraints.txt'
Set-Content -Path $constraints -Value 'numpy<2' -Encoding Ascii
$env:PIP_CONSTRAINT = $constraints   # 後続の pip / mim に効く

if (-not (Test-Path $UvExe)) { throw "uv.exe が見つかりません: $UvExe" }

# --- GPU 判定 (Cpu/Gpu 指定で上書き) ---
$useGpu = $false
if ($Gpu) {
    $useGpu = $true
} elseif (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        try { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $useGpu = $true } } catch {}
    }
}

# --- 1. 専用 Python 3.11 + venv (pip 同梱) ---
Write-Step '専用 Python 3.11 と仮想環境を準備'
& $UvExe venv --seed --python 3.11 $envDir
if ($LASTEXITCODE -ne 0) { throw '仮想環境の作成に失敗しました' }

# 注意: パラメータ名に $Args は使わない (PowerShell の自動変数 $args と衝突し、
# スプラッティングが空になって `python` が引数なし起動 = 何もせず成功してしまう)
function Invoke-UvPip([string[]]$PipArgs) {
    & $UvExe pip install --python $python @PipArgs --constraint $constraints
    if ($LASTEXITCODE -ne 0) { throw ("uv pip install に失敗: " + ($PipArgs -join ' ')) }
}
function Invoke-Py([string[]]$PyArgs) {
    & $python @PyArgs
    if ($LASTEXITCODE -ne 0) { throw ("python 実行に失敗: " + ($PyArgs -join ' ')) }
}

# --- 2. PyTorch (大きいので uv の高速・レジューム対応 DL を使う) ---
if ($useGpu) {
    Write-Step 'NVIDIA GPU を検出 → CUDA 11.8 版 PyTorch を導入 (約2GB)'
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
} else {
    Write-Step 'GPU 未検出 → CPU 版 PyTorch を導入 (軽量・約200MB)'
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
}
Invoke-UvPip @('torch==2.1.2', 'torchvision==0.16.2', '--index-url', $torchIndex)

# --- 3. chumpy (mmpose の依存。壊れた sdist 対策で分離ビルドを無効化) ---
Write-Step 'mmpose の依存 (chumpy) を導入'
Invoke-Py @('-m', 'pip', 'install', '--no-input', 'scipy')
Invoke-Py @('-m', 'pip', 'install', '--no-input', 'chumpy==0.70', '--no-build-isolation')

# --- 4. OpenMMLab (mim が torch の CUDA/CPU に合う mmcv を自動選択) ---
Write-Step 'OpenMMLab (mmengine / mmcv / mmdet / mmpose) を導入'
Invoke-Py @('-m', 'pip', 'install', '--no-input', '-U', 'openmim')
Invoke-Py @('-m', 'mim', 'install', 'mmengine', 'mmcv==2.1.0', 'mmdet==3.2.0', 'mmpose==1.3.2')

# --- 5. poselab (同梱 wheel) ---
Write-Step 'poselab を導入'
$wheel = Get-ChildItem -Path $Wheelhouse -Filter 'poselab_toolkit-*.whl' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $wheel) { throw "poselab の wheel が $Wheelhouse に見つかりません" }
Invoke-UvPip @($wheel.FullName)

# --- 6. 動作確認 ---
Write-Step 'セットアップの動作確認'
Invoke-Py @('-c', 'import poselab; from poselab.studio.server import build_app_js; from mmpose.apis.inferencers import Pose3DInferencer; print("PoseLab Studio env OK:", poselab.__version__)')

Write-Host ''
Write-Host 'セットアップが完了しました。' -ForegroundColor Green
if (-not $useGpu) {
    Write-Host '(GPU 未検出のため CPU で動作します。NVIDIA GPU 搭載機で再インストールすると CUDA 版になります)' -ForegroundColor Yellow
}
exit 0
