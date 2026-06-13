# PoseLab Studio ローカルインストーラー (署名済み powershell + Python のみを使う)。
#
# 未署名の .exe を実行しないため Smart App Control にブロックされない。
# リポジトリ直下で実行し、専用 venv に GPU(または CPU)版 PyTorch・mmpose・
# poselab を入れて、スタートメニュー / デスクトップにショートカットを作る。
#
# 使い方 (どちらでも):
#   ・packaging\installer\Install-PoseLabStudio.cmd をダブルクリック
#   ・powershell -ExecutionPolicy Bypass -File packaging\installer\install_local.ps1
#
# Python 3.11 が必要 (mmcv の Windows wheel が 3.12 に無い)。無ければ winget で導入を試みる。

[CmdletBinding()]
param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'PoseLab Studio'),
    [switch]$Cpu,   # CPU 版 PyTorch を強制
    [switch]$Gpu    # CUDA 版 PyTorch を強制
)

$ErrorActionPreference = 'Stop'
function Write-Step($m) { Write-Host "==> $m" -ForegroundColor Cyan }

# リポジトリのルート (このスクリプトは packaging/installer/ にある)
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path

# --- Python 3.11 を探す (無ければ winget で導入) ---
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

Write-Step 'Python 3.11 を確認'
$py311 = Resolve-Py311
if (-not $py311) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Step 'Python 3.11 が無いため winget で導入します'
        winget install -e --id Python.Python.3.11 --silent `
            --accept-package-agreements --accept-source-agreements
        $py311 = Resolve-Py311
    }
}
if (-not $py311) {
    throw 'Python 3.11 が見つかりません。`winget install -e --id Python.Python.3.11` で導入してから再実行してください。'
}
Write-Host "  使用する Python: $py311"

# --- venv 準備 ---
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$venv = Join-Path $InstallDir 'env'
$python = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Step '専用の仮想環境を作成'
    & $py311 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'venv の作成に失敗しました' }
}

# numpy<2 を全インストールに強制 (torch 2.1 系は NumPy 1.x ビルド)
$constraints = Join-Path $InstallDir 'constraints.txt'
Set-Content -Path $constraints -Value 'numpy<2' -Encoding Ascii
$env:PIP_CONSTRAINT = $constraints

function Invoke-Pip([string[]]$PipArgs) {
    & $python -m pip install --no-input @PipArgs
    if ($LASTEXITCODE -ne 0) { throw ("pip install に失敗: " + ($PipArgs -join ' ')) }
}

& $python -m pip install --no-input -U pip | Out-Null

# --- GPU 判定 ---
$useGpu = $false
if ($Gpu) { $useGpu = $true }
elseif (-not $Cpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) { try { & $smi.Source | Out-Null; if ($LASTEXITCODE -eq 0) { $useGpu = $true } } catch {} }
}

# --- PyTorch ---
if ($useGpu) {
    Write-Step 'NVIDIA GPU を検出 → CUDA 11.8 版 PyTorch を導入 (約2GB)'
    $torchIndex = 'https://download.pytorch.org/whl/cu118'
} else {
    Write-Step 'GPU 未検出 → CPU 版 PyTorch を導入 (軽量)'
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
}
Invoke-Pip @('torch==2.1.2', 'torchvision==0.16.2', '--index-url', $torchIndex)

# --- mmpose の依存 chumpy (壊れた sdist 対策) ---
Write-Step 'mmpose の依存 (chumpy) を導入'
Invoke-Pip @('numpy<2', 'scipy')
Invoke-Pip @('chumpy==0.70', '--no-build-isolation')

# --- OpenMMLab (mim が torch に合う mmcv を自動選択) ---
Write-Step 'OpenMMLab (mmengine / mmcv / mmdet / mmpose) を導入'
Invoke-Pip @('-U', 'openmim')
& $python -m mim install mmengine 'mmcv==2.1.0' 'mmdet==3.2.0' 'mmpose==1.3.2'
if ($LASTEXITCODE -ne 0) { throw 'mim install に失敗しました' }

# --- poselab 本体 (リポジトリから) ---
Write-Step 'poselab を導入'
Invoke-Pip @($RepoRoot)

# --- ランチャと動作確認 ---
$launcher = Join-Path $InstallDir 'PoseLab Studio.cmd'
Set-Content -Path $launcher -Encoding Ascii -Value @"
@echo off
"$python" -m poselab.studio %*
"@

Write-Step 'セットアップの動作確認'
& $python -c "import poselab; from poselab.studio import build_app_js; from poselab.studio.server import gpu_info; from mmpose.apis.inferencers import Pose3DInferencer; print('PoseLab Studio env OK:', poselab.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'セットアップ後の動作確認に失敗しました' }

# --- ショートカット (スタートメニュー / デスクトップ) ---
function New-Shortcut($linkPath, $target) {
    $sh = New-Object -ComObject WScript.Shell
    $sc = $sh.CreateShortcut($linkPath)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = (Split-Path $target)
    $sc.Description = 'PoseLab Studio'
    $sc.Save()
}
Write-Step 'ショートカットを作成'
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) 'PoseLab Studio.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) 'PoseLab Studio.lnk'
try { New-Shortcut $startMenu $launcher; New-Shortcut $desktop $launcher } catch {
    Write-Host "  (ショートカット作成をスキップ: $_)" -ForegroundColor Yellow
}

Write-Host ''
Write-Host 'インストールが完了しました。' -ForegroundColor Green
Write-Host "  起動: スタートメニュー / デスクトップの「PoseLab Studio」"
Write-Host "  または: `"$launcher`""
if (-not $useGpu) {
    Write-Host '  (GPU 未検出のため CPU 動作。NVIDIA GPU 機なら自動で CUDA 版になります)' -ForegroundColor Yellow
}
exit 0
