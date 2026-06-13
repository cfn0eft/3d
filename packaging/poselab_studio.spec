# -*- mode: python ; coding: utf-8 -*-
"""PoseLab Studio (Windows 配布版) の PyInstaller spec。

ビルド: pyinstaller packaging/poselab_studio.spec --noconfirm
(通常は .github/workflows/build-exe.yml が CI で実行する)

ポイント:
- OpenMMLab 系 (mmengine/mmcv/mmdet/mmpose) は実行時に .py コンフィグを
  読む (mmengine.Config が exec する) ため、module_collection_mode="py"
  でソースのまま _internal/ に展開する (旧 Pose3DStudio.exe と同じ方式)
- torch (CUDA 同梱) の DLL 収集は pyinstaller-hooks-contrib の
  標準フックに任せる
- poselab のパッケージデータ (webviewer/static, studio/gui) を同梱し、
  GUI の app.js はサーバーが起動時に連結生成する
"""

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
)

APP_NAME = "PoseLabStudio"
MM_PACKAGES = ("mmengine", "mmcv", "mmdet", "mmpose")

datas = []
binaries = []
# poselab の全サブモジュール (server が pose3d / backends 等を遅延 import する)
hiddenimports = collect_submodules("poselab")

for pkg in MM_PACKAGES:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# poselab のパッケージデータ (webviewer/static, studio/gui)
datas += collect_data_files("poselab")

a = Analysis(
    ["studio_entry.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib.tests", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    module_collection_mode={pkg: "py" for pkg in MM_PACKAGES},
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # ログが見えるようコンソール付き (閉じると終了)
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
