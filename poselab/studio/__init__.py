"""Pose3DStudio 後継 GUI の起動と、GUI 一式のビルド / 配備。

起動: poselab-studio  (または poselab-studio serve / python -m poselab.studio)

サブコマンド:
- serve  : Web GUI をローカルサーバーで起動 (引数なしの既定。
           poselab 自身のパイプラインで推定する。旧 exe は不要)
- build  : GUI 一式 (index.html / app.css / app.js) をフォルダへ書き出す
- deploy : 旧 Pose3DStudio.exe の `_internal/gui` へ配備する (レガシー。
           exe はこの 3 ファイルをディスクから配信するため差し替えだけで
           GUI を更新できる)

ソース構成 (このリポジトリが唯一のソース):
- 3D エンジン  : poselab/webviewer/static/engine.js
                 (ビューアと共通。二重管理しない)
- GUI 本体     : poselab/studio/gui/app_main.js
- HTML / CSS   : poselab/studio/gui/index.html, app.css
- サーバー     : poselab/studio/server.py (GUI が叩く API の独自実装)

ビルドは エンジンを IIFE (window.PoseLab3D) で包み、app_main.js を
連結して app.js を生成します。

使い方:
    poselab-studio                       # GUI を起動 (serve と同じ)
    poselab-studio serve --port 7860
    poselab-studio build --out dist/studio-gui
    poselab-studio deploy "C:\\path\\to\\Pose3DStudio\\_internal\\gui"
    # deploy の配備先は環境変数 POSE3DSTUDIO_GUI でも指定可
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

GUI_DIR = Path(__file__).resolve().parent / "gui"
ENGINE_SOURCE = (
    Path(__file__).resolve().parents[1] / "webviewer" / "static" / "engine.js"
)

_IIFE_HEADER = """/* PoseLab 3D viewer engine — poselab/webviewer/static/engine.js を
   poselab-studio build で連結したもの (スケルトン定義 / パーサ /
   PoseStage レンダラ / エクスポート)。window.PoseLab3D として公開。
   ※ 自動生成ファイル: 編集はリポジトリ側で行うこと */
(() => {
"""

_IIFE_FOOTER = """
window.PoseLab3D = {
  PoseStage, parseAny, demoModel, findJoint, sideOf, PERSON_HUES,
  collectExport, exportWideCsv, exportLongCsv, exportPoselabJson,
  exportMmposeJson, EXPORT_FORMATS,
};
})();
"""

DEPLOY_FILES = ("index.html", "app.css", "app.js")


def read_engine() -> str:
    """ビューアと共通の 3D エンジン (engine.js) のソースを読み込む。"""
    return ENGINE_SOURCE.read_text(encoding="utf-8")


def build_app_js() -> str:
    """エンジン + GUI 本体を連結した app.js を生成する。"""
    app_main = (GUI_DIR / "app_main.js").read_text(encoding="utf-8")
    return _IIFE_HEADER + read_engine() + _IIFE_FOOTER + app_main


def build(out_dir: "str | Path") -> Path:
    """GUI 一式 (index.html / app.css / app.js) を out_dir に書き出す。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "app.js").write_text(build_app_js(), encoding="utf-8", newline="\n")
    for name in ("index.html", "app.css"):
        (out / name).write_text(
            (GUI_DIR / name).read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n",
        )
    return out


def deploy(target_dir: "str | Path", backup: bool = True) -> Path:
    """ビルドして Pose3DStudio の gui フォルダへ配備する。

    backup=True なら既存ファイルを <target>/.backup/ に退避してから
    上書きする (毎回上書き保存)。
    """
    target = Path(target_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"配備先フォルダがありません: {target}")
    if backup:
        backup_dir = target / ".backup"
        backup_dir.mkdir(exist_ok=True)
        for name in DEPLOY_FILES:
            src = target / name
            if src.is_file():
                shutil.copy2(src, backup_dir / name)
    build(target)
    return target


def default_target() -> Optional[str]:
    """環境変数 POSE3DSTUDIO_GUI から既定の配備先を得る。"""
    return os.environ.get("POSE3DSTUDIO_GUI")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    # 引数なし、または serve はサーバー起動 (オプションは server 側で解釈)
    if not args_list or args_list[0] == "serve":
        from poselab.studio.server import main_serve

        rest = args_list[1:] if args_list else []
        return main_serve(rest)

    parser = argparse.ArgumentParser(
        prog="poselab-studio",
        description=(
            "Pose3DStudio 後継の Web GUI を起動 / ビルド / 配備します。"
            "引数なし (または serve) で GUI を起動します。"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve", help="Web GUI をローカルサーバーで起動する (既定)")

    p_build = sub.add_parser("build", help="GUI 一式をフォルダへ書き出す")
    p_build.add_argument(
        "--out", default="dist/studio-gui",
        help="出力先フォルダ (既定: dist/studio-gui)",
    )

    p_deploy = sub.add_parser(
        "deploy", help="Pose3DStudio の _internal/gui へビルドして配備する"
    )
    p_deploy.add_argument(
        "target", nargs="?", default=None,
        help="exe の gui フォルダ (省略時: 環境変数 POSE3DSTUDIO_GUI)",
    )
    p_deploy.add_argument(
        "--no-backup", action="store_true",
        help="既存ファイルの .backup/ への退避を行わない",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        out = build(args.out)
        print(f"ビルドしました: {out}")
        for name in DEPLOY_FILES:
            print(f"  {name}: {(out / name).stat().st_size:,} bytes")
        return 0

    if args.command == "deploy":
        target = args.target or default_target()
        if not target:
            print(
                "エラー: 配備先を指定してください "
                "(引数または環境変数 POSE3DSTUDIO_GUI)",
                file=sys.stderr,
            )
            return 2
        try:
            deployed = deploy(target, backup=not args.no_backup)
        except FileNotFoundError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 2
        print(f"配備しました: {deployed}")
        print("  Pose3DStudio のブラウザ画面を再読み込み (F5) すると反映されます")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
