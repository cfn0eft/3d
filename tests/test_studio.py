"""poselab.studio (Pose3DStudio GUI のビルド / 配備) のテスト。"""

from __future__ import annotations

import pytest

from poselab.studio import (
    DEPLOY_FILES,
    ENGINE_SOURCE,
    GUI_DIR,
    build,
    build_app_js,
    deploy,
    main,
    read_engine,
)


def test_sources_exist():
    for name in ("index.html", "app.css", "app_main.js"):
        path = GUI_DIR / name
        assert path.is_file(), name
        assert path.stat().st_size > 1000, name
    # エンジンはビューアと共通の engine.js (唯一のソース)
    assert ENGINE_SOURCE.is_file()
    assert ENGINE_SOURCE.name == "engine.js"


def test_engine_contains_core_symbols():
    engine = read_engine()
    for symbol in ("class PoseStage", "function parseAny", "function demoModel",
                   "function collectExport", "EXPORT_FORMATS"):
        assert symbol in engine, symbol
    # ビューアのアプリ配線 (app.js 側) は含まれない
    assert 'new PoseStage($("stage-canvas"))' not in engine
    assert "getElementById" not in engine.split("*/", 1)[1]


def test_build_app_js_structure():
    js = build_app_js()
    assert js.startswith("/* PoseLab 3D viewer engine")
    assert "window.PoseLab3D = {" in js
    assert "Pose3D Studio GUI 本体" in js
    # エンジン → 公開 → GUI 本体 の順
    assert js.index("class PoseStage") < js.index("window.PoseLab3D = {") < js.index(
        "Pose3D Studio GUI 本体"
    )
    # GUI 本体はエンジンの公開シンボルを使う
    assert "PoseLab3D.parseAny" in js
    assert "PoseLab3D.collectExport" in js


def test_build_writes_files(tmp_path):
    out = build(tmp_path / "gui")
    for name in DEPLOY_FILES:
        assert (out / name).is_file(), name
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "viewer-canvas" in html
    assert (out / "app.js").stat().st_size > 50_000


def test_deploy_with_backup(tmp_path):
    target = tmp_path / "exe_gui"
    target.mkdir()
    (target / "app.js").write_text("old-app", encoding="utf-8")
    (target / "index.html").write_text("old-html", encoding="utf-8")

    deploy(target, backup=True)

    # 新ファイルが配備され、旧ファイルは .backup へ
    assert (target / "app.js").stat().st_size > 50_000
    assert (target / ".backup" / "app.js").read_text(encoding="utf-8") == "old-app"
    assert (target / ".backup" / "index.html").read_text(encoding="utf-8") == "old-html"
    # app.css は旧ファイルが無かったのでバックアップも無い
    assert not (target / ".backup" / "app.css").exists()


def test_deploy_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        deploy(tmp_path / "nope")


def test_cli_build_and_deploy(tmp_path, capsys, monkeypatch):
    out = tmp_path / "dist"
    assert main(["build", "--out", str(out)]) == 0
    assert (out / "app.js").is_file()

    target = tmp_path / "gui"
    target.mkdir()
    assert main(["deploy", str(target), "--no-backup"]) == 0
    assert (target / "index.html").is_file()

    # 引数なし + 環境変数
    monkeypatch.setenv("POSE3DSTUDIO_GUI", str(target))
    assert main(["deploy"]) == 0

    monkeypatch.delenv("POSE3DSTUDIO_GUI")
    assert main(["deploy"]) == 2
    assert "POSE3DSTUDIO_GUI" in capsys.readouterr().err
