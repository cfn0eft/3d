"""poselab.webviewer (ブラウザ 3D ビューア) のテスト。

ブラウザ側 (app.js) の描画はテスト対象外。Python 側のサーバー /
自己完結 HTML 生成 / ファイル収集をテストする。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from poselab.webviewer import (
    STATIC_DIR,
    build_single_html,
    collect_data_files,
    main,
    serve,
)


def test_static_assets_exist():
    for name in ("index.html", "app.css", "app.js"):
        path = STATIC_DIR / name
        assert path.is_file(), name
        assert path.stat().st_size > 1000, name


def test_build_single_html_inlines_assets():
    html = build_single_html()
    # 外部参照が消えてインライン化されている
    assert './app.css' not in html
    assert './app.js' not in html
    assert "<style>" in html
    assert "<script>" in html
    # 主要な実装が含まれている
    assert "PoseStage" in html
    assert "meta_info" in html  # MMPose 形式対応
    assert "keypoint_name" in html  # poselab ロング CSV 対応


def test_collect_data_files(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "b.csv").write_text("frame\n0\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")
    files = collect_data_files([tmp_path])
    assert [f.name for f in files] == ["a.json", "b.csv"]
    # 個別ファイル指定はそのまま
    files = collect_data_files([tmp_path / "a.json"])
    assert [f.name for f in files] == ["a.json"]
    with pytest.raises(FileNotFoundError):
        collect_data_files([tmp_path / "missing.json"])


@pytest.fixture()
def viewer_server(tmp_path):
    data = tmp_path / "result.json"
    data.write_text(
        json.dumps({"metadata": {}, "frames": []}), encoding="utf-8"
    )
    server = serve([data], host="127.0.0.1", port=0, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}"
    server.shutdown()
    server.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as res:
        return res.status, res.read(), res.headers.get("Content-Type", "")


def test_server_serves_index_and_assets(viewer_server):
    status, body, ctype = _get(viewer_server + "/")
    assert status == 200
    assert b"poselab" in body
    assert "text/html" in ctype

    status, body, ctype = _get(viewer_server + "/app.js")
    assert status == 200
    assert "javascript" in ctype

    status, body, ctype = _get(viewer_server + "/app.css")
    assert status == 200
    assert "css" in ctype


def test_server_manifest_and_data(viewer_server):
    status, body, _ = _get(viewer_server + "/manifest.json")
    assert status == 200
    manifest = json.loads(body)
    assert manifest["files"] == [{"index": 0, "name": "result.json"}]

    status, body, ctype = _get(viewer_server + "/data/0")
    assert status == 200
    assert json.loads(body) == {"metadata": {}, "frames": []}
    assert "json" in ctype


def test_server_rejects_traversal_and_unknown(viewer_server):
    for path in ("/data/99", "/../pyproject.toml", "/%2e%2e/pyproject.toml", "/nope.js"):
        try:
            status, _, _ = _get(viewer_server + path)
        except urllib.error.HTTPError as err:  # urlopen は 404 で例外
            status = err.code
        assert status == 404, path


def test_export_html_cli(tmp_path):
    out = tmp_path / "viewer.html"
    code = main(["--export-html", str(out)])
    assert code == 0
    html = out.read_text(encoding="utf-8")
    assert "PoseStage" in html
    assert "<style>" in html


def test_main_missing_file_returns_error(tmp_path):
    code = main([str(tmp_path / "none.json"), "--no-browser", "--export-html",
                 str(tmp_path / "x.html")])
    # --export-html はデータ引数より先に処理されるため成功する
    assert code == 0
    code = main([str(tmp_path / "none.json"), "--no-browser"])
    assert code == 2
