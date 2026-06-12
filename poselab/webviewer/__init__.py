"""ブラウザで動く 3D ポーズビューア。

起動: poselab-viewer  (または python -m poselab.webviewer)

機能:
- 推定結果 (poselab の JSON / long CSV / wide CSV、および MMPose 系
  ツールが出力する meta_info/instance_info 形式 JSON) をブラウザ上で
  3D 再生・回転・ズームできるビューア
- 依存ライブラリゼロ (標準ライブラリのみ)。描画はブラウザ側の
  Canvas 2D で行い、WebGL や外部 CDN を使わないためオフラインでも動く
- ファイルのドラッグ & ドロップ / 起動時引数での事前読み込みに対応
- `--export-html` で CSS / JS を 1 ファイルに埋め込んだ自己完結 HTML を
  書き出せる (GitHub Pages などの静的ホスティングにそのまま置ける)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import unquote, urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}

# ビューアが解釈できるデータファイルの拡張子
DATA_SUFFIXES = (".json", ".csv")


def build_single_html() -> str:
    """CSS / JS をインライン展開した自己完結 HTML を返す。"""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "app.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html = re.sub(
        r'<link rel="stylesheet" href="\./app\.css"\s*/?>',
        lambda _: f"<style>\n{css}\n</style>",
        html,
        count=1,
    )
    html = re.sub(
        r'<script src="\./app\.js"></script>',
        lambda _: f"<script>\n{js}\n</script>",
        html,
        count=1,
    )
    return html


def collect_data_files(paths: Sequence["str | Path"]) -> List[Path]:
    """引数のファイル / フォルダからビューアで読めるファイルを集める。"""
    found: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in DATA_SUFFIXES and child.is_file():
                    found.append(child)
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
    return found


class _ViewerHandler(BaseHTTPRequestHandler):
    """静的アセットと事前読み込みデータを配信するハンドラ。"""

    # main() からセットされる
    data_files: List[Path] = []
    quiet = True

    def log_message(self, fmt, *args):  # noqa: D102 - 静かに
        if not self.quiet:
            super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(HTTPStatus.OK, body, _MIME[".json"])

    def do_GET(self) -> None:  # noqa: N802 (http.server の規約)
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path == "/manifest.json":
            return self._send_json(
                {
                    "files": [
                        {"index": i, "name": p.name}
                        for i, p in enumerate(self.data_files)
                    ]
                }
            )
        match = re.fullmatch(r"/data/(\d+)", path)
        if match:
            index = int(match.group(1))
            if 0 <= index < len(self.data_files):
                file = self.data_files[index]
                body = file.read_bytes()
                mime = _MIME.get(file.suffix.lower(), "application/octet-stream")
                return self._send(HTTPStatus.OK, body, mime)
            return self._send(
                HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8"
            )
        return self._serve_static(path.lstrip("/"))

    def _serve_static(self, name: str) -> None:
        target = (STATIC_DIR / name).resolve()
        # static ディレクトリ外へのトラバーサルを拒否
        if STATIC_DIR not in target.parents or not target.is_file():
            return self._send(
                HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8"
            )
        body = target.read_bytes()
        mime = _MIME.get(target.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, body, mime)


def serve(
    data_files: Sequence[Path],
    host: str = "127.0.0.1",
    port: int = 0,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    """ビューアサーバーを起動して返す (port=0 で空きポート自動選択)。"""
    handler = type(
        "_BoundViewerHandler",
        (_ViewerHandler,),
        {"data_files": list(data_files), "quiet": quiet},
    )
    server = ThreadingHTTPServer((host, port), handler)
    return server


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="poselab-viewer",
        description=(
            "推定結果 (JSON / CSV) をブラウザで 3D 再生するビューア。"
            "ファイルを渡さずに起動してドラッグ & ドロップで読み込むこともできます。"
        ),
    )
    parser.add_argument(
        "data",
        nargs="*",
        help="事前に読み込む JSON / CSV (フォルダ指定で中の対応ファイル全部)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="バインド先 (既定: ローカルのみ)")
    parser.add_argument("--port", type=int, default=7870, help="ポート (0 で自動選択)")
    parser.add_argument(
        "--no-browser", action="store_true", help="ブラウザを自動で開かない"
    )
    parser.add_argument(
        "--export-html",
        type=Path,
        metavar="OUT.html",
        help="自己完結 HTML を書き出して終了 (サーバーは起動しない)",
    )
    args = parser.parse_args(argv)

    if args.export_html is not None:
        html = build_single_html()
        args.export_html.parent.mkdir(parents=True, exist_ok=True)
        args.export_html.write_text(html, encoding="utf-8")
        print(f"書き出しました: {args.export_html}")
        return 0

    try:
        data_files = collect_data_files(args.data)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2

    try:
        server = serve(data_files, host=args.host, port=args.port)
    except OSError:
        # 指定ポートが使用中なら自動選択で再試行
        server = serve(data_files, host=args.host, port=0)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(f"poselab 3D viewer: {url}  (Ctrl+C で終了)")
    if data_files:
        for file in data_files:
            print(f"  preload: {file}")

    if not args.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
