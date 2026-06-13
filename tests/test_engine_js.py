"""3D エンジン (poselab/webviewer/static/engine.js) の Node スモークテスト。

実体は tests/engine_smoke.mjs (デモ生成 → 全形式エクスポート →
再パースのラウンドトリップ)。node が無い環境ではスキップする
(GitHub Actions のランナーには同梱されている)。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
SMOKE_SCRIPT = Path(__file__).resolve().parent / "engine_smoke.mjs"


@pytest.mark.skipif(NODE is None, reason="node が見つからない環境ではスキップ")
def test_engine_smoke_roundtrip():
    proc = subprocess.run(
        [NODE, str(SMOKE_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    detail = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, f"engine_smoke.mjs が失敗:\n{detail}"
    assert "engine smoke: OK" in proc.stdout
