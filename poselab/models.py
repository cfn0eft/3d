"""推定モデルファイルの取得とキャッシュ管理。

MediaPipe Pose Landmarker のモデル (Apache-2.0) を Google の公開
ストレージから初回のみダウンロードし、ローカルにキャッシュします。
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)

MODEL_VARIANTS = ("lite", "full", "heavy")


def default_cache_dir() -> Path:
    base = os.environ.get("POSELAB_CACHE_DIR")
    if base:
        return Path(base)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "poselab"
    return Path.home() / ".cache" / "poselab"


def get_model_path(variant: str = "full", cache_dir: "Path | None" = None) -> Path:
    """モデルファイルのパスを返す。未取得ならダウンロードする。"""
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"unknown model variant: {variant!r} (choose from {MODEL_VARIANTS})"
        )
    cache = Path(cache_dir) if cache_dir else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"pose_landmarker_{variant}.task"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = _BASE_URL.format(variant=variant)
    tmp = path.with_suffix(".task.part")
    print(f"[poselab] downloading model '{variant}' -> {path}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path
