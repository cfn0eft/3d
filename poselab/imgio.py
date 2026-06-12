"""Unicode パス対応の画像入出力ヘルパー。

OpenCV の cv2.imread / cv2.imwrite は Windows で非 ASCII パス
(日本語のフォルダ名・ファイル名等) を扱えないため、Python 側で
ファイル I/O を行い imdecode / imencode と組み合わせる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread(path: "str | Path") -> Optional[np.ndarray]:
    """画像を BGR で読み込む。失敗時は None (cv2.imread と同じ規約)。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: "str | Path", image: np.ndarray) -> None:
    """画像を書き出す。拡張子からフォーマットを決定 (省略時は PNG)。"""
    path = Path(path)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"failed to encode image as {ext!r}: {path}")
    buf.tofile(str(path))
