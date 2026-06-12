"""フレーム入力ソース (画像・動画・カメラ) の統一インターフェース。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from poselab.imgio import imread

# (frame_index, timestamp_ms, frame_bgr)
Frame = Tuple[int, float, np.ndarray]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class FrameSource:
    """フレームを順に返すイテレータの基底クラス。"""

    description: str = ""
    fps: Optional[float] = None
    frame_count: Optional[int] = None  # 不明 (カメラ等) なら None
    is_live: bool = False

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ImageSource(FrameSource):
    """1 枚以上の静止画。タイムスタンプはフレーム番号 (ms 扱い)。"""

    def __init__(self, paths: "List[str] | List[Path]") -> None:
        self.paths = [Path(p) for p in paths]
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(p)
        self.frame_count = len(self.paths)
        self.description = ", ".join(p.name for p in self.paths[:3])

    def __iter__(self) -> Iterator[Frame]:
        for i, path in enumerate(self.paths):
            frame = imread(path)
            if frame is None:
                raise IOError(f"failed to read image: {path}")
            yield i, float(i), frame


class VideoSource(FrameSource):
    """動画ファイル。"""

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise IOError(f"failed to open video: {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or None
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = count if count > 0 else None
        self.description = self.path.name

    def __iter__(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if ts <= 0 and self.fps:
                ts = index * 1000.0 / self.fps
            yield index, float(ts), frame
            index += 1

    def close(self) -> None:
        self.cap.release()


class CameraSource(FrameSource):
    """接続カメラ (Web カメラ等)。タイムスタンプは取得時刻ベース。

    mirror=True で左右反転 (鏡像) 表示。反転した画像で推定するため、
    キーポイントの left/right は被写体実際の左右と逆になる点に注意。
    """

    is_live = True

    def __init__(
        self,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        mirror: bool = False,
    ) -> None:
        self.mirror = mirror
        self.cap = _open_camera(index)
        if self.cap is None:
            raise IOError(
                f"カメラ {index} を開けませんでした。\n"
                "・カメラが接続されているか\n"
                "・他のアプリ (Zoom / Teams 等) が使用中でないか\n"
                "・OS のカメラアクセス許可 (Windows: 設定 → プライバシー →\n"
                "  カメラ → デスクトップアプリにカメラへのアクセスを許可)\n"
                "を確認し、別のカメラ番号 (1, 2, ...) も試してください。"
            )
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or None
        self.description = f"camera:{index}"

    def __iter__(self) -> Iterator[Frame]:
        import time

        start = time.monotonic()
        index = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            ts = (time.monotonic() - start) * 1000.0
            yield index, ts, frame
            index += 1

    def close(self) -> None:
        self.cap.release()


def _open_camera(index: int) -> Optional[cv2.VideoCapture]:
    """カメラを開く。Windows では MSMF 失敗時に DirectShow で再試行する。"""
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap
    cap.release()
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def scan_cameras(max_index: int = 8) -> List[dict]:
    """利用可能なカメラを検索する。

    Returns
    -------
    list of dict: {"index", "width", "height"} (開けたカメラのみ)
    """
    found = []
    for index in range(max_index + 1):
        cap = _open_camera(index)
        if cap is None:
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append(
                {
                    "index": index,
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            )
        cap.release()
    return found


def open_source(
    spec: str,
    camera_width: Optional[int] = None,
    camera_height: Optional[int] = None,
    camera_mirror: bool = False,
) -> FrameSource:
    """入力指定文字列からソースを生成する。

    - "camera:0" / "cam:0" → カメラ
    - 画像拡張子のパス → 静止画
    - それ以外のパス → 動画
    """
    low = spec.lower()
    if low.startswith(("camera:", "cam:")):
        index = int(spec.split(":", 1)[1])
        return CameraSource(index, camera_width, camera_height, mirror=camera_mirror)
    path = Path(spec)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSource([path])
    return VideoSource(path)
