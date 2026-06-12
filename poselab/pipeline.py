"""入力 → 推定 → エクスポート/描画 を束ねる処理パイプライン。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

from poselab.backends.base import PoseBackend
from poselab.exporters import Exporter
from poselab.imgio import imwrite
from poselab.sources import FrameSource
from poselab.tracking import PersonTracker
from poselab.types import FrameResult
from poselab.visualize import TrajectoryOverlay, draw_result, draw_status


class VideoWriter:
    """注釈付き動画の書き出し (サイズは最初のフレームで決定)。"""

    def __init__(self, path: "str | Path", fps: float = 30.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
            if not self._writer.isOpened():
                raise IOError(f"failed to open video writer: {self.path}")
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def run_pipeline(
    source: FrameSource,
    backend: PoseBackend,
    exporters: Sequence[Exporter] = (),
    video_writer: Optional[VideoWriter] = None,
    image_output: Optional[Path] = None,
    draw: bool = True,
    draw_labels: bool = False,
    trajectory: Optional[TrajectoryOverlay] = None,
    tracker: Optional[PersonTracker] = None,
    draw_ids: Optional[bool] = None,
    min_visibility: float = 0.3,
    show: bool = False,
    max_frames: Optional[int] = None,
    on_frame: Optional[Callable[[FrameResult, np.ndarray], bool]] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> List[FrameResult]:
    """ソースの全フレームを処理する。

    on_frame が False を返すと処理を中断する (GUI の停止操作用)。
    戻り値はエクスポータを使わない場合のための全フレーム結果。
    show=True のときは OpenCV ウィンドウでプレビューし、q で終了。
    """
    if draw_ids is None:
        draw_ids = tracker is not None
    results: List[FrameResult] = []
    fps_t0 = time.monotonic()
    fps_count = 0
    fps_value = 0.0

    try:
        for frame_index, timestamp_ms, frame in source:
            if max_frames is not None and frame_index >= max_frames:
                break
            persons = backend.process(frame, timestamp_ms)
            h, w = frame.shape[:2]
            if tracker is not None:
                persons = tracker.assign(
                    persons, w, h,
                    frame=frame, frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                )
            result = FrameResult(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                width=w,
                height=h,
                persons=persons,
                source=source.description,
            )
            results.append(result)
            for exporter in exporters:
                exporter.add(result)

            annotated = frame
            if trajectory is not None:
                trajectory.update(result)
                annotated = trajectory.draw(annotated)
            if draw:
                annotated = draw_result(
                    annotated, result, backend.skeleton,
                    min_visibility=min_visibility, draw_labels=draw_labels,
                    draw_ids=draw_ids,
                )

            fps_count += 1
            elapsed = time.monotonic() - fps_t0
            if elapsed >= 1.0:
                fps_value = fps_count / elapsed
                fps_count = 0
                fps_t0 = time.monotonic()

            if video_writer is not None:
                video_writer.write(annotated)
            if image_output is not None:
                imwrite(image_output, annotated)
            if show:
                preview = annotated.copy()
                draw_status(
                    preview,
                    f"frame {frame_index}  persons {len(persons)}  fps {fps_value:.1f}",
                )
                cv2.imshow("poselab", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if progress is not None:
                progress(frame_index + 1, source.frame_count)
            if on_frame is not None and on_frame(result, annotated) is False:
                break
    finally:
        for exporter in exporters:
            exporter.close()
        if video_writer is not None:
            video_writer.close()
        if show:
            cv2.destroyAllWindows()
        source.close()

    return results
