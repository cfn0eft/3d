"""CLI 向けの進捗表示 (%, fps, 残り時間)。"""

from __future__ import annotations

import sys
import time
from typing import Optional, TextIO


def format_duration(seconds: float) -> str:
    """秒数を h:mm:ss / m:ss 形式の文字列にする。"""
    seconds = max(0, int(round(seconds)))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class ProgressReporter:
    """ターミナルに 1 行で進捗を描画する。

    総数が分かる場合はバー + % + 残り時間、不明 (カメラ等) の場合は
    処理フレーム数 + fps + 経過時間を表示する。
    """

    BAR_WIDTH = 24

    def __init__(
        self,
        total: Optional[int] = None,
        stream: Optional[TextIO] = None,
        min_interval: float = 0.1,
        enabled: bool = True,
    ) -> None:
        self.total = total
        self.stream = stream if stream is not None else sys.stderr
        self.min_interval = min_interval
        self.enabled = enabled
        self._start = time.monotonic()
        self._last_render = 0.0
        self._done = 0

    def update(self, done: int) -> None:
        self._done = done
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_render < self.min_interval:
            return
        self._last_render = now
        self.stream.write("\r" + self.render())
        self.stream.flush()

    def render(self) -> str:
        elapsed = time.monotonic() - self._start
        fps = self._done / elapsed if elapsed > 0 else 0.0
        if self.total:
            ratio = min(self._done / self.total, 1.0)
            filled = int(ratio * self.BAR_WIDTH)
            bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
            eta = (self.total - self._done) / fps if fps > 0 else 0.0
            return (
                f"[{bar}] {ratio * 100:5.1f}%  "
                f"{self._done}/{self.total}  "
                f"{fps:.1f} fps  残り {format_duration(eta)}"
            )
        return (
            f"{self._done} フレーム  {fps:.1f} fps  "
            f"経過 {format_duration(elapsed)}"
        )

    def finish(self) -> None:
        if not self.enabled:
            return
        self._last_render = 0.0
        self.stream.write("\r" + self.render() + "\n")
        self.stream.flush()
