"""バックエンドの共通インターフェース。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Tuple

import numpy as np

from poselab.types import PersonPose


class PoseBackend(ABC):
    """骨格推定バックエンドの抽象基底クラス。

    実装クラスは BGR フレームを受け取り PersonPose のリストを返す。
    """

    name: str = "base"

    @property
    @abstractmethod
    def keypoint_names(self) -> Sequence[str]:
        """index 順のキーポイント名。"""

    @property
    @abstractmethod
    def skeleton(self) -> Sequence[Tuple[int, int]]:
        """描画・解析用の骨格エッジ (キーポイント index の組)。"""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> List[PersonPose]:
        """1 フレームを推定する。timestamp_ms は単調増加であること。"""

    def close(self) -> None:
        """リソースを解放する。"""

    def __enter__(self) -> "PoseBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
