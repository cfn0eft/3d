"""推定結果を表すデータ型。バックエンド非依存。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Keypoint:
    """画像座標系のキーポイント。

    x_norm / y_norm は画像サイズで正規化された値 (0.0-1.0)、
    x_px / y_px はピクセル座標。z は腰中心を原点とする相対深度
    (画像幅と同程度のスケール、カメラに近いほど負)。
    visibility は「画面内に写っていて遮蔽されていない」推定確率。
    """

    index: int
    name: str
    x_norm: float
    y_norm: float
    z: float
    visibility: float
    presence: float
    x_px: float
    y_px: float


@dataclass
class WorldKeypoint:
    """実世界座標系 (メートル単位、腰中心が原点) のキーポイント。"""

    index: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PersonPose:
    """1 人分の推定結果。"""

    person_index: int
    keypoints: List[Keypoint] = field(default_factory=list)
    world_keypoints: List[WorldKeypoint] = field(default_factory=list)


@dataclass
class FrameResult:
    """1 フレーム分の推定結果。"""

    frame_index: int
    timestamp_ms: float
    width: int
    height: int
    persons: List[PersonPose] = field(default_factory=list)
    source: Optional[str] = None
