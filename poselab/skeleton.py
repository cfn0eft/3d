"""33 点ランドマークモデルの骨格トポロジー定義。

ランドマークの並び順と接続関係は MediaPipe Pose Landmarker の
公開ドキュメントに記載されている仕様 (事実情報) に基づき、
本ファイルの実装は独自に記述したものです。
"""

from __future__ import annotations

from typing import List, Tuple

# index 順のランドマーク名 (33 点)
LANDMARK_NAMES: List[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

NUM_LANDMARKS = len(LANDMARK_NAMES)

# 骨格の接続 (描画・解析用エッジリスト)
SKELETON_EDGES: List[Tuple[int, int]] = [
    # 顔まわり
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # 体幹
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左腕・左手
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # 右腕・右手
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # 左脚・左足
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    # 右脚・右足
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def landmark_side(name: str) -> str:
    """ランドマークの体側を返す ('left' / 'right' / 'center')。"""
    if name.startswith("left_") or name.endswith("_left"):
        return "left"
    if name.startswith("right_") or name.endswith("_right"):
        return "right"
    return "center"


# ---------------------------------------------------------------------------
# 追加の骨格カタログ (MMPose バックエンド用)
#
# COCO 17 点 / Human3.6M 17 点の並び順と接続関係は、各データセットの
# 公開仕様 (事実情報) に基づき独自に記述したものです。
# ---------------------------------------------------------------------------

# COCO キーポイント (RTMPose など 2D モデルの標準出力)
COCO17_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_EDGES: List[Tuple[int, int]] = [
    # 顔まわり
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 体幹
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 腕
    (5, 7), (7, 9), (6, 8), (8, 10),
    # 脚
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Human3.6M キーポイント (VideoPose3D など 3D リフタの標準出力)
H36M17_NAMES: List[str] = [
    "root",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

H36M17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3),       # 右脚
    (0, 4), (4, 5), (5, 6),       # 左脚
    (0, 7), (7, 8), (8, 9), (9, 10),   # 体幹・頭
    (8, 11), (11, 12), (12, 13),  # 左腕
    (8, 14), (14, 15), (15, 16),  # 右腕
]
