"""関節角度などの解析機能。

角度は「中点を頂点とする 3 点のなす角」(度) として計算します。
ワールド座標 (3D) があればそれを優先し、なければピクセル座標
(2D) で計算します。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from poselab.exporters import Exporter
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, PersonPose

_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}

# 角度名 -> (始点, 頂点, 終点) のランドマーク名
ANGLE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
}


def joint_angle(
    a: np.ndarray, vertex: np.ndarray, c: np.ndarray
) -> float:
    """vertex を頂点とした a-vertex-c のなす角を度で返す (0-180)。"""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(vertex, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cosine = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_person_angles(
    person: PersonPose,
) -> Dict[str, Tuple[float, float, str]]:
    """1 人分の関節角度を計算する。

    Returns
    -------
    dict: 角度名 -> (角度 [度], 3 点の visibility の最小値, 座標系)
        座標系は "world" (3D) または "pixel" (2D)。
    """
    use_world = bool(person.world_keypoints)
    angles: Dict[str, Tuple[float, float, str]] = {}
    for angle_name, (na, nb, nc) in ANGLE_DEFINITIONS.items():
        ia, ib, ic = _INDEX[na], _INDEX[nb], _INDEX[nc]
        if use_world:
            pts = [person.world_keypoints[i] for i in (ia, ib, ic)]
            coords = [np.array([p.x, p.y, p.z]) for p in pts]
            system = "world"
        else:
            pts = [person.keypoints[i] for i in (ia, ib, ic)]
            coords = [np.array([p.x_px, p.y_px]) for p in pts]
            system = "pixel"
        visibility = min(p.visibility for p in pts)
        angles[angle_name] = (
            joint_angle(coords[0], coords[1], coords[2]),
            visibility,
            system,
        )
    return angles


class AngleCsvExporter(Exporter):
    """関節角度のロング形式 CSV (1 行 = 1 角度)。"""

    FIELDS = [
        "frame",
        "timestamp_ms",
        "person",
        "angle_name",
        "angle_deg",
        "min_visibility",
        "coordinates",
    ]

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def add(self, result: FrameResult) -> None:
        for person in result.persons:
            for name, (deg, vis, system) in compute_person_angles(person).items():
                self._writer.writerow(
                    [
                        result.frame_index,
                        f"{result.timestamp_ms:.3f}",
                        person.person_index,
                        name,
                        f"{deg:.2f}" if not np.isnan(deg) else "",
                        f"{vis:.4f}",
                        system,
                    ]
                )

    def close(self) -> None:
        self._file.close()
