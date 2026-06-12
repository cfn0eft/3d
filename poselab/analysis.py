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


def summarize_results(results) -> Dict[str, object]:
    """処理結果全体のサマリ統計を返す。

    検出率・平均人数・平均 visibility など、データの品質確認に
    使える指標をまとめる。
    """
    n_frames = len(results)
    detected = [r for r in results if r.persons]
    visibilities = [
        kp.visibility
        for r in detected
        for p in r.persons
        for kp in p.keypoints
    ]
    duration_ms = 0.0
    if n_frames >= 2:
        duration_ms = results[-1].timestamp_ms - results[0].timestamp_ms
    return {
        "total_frames": n_frames,
        "detected_frames": len(detected),
        "detection_rate": len(detected) / n_frames if n_frames else 0.0,
        "max_persons": max((len(r.persons) for r in results), default=0),
        "mean_persons": (
            sum(len(r.persons) for r in results) / n_frames if n_frames else 0.0
        ),
        "mean_visibility": (
            float(np.mean(visibilities)) if visibilities else 0.0
        ),
        "duration_s": duration_ms / 1000.0,
    }


class VelocityCsvExporter(Exporter):
    """キーポイント速度のロング形式 CSV (1 行 = 1 キーポイント)。

    前フレームとの差分から速度を計算する。ピクセル座標系の
    vx / vy / 速さ (px/s) と、ワールド座標系の速さ (m/s) を出力。
    最初のフレーム (差分が取れない) は行を生成しない。
    """

    FIELDS = [
        "frame",
        "timestamp_ms",
        "person",
        "keypoint_id",
        "keypoint_name",
        "vx_px_per_s",
        "vy_px_per_s",
        "speed_px_per_s",
        "speed_m_per_s",
    ]

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)
        # person_index -> (timestamp_ms, keypoints, world_keypoints)
        self._prev: Dict[int, tuple] = {}

    def add(self, result: FrameResult) -> None:
        current = {}
        for person in result.persons:
            pi = person.person_index
            prev = self._prev.get(pi)
            if prev is not None:
                prev_ts, prev_kps, prev_world = prev
                dt = (result.timestamp_ms - prev_ts) / 1000.0
                if dt > 0:
                    prev_w = {wk.index: wk for wk in prev_world}
                    cur_w = {wk.index: wk for wk in person.world_keypoints}
                    for kp, pkp in zip(person.keypoints, prev_kps):
                        vx = (kp.x_px - pkp.x_px) / dt
                        vy = (kp.y_px - pkp.y_px) / dt
                        speed_m = ""
                        wk, pwk = cur_w.get(kp.index), prev_w.get(kp.index)
                        if wk is not None and pwk is not None:
                            speed_m = "%.6f" % (
                                float(
                                    np.linalg.norm(
                                        [
                                            wk.x - pwk.x,
                                            wk.y - pwk.y,
                                            wk.z - pwk.z,
                                        ]
                                    )
                                )
                                / dt
                            )
                        self._writer.writerow(
                            [
                                result.frame_index,
                                f"{result.timestamp_ms:.3f}",
                                pi,
                                kp.index,
                                kp.name,
                                f"{vx:.3f}",
                                f"{vy:.3f}",
                                f"{float(np.hypot(vx, vy)):.3f}",
                                speed_m,
                            ]
                        )
            current[pi] = (
                result.timestamp_ms,
                person.keypoints,
                person.world_keypoints,
            )
        self._prev = current

    def close(self) -> None:
        self._file.close()


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
