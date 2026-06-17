"""関節角度などの解析機能。

角度は「中点を頂点とする 3 点のなす角」(度) として計算します。
ワールド座標 (3D) があればそれを優先し、なければピクセル座標
(2D) で計算します。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Tuple

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
    """キーポイントの速度・加速度・ジャークの CSV (1 行 = 1 キーポイント)。

    フレーム間の有限差分から速度 (1 階)、加速度 (2 階)、ジャーク (3 階) を
    計算する。ピクセル座標の vx / vy / 速さ (px/s)・加速度 (px/s^2) と、
    ワールド座標の速さ (m/s)・加速度 (m/s^2)・ジャーク (m/s^3) を出力する。
    高次の量は前 2〜3 フレームが必要なため最初の数フレームは空欄になる。
    高次微分はノイズに敏感なので --smooth との併用を推奨。
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
        "accel_px_per_s2",
        "accel_m_per_s2",
        "jerk_m_per_s3",
    ]

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)
        # person_index -> (timestamp_ms, keypoints, world_keypoints)
        self._prev: Dict[int, tuple] = {}
        # (person, keypoint_id) -> (vx_px, vy_px, world_vel[3] or None)
        self._prev_vel: Dict[tuple, tuple] = {}
        # (person, keypoint_id) -> world_accel[3]
        self._prev_accel: Dict[tuple, np.ndarray] = {}

    def add(self, result: FrameResult) -> None:
        current = {}
        cur_vel: Dict[tuple, tuple] = {}
        cur_accel: Dict[tuple, np.ndarray] = {}
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
                        key = (pi, kp.index)
                        vx = (kp.x_px - pkp.x_px) / dt
                        vy = (kp.y_px - pkp.y_px) / dt
                        speed_m = ""
                        world_vel = None
                        wk, pwk = cur_w.get(kp.index), prev_w.get(kp.index)
                        if wk is not None and pwk is not None:
                            world_vel = (
                                np.array([wk.x, wk.y, wk.z])
                                - np.array([pwk.x, pwk.y, pwk.z])
                            ) / dt
                            speed_m = "%.6f" % float(np.linalg.norm(world_vel))

                        # 2 階 (加速度) / 3 階 (ジャーク)
                        accel_px = accel_m = jerk_m = ""
                        pv = self._prev_vel.get(key)
                        if pv is not None:
                            pvx, pvy, pworld_vel = pv
                            a_px = np.array([(vx - pvx) / dt, (vy - pvy) / dt])
                            accel_px = f"{float(np.linalg.norm(a_px)):.3f}"
                            if world_vel is not None and pworld_vel is not None:
                                a_w = (world_vel - pworld_vel) / dt
                                accel_m = "%.6f" % float(np.linalg.norm(a_w))
                                cur_accel[key] = a_w
                                pa = self._prev_accel.get(key)
                                if pa is not None:
                                    jerk = (a_w - pa) / dt
                                    jerk_m = "%.6f" % float(np.linalg.norm(jerk))
                        cur_vel[key] = (vx, vy, world_vel)

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
                                accel_px,
                                accel_m,
                                jerk_m,
                            ]
                        )
            current[pi] = (
                result.timestamp_ms,
                person.keypoints,
                person.world_keypoints,
            )
        self._prev = current
        self._prev_vel = cur_vel
        self._prev_accel = cur_accel

    def close(self) -> None:
        self._file.close()


class AngleCsvExporter(Exporter):
    """関節角度のロング形式 CSV (1 行 = 1 角度)。

    角度 (度) に加え、前フレームとの差分から角速度 (度/秒) も出力する。
    最初のフレームや角度が欠損したフレームでは角速度は空欄になる。
    """

    FIELDS = [
        "frame",
        "timestamp_ms",
        "person",
        "angle_name",
        "angle_deg",
        "angular_velocity_deg_per_s",
        "min_visibility",
        "coordinates",
    ]

    def __init__(self, path: "str | Path") -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)
        # (person, angle_name) -> (angle_deg, timestamp_ms)
        self._prev: Dict[tuple, tuple] = {}

    def add(self, result: FrameResult) -> None:
        for person in result.persons:
            pi = person.person_index
            for name, (deg, vis, system) in compute_person_angles(person).items():
                ang_vel = ""
                prev = self._prev.get((pi, name))
                if prev is not None and not np.isnan(deg):
                    prev_deg, prev_ts = prev
                    dt = (result.timestamp_ms - prev_ts) / 1000.0
                    if dt > 0 and not np.isnan(prev_deg):
                        ang_vel = f"{(deg - prev_deg) / dt:.3f}"
                if not np.isnan(deg):
                    self._prev[(pi, name)] = (deg, result.timestamp_ms)
                self._writer.writerow(
                    [
                        result.frame_index,
                        f"{result.timestamp_ms:.3f}",
                        pi,
                        name,
                        f"{deg:.2f}" if not np.isnan(deg) else "",
                        ang_vel,
                        f"{vis:.4f}",
                        system,
                    ]
                )

    def close(self) -> None:
        self._file.close()


class SymmetryCsvExporter(Exporter):
    """左右対称性のロング形式 CSV (1 行 = 1 関節ペア × フレーム)。

    対応する左右の関節角度から Symmetry Index を計算する。0 が左右対称、
    値が大きいほど非対称 (リハビリ・左右差評価向け)。どちらかの角度が
    欠損したペアは行を生成しない。
    """

    FIELDS = [
        "frame",
        "timestamp_ms",
        "person",
        "joint",
        "left_deg",
        "right_deg",
        "symmetry_index",
    ]

    def __init__(self, path: "str | Path") -> None:
        from poselab.kinematics import SYMMETRY_PAIRS

        self.path = Path(path)
        self._pairs = SYMMETRY_PAIRS
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def add(self, result: FrameResult) -> None:
        from poselab.kinematics import symmetry_index

        for person in result.persons:
            angles = compute_person_angles(person)
            for joint, left_name, right_name in self._pairs:
                left = angles.get(left_name)
                right = angles.get(right_name)
                if not left or not right:
                    continue
                left_deg, right_deg = left[0], right[0]
                if np.isnan(left_deg) or np.isnan(right_deg):
                    continue
                self._writer.writerow(
                    [
                        result.frame_index,
                        f"{result.timestamp_ms:.3f}",
                        person.person_index,
                        joint,
                        f"{left_deg:.2f}",
                        f"{right_deg:.2f}",
                        f"{symmetry_index(left_deg, right_deg):.4f}",
                    ]
                )

    def close(self) -> None:
        self._file.close()
