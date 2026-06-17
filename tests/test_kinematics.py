"""高次運動学 (kinematics) と関連エクスポータのテスト。"""

import csv
import math

import numpy as np
import pytest

from poselab.analysis import (
    AngleCsvExporter,
    SymmetryCsvExporter,
    VelocityCsvExporter,
)
from poselab.kinematics import (
    estimate_cadence,
    estimate_period,
    gait_summary,
    symmetry_index,
)
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint

_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def _person(pixels=None, world=None):
    """テスト用の人物。world を渡したときだけ world_keypoints を埋める
    (空なら compute_person_angles はピクセル座標へフォールバックする)。"""
    pixels = pixels or {}
    kps = []
    for i, name in enumerate(LANDMARK_NAMES):
        px, py = pixels.get(name, (0.0, 0.0))
        kps.append(Keypoint(index=i, name=name, x_norm=0, y_norm=0, z=0,
                            visibility=0.9, presence=0.9, x_px=px, y_px=py))
    wks = []
    if world:
        for i, name in enumerate(LANDMARK_NAMES):
            wx, wy, wz = world.get(name, (0.0, 0.0, 0.0))
            wks.append(
                WorldKeypoint(index=i, name=name, x=wx, y=wy, z=wz, visibility=0.9)
            )
    return PersonPose(person_index=0, keypoints=kps, world_keypoints=wks)


# ----------------------------------------------------------------- 純関数


def test_symmetry_index():
    assert symmetry_index(90, 90) == pytest.approx(0.0)
    assert symmetry_index(0, 0) == pytest.approx(0.0)
    assert symmetry_index(100, 80) == pytest.approx(20 / 90)


def test_estimate_period_sine():
    period = 1.0
    t = np.arange(0, 4, 0.05)
    signal = np.sin(2 * math.pi * t / period)
    est = estimate_period(t, signal)
    assert est == pytest.approx(period, abs=0.1)
    assert estimate_cadence(t, signal) == pytest.approx(60.0, abs=6.0)


def test_estimate_period_insufficient():
    assert estimate_period([0, 1, 2], [0, 1, 0]) is None
    # 周期性のない信号
    assert estimate_period(np.arange(0, 2, 0.05), np.arange(0, 2, 0.05)) is None


def test_gait_summary():
    period = 0.8
    frames = []
    for i in range(120):
        t_ms = i * (1000 / 30.0)
        y = math.sin(2 * math.pi * (t_ms / 1000.0) / period)
        person = _person(world={"left_ankle": (0.0, y, 0.0)})
        frames.append(FrameResult(i, t_ms, 640, 480, persons=[person]))
    gait = gait_summary(frames)
    assert "left_ankle" in gait
    assert gait["left_ankle"]["cycle_time_s"] == pytest.approx(period, abs=0.1)


# ----------------------------------------------------------------- 加速度


def test_velocity_acceleration_columns(tmp_path):
    path = tmp_path / "vel.csv"
    exp = VelocityCsvExporter(path)
    # x: 0 -> 10 -> 30 (dt=0.1s) => v: 100, 200 => a=(200-100)/0.1=1000
    for i, x in enumerate((0.0, 10.0, 30.0)):
        exp.add(FrameResult(i, i * 100.0, 640, 480,
                            persons=[_person(pixels={"nose": (x, 0.0)})]))
    exp.close()
    rows = list(csv.DictReader(open(path)))
    nose_rows = [r for r in rows if r["keypoint_name"] == "nose"]
    assert nose_rows[0]["accel_px_per_s2"] == ""          # 2 フレーム目は未定
    assert float(nose_rows[1]["accel_px_per_s2"]) == pytest.approx(1000.0)


def test_angle_angular_velocity(tmp_path):
    path = tmp_path / "ang.csv"
    exp = AngleCsvExporter(path)
    straight = {"left_shoulder": (0, 0), "left_elbow": (10, 0), "left_wrist": (20, 0)}
    bent = {"left_shoulder": (0, 0), "left_elbow": (10, 0), "left_wrist": (10, 10)}
    exp.add(FrameResult(0, 0.0, 640, 480, persons=[_person(pixels=straight)]))
    exp.add(FrameResult(1, 100.0, 640, 480, persons=[_person(pixels=bent)]))
    exp.close()
    rows = list(csv.DictReader(open(path)))
    elbow = [r for r in rows if r["angle_name"] == "left_elbow"]
    assert elbow[0]["angular_velocity_deg_per_s"] == ""     # 初フレーム
    # 180度 -> 90度 / 0.1s = -900 deg/s
    assert float(elbow[1]["angular_velocity_deg_per_s"]) == pytest.approx(-900.0)


def test_symmetry_csv(tmp_path):
    path = tmp_path / "sym.csv"
    # 左肘=直角(90), 右肘=直線(180) で非対称
    pixels = {
        "left_shoulder": (0, 0), "left_elbow": (10, 0), "left_wrist": (10, 10),
        "right_shoulder": (0, 0), "right_elbow": (10, 0), "right_wrist": (20, 0),
    }
    exp = SymmetryCsvExporter(path)
    exp.add(FrameResult(0, 0.0, 640, 480, persons=[_person(pixels=pixels)]))
    exp.close()
    rows = list(csv.DictReader(open(path)))
    elbow = next(r for r in rows if r["joint"] == "elbow")
    assert float(elbow["left_deg"]) == pytest.approx(90.0)
    assert float(elbow["right_deg"]) == pytest.approx(180.0)
    assert float(elbow["symmetry_index"]) == pytest.approx(
        symmetry_index(90, 180), abs=1e-3
    )
