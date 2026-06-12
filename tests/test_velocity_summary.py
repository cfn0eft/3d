import csv

import pytest

from poselab.analysis import VelocityCsvExporter, summarize_results
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


def _frame(frame_index: int, ts_ms: float, x_px: float, wx: float) -> FrameResult:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=0.5, y_norm=0.5, z=0.0,
            visibility=0.9, presence=0.9, x_px=x_px, y_px=100.0,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    world = [
        WorldKeypoint(index=i, name=name, x=wx, y=0.0, z=0.0, visibility=0.9)
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return FrameResult(
        frame_index=frame_index, timestamp_ms=ts_ms, width=640, height=480,
        persons=[PersonPose(0, keypoints, world)],
    )


def test_velocity_exporter(tmp_path):
    path = tmp_path / "vel.csv"
    exporter = VelocityCsvExporter(path)
    # 100 ms で 10 px / 0.05 m 移動 → 100 px/s, 0.5 m/s
    exporter.add(_frame(0, 0.0, 100.0, 0.0))
    exporter.add(_frame(1, 100.0, 110.0, 0.05))
    exporter.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 33  # 最初のフレームは行なし
    assert float(rows[0]["vx_px_per_s"]) == pytest.approx(100.0)
    assert float(rows[0]["speed_px_per_s"]) == pytest.approx(100.0)
    assert float(rows[0]["speed_m_per_s"]) == pytest.approx(0.5)


def test_velocity_skips_detection_gaps(tmp_path):
    path = tmp_path / "vel.csv"
    exporter = VelocityCsvExporter(path)
    exporter.add(_frame(0, 0.0, 100.0, 0.0))
    exporter.add(FrameResult(1, 100.0, 640, 480))  # 未検出フレーム
    exporter.add(_frame(2, 200.0, 120.0, 0.1))
    exporter.close()
    rows = list(csv.DictReader(open(path)))
    assert rows == []  # 連続検出がないため速度は計算されない


def test_summarize_results():
    results = [
        _frame(0, 0.0, 100.0, 0.0),
        _frame(1, 100.0, 110.0, 0.05),
        FrameResult(2, 200.0, 640, 480),  # 未検出
    ]
    summary = summarize_results(results)
    assert summary["total_frames"] == 3
    assert summary["detected_frames"] == 2
    assert summary["detection_rate"] == pytest.approx(2 / 3)
    assert summary["max_persons"] == 1
    assert summary["duration_s"] == pytest.approx(0.2)
    assert summary["mean_visibility"] == pytest.approx(0.9)


def test_summarize_empty():
    summary = summarize_results([])
    assert summary["total_frames"] == 0
    assert summary["detection_rate"] == 0.0
