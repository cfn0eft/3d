import numpy as np
import pytest

from poselab.filters import smooth_results
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


def _frame(frame_index: int, x: float) -> FrameResult:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=x / 640.0, y_norm=0.5, z=0.0,
            visibility=0.9, presence=0.9, x_px=x, y_px=240.0,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    world = [
        WorldKeypoint(index=i, name=name, x=x / 100.0, y=0.0, z=0.0, visibility=0.9)
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    person = PersonPose(person_index=0, keypoints=keypoints, world_keypoints=world)
    return FrameResult(
        frame_index=frame_index, timestamp_ms=frame_index * 33.3,
        width=640, height=480, persons=[person],
    )


def test_smooth_constant_signal_unchanged():
    results = [_frame(i, 100.0) for i in range(10)]
    smooth_results(results, window=5)
    for r in results:
        assert r.persons[0].keypoints[0].x_px == pytest.approx(100.0)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(0)
    base = 100.0
    noisy = base + rng.normal(0, 5.0, size=30)
    results = [_frame(i, float(v)) for i, v in enumerate(noisy)]
    smooth_results(results, window=5)
    smoothed = np.array([r.persons[0].keypoints[0].x_px for r in results])
    assert np.std(smoothed) < np.std(noisy)
    # x_norm も再計算されている
    assert results[5].persons[0].keypoints[0].x_norm == pytest.approx(
        smoothed[5] / 640.0
    )


def test_smooth_handles_missing_frames():
    results = [_frame(0, 100.0), _frame(1, 110.0)]
    results.append(
        FrameResult(frame_index=2, timestamp_ms=66.6, width=640, height=480)
    )
    results.append(_frame(3, 120.0))
    out = smooth_results(results, window=3)
    assert out[2].persons == []  # 未検出フレームはそのまま
    assert not np.isnan(out[3].persons[0].keypoints[0].x_px)


def test_smooth_window_one_noop():
    results = [_frame(i, float(i)) for i in range(3)]
    smooth_results(results, window=1)
    assert results[2].persons[0].keypoints[0].x_px == pytest.approx(2.0)
