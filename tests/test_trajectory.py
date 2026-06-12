import numpy as np
import pytest

from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose
from poselab.visualize import TrajectoryOverlay


def _frame(frame_index: int, x: float, visibility: float = 0.9) -> FrameResult:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=0.5, y_norm=0.5, z=0.0,
            visibility=visibility, presence=0.9, x_px=x, y_px=100.0,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return FrameResult(
        frame_index=frame_index, timestamp_ms=frame_index * 33.3,
        width=640, height=480, persons=[PersonPose(0, keypoints)],
    )


def test_trajectory_draws_trail():
    overlay = TrajectoryOverlay(["right_wrist"], length=10)
    for i in range(5):
        overlay.update(_frame(i, 100.0 + i * 20))
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay.draw(canvas)
    assert canvas.sum() > 0  # 何かしら描画されている
    # 軌跡は y=100 付近の帯にだけ存在する
    assert canvas[:90].sum() == 0
    assert canvas[110:].sum() == 0


def test_trajectory_breaks_on_gap():
    overlay = TrajectoryOverlay(["right_wrist"], length=10)
    overlay.update(_frame(0, 100.0))
    overlay.update(FrameResult(1, 33.3, 640, 480))  # 未検出
    overlay.update(_frame(2, 500.0))
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay.draw(canvas)
    # 切れ目を跨いだ線 (x=100..500) は引かれない
    assert canvas[:, 200:400].sum() == 0


def test_trajectory_low_visibility_breaks():
    overlay = TrajectoryOverlay(["right_wrist"], length=10, min_visibility=0.5)
    overlay.update(_frame(0, 100.0))
    overlay.update(_frame(1, 300.0, visibility=0.1))
    overlay.update(_frame(2, 500.0))
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay.draw(canvas)
    assert canvas[:, 150:450].sum() == 0


def test_trajectory_all_keyword():
    overlay = TrajectoryOverlay(["all"], length=5)
    assert len(overlay.indices) == len(LANDMARK_NAMES)


def test_trajectory_unknown_name():
    with pytest.raises(ValueError):
        TrajectoryOverlay(["no_such_point"])


def test_trajectory_length_limit():
    overlay = TrajectoryOverlay(["right_wrist"], length=3)
    for i in range(10):
        overlay.update(_frame(i, 100.0 + i))
    trail = overlay._trails[(0, LANDMARK_NAMES.index("right_wrist"))]
    assert len(trail) == 3


def test_trajectory_reset():
    overlay = TrajectoryOverlay(["right_wrist"], length=5)
    overlay.update(_frame(0, 100.0))
    overlay.reset()
    assert not overlay._trails
