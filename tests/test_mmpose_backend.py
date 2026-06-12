"""MMPose バックエンドのテスト (mmpose 本体は不要、フェイク注入)。"""

from __future__ import annotations

import numpy as np
import pytest

from poselab.backends.mmpose_backend import (
    DEFAULT_POSE2D,
    MMPoseBackend,
    instances_to_persons,
)
from poselab.skeleton import COCO17_EDGES, COCO17_NAMES


def make_instance(x0: float, y0: float, score: float, n: int = 17) -> dict:
    return {
        "keypoints": [[x0 + i, y0 + i] for i in range(n)],
        "keypoint_scores": [score] * n,
    }


class FakePose2DInferencer:
    """MMPose Pose2DInferencer の出力形式を模したフェイク。"""

    def __init__(self, instances_per_call):
        self.instances_per_call = instances_per_call
        self.calls = []

    def __call__(self, inputs, return_datasamples=False, **kwargs):
        self.calls.append(kwargs)
        yield {"predictions": [self.instances_per_call]}


def test_instances_to_persons_basic():
    persons = instances_to_persons(
        [make_instance(10, 20, 0.9)], width=640, height=480
    )
    assert len(persons) == 1
    person = persons[0]
    assert len(person.keypoints) == 17
    kp = person.keypoints[0]
    assert kp.name == "nose"
    assert kp.x_px == 10.0 and kp.y_px == 20.0
    assert kp.x_norm == pytest.approx(10 / 640)
    assert kp.y_norm == pytest.approx(20 / 480)
    assert kp.visibility == pytest.approx(0.9)
    assert person.world_keypoints == []


def test_instances_to_persons_caps_and_sorts_by_score():
    instances = [
        make_instance(0, 0, 0.5),
        make_instance(100, 100, 0.9),
        make_instance(200, 200, 0.7),
    ]
    persons = instances_to_persons(instances, 640, 480, max_persons=2)
    assert len(persons) == 2
    # スコア順 (0.9 → 0.7) に並ぶ
    assert persons[0].keypoints[0].x_px == 100.0
    assert persons[1].keypoints[0].x_px == 200.0
    # person_index は振り直し
    assert [p.person_index for p in persons] == [0, 1]


def test_instances_to_persons_skips_empty():
    persons = instances_to_persons(
        [{"keypoints": [], "keypoint_scores": []}], 640, 480
    )
    assert persons == []


def test_backend_process_with_fake_inferencer():
    fake = FakePose2DInferencer([make_instance(10, 20, 0.8)])
    backend = MMPoseBackend(num_poses=1, inferencer=fake)
    assert list(backend.keypoint_names) == list(COCO17_NAMES)
    assert list(backend.skeleton) == list(COCO17_EDGES)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = backend.process(frame, 0.0)
    assert len(persons) == 1
    assert persons[0].keypoints[5].name == "left_shoulder"
    # しきい値が inferencer へ渡る
    assert fake.calls[0]["bbox_thr"] == pytest.approx(0.3)
    backend.close()


def test_backend_factory_requires_mmpose():
    from poselab.backends import create_backend

    try:
        import mmpose  # noqa: F401

        pytest.skip("mmpose がインストールされている環境ではスキップ")
    except ImportError:
        pass
    with pytest.raises(ImportError) as excinfo:
        create_backend("mmpose")
    assert "mim install" in str(excinfo.value)


def test_default_model_names():
    # 既定モデルは MMPose model zoo のコンフィグ名 (英数字とハイフン等のみ)
    assert "rtmpose" in DEFAULT_POSE2D
