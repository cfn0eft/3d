import csv
import json

import numpy as np
import pytest

from poselab.exporters import CsvExporter, JsonExporter, NpzExporter, export_results
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


@pytest.fixture
def sample_results():
    results = []
    for frame in range(3):
        keypoints = [
            Keypoint(
                index=i,
                name=name,
                x_norm=0.5,
                y_norm=0.25,
                z=-0.1,
                visibility=0.9,
                presence=0.95,
                x_px=320.0,
                y_px=120.0,
            )
            for i, name in enumerate(LANDMARK_NAMES)
        ]
        world = [
            WorldKeypoint(index=i, name=name, x=0.1, y=-0.2, z=0.05, visibility=0.9)
            for i, name in enumerate(LANDMARK_NAMES)
        ]
        person = PersonPose(person_index=0, keypoints=keypoints, world_keypoints=world)
        results.append(
            FrameResult(
                frame_index=frame,
                timestamp_ms=frame * 33.3,
                width=640,
                height=480,
                persons=[person],
            )
        )
    # 検出なしフレーム
    results.append(
        FrameResult(frame_index=3, timestamp_ms=99.9, width=640, height=480)
    )
    return results


def test_csv_exporter(tmp_path, sample_results):
    path = tmp_path / "out.csv"
    export_results(sample_results, [CsvExporter(path)])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * 33  # 検出なしフレームは行を生成しない
    assert rows[0]["keypoint_name"] == "nose"
    assert float(rows[0]["x_px"]) == pytest.approx(320.0)
    assert float(rows[0]["world_z"]) == pytest.approx(0.05)


def test_json_exporter(tmp_path, sample_results):
    path = tmp_path / "out.json"
    export_results(
        sample_results, [JsonExporter(path, LANDMARK_NAMES, {"tool": "test"})]
    )
    with open(path) as f:
        data = json.load(f)
    assert data["metadata"]["tool"] == "test"
    assert data["metadata"]["keypoint_names"] == LANDMARK_NAMES
    assert len(data["frames"]) == 4
    assert data["frames"][0]["persons"][0]["keypoints"][0]["name"] == "nose"
    assert data["frames"][3]["persons"] == []


def test_npz_exporter(tmp_path, sample_results):
    path = tmp_path / "out.npz"
    export_results(sample_results, [NpzExporter(path, LANDMARK_NAMES, max_persons=1)])
    data = np.load(path, allow_pickle=False)
    assert data["keypoints"].shape == (4, 1, 33, 5)
    assert data["world"].shape == (4, 1, 33, 4)
    assert np.isnan(data["keypoints"][3]).all()  # 検出なしフレームは NaN
    assert data["keypoints"][0, 0, 0, 0] == pytest.approx(320.0)
    assert list(data["keypoint_names"]) == LANDMARK_NAMES
