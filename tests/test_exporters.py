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


@pytest.fixture
def mixed_visibility_result():
    """1 フレーム・1 人で、最初の点が低 visibility のデータ。"""
    keypoints = []
    world = []
    for i, name in enumerate(LANDMARK_NAMES):
        vis = 0.1 if i == 0 else 0.9
        keypoints.append(
            Keypoint(index=i, name=name, x_norm=0.5, y_norm=0.25, z=-0.1,
                     visibility=vis, presence=vis, x_px=320.0, y_px=120.0)
        )
        world.append(
            WorldKeypoint(index=i, name=name, x=0.1, y=-0.2, z=0.05, visibility=vis)
        )
    person = PersonPose(person_index=0, keypoints=keypoints, world_keypoints=world)
    return [FrameResult(frame_index=0, timestamp_ms=0.0, width=640, height=480,
                        persons=[person])]


def test_csv_mask_visibility(tmp_path, mixed_visibility_result):
    path = tmp_path / "masked.csv"
    export_results(mixed_visibility_result, [CsvExporter(path, mask_visibility=0.5)])
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # 低 visibility の点 (index 0) は座標が空欄、信頼度は残る
    assert rows[0]["x_px"] == ""
    assert rows[0]["world_x"] == ""
    assert float(rows[0]["visibility"]) == pytest.approx(0.1)
    # 高 visibility の点は通常どおり
    assert float(rows[1]["x_px"]) == pytest.approx(320.0)


def test_npz_mask_visibility(tmp_path, mixed_visibility_result):
    path = tmp_path / "masked.npz"
    export_results(
        mixed_visibility_result,
        [NpzExporter(path, LANDMARK_NAMES, max_persons=1, mask_visibility=0.5)],
    )
    data = np.load(path, allow_pickle=False)
    # 低 visibility の点は座標 NaN、visibility は保持
    assert np.isnan(data["keypoints"][0, 0, 0, 0])  # x_px
    assert data["keypoints"][0, 0, 0, 3] == pytest.approx(0.1)  # visibility
    assert np.isnan(data["world"][0, 0, 0, 0])
    # 高 visibility の点は座標が残る
    assert data["keypoints"][0, 0, 1, 0] == pytest.approx(320.0)


def test_mask_visibility_off_is_unchanged(tmp_path, mixed_visibility_result):
    path = tmp_path / "raw.csv"
    export_results(mixed_visibility_result, [CsvExporter(path)])  # 既定 0.0
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["x_px"]) == pytest.approx(320.0)  # マスクされない


def test_npz_exporter_metadata(tmp_path, sample_results):
    path = tmp_path / "out.npz"
    meta = {"tool": "test", "units": {"world_x": "meter"}}
    export_results(
        sample_results,
        [NpzExporter(path, LANDMARK_NAMES, max_persons=1, metadata=meta)],
    )
    data = np.load(path, allow_pickle=False)
    assert "metadata_json" in data
    loaded = json.loads(str(data["metadata_json"]))
    assert loaded["tool"] == "test"
    assert loaded["units"]["world_x"] == "meter"
