import math

import pytest

from poselab.analysis import (
    ANGLE_DEFINITIONS,
    AngleCsvExporter,
    compute_person_angles,
    joint_angle,
)
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose


def test_joint_angle_straight():
    assert joint_angle([0, 0], [1, 0], [2, 0]) == pytest.approx(180.0)


def test_joint_angle_right():
    assert joint_angle([1, 0], [0, 0], [0, 1]) == pytest.approx(90.0)


def test_joint_angle_3d():
    assert joint_angle([1, 0, 0], [0, 0, 0], [0, 0, 1]) == pytest.approx(90.0)


def test_joint_angle_degenerate():
    assert math.isnan(joint_angle([0, 0], [0, 0], [1, 1]))


def _person_with_pixel_coords(coords: dict) -> PersonPose:
    keypoints = []
    for i, name in enumerate(LANDMARK_NAMES):
        x, y = coords.get(name, (0.0, 0.0))
        keypoints.append(
            Keypoint(
                index=i, name=name, x_norm=0, y_norm=0, z=0,
                visibility=0.9, presence=0.9, x_px=x, y_px=y,
            )
        )
    return PersonPose(person_index=0, keypoints=keypoints)


def test_compute_person_angles_pixel_fallback():
    # 肘を直角に曲げた左腕
    person = _person_with_pixel_coords(
        {
            "left_shoulder": (0.0, 0.0),
            "left_elbow": (10.0, 0.0),
            "left_wrist": (10.0, 10.0),
        }
    )
    angles = compute_person_angles(person)
    assert set(angles) == set(ANGLE_DEFINITIONS)
    deg, vis, system = angles["left_elbow"]
    assert deg == pytest.approx(90.0)
    assert system == "pixel"
    assert vis == pytest.approx(0.9)


def test_angle_csv_exporter(tmp_path):
    person = _person_with_pixel_coords(
        {
            "left_shoulder": (0.0, 0.0),
            "left_elbow": (10.0, 0.0),
            "left_wrist": (20.0, 0.0),
        }
    )
    result = FrameResult(
        frame_index=0, timestamp_ms=0.0, width=640, height=480, persons=[person]
    )
    path = tmp_path / "angles.csv"
    exporter = AngleCsvExporter(path)
    exporter.add(result)
    exporter.close()
    import csv

    rows = list(csv.DictReader(open(path)))
    assert len(rows) == len(ANGLE_DEFINITIONS)
    elbow = next(r for r in rows if r["angle_name"] == "left_elbow")
    assert float(elbow["angle_deg"]) == pytest.approx(180.0)
