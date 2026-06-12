import csv

import pytest

from poselab.features import DistanceCsvExporter, parse_pair
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


def _frame(frame_index: int = 0) -> FrameResult:
    keypoints = []
    for i, name in enumerate(LANDMARK_NAMES):
        # nose は (0, 0)、right_wrist (index 16) は (30, 40) に置く
        x, y = (30.0, 40.0) if name == "right_wrist" else (0.0, 0.0)
        keypoints.append(
            Keypoint(
                index=i, name=name, x_norm=0, y_norm=0, z=0,
                visibility=0.9, presence=0.9, x_px=x, y_px=y,
            )
        )
    world = [
        WorldKeypoint(
            index=i, name=name,
            x=0.3 if name == "right_wrist" else 0.0, y=0.0, z=0.0,
            visibility=0.9,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return FrameResult(
        frame_index=frame_index, timestamp_ms=frame_index * 33.3,
        width=640, height=480,
        persons=[PersonPose(0, keypoints, world)],
    )


def test_parse_pair():
    assert parse_pair("right_wrist:nose") == ("right_wrist", "nose")
    with pytest.raises(ValueError):
        parse_pair("right_wrist")
    with pytest.raises(ValueError):
        parse_pair("no_such:nose")
    with pytest.raises(ValueError):
        parse_pair("nose:nose")


def test_distance_exporter(tmp_path):
    path = tmp_path / "dist.csv"
    exporter = DistanceCsvExporter(path, [("right_wrist", "nose")])
    exporter.add(_frame())
    exporter.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 1
    assert rows[0]["pair"] == "right_wrist-nose"
    assert float(rows[0]["distance_px"]) == pytest.approx(50.0)  # 3-4-5 三角形
    assert float(rows[0]["distance_m"]) == pytest.approx(0.3)


def _write_coords_csv(path):
    from poselab.exporters import CsvExporter

    exporter = CsvExporter(path)
    for i in range(10):
        exporter.add(_frame(i))
    exporter.close()


def test_plot_timeseries_and_trajectory_and_heatmap(tmp_path):
    pytest.importorskip("matplotlib")
    from poselab.plots import plot_csv

    coords = tmp_path / "coords.csv"
    _write_coords_csv(coords)
    for kind in (None, "trajectory", "heatmap"):
        out = plot_csv(coords, tmp_path / f"out_{kind}.png", kind=kind,
                       keypoints=["right_wrist"])
        assert out.exists() and out.stat().st_size > 0


def test_plot_distance_csv(tmp_path):
    pytest.importorskip("matplotlib")
    from poselab.plots import plot_csv

    dist = tmp_path / "dist.csv"
    exporter = DistanceCsvExporter(dist, [("right_wrist", "nose")])
    for i in range(5):
        exporter.add(_frame(i))
    exporter.close()
    out = plot_csv(dist)
    assert out.exists()


def test_detect_csv_type():
    from poselab.plots import detect_csv_type

    assert detect_csv_type(["frame", "timestamp_ms", "person",
                            "keypoint_name", "x_px", "y_px"]) == "coords"
    assert detect_csv_type(["timestamp_ms", "person", "angle_name",
                            "angle_deg"]) == "angles"
    assert detect_csv_type(["timestamp_ms", "person", "keypoint_name",
                            "speed_px_per_s"]) == "velocity"
    assert detect_csv_type(["timestamp_ms", "person", "pair",
                            "distance_px"]) == "distance"
    with pytest.raises(SystemExit):
        detect_csv_type(["foo", "bar"])
