import csv

import numpy as np
import pytest

from poselab.exporters import WideCsvExporter
from poselab.skeleton import LANDMARK_NAMES, NUM_LANDMARKS
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


def _frame(frame_index: int = 0) -> FrameResult:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=0.5, y_norm=0.5, z=-0.1,
            visibility=0.9, presence=0.9, x_px=100.0 + i, y_px=200.0,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    world = [
        WorldKeypoint(index=i, name=name, x=0.1, y=-0.2, z=0.05, visibility=0.9)
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return FrameResult(
        frame_index=frame_index, timestamp_ms=frame_index * 33.3,
        width=640, height=480,
        persons=[PersonPose(0, keypoints, world)],
    )


def test_wide_csv(tmp_path):
    path = tmp_path / "wide.csv"
    exporter = WideCsvExporter(path, LANDMARK_NAMES)
    exporter.add(_frame(0))
    exporter.add(_frame(1))
    exporter.close()
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2  # 1 行 = 1 フレーム
    assert len(rows[0]) == 3 + NUM_LANDMARKS * 7
    assert float(rows[0]["nose_x"]) == pytest.approx(100.0)
    assert float(rows[0]["right_wrist_x"]) == pytest.approx(116.0)
    assert float(rows[0]["nose_world_z"]) == pytest.approx(0.05)


def test_find_ffmpeg_path(monkeypatch):
    from poselab import pipeline

    # PATH 上に ffmpeg があればそれを使う
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert pipeline.find_ffmpeg() == "/usr/bin/ffmpeg"


def test_find_ffmpeg_imageio_fallback(monkeypatch):
    import sys
    import types

    from poselab import pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: "/bundled/ffmpeg"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    assert pipeline.find_ffmpeg() == "/bundled/ffmpeg"


def test_find_ffmpeg_none(monkeypatch):
    import sys

    from poselab import pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    assert pipeline.find_ffmpeg() is None


def test_reencode_h264(tmp_path):
    import cv2

    from poselab.pipeline import find_ffmpeg, reencode_h264

    if find_ffmpeg() is None:
        pytest.skip("ffmpeg not available (PATH も imageio-ffmpeg も無し)")

    path = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48)
    )
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    assert reencode_h264(path) is True
    cap = cv2.VideoCapture(str(path))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(
        errors="replace")
    cap.release()
    assert fourcc in ("avc1", "h264", "H264")


def test_plot_pose3d(tmp_path):
    pytest.importorskip("matplotlib")
    from poselab.exporters import CsvExporter
    from poselab.plots import plot_csv

    coords = tmp_path / "coords.csv"
    exporter = CsvExporter(coords)
    for i in range(5):
        exporter.add(_frame(i))
    exporter.close()
    out = plot_csv(coords, tmp_path / "pose3d.png", kind="pose3d")
    assert out.exists() and out.stat().st_size > 0
    # 特定フレーム指定
    out2 = plot_csv(coords, tmp_path / "pose3d_f0.png", kind="pose3d", frame=0)
    assert out2.exists()


def test_auto_output_paths(tmp_path, monkeypatch):
    """--auto-output が動画ごとのフォルダ・ファイル名を構成すること。"""
    import poselab.cli as cli_mod

    video = tmp_path / "実験動画.mp4"
    video.write_bytes(b"dummy")
    captured = []

    def fake_run_job(parser, args, specs):
        captured.append((argspaths(args), list(specs)))
        return 0

    def argspaths(args):
        return {
            "csv": str(args.csv), "wide": str(args.wide_csv),
            "video": str(args.save_video), "h264": args.h264,
        }

    monkeypatch.setattr(cli_mod, "_run_job", fake_run_job)
    code = cli_mod.main(["--input", str(video), "--auto-output", "-q"])
    assert code == 0
    paths, specs = captured[0]
    out_dir = tmp_path / "実験動画_poselab"
    assert paths["csv"] == str(out_dir / "実験動画_long.csv")
    assert paths["wide"] == str(out_dir / "実験動画_wide.csv")
    assert paths["video"] == str(out_dir / "実験動画_annotated.mp4")
    assert paths["h264"] is True
    assert out_dir.is_dir()
