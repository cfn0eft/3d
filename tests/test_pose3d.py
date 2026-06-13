"""poselab.pose3d (3D リフティングランナー) のテスト。

mmpose 本体は不要。inferencer をフェイクで注入し、FrameResult 変換・
MMPose 互換 JSON 出力・エクスポータ連携・可視化動画の回収を確認する。
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from poselab.exporters import CsvExporter, WideCsvExporter
from poselab.pose3d import (
    build_meta_info,
    collect_vis_output,
    instances_to_frame_result,
    keypoint_names_from_meta,
    run_pose3d,
)
from poselab.skeleton import H36M17_EDGES, H36M17_NAMES


def make_instance_3d(base: float, score: float = 0.9, n: int = 17) -> dict:
    return {
        "keypoints": [[base + i, base + i + 0.5, float(i)] for i in range(n)],
        "keypoint_scores": [score] * n,
    }


H36M_META = {
    "dataset_name": "h36m",
    "num_keypoints": 17,
    "keypoint_id2name": {i: n for i, n in enumerate(H36M17_NAMES)},
    "skeleton_links": [list(e) for e in H36M17_EDGES],
    "keypoint_colors": np.zeros((17, 3)),  # JSON 化できない値が混ざっても良い
}


class FakeModel:
    dataset_meta = H36M_META


class FakePose3DInferencer:
    """動画パスを受け取りフレームごとに predictions を yield するフェイク。"""

    model = FakeModel()

    def __init__(self, frames_instances):
        self.frames_instances = frames_instances
        self.calls = []
        self._buffer = {"pose_est_results_list": []}

    def __call__(self, inputs, **kwargs):
        self.calls.append({"inputs": inputs, **kwargs})
        for instances in self.frames_instances:
            yield {"predictions": [instances]}


@pytest.fixture()
def fake_video(tmp_path):
    """cv2 で読める小さな実動画を作る。"""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    assert path.is_file() and path.stat().st_size > 0
    return path


def test_keypoint_names_from_meta():
    assert keypoint_names_from_meta(H36M_META) == list(H36M17_NAMES)
    assert keypoint_names_from_meta({}) == list(H36M17_NAMES)
    assert keypoint_names_from_meta({"num_keypoints": 2}) == ["root", "right_hip"]


def test_build_meta_info_is_json_safe():
    info = build_meta_info(H36M_META)
    text = json.dumps(info)  # numpy が混ざっていても例外にならないこと
    assert "keypoint_id2name" in text
    assert info["num_keypoints"] == 17
    assert info["skeleton_links"] == [list(e) for e in H36M17_EDGES]


def test_instances_to_frame_result_world_and_px():
    px = [np.array([[float(i * 2), float(i * 3)] for i in range(17)])]
    result = instances_to_frame_result(
        [make_instance_3d(1.0)],
        frame_index=4,
        timestamp_ms=133.3,
        width=640,
        height=480,
        names=H36M17_NAMES,
        keypoints_2d=px,
    )
    assert result.frame_index == 4
    person = result.persons[0]
    wk = person.world_keypoints[2]
    assert wk.name == "right_knee"
    assert wk.x == pytest.approx(3.0)
    assert wk.y == pytest.approx(3.5)
    assert wk.z == pytest.approx(2.0)
    kp = person.keypoints[2]
    assert kp.x_px == pytest.approx(4.0)
    assert kp.y_px == pytest.approx(6.0)


def test_run_pose3d_writes_json_and_csv(tmp_path, fake_video):
    frames = [
        [make_instance_3d(0.0)],
        [make_instance_3d(1.0), make_instance_3d(5.0, score=0.7)],
        [],
    ]
    fake = FakePose3DInferencer(frames)
    json_path = tmp_path / "out" / "result.json"
    csv_path = tmp_path / "long.csv"
    wide_path = tmp_path / "wide.csv"
    progressed = []

    results = run_pose3d(
        fake_video,
        json_path=json_path,
        exporters=[
            CsvExporter(csv_path),
            WideCsvExporter(wide_path, H36M17_NAMES),
        ],
        progress=lambda done, total: progressed.append((done, total)),
        quiet=True,
        inferencer=fake,
    )

    # FrameResult
    assert len(results) == 3
    assert len(results[0].persons) == 1
    assert len(results[1].persons) == 2
    assert results[2].persons == []
    assert results[1].timestamp_ms == pytest.approx(100.0)  # 10fps の 2 フレーム目

    # 進捗コールバック
    assert progressed[0][0] == 1 and progressed[-1][0] == 3
    assert progressed[0][1] == 3

    # MMPose 互換 JSON
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["meta_info"]["num_keypoints"] == 17
    info = payload["instance_info"]
    assert [f["frame_id"] for f in info] == [0, 1, 2]
    assert len(info[1]["instances"]) == 2
    assert info[0]["instances"][0]["keypoints"][2][2] == pytest.approx(2.0)

    # CSV (world_x に 3D が入る)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["keypoint_name"] == "root"
    assert float(rows[2]["world_z"]) == pytest.approx(2.0)
    wide_rows = list(csv.DictReader(wide_path.open(encoding="utf-8")))
    assert "root_world_x" in wide_rows[0]
    assert len(wide_rows) == 3  # 1 + 2 + 0 人


def test_run_pose3d_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pose3d(
            tmp_path / "none.mp4",
            inferencer=FakePose3DInferencer([]),
            quiet=True,
        )


def test_collect_vis_output(tmp_path):
    vis_dir = tmp_path / "vis"
    vis_dir.mkdir()
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"src")
    (vis_dir / "walk.mp4").write_bytes(b"annotated")
    target = tmp_path / "out" / "walk_annotated.mp4"
    assert collect_vis_output(vis_dir, video, target)
    assert target.read_bytes() == b"annotated"
    # 見つからない場合は False
    assert not collect_vis_output(vis_dir, video, target)


def test_rgb_bytes_from_canvas_strips_alpha():
    """buffer_rgba の RGBA から行優先 RGB バイト列を作れること。"""
    from poselab._mpl_compat import _rgb_bytes_from_canvas

    rgba = np.array(
        [[[10, 20, 30, 255], [40, 50, 60, 128]]], dtype=np.uint8
    )

    class FakeCanvas:
        def buffer_rgba(self):
            return rgba

    out = _rgb_bytes_from_canvas(FakeCanvas())
    assert out == bytes([10, 20, 30, 40, 50, 60])
    assert len(out) == rgba.shape[0] * rgba.shape[1] * 3


def test_needs_agg_switch_matches_interactive_backends():
    """対話 Agg 系 (TkAgg 等) は切替対象、純 Agg だけ据え置き。"""
    from poselab._mpl_compat import _needs_agg_switch

    # 部分文字列 "agg" を含む対話バックエンドも切替対象になること
    assert _needs_agg_switch("TkAgg")
    assert _needs_agg_switch("QtAgg")
    assert _needs_agg_switch("module://ipympl.backend_nbagg")
    assert _needs_agg_switch("")
    # 非対話の純 Agg は切替不要
    assert not _needs_agg_switch("agg")
    assert not _needs_agg_switch("Agg")


def test_ensure_matplotlib_canvas_compat_real_figure():
    """matplotlib 3.10+ でも tostring_rgb が使え、Agg に切り替わること。"""
    matplotlib = pytest.importorskip("matplotlib")
    from poselab._mpl_compat import ensure_matplotlib_canvas_compat

    ensure_matplotlib_canvas_compat()
    # 部分一致ではなく非対話 Agg そのものへ切り替わっていること
    assert (matplotlib.get_backend() or "").lower() == "agg"

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    assert hasattr(FigureCanvasAgg, "tostring_rgb")

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2, 2), dpi=10)
    try:
        fig.canvas.draw()
        raw = fig.canvas.tostring_rgb()  # mmpose と同じ呼び出し
        rgba = np.asarray(fig.canvas.buffer_rgba())
        assert raw == rgba[..., :3].tobytes()
        assert len(raw) == rgba.shape[0] * rgba.shape[1] * 3
    finally:
        plt.close(fig)


def test_run_pose3d_collects_visualization(tmp_path, fake_video):
    class VisWritingFake(FakePose3DInferencer):
        def __call__(self, inputs, **kwargs):
            vis_dir = kwargs.get("vis_out_dir")
            if vis_dir:
                from pathlib import Path

                out = Path(vis_dir) / Path(str(inputs)).name
                out.write_bytes(b"vis-video")
            yield from super().__call__(inputs, **kwargs)

    fake = VisWritingFake([[make_instance_3d(0.0)]])
    save_video = tmp_path / "annotated.mp4"
    run_pose3d(
        fake_video,
        save_video=save_video,
        quiet=True,
        inferencer=fake,
    )
    assert save_video.read_bytes() == b"vis-video"
    assert fake.calls[0]["num_instances"] == -1
    # 一時 vis フォルダは消えている
    assert not (save_video.parent / "annotated_mmpose_vis").exists()
