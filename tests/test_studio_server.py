"""poselab.studio.server (Pose3DStudio 後継 GUI サーバー) のテスト。

実推定 (mmpose) は使わず、ジョブ実行コマンドをフェイクの Python
スクリプトへ差し替えて、キュー実行・進捗 / ログ配信・HTTP API を検証する。
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pytest

from poselab.studio.server import (
    JobManager,
    build_command,
    config_to_model_name,
    job_output_paths,
    normalize_job,
    postprocess_csv_outputs,
    preflight,
    serve,
    summarize_results_json,
)

# フェイクのパイプライン: 進捗 2 行 + ログを出し、出力ファイルを作る
FAKE_PIPELINE = """
import sys, time
from pathlib import Path
out = Path(sys.argv[1]); stem = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "ok"
print("fake pipeline start", flush=True)
if mode == "slow":
    time.sleep(30)
print("[############------------]  50.0%  5/10  9.9 fps  \\u6b8b\\u308a 0:01",
      flush=True)
if mode == "fail":
    print("something broke", flush=True)
    sys.exit(3)
out.mkdir(parents=True, exist_ok=True)
(out / f"results_{stem}.json").write_text(json.dumps({
    "meta_info": {"num_keypoints": 1, "keypoint_id2name": {"0": "root"}},
    "instance_info": [
        {"frame_id": 0, "instances": [
            {"keypoints": [[0, 0, 0]], "keypoint_scores": [0.8]},
            {"keypoints": [[1, 1, 1]], "keypoint_scores": [0.6]},
        ]},
        {"frame_id": 1, "instances": [
            {"keypoints": [[0, 0, 0]], "keypoint_scores": [1.0]},
        ]},
    ],
}), encoding="utf-8")
(out / f"{stem}_wide.csv").write_text("frame,timestamp_ms,person\\n0,0.0,0\\n",
                                      encoding="utf-8")
print("[########################] 100.0%  10/10  9.9 fps  \\u6b8b\\u308a 0:00",
      flush=True)
print("fake pipeline done", flush=True)
"""
FAKE_PIPELINE = "import json\n" + FAKE_PIPELINE


@pytest.fixture()
def fake_env(tmp_path):
    """フェイクのジョブスクリプトと入力動画ファイルを用意する。"""
    script = tmp_path / "fake_pipeline.py"
    script.write_text(FAKE_PIPELINE, encoding="utf-8")
    video = tmp_path / "walk.mp4"
    video.write_bytes(b"\x00" * 256)

    def builder(mode="ok"):
        def build(job):
            return [
                sys.executable, str(script),
                job["output_root"], Path(job["input"]).stem, mode,
            ]
        return build

    return {"script": script, "video": video, "builder": builder,
            "tmp": tmp_path}


# ---------------------------------------------------------------- payload


def test_config_to_model_name_translates_gui_profiles():
    # GUI の accuracy プロファイルは旧 exe 内のコンフィグパスを送る
    assert config_to_model_name(
        "configs/body_2d_keypoint/rtmpose/coco/"
        "rtmpose-l_8xb256-420e_aic-coco-384x288.py"
    ) == "rtmpose-l_8xb256-420e_aic-coco-384x288"
    assert config_to_model_name(
        "configs/body_3d_keypoint/video_pose_lift/h36m/"
        "video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py"
    ) == "video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m"
    # mmpose デモ用コンフィグ名は mmdet model zoo の正式名へ読み替える
    assert config_to_model_name(
        "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
    ) == "faster-rcnn_r50_fpn_1x_coco"
    assert config_to_model_name(None) is None
    assert config_to_model_name("") is None


def test_normalize_job_defaults_and_output_root():
    job = normalize_job({"input": "C:/data/walk.mp4"})
    assert job["input"] == "C:/data/walk.mp4"
    # output_root は <フォルダ>/<語幹> が既定 (GUI と同じ導出)
    assert Path(job["output_root"]) == Path("C:/data/walk")
    assert job["csv_format"] == "both"
    assert job["reencode"] is True
    assert job["pose2d_model"] is None


def test_build_command_flags(fake_env):
    job = normalize_job({
        "input": str(fake_env["video"]),
        "output_root": str(fake_env["tmp"] / "out"),
        "csv_format": "wide",
        "reencode": True,
        "progress": True,
        "pose2d_config": "configs/x/rtmpose-l_8xb256-420e_aic-coco-384x288.py",
    })
    cmd = build_command(job)
    text = " ".join(cmd)
    assert "--pose3d" in cmd
    assert "--h264" in cmd
    assert "--wide-csv" in cmd
    assert "--csv" not in cmd  # csv_format=wide ではロング CSV を出さない
    assert "--pose2d-model rtmpose-l_8xb256-420e_aic-coco-384x288" in text
    assert "--quiet" not in cmd

    job2 = normalize_job({
        "input": str(fake_env["video"]),
        "csv_format": "long",
        "reencode": False,
        "progress": False,
    })
    cmd2 = build_command(job2)
    assert "--h264" not in cmd2
    assert "--csv" in cmd2
    assert "--wide-csv" not in cmd2
    assert "--quiet" in cmd2


def test_job_output_paths_layout(tmp_path):
    job = normalize_job({
        "input": str(tmp_path / "dance.mp4"),
        "output_root": str(tmp_path / "dance"),
    })
    paths = job_output_paths(job)
    assert paths["json"].name == "results_dance.json"
    assert paths["video"].name == "dance_2d3d.mp4"
    assert paths["wide_csv"].parent == tmp_path / "dance"
    assert paths["long_csv"].name == "dance_long.csv"


# ---------------------------------------------------------------- preflight


def test_preflight_warnings(tmp_path, fake_env):
    res = preflight({"input": ""})
    assert any("Input" in w for w in res["warnings"])

    res = preflight({"input": str(tmp_path / "missing.mp4")})
    assert any("not found" in w for w in res["warnings"])

    res = preflight({
        "input": str(fake_env["video"]),
        "output_root": str(tmp_path / "no_such_dir" / "out"),
    })
    assert any("Output folder" in w for w in res["warnings"])
    assert res["ok"] is True


def test_summarize_results_json(tmp_path):
    path = tmp_path / "results_x.json"
    path.write_text(json.dumps({
        "meta_info": {},
        "instance_info": [
            {"instances": [
                {"keypoint_scores": [1.0, 0.5]},
                {"keypoint_scores": [0.5]},
            ]},
            {"instances": [{"keypoint_scores": [1.0]}]},
        ],
    }), encoding="utf-8")
    summary = summarize_results_json(path)
    assert summary["frames"] == 2
    assert summary["avg_instances"] == 1.5
    assert summary["avg_score"] == 0.75


# ---------------------------------------------------------------- manager


def collect_events(events_queue, until_idle, timeout=20.0):
    """購読キューからイベントを集める (実行終了まで)。"""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            events.append(json.loads(events_queue.get(timeout=0.2)))
        except queue.Empty:
            if until_idle():
                break
    return events


def test_jobmanager_runs_queue_and_emits(fake_env):
    manager = JobManager(command_builder=fake_env["builder"]("ok"))
    out_root = fake_env["tmp"] / "out_run"
    payload = {
        "input": str(fake_env["video"]),
        "output_root": str(out_root),
        "csv_format": "wide",
    }
    # 取りこぼしが無いよう実行前に購読する
    events_queue = manager.subscribe()
    assert manager.enqueue(payload)["ok"]
    assert manager.run({})["ok"]
    events = collect_events(
        events_queue, until_idle=lambda: not manager.status()["running"]
    )
    manager.unsubscribe(events_queue)
    manager.wait(15)

    status = manager.status()
    assert status["running"] is False
    assert status["completed"], "completed 履歴があるはず"
    assert status["completed"][-1]["return_code"] == 0
    assert (out_root / "results_walk.json").is_file()

    types = {e.get("type") for e in events}
    assert "log" in types
    assert "progress" in types
    assert "status" in types
    progress_values = [
        e["percent"] for e in events if e.get("type") == "progress"
    ]
    assert 100 in progress_values
    output_events = [e for e in events if e.get("type") == "output"]
    kinds = {e["kind"] for e in output_events}
    assert {"json", "csv"} <= kinds


def test_jobmanager_failure_recorded(fake_env):
    manager = JobManager(command_builder=fake_env["builder"]("fail"))
    payload = {"input": str(fake_env["video"]),
               "output_root": str(fake_env["tmp"] / "out_fail")}
    assert manager.run(payload)["ok"]
    manager.wait(15)
    status = manager.status()
    assert status["completed"][-1]["return_code"] == 3


def test_jobmanager_cancel(fake_env):
    manager = JobManager(command_builder=fake_env["builder"]("slow"))
    payload = {"input": str(fake_env["video"]),
               "output_root": str(fake_env["tmp"] / "out_slow")}
    assert manager.run(payload)["ok"]
    deadline = time.monotonic() + 10
    while manager.status()["current_job"] is None:
        assert time.monotonic() < deadline, "ジョブが開始しない"
        time.sleep(0.05)
    time.sleep(0.2)
    assert manager.cancel()["ok"]
    manager.wait(15)
    status = manager.status()
    assert status["running"] is False
    assert status["completed"][-1]["return_code"] != 0


def test_jobmanager_queue_operations(fake_env):
    manager = JobManager(command_builder=fake_env["builder"]("ok"))
    videos = []
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        v = fake_env["tmp"] / name
        v.write_bytes(b"\x00")
        videos.append(str(v))
    res = manager.enqueue({"inputs": videos, "output_root": ""})
    assert res == {"ok": True, "queued": 3}
    # 複数入力は output_root を入力ごとに導出する
    inputs = [j["input"] for j in manager.status()["queue"]]
    assert [Path(p).name for p in inputs] == ["a.mp4", "b.mp4", "c.mp4"]

    assert manager.move(2, -1)["ok"]
    inputs = [Path(j["input"]).name for j in manager.status()["queue"]]
    assert inputs == ["a.mp4", "c.mp4", "b.mp4"]

    assert manager.clear()["ok"]
    assert manager.status()["queue"] == []

    missing = manager.enqueue({"input": str(fake_env["tmp"] / "nope.mp4")})
    assert missing["ok"] is False


# ---------------------------------------------------------------- transform


def test_postprocess_csv_center_and_normalize(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    wide = out / "x_wide.csv"
    wide.write_text(
        "frame,timestamp_ms,person,"
        "root_x,root_y,root_z,root_visibility,"
        "root_world_x,root_world_y,root_world_z,"
        "head_x,head_y,head_z,head_visibility,"
        "head_world_x,head_world_y,head_world_z\n"
        "0,0.0,0,0,0,0,1,1.0,2.0,3.0,0,0,0,1,1.0,2.0,5.0\n",
        encoding="utf-8",
    )
    long_csv = out / "x_long.csv"
    long_csv.write_text(
        "frame,timestamp_ms,person,keypoint_id,keypoint_name,"
        "x_px,y_px,x_norm,y_norm,z,visibility,presence,"
        "world_x,world_y,world_z\n"
        "0,0.0,0,0,root,0,0,0,0,0,1,1,1.0,2.0,3.0\n"
        "0,0.0,0,1,head,0,0,0,0,0,1,1,1.0,2.0,5.0\n",
        encoding="utf-8",
    )
    job = {"center_root": True, "normalize_scale": True}
    paths = {"wide_csv": wide, "long_csv": long_csv}
    changed = postprocess_csv_outputs(job, paths)
    assert set(changed) == {"wide CSV", "long CSV"}

    rows = wide.read_text(encoding="utf-8").splitlines()[1].split(",")
    header = wide.read_text(encoding="utf-8").splitlines()[0].split(",")
    get = lambda col: float(rows[header.index(col)])  # noqa: E731
    # root が原点、head は (0,0,2) を正規化して長さ 1
    assert get("root_world_x") == 0.0
    assert get("root_world_z") == 0.0
    assert abs(get("head_world_z") - 1.0) < 1e-6

    lines = long_csv.read_text(encoding="utf-8").splitlines()
    head_row = lines[2].split(",")
    assert abs(float(head_row[-1]) - 1.0) < 1e-6


# ---------------------------------------------------------------- HTTP


@pytest.fixture()
def studio_server(fake_env):
    manager = JobManager(command_builder=fake_env["builder"]("ok"))
    server = serve(host="127.0.0.1", port=0, manager=manager, quiet=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    yield f"http://{host}:{port}", manager
    server.shutdown()
    server.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as res:
        return res.status, res.read(), dict(res.headers)


def _post(url, payload=None):
    body = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as res:
        return json.loads(res.read())


def test_http_static_and_status(studio_server):
    base, _ = studio_server
    status, body, _ = _get(base + "/")
    assert status == 200 and b"viewer-canvas" in body

    status, body, _ = _get(base + "/app.js")
    assert status == 200
    assert b"PoseLab3D" in body  # エンジン連結済み

    status, body, _ = _get(base + "/app.css")
    assert status == 200

    status, body, _ = _get(base + "/status")
    data = json.loads(body)
    assert data["running"] is False
    assert data["queue"] == []

    status, body, _ = _get(base + "/gpu")
    assert "available" in json.loads(body)


def test_http_run_roundtrip(studio_server, fake_env):
    base, manager = studio_server
    out_root = fake_env["tmp"] / "out_http"
    res = _post(base + "/run", {
        "input": str(fake_env["video"]),
        "output_root": str(out_root),
        "csv_format": "wide",
        "reencode": False,
    })
    assert res["ok"], res
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        data = json.loads(_get(base + "/status")[1])
        if not data["running"] and data["completed"]:
            break
        time.sleep(0.1)
    assert data["completed"][-1]["return_code"] == 0
    json_path = out_root / "results_walk.json"
    assert json_path.is_file()

    # summary
    res = _post(base + "/preflight", {"input": str(fake_env["video"])})
    assert res["ok"]
    summary = json.loads(
        _get(base + "/summary?path=" + quote(str(json_path)))[1]
    )
    assert summary["ok"] and summary["summary"]["frames"] == 2


def test_http_file_serving_with_range(studio_server, fake_env):
    base, _ = studio_server
    data_file = fake_env["tmp"] / "data.csv"
    data_file.write_text("0123456789", encoding="utf-8")

    file_url = base + "/file?path=" + quote(str(data_file))
    status, body, headers = _get(file_url)
    assert status == 200 and body == b"0123456789"
    assert headers.get("Accept-Ranges") == "bytes"

    status, body, headers = _get(file_url, headers={"Range": "bytes=2-5"})
    assert status == 206
    assert body == b"2345"
    assert headers.get("Content-Range") == "bytes 2-5/10"

    try:
        status, _, _ = _get(base + "/file?path=/no/such/file.csv")
    except urllib.error.HTTPError as err:
        status = err.code
    assert status == 404


def test_http_queue_endpoints(studio_server, fake_env):
    base, _ = studio_server
    videos = []
    for name in ("q1.mp4", "q2.mp4"):
        v = fake_env["tmp"] / name
        v.write_bytes(b"\x00")
        videos.append(str(v))
    res = _post(base + "/enqueue", {"inputs": videos})
    assert res["ok"]
    data = json.loads(_get(base + "/status")[1])
    assert len(data["queue"]) == 2
    assert _post(base + "/queue-move", {"index": 1, "offset": -1})["ok"]
    assert _post(base + "/clear-queue")["ok"]
    data = json.loads(_get(base + "/status")[1])
    assert data["queue"] == []
    assert _post(base + "/cancel")["ok"]
