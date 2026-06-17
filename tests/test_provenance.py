"""provenance (実行来歴) モジュールのテスト。

cv2 / mediapipe / mmpose に依存しないので CI でそのまま実行できる。
"""

import argparse
import json

from poselab import provenance


def test_environment_info_keys():
    info = provenance.environment_info()
    assert "poselab" in info
    assert "python" in info
    assert "platform" in info
    # numpy はテスト環境に必ず入っている
    assert "numpy" in info


def test_hash_file_roundtrip(tmp_path):
    f = tmp_path / "video.bin"
    f.write_bytes(b"hello poselab")
    digest = provenance.hash_file(f)
    assert isinstance(digest, str) and len(digest) == 64
    # 同じ内容なら同じハッシュ (決定性)
    assert provenance.hash_file(f) == digest
    # ファイルでなければ None
    assert provenance.hash_file(tmp_path / "missing.bin") is None
    assert provenance.hash_file(tmp_path) is None


def test_input_provenance(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00\x01\x02\x03")
    records = provenance.input_provenance([str(f), "camera:0", "no_such.mp4"])
    assert records[0]["sha256"] is not None
    assert records[0]["bytes"] == 4
    assert records[1]["kind"] == "camera"
    assert "note" in records[2]


def test_coordinate_system():
    cs_mp = provenance.coordinate_system("mediapipe")
    assert cs_mp["world"]["viewer_axis"] == "ydown"
    cs_mm = provenance.coordinate_system("mmpose")
    assert cs_mm["world"] is None  # mmpose 2D は world 無し
    cs_3d = provenance.coordinate_system("mmpose", pose3d=True)
    assert cs_3d["world"]["viewer_axis"] == "zup"


def test_build_manifest_fields(tmp_path):
    f = tmp_path / "walk.mp4"
    f.write_bytes(b"data")
    args = argparse.Namespace(input=[str(f)], smooth=0, num_poses=2)
    manifest = provenance.build_manifest(
        args=args,
        backend="mediapipe",
        model="full",
        inputs=[str(f)],
        fps=30.0,
        source_type="VideoSource",
        timestamp_source="VideoSource",
        tracking={"enabled": True},
    )
    assert manifest["schema"] == provenance.SCHEMA
    assert manifest["backend"] == "mediapipe"
    assert manifest["model"] == "full"
    assert manifest["created_at"].endswith("Z")
    assert isinstance(manifest["command"], list)
    assert manifest["units"]["world_x"] == "meter"
    assert manifest["source"]["fps"] == 30.0
    assert manifest["tracking"]["enabled"] is True
    # 全引数が丸ごと残る (再現用)
    assert manifest["args"]["num_poses"] == 2
    assert manifest["inputs"][0]["sha256"] is not None


def test_embed_metadata_drops_command_and_args():
    manifest = provenance.build_manifest(
        args={"a": 1}, backend="mediapipe", model="full",
        inputs=["x.mp4"],
    )
    meta = provenance.embed_metadata(manifest)
    assert "command" not in meta
    assert "args" not in meta
    assert "units" in meta
    assert "coordinate_system" in meta
    assert meta["input"] == ["x.mp4"]  # 互換キー


def test_write_manifest_roundtrip(tmp_path):
    manifest = provenance.build_manifest(backend="mmpose", model="rtmpose-m")
    out = provenance.write_manifest(tmp_path / "sub" / "run.json", manifest)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["backend"] == "mmpose"
    assert loaded["schema"] == provenance.SCHEMA


def test_units_cover_csv_fields():
    from poselab.exporters import CSV_FIELDS

    measured = set(CSV_FIELDS) - {"frame", "person", "keypoint_id", "keypoint_name"}
    assert measured <= set(provenance.UNITS)
