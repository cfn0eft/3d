from poselab.cli import build_parser, main


def test_list_keypoints(capsys):
    assert main(["--list-keypoints"]) == 0
    out = capsys.readouterr().out
    assert "nose" in out
    assert "right_foot_index" in out


def test_parser_defaults():
    args = build_parser().parse_args(["--input", "video.mp4"])
    assert args.model == "full"
    assert args.num_poses == 1
    assert not args.show


def test_list_keypoints_mmpose(capsys):
    assert main(["--backend", "mmpose", "--list-keypoints"]) == 0
    out = capsys.readouterr().out
    assert "left_shoulder" in out  # COCO 17
    assert "neck_base" in out  # H36M 17


def test_parser_pose3d_defaults():
    args = build_parser().parse_args(["--input", "video.mp4", "--pose3d"])
    assert args.pose3d
    assert args.backend == "mediapipe"  # --pose3d は backend 指定なしでも使える
    assert args.lift_model is None


def test_pose3d_rejects_camera_input(capsys):
    import pytest

    with pytest.raises(SystemExit):
        main(["--input", "camera:0", "--pose3d"])
    assert "カメラ入力では使用できません" in capsys.readouterr().err


def test_pose3d_rejects_image_input(tmp_path, capsys):
    import pytest

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--input", str(img), "--pose3d"])
    assert "動画入力専用" in capsys.readouterr().err


def test_pose3d_rejects_unsupported_outputs(tmp_path, capsys):
    import pytest

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--input", str(video), "--pose3d", "--npz", str(tmp_path / "o.npz")])
    assert "--npz" in capsys.readouterr().err
