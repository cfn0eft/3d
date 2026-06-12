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


def test_parse_outputs():
    import pytest

    from poselab.cli import AUTO_OUTPUT_FORMATS, parse_outputs

    parser = build_parser()
    assert parse_outputs(None, parser) == set(AUTO_OUTPUT_FORMATS)
    assert parse_outputs("json,wide", parser) == {"json", "wide"}
    assert parse_outputs(" json , long ", parser) == {"json", "long"}
    with pytest.raises(SystemExit):
        parse_outputs("json,bogus", parser)
    with pytest.raises(SystemExit):
        parse_outputs(" , ", parser)


def test_auto_output_paths_selection(tmp_path):
    from poselab.cli import auto_output_paths

    video = tmp_path / "walk.mp4"
    paths = auto_output_paths(video, {"json", "wide"}, is_image=False)
    assert paths["out_dir"].name == "walk_poselab"
    assert paths["json"].name == "walk.json"
    assert paths["wide_csv"].name == "walk_wide.csv"
    assert paths["csv"] is None
    assert paths["angles_csv"] is None
    assert paths["summary_json"] is None
    assert paths["save_video"] is None

    paths = auto_output_paths(video, {"video"}, is_image=False)
    assert paths["save_video"].name == "walk_annotated.mp4"
    assert paths["json"] is None

    image = tmp_path / "p.jpg"
    paths = auto_output_paths(image, {"video"}, is_image=True)
    assert paths["save_video"] is None and paths["save_image"] is None
    paths = auto_output_paths(image, {"image"}, is_image=True)
    assert paths["save_image"].name == "p_annotated.png"


def test_outputs_requires_auto_output(tmp_path, capsys):
    import pytest

    video = tmp_path / "c.mp4"
    video.write_bytes(b"x")
    with pytest.raises(SystemExit):
        main(["--input", str(video), "--outputs", "json"])
    assert "--auto-output" in capsys.readouterr().err
