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
