from poselab.sources import CameraSource, scan_cameras


def test_scan_cameras_returns_list():
    # CI コンテナにはカメラがないため空リストが返る (例外を出さないこと)
    result = scan_cameras(max_index=1)
    assert isinstance(result, list)


def test_camera_error_message_is_helpful():
    try:
        CameraSource(99)
    except IOError as e:
        message = str(e)
        assert "カメラ 99" in message
        assert "アクセス許可" in message
    else:  # カメラ 99 が実在する環境ではスキップ相当
        pass
