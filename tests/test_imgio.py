import numpy as np

from poselab.imgio import imread, imwrite
from poselab.sources import ImageSource


def test_roundtrip_ascii_path(tmp_path):
    image = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    imwrite(path, image)
    loaded = imread(path)
    assert loaded is not None
    assert (loaded == image).all()  # PNG はロスレス


def test_roundtrip_japanese_path(tmp_path):
    image = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
    path = tmp_path / "実験データ" / "骨格推定 結果.png"
    path.parent.mkdir()
    imwrite(path, image)
    loaded = imread(path)
    assert loaded is not None
    assert (loaded == image).all()


def test_imread_missing_returns_none(tmp_path):
    assert imread(tmp_path / "存在しない.png") is None


def test_imwrite_jpeg(tmp_path):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    path = tmp_path / "出力.jpg"
    imwrite(path, image)
    assert imread(path) is not None


def test_image_source_japanese_path(tmp_path):
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    path = tmp_path / "テスト画像.png"
    imwrite(path, image)
    source = ImageSource([path])
    frames = list(source)
    assert len(frames) == 1
    assert frames[0][2].shape == (24, 24, 3)
