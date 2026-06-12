from poselab import config


def test_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("POSELAB_CONFIG_DIR", str(tmp_path))
    config.save_settings({"model": "heavy", "num_poses": 3, "mirror": True})
    loaded = config.load_settings()
    assert loaded == {"model": "heavy", "num_poses": 3, "mirror": True}


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("POSELAB_CONFIG_DIR", str(tmp_path / "nope"))
    assert config.load_settings() == {}


def test_load_corrupt_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("POSELAB_CONFIG_DIR", str(tmp_path))
    (tmp_path / "settings.json").write_text("{broken", encoding="utf-8")
    assert config.load_settings() == {}
