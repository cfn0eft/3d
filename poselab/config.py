"""GUI 設定の保存・復元。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def config_path() -> Path:
    base = os.environ.get("POSELAB_CONFIG_DIR")
    if base:
        return Path(base) / "settings.json"
    appdata = os.environ.get("APPDATA")  # Windows
    if appdata:
        return Path(appdata) / "poselab" / "settings.json"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "poselab" / "settings.json"
    return Path.home() / ".config" / "poselab" / "settings.json"


def load_settings() -> Dict[str, Any]:
    """保存済み設定を読み込む。なければ・壊れていれば空 dict。"""
    path = config_path()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except OSError:
        pass  # 設定保存の失敗でアプリを落とさない
