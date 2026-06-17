"""実行来歴 (プロベナンス) の記録。

研究で結果を再現・引用できるよう、推定実行のメタ情報を 1 つの
「run-manifest」にまとめる。記録対象は次のとおり:

- 実行日時 (UTC)・コマンドライン・全 CLI 引数
- バックエンド / モデル名 / 明示指定した重み
- 入力ファイルの SHA-256 とサイズ (改ざん・取り違え検知用)
- 実行環境 (Python / OpenCV / MediaPipe / mmpose / PyTorch のバージョン)
- 出力データの単位 (px / m / 度 / 確率) と座標系の定義
- フレームレート / タイムスタンプの由来・トラッキング設定

このモジュールは標準ライブラリのみに依存し、cv2 / mediapipe / mmpose が
未導入の環境でも import できる (バージョン検出は best-effort)。
"""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from poselab import __version__

SCHEMA = "poselab.run-manifest/1"

# 出力列が持つ物理単位 (CSV/JSON/NPZ 共通)。
UNITS: Dict[str, str] = {
    "timestamp_ms": "millisecond",
    "x_px": "pixel",
    "y_px": "pixel",
    "x_norm": "normalized fraction of image width/height (0-1)",
    "y_norm": "normalized fraction of image width/height (0-1)",
    "z": "relative depth in image-plane units (~image width scale; "
         "smaller = closer to camera)",
    "visibility": "probability (0-1)",
    "presence": "probability (0-1)",
    "world_x": "meter",
    "world_y": "meter",
    "world_z": "meter",
}

_IMAGE_FRAME = {"x": "right", "y": "down", "origin": "top-left pixel"}


def coordinate_system(backend: Optional[str], pose3d: bool = False) -> Dict[str, Any]:
    """出力座標の軸の向き・原点・規約を記述した dict を返す。

    座標規約は CLAUDE.md / README と一致させている:
    - MediaPipe world / mmpose カメラ系 → y 下向き
    - 3D リフタ (--pose3d) → z 上向き・x 反転・床基準 (Human3.6M 規約)
    """
    if pose3d:
        return {
            "image": dict(_IMAGE_FRAME),
            "world": {
                "convention": "MMPose 3D / Human3.6M",
                "axes": "z-up, x-flipped, y-depth",
                "origin": "floor-referenced (pelvis-rooted)",
                "viewer_axis": "zup",
            },
        }
    cs: Dict[str, Any] = {"image": dict(_IMAGE_FRAME)}
    if backend == "mediapipe":
        cs["world"] = {
            "convention": "MediaPipe world landmarks",
            "axes": "y-down (camera frame)",
            "origin": "pelvis center (hip midpoint)",
            "viewer_axis": "ydown",
        }
    else:
        # mmpose 2D (RTMPose) は画像座標のみ。world 列は空になる。
        cs["world"] = None
    return cs


def _safe_version(module_name: str) -> Optional[str]:
    """モジュールを import してバージョン文字列を返す (失敗時 None)。"""
    try:
        module = __import__(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:  # __version__ を持たないパッケージ向けのフォールバック
        from importlib.metadata import version as _meta_version

        return str(_meta_version(module_name))
    except Exception:
        return None


def environment_info() -> Dict[str, Any]:
    """再現に必要な実行環境のバージョン情報を集める (best-effort)。"""
    import platform

    info: Dict[str, Any] = {
        "poselab": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name, key in (
        ("numpy", "numpy"),
        ("cv2", "opencv"),
        ("mediapipe", "mediapipe"),
        ("mmpose", "mmpose"),
        ("torch", "torch"),
    ):
        version = _safe_version(module_name)
        if version is not None:
            info[key] = version
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def hash_file(path: "str | Path", algo: str = "sha256") -> Optional[str]:
    """ファイルのハッシュ (16 進) を返す。ファイルでなければ None。"""
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_provenance(specs: Sequence[Any]) -> List[Dict[str, Any]]:
    """入力指定ごとにパス・ハッシュ・サイズを記録する。

    カメラ入力 (camera:0 等) や存在しないパスはハッシュを付けずに残す。
    """
    records: List[Dict[str, Any]] = []
    for spec in specs:
        text = str(spec)
        record: Dict[str, Any] = {"path": text}
        lowered = text.lower()
        if lowered.startswith(("camera:", "cam:")):
            record["kind"] = "camera"
        else:
            p = Path(text)
            if p.is_file():
                record["sha256"] = hash_file(p)
                record["bytes"] = p.stat().st_size
            else:
                record["note"] = "not a regular file at manifest time"
        records.append(record)
    return records


def _sanitize(obj: Any) -> Any:
    """JSON 化できる形へ再帰的に変換する。"""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_sanitize(v) for v in obj)
    return str(obj)


def build_manifest(
    *,
    args: Any = None,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    inputs: Optional[Sequence[Any]] = None,
    fps: Optional[float] = None,
    source_type: Optional[str] = None,
    timestamp_source: Optional[str] = None,
    pose3d: bool = False,
    tracking: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """実行来歴 (run-manifest) を組み立てる。

    args は argparse.Namespace か dict。全引数を ``args`` キーに丸ごと
    記録するので、後から正確に再現できる。
    """
    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "tool": f"poselab {__version__}",
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "command": list(sys.argv),
        "backend": backend,
        "model": model,
        "environment": environment_info(),
        "units": dict(UNITS),
        "coordinate_system": coordinate_system(backend, pose3d=pose3d),
    }
    if inputs is not None:
        manifest["inputs"] = input_provenance(inputs)
    source: Dict[str, Any] = {}
    if source_type is not None:
        source["type"] = source_type
    if fps is not None:
        source["fps"] = fps
    if timestamp_source is not None:
        source["timestamp_source"] = timestamp_source
    if source:
        manifest["source"] = source
    if tracking is not None:
        manifest["tracking"] = _sanitize(tracking)
    if args is not None:
        raw = vars(args) if not isinstance(args, dict) else args
        manifest["args"] = _sanitize(raw)
    if extra:
        manifest.update(_sanitize(extra))
    return manifest


def embed_metadata(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """データファイル (JSON/NPZ/サマリ) に埋め込む軽量メタを作る。

    冗長な command / args はサイドカーにのみ残し、ここでは省く。
    """
    skip = {"command", "args"}
    meta = {k: v for k, v in manifest.items() if k not in skip}
    inputs = manifest.get("inputs")
    if inputs:
        # 旧来の互換キー (入力パスのリスト)
        meta["input"] = [rec.get("path") for rec in inputs]
    return meta


def write_manifest(path: "str | Path", manifest: Dict[str, Any]) -> Path:
    """run-manifest を JSON ファイルへ書き出す。"""
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return out
