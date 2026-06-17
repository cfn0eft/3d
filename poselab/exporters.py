"""座標データのエクスポータ (CSV / JSON / NPZ)。

CSV はロング形式 (1 行 = 1 キーポイント) で、pandas 等で
そのまま読み込んで解析できます。JSON はフレーム単位の構造化
データ、NPZ は NumPy 配列 (フレーム × 人 × キーポイント) です。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from poselab.types import FrameResult

CSV_FIELDS = [
    "frame",
    "timestamp_ms",
    "person",
    "keypoint_id",
    "keypoint_name",
    "x_px",
    "y_px",
    "x_norm",
    "y_norm",
    "z",
    "visibility",
    "presence",
    "world_x",
    "world_y",
    "world_z",
]


def _kp_masked(kp, threshold: float) -> bool:
    """信頼度マスキング: visibility が閾値未満なら座標を欠損扱いにする。"""
    return threshold > 0.0 and kp.visibility < threshold


def _wk_masked(wk, threshold: float) -> bool:
    """world キーポイントが無い、または低信頼度なら欠損扱い。"""
    return wk is None or (threshold > 0.0 and wk.visibility < threshold)


class Exporter:
    """フレーム結果を逐次受け取って書き出す基底クラス。"""

    def add(self, result: FrameResult) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "Exporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CsvExporter(Exporter):
    """ロング形式 CSV。

    mask_visibility > 0 のとき、visibility が閾値未満のキーポイント座標は
    空欄 (= 欠損) として書き出す。信頼度・presence 列は残すので、なぜ欠損
    したかは追跡できる。pandas で読むと空欄は NaN になる。
    """

    def __init__(self, path: "str | Path", mask_visibility: float = 0.0) -> None:
        self.path = Path(path)
        self.mask = mask_visibility
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_FIELDS)

    def add(self, result: FrameResult) -> None:
        for person in result.persons:
            world = {wk.index: wk for wk in person.world_keypoints}
            for kp in person.keypoints:
                wk = world.get(kp.index)
                kp_missing = _kp_masked(kp, self.mask)
                wk_missing = _wk_masked(wk, self.mask)
                self._writer.writerow(
                    [
                        result.frame_index,
                        f"{result.timestamp_ms:.3f}",
                        person.person_index,
                        kp.index,
                        kp.name,
                        "" if kp_missing else f"{kp.x_px:.3f}",
                        "" if kp_missing else f"{kp.y_px:.3f}",
                        "" if kp_missing else f"{kp.x_norm:.6f}",
                        "" if kp_missing else f"{kp.y_norm:.6f}",
                        "" if kp_missing else f"{kp.z:.6f}",
                        f"{kp.visibility:.4f}",
                        f"{kp.presence:.4f}",
                        "" if wk_missing else f"{wk.x:.6f}",
                        "" if wk_missing else f"{wk.y:.6f}",
                        "" if wk_missing else f"{wk.z:.6f}",
                    ]
                )

    def close(self) -> None:
        self._file.close()


class WideCsvExporter(Exporter):
    """ワイド形式 CSV (1 行 = 1 フレーム × 1 人物)。

    キーポイントごとに x / y / z / visibility / world_x / world_y /
    world_z の列を持つ。Excel や MATLAB でそのまま扱いやすい形式。
    """

    _PER_KP = ("x", "y", "z", "visibility", "world_x", "world_y", "world_z")

    def __init__(
        self,
        path: "str | Path",
        keypoint_names: Sequence[str],
        mask_visibility: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.names = list(keypoint_names)
        self.mask = mask_visibility
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        header = ["frame", "timestamp_ms", "person"]
        for name in self.names:
            header.extend(f"{name}_{suffix}" for suffix in self._PER_KP)
        self._writer.writerow(header)

    def add(self, result: FrameResult) -> None:
        for person in result.persons:
            world = {wk.index: wk for wk in person.world_keypoints}
            row = [
                result.frame_index,
                f"{result.timestamp_ms:.3f}",
                person.person_index,
            ]
            for kp in person.keypoints:
                wk = world.get(kp.index)
                kp_missing = _kp_masked(kp, self.mask)
                wk_missing = _wk_masked(wk, self.mask)
                row.extend(
                    [
                        "" if kp_missing else f"{kp.x_px:.3f}",
                        "" if kp_missing else f"{kp.y_px:.3f}",
                        "" if kp_missing else f"{kp.z:.6f}",
                        f"{kp.visibility:.4f}",
                        "" if wk_missing else f"{wk.x:.6f}",
                        "" if wk_missing else f"{wk.y:.6f}",
                        "" if wk_missing else f"{wk.z:.6f}",
                    ]
                )
            self._writer.writerow(row)

    def close(self) -> None:
        self._file.close()


class JsonExporter(Exporter):
    """フレーム単位の構造化 JSON (ストリーミング書き込み)。"""

    def __init__(
        self,
        path: "str | Path",
        keypoint_names: Optional[Sequence[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.path = Path(path)
        self._file = open(self.path, "w", encoding="utf-8")
        self._first = True
        meta = dict(metadata or {})
        if keypoint_names is not None:
            meta["keypoint_names"] = list(keypoint_names)
        self._file.write('{\n"metadata": ')
        json.dump(meta, self._file, ensure_ascii=False)
        self._file.write(',\n"frames": [\n')

    def add(self, result: FrameResult) -> None:
        record = {
            "frame": result.frame_index,
            "timestamp_ms": round(result.timestamp_ms, 3),
            "persons": [
                {
                    "person": p.person_index,
                    "keypoints": [
                        {
                            "id": kp.index,
                            "name": kp.name,
                            "x_px": round(kp.x_px, 3),
                            "y_px": round(kp.y_px, 3),
                            "x_norm": round(kp.x_norm, 6),
                            "y_norm": round(kp.y_norm, 6),
                            "z": round(kp.z, 6),
                            "visibility": round(kp.visibility, 4),
                            "presence": round(kp.presence, 4),
                        }
                        for kp in p.keypoints
                    ],
                    "world_keypoints": [
                        {
                            "id": wk.index,
                            "name": wk.name,
                            "x": round(wk.x, 6),
                            "y": round(wk.y, 6),
                            "z": round(wk.z, 6),
                            "visibility": round(wk.visibility, 4),
                        }
                        for wk in p.world_keypoints
                    ],
                }
                for p in result.persons
            ],
        }
        if not self._first:
            self._file.write(",\n")
        self._first = False
        json.dump(record, self._file, ensure_ascii=False)

    def close(self) -> None:
        self._file.write("\n]\n}\n")
        self._file.close()


class NpzExporter(Exporter):
    """NumPy .npz 形式。

    keypoints: (n_frames, max_persons, n_keypoints, 5)
        最後の軸は [x_px, y_px, z, visibility, presence]。
    world: (n_frames, max_persons, n_keypoints, 4)
        最後の軸は [x, y, z, visibility]。
    未検出の人物枠は NaN で埋められます。
    """

    def __init__(
        self,
        path: "str | Path",
        keypoint_names: Sequence[str],
        max_persons: int = 1,
        metadata: Optional[dict] = None,
        mask_visibility: float = 0.0,
    ) -> None:
        self.path = Path(path)
        self.names = list(keypoint_names)
        self.max_persons = max_persons
        self.metadata = metadata
        self.mask = mask_visibility
        self._frames: List[np.ndarray] = []
        self._world: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._frame_indices: List[int] = []

    def add(self, result: FrameResult) -> None:
        n = len(self.names)
        kp_arr = np.full((self.max_persons, n, 5), np.nan, dtype=np.float32)
        w_arr = np.full((self.max_persons, n, 4), np.nan, dtype=np.float32)
        for p in result.persons:
            # トラッキング ID が枠を超えた人物は記録しない (CSV/JSON には残る)
            if p.person_index >= self.max_persons:
                continue
            for kp in p.keypoints:
                # 座標はマスクしても visibility/presence は残す
                if _kp_masked(kp, self.mask):
                    kp_arr[p.person_index, kp.index] = (
                        np.nan, np.nan, np.nan, kp.visibility, kp.presence,
                    )
                else:
                    kp_arr[p.person_index, kp.index] = (
                        kp.x_px, kp.y_px, kp.z, kp.visibility, kp.presence,
                    )
            for wk in p.world_keypoints:
                if _wk_masked(wk, self.mask):
                    w_arr[p.person_index, wk.index] = (
                        np.nan, np.nan, np.nan, wk.visibility,
                    )
                else:
                    w_arr[p.person_index, wk.index] = (
                        wk.x, wk.y, wk.z, wk.visibility,
                    )
        self._frames.append(kp_arr)
        self._world.append(w_arr)
        self._timestamps.append(result.timestamp_ms)
        self._frame_indices.append(result.frame_index)

    def close(self) -> None:
        arrays = dict(
            keypoints=np.stack(self._frames) if self._frames else np.empty((0,)),
            world=np.stack(self._world) if self._world else np.empty((0,)),
            timestamps_ms=np.asarray(self._timestamps, dtype=np.float64),
            frame_indices=np.asarray(self._frame_indices, dtype=np.int64),
            keypoint_names=np.asarray(self.names),
        )
        if self.metadata is not None:
            # メタ情報 (単位・座標系・来歴) を JSON 文字列として同梱する
            arrays["metadata_json"] = np.asarray(
                json.dumps(self.metadata, ensure_ascii=False)
            )
        np.savez_compressed(self.path, **arrays)


def export_results(
    results: Iterable[FrameResult],
    exporters: Sequence[Exporter],
) -> None:
    """蓄積済みの結果リストを複数のエクスポータへ書き出す (GUI 用)。"""
    for result in results:
        for exporter in exporters:
            exporter.add(result)
    for exporter in exporters:
        exporter.close()
