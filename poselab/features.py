"""キーポイント間の特徴量計算 (2 点間距離など)。

行動観察でよく使う「手首と鼻の距離」「両手首間の距離」のような
時系列特徴量を、推定と同時に CSV へ書き出す。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from poselab.exporters import Exporter
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult

_INDEX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def parse_pair(spec: str) -> Tuple[str, str]:
    """"right_wrist:nose" 形式のペア指定をパースする。"""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"距離ペアは 'キーポイント名:キーポイント名' 形式で指定してください: {spec!r}"
        )
    a, b = (p.strip() for p in parts)
    for name in (a, b):
        if name not in _INDEX:
            raise ValueError(
                f"不明なキーポイント名: {name!r} "
                "(poselab --list-keypoints で一覧を確認できます)"
            )
    if a == b:
        raise ValueError(f"同じキーポイント同士の距離は指定できません: {spec!r}")
    return a, b


class DistanceCsvExporter(Exporter):
    """キーポイント 2 点間距離のロング形式 CSV (1 行 = 1 ペア)。

    ピクセル座標の距離と、ワールド座標 (m) の距離を出力する。
    """

    FIELDS = [
        "frame",
        "timestamp_ms",
        "person",
        "pair",
        "distance_px",
        "distance_m",
    ]

    def __init__(self, path: "str | Path", pairs: Sequence[Tuple[str, str]]) -> None:
        if not pairs:
            raise ValueError("距離ペアが指定されていません")
        self.path = Path(path)
        self.pairs: List[Tuple[str, str]] = list(pairs)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.FIELDS)

    def add(self, result: FrameResult) -> None:
        for person in result.persons:
            world = {wk.index: wk for wk in person.world_keypoints}
            for a, b in self.pairs:
                ia, ib = _INDEX[a], _INDEX[b]
                ka, kb = person.keypoints[ia], person.keypoints[ib]
                d_px = float(np.hypot(ka.x_px - kb.x_px, ka.y_px - kb.y_px))
                d_m = ""
                wa, wb = world.get(ia), world.get(ib)
                if wa is not None and wb is not None:
                    d_m = "%.6f" % float(
                        np.linalg.norm(
                            [wa.x - wb.x, wa.y - wb.y, wa.z - wb.z]
                        )
                    )
                self._writer.writerow(
                    [
                        result.frame_index,
                        f"{result.timestamp_ms:.3f}",
                        person.person_index,
                        f"{a}-{b}",
                        f"{d_px:.3f}",
                        d_m,
                    ]
                )

    def close(self) -> None:
        self._file.close()
