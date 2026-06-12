"""時系列座標の平滑化フィルタ。

推定ノイズの低減のため、フレーム方向の移動平均 (NaN・欠損対応)
を座標に適用します。visibility / presence は平滑化しません。

注意: 複数人検出時の person インデックスはトラッキング ID では
ないため、平滑化は同一インデックスが同一人物である前提です。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from poselab.skeleton import NUM_LANDMARKS
from poselab.types import FrameResult


def _nan_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """先頭軸 (時間) 方向の NaN 対応移動平均。NaN 位置は NaN のまま。"""
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    valid = ~np.isnan(flat)
    filled = np.where(valid, flat, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    num = np.empty_like(flat, dtype=np.float64)
    den = np.empty_like(flat, dtype=np.float64)
    for col in range(flat.shape[1]):
        num[:, col] = np.convolve(filled[:, col], kernel, mode="same")
        den[:, col] = np.convolve(valid[:, col].astype(np.float64), kernel, mode="same")
    out = num / np.maximum(den, 1e-9)
    out[~valid] = np.nan
    return out.reshape(arr.shape)


def smooth_results(
    results: Sequence[FrameResult], window: int
) -> List[FrameResult]:
    """フレーム結果列の座標を移動平均で平滑化する (インプレース)。

    window が 1 以下なら何もしない。x_px / y_px / z とワールド座標を
    平滑化し、x_norm / y_norm は平滑化後のピクセル座標から再計算する。
    """
    results = list(results)
    if window <= 1 or len(results) < 2:
        return results

    n = len(results)
    max_persons = max((len(r.persons) for r in results), default=0)
    if max_persons == 0:
        return results

    for pi in range(max_persons):
        px = np.full((n, NUM_LANDMARKS, 3), np.nan)
        world = np.full((n, NUM_LANDMARKS, 3), np.nan)
        for fi, result in enumerate(results):
            for person in result.persons:
                if person.person_index != pi:
                    continue
                for kp in person.keypoints:
                    px[fi, kp.index] = (kp.x_px, kp.y_px, kp.z)
                for wk in person.world_keypoints:
                    world[fi, wk.index] = (wk.x, wk.y, wk.z)

        px_s = _nan_moving_average(px, window)
        world_s = _nan_moving_average(world, window)

        for fi, result in enumerate(results):
            for person in result.persons:
                if person.person_index != pi:
                    continue
                for kp in person.keypoints:
                    if np.isnan(px_s[fi, kp.index, 0]):
                        continue
                    kp.x_px, kp.y_px, kp.z = (float(v) for v in px_s[fi, kp.index])
                    if result.width:
                        kp.x_norm = kp.x_px / result.width
                    if result.height:
                        kp.y_norm = kp.y_px / result.height
                for wk in person.world_keypoints:
                    if np.isnan(world_s[fi, wk.index, 0]):
                        continue
                    wk.x, wk.y, wk.z = (float(v) for v in world_s[fi, wk.index])
    return results
