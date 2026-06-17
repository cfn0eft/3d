"""時系列座標の平滑化フィルタ。

推定ノイズの低減のため、フレーム方向のフィルタを座標に適用する。
visibility / presence そのものは平滑化しない。

対応する手法 (method):
- "moving" : NaN 対応の中央移動平均 (既定。対称カーネルなので位相遅延なし)
- "median" : NaN 対応の移動メディアン (外れ値・スパイクに頑健)
- "butter" : ゼロ位相 Butterworth ローパス (filtfilt、生体力学の標準。
             カットオフ周波数 cutoff [Hz] と fps が必要)

weighted=True で visibility を重みにした加重平均にできる (低信頼度フレーム
の寄与を下げる)。Butterworth は scipy 非依存の純 numpy 実装。

注意: 複数人検出時の person インデックスはトラッキング ID ではないため、
平滑化は同一インデックスが同一人物である前提。高次微分 (加速度等) を取る
場合は平滑化との併用を推奨。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from poselab.skeleton import NUM_LANDMARKS
from poselab.types import FrameResult

SMOOTH_METHODS = ("moving", "median", "butter")


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


def _weighted_moving_average(
    arr: np.ndarray, window: int, weights: np.ndarray
) -> np.ndarray:
    """visibility を重みにした NaN 対応移動平均。

    arr: (n, K, C)、weights: (n, K)。低信頼度フレームの寄与を下げる。
    """
    n = arr.shape[0]
    valid = ~np.isnan(arr)
    w = np.where(np.isnan(weights), 0.0, np.clip(weights, 0.0, None))[:, :, None]
    eff_w = np.where(valid, w, 0.0)
    filled = np.where(valid, arr, 0.0)
    kernel = np.ones(window, dtype=np.float64)
    fw = (filled * eff_w).reshape(n, -1)
    ww = eff_w.reshape(n, -1)
    num = np.empty_like(fw)
    den = np.empty_like(ww)
    for col in range(fw.shape[1]):
        num[:, col] = np.convolve(fw[:, col], kernel, mode="same")
        den[:, col] = np.convolve(ww[:, col], kernel, mode="same")
    out = (num / np.maximum(den, 1e-9)).reshape(arr.shape)
    out[~valid] = np.nan
    return out


def _nan_median_filter(arr: np.ndarray, window: int) -> np.ndarray:
    """先頭軸方向の NaN 対応移動メディアン (外れ値に頑健)。"""
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    valid = ~np.isnan(flat)
    half = window // 2
    out = flat.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = flat[lo:hi]
        col_valid = np.any(~np.isnan(seg), axis=0)
        med = np.full(flat.shape[1], np.nan)
        if np.any(col_valid):
            med[col_valid] = np.nanmedian(seg[:, col_valid], axis=0)
        out[i] = med
    out[~valid] = np.nan
    return out.reshape(arr.shape)


def _butter2_coeffs(cutoff: float, fps: float):
    """2 次 Butterworth ローパスの双 1 次変換係数 (b, a)。"""
    if cutoff <= 0 or fps <= 0 or cutoff >= fps / 2.0:
        raise ValueError(
            f"カットオフ周波数が不正です: cutoff={cutoff}Hz, fps={fps} "
            "(0 < cutoff < fps/2 が必要)"
        )
    k = math.tan(math.pi * cutoff / fps)
    k2 = k * k
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k2)
    b = (k2 * norm, 2.0 * k2 * norm, k2 * norm)
    a = (1.0, 2.0 * (k2 - 1.0) * norm, (1.0 - math.sqrt(2.0) * k + k2) * norm)
    return b, a


def _lfilter(b, a, x: np.ndarray) -> np.ndarray:
    """直接 II 型転置による IIR フィルタ (a[0]=1 前提)。"""
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    y = np.empty_like(x)
    z1 = 0.0
    z2 = 0.0
    for n in range(x.shape[0]):
        xn = x[n]
        yn = b0 * xn + z1
        z1 = b1 * xn - a1 * yn + z2
        z2 = b2 * xn - a2 * yn
        y[n] = yn
    return y


def _filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    """前後 2 回適用によるゼロ位相フィルタ (奇対称パディング付き)。"""
    pad = min(3 * (len(a) - 1), x.shape[0] - 1)
    if pad > 0:
        left = 2.0 * x[0] - x[pad:0:-1]
        right = 2.0 * x[-1] - x[-2:-pad - 2:-1]
        ext = np.concatenate([left, x, right])
    else:
        ext = x
    y = _lfilter(b, a, ext)
    y = _lfilter(b, a, y[::-1])[::-1]
    return y[pad:pad + x.shape[0]] if pad > 0 else y


def _butter_filter(arr: np.ndarray, cutoff: float, fps: float) -> np.ndarray:
    """各チャンネルにゼロ位相 Butterworth を適用 (NaN は内部補間)。

    欠損フレームは線形補間してフィルタの連続性を保つが、書き戻すのは元から
    有効だった位置だけ (欠損は欠損のまま)。
    """
    b, a = _butter2_coeffs(cutoff, fps)
    n = arr.shape[0]
    flat = arr.reshape(n, -1)
    out = flat.copy()
    idx = np.arange(n)
    for col in range(flat.shape[1]):
        series = flat[:, col]
        valid = ~np.isnan(series)
        if valid.sum() < 4:
            continue
        filled = np.interp(idx, idx[valid], series[valid])
        filtered = _filtfilt(b, a, filled)
        out[valid, col] = filtered[valid]
    return out.reshape(arr.shape)


def smooth_results(
    results: Sequence[FrameResult],
    window: int = 0,
    *,
    method: str = "moving",
    weighted: bool = False,
    cutoff: Optional[float] = None,
    fps: Optional[float] = None,
) -> List[FrameResult]:
    """フレーム結果列の座標を平滑化する (インプレース)。

    x_px / y_px / z とワールド座標を平滑化し、x_norm / y_norm は平滑化後の
    ピクセル座標から再計算する。method / weighted / cutoff / fps で手法を選ぶ。
    既定 (method="moving") は従来どおりの中央移動平均。
    """
    results = list(results)
    if method not in SMOOTH_METHODS:
        raise ValueError(
            f"unknown smooth method: {method!r} (choose from {SMOOTH_METHODS})"
        )
    if len(results) < 2:
        return results
    if method == "butter":
        if not cutoff or not fps:
            raise ValueError("method='butter' には cutoff [Hz] と fps が必要です")
    elif window <= 1:
        return results

    person_ids = sorted({p.person_index for r in results for p in r.persons})
    if not person_ids:
        return results

    n = len(results)

    def _apply(arr: np.ndarray, vis: np.ndarray) -> np.ndarray:
        if method == "butter":
            return _butter_filter(arr, cutoff, fps)
        if method == "median":
            return _nan_median_filter(arr, window)
        if weighted:
            return _weighted_moving_average(arr, window, vis)
        return _nan_moving_average(arr, window)

    for pi in person_ids:
        px = np.full((n, NUM_LANDMARKS, 3), np.nan)
        world = np.full((n, NUM_LANDMARKS, 3), np.nan)
        px_vis = np.full((n, NUM_LANDMARKS), np.nan)
        world_vis = np.full((n, NUM_LANDMARKS), np.nan)
        for fi, result in enumerate(results):
            for person in result.persons:
                if person.person_index != pi:
                    continue
                for kp in person.keypoints:
                    px[fi, kp.index] = (kp.x_px, kp.y_px, kp.z)
                    px_vis[fi, kp.index] = kp.visibility
                for wk in person.world_keypoints:
                    world[fi, wk.index] = (wk.x, wk.y, wk.z)
                    world_vis[fi, wk.index] = wk.visibility

        px_s = _apply(px, px_vis)
        world_s = _apply(world, world_vis)

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
