"""高次の運動学的特徴量 (純関数)。

速度より高次の量 (加速度・ジャーク)、左右対称性、歩行リズムなど、
研究でよく使う指標を計算する。バックエンドに依存せず numpy のみ。

注意: 有限差分による高次微分 (特にジャーク) は推定ノイズに敏感なので、
研究では平滑化 (--smooth) と併用することを推奨する。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# 左右対称性を評価する関節角度ペア (左, 右)。
SYMMETRY_PAIRS = (
    ("elbow", "left_elbow", "right_elbow"),
    ("shoulder", "left_shoulder", "right_shoulder"),
    ("hip", "left_hip", "right_hip"),
    ("knee", "left_knee", "right_knee"),
    ("ankle", "left_ankle", "right_ankle"),
)


def symmetry_index(left: float, right: float) -> float:
    """左右の値の非対称度を 0 (完全対称) を基準に返す。

    Symmetry Index = |L - R| / ((|L| + |R|) / 2)。リハビリ等で使われる
    正規化指標で、0 が左右対称、値が大きいほど非対称。両方 0 のときは 0。
    """
    left = float(left)
    right = float(right)
    denom = (abs(left) + abs(right)) / 2.0
    if denom < 1e-9:
        return 0.0
    return abs(left - right) / denom


def derivative(prev: float, cur: float, dt: float) -> Optional[float]:
    """後退差分による 1 階微分。dt<=0 なら None。"""
    if dt <= 0:
        return None
    return (cur - prev) / dt


def estimate_period(times_s: Sequence[float], signal: Sequence[float]) -> Optional[float]:
    """1 次元周期信号の周期 [秒] を自己相関で推定する。

    歩行などの周期運動の 1 周期 (例: 足首の上下動) を求めるのに使う。
    周期性が弱い/データが短い場合は None を返す。等間隔サンプリングを仮定。
    """
    t = np.asarray(times_s, dtype=np.float64)
    x = np.asarray(signal, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    n = x.size
    if n < 8:
        return None
    duration = t[-1] - t[0]
    if duration <= 0:
        return None
    dt = duration / (n - 1)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return None
    # 自己相関 (非負ラグ) を計算し、最初の有意なピークを周期とみなす
    corr = np.correlate(x, x, mode="full")[n - 1:]
    corr = corr / corr[0]
    # ゼロ交差後に現れる最大ピークのラグを探す
    zero_cross = np.argmax(corr < 0) if np.any(corr < 0) else 1
    if zero_cross < 1:
        zero_cross = 1
    search = corr[zero_cross:]
    if search.size == 0:
        return None
    peak_lag = int(np.argmax(search)) + zero_cross
    if corr[peak_lag] < 0.3:  # 周期性が弱い
        return None
    return float(peak_lag * dt)


def estimate_cadence(times_s: Sequence[float], signal: Sequence[float]) -> Optional[float]:
    """周期信号から毎分のサイクル数 (cadence) を推定する。None なら不定。"""
    period = estimate_period(times_s, signal)
    if not period or period <= 0:
        return None
    return 60.0 / period


def gait_summary(results) -> Dict[str, object]:
    """足首の上下動から歩行リズム (cadence・1 周期) を推定する。

    左右どちらかの足首が十分に検出されていれば推定する。world 座標が
    あれば鉛直成分 (y) を、なければピクセル y を信号として使う。推定でき
    なければ空 dict を返す (誤った値を出さない)。
    """
    from poselab.skeleton import LANDMARK_NAMES

    index = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    out: Dict[str, object] = {}
    for side in ("left_ankle", "right_ankle"):
        ankle = index.get(side)
        if ankle is None:
            continue
        times: List[float] = []
        signal: List[float] = []
        for r in results:
            person = next((p for p in r.persons if p.person_index == 0), None)
            if person is None:
                continue
            value = None
            if person.world_keypoints and ankle < len(person.world_keypoints):
                value = person.world_keypoints[ankle].y
            elif ankle < len(person.keypoints):
                value = person.keypoints[ankle].y_px
            if value is None:
                continue
            times.append(r.timestamp_ms / 1000.0)
            signal.append(value)
        cadence = estimate_cadence(times, signal)
        if cadence is not None:
            out[side] = {
                "cadence_per_min": round(cadence, 1),
                "cycle_time_s": round(60.0 / cadence, 3),
                "method": "ankle vertical autocorrelation (estimate)",
            }
    return out
