import numpy as np
import pytest

from poselab.filters import smooth_results
from poselab.skeleton import LANDMARK_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint


def _frame(frame_index: int, x: float) -> FrameResult:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=x / 640.0, y_norm=0.5, z=0.0,
            visibility=0.9, presence=0.9, x_px=x, y_px=240.0,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    world = [
        WorldKeypoint(index=i, name=name, x=x / 100.0, y=0.0, z=0.0, visibility=0.9)
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    person = PersonPose(person_index=0, keypoints=keypoints, world_keypoints=world)
    return FrameResult(
        frame_index=frame_index, timestamp_ms=frame_index * 33.3,
        width=640, height=480, persons=[person],
    )


def test_smooth_constant_signal_unchanged():
    results = [_frame(i, 100.0) for i in range(10)]
    smooth_results(results, window=5)
    for r in results:
        assert r.persons[0].keypoints[0].x_px == pytest.approx(100.0)


def test_smooth_reduces_noise():
    rng = np.random.default_rng(0)
    base = 100.0
    noisy = base + rng.normal(0, 5.0, size=30)
    results = [_frame(i, float(v)) for i, v in enumerate(noisy)]
    smooth_results(results, window=5)
    smoothed = np.array([r.persons[0].keypoints[0].x_px for r in results])
    assert np.std(smoothed) < np.std(noisy)
    # x_norm も再計算されている
    assert results[5].persons[0].keypoints[0].x_norm == pytest.approx(
        smoothed[5] / 640.0
    )


def test_smooth_handles_missing_frames():
    results = [_frame(0, 100.0), _frame(1, 110.0)]
    results.append(
        FrameResult(frame_index=2, timestamp_ms=66.6, width=640, height=480)
    )
    results.append(_frame(3, 120.0))
    out = smooth_results(results, window=3)
    assert out[2].persons == []  # 未検出フレームはそのまま
    assert not np.isnan(out[3].persons[0].keypoints[0].x_px)


def test_smooth_window_one_noop():
    results = [_frame(i, float(i)) for i in range(3)]
    smooth_results(results, window=1)
    assert results[2].persons[0].keypoints[0].x_px == pytest.approx(2.0)


def test_median_filter_rejects_outlier():
    # 単発スパイクを含む信号: メディアンは外れ値を除去、平均は引きずられる
    values = [100.0, 100.0, 1000.0, 100.0, 100.0]
    res_med = [_frame(i, v) for i, v in enumerate(values)]
    smooth_results(res_med, window=3, method="median")
    assert res_med[2].persons[0].keypoints[0].x_px == pytest.approx(100.0)

    res_mean = [_frame(i, v) for i, v in enumerate(values)]
    smooth_results(res_mean, window=3, method="moving")
    assert res_mean[2].persons[0].keypoints[0].x_px > 300.0  # 外れ値に引きずられる


def test_weighted_moving_average_downweights_low_visibility():
    # 中央フレームを低 visibility・外れ値にする → 加重平均では影響が小さい
    results = [_frame(i, 100.0) for i in range(5)]
    mid = results[2].persons[0]
    for kp in mid.keypoints:
        kp.x_px = 1000.0
        kp.visibility = 0.01
    smooth_results(results, window=3, method="moving", weighted=True)
    # 近傍 (index 1) は低信頼の外れ値をほぼ無視する
    assert results[1].persons[0].keypoints[0].x_px == pytest.approx(100.0, abs=5.0)


def test_butter_zero_phase_lowpass():
    # 低周波 + 高周波ノイズ。Butterworth で高周波が減衰し、位相は保たれる
    fps = 30.0
    t = np.arange(120) / fps
    low = 50.0 * np.sin(2 * np.pi * 0.7 * t)        # 0.7 Hz (通過帯)
    high = 10.0 * np.sin(2 * np.pi * 8.0 * t)       # 8 Hz (阻止帯)
    signal = 320.0 + low + high
    results = [_frame(i, float(signal[i])) for i in range(len(t))]
    smooth_results(results, method="butter", cutoff=3.0, fps=fps)
    out = np.array([r.persons[0].keypoints[0].x_px for r in results])
    # 高周波成分が減衰している
    assert np.std(out - 320.0) < np.std(signal - 320.0)
    # ゼロ位相: 低周波基準波形との相関が高い (遅延なし)
    ref = 320.0 + low
    inner = len(t) // 4
    corr = np.corrcoef(out[inner:-inner], ref[inner:-inner])[0, 1]
    assert corr > 0.98


def test_butter_requires_cutoff_and_fps():
    results = [_frame(i, 100.0) for i in range(5)]
    with pytest.raises(ValueError):
        smooth_results(results, method="butter")
