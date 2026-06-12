"""複数人検出時の人物 ID トラッキング。

推定バックエンドはフレームごとの検出順を保証しないため、
フレーム間の人物 ID を安定させる。マッチングには以下を併用する:

- 等速モデルによる位置予測 (交差・すれ違いに強い)
- 胴体領域の色ヒストグラム (服装の違いによる識別)

また、ID が入れ替わるリスクの高い事象 (人物同士の接近、
長いオクルージョン後の再出現) を記録し、処理後に警告として
取得できる (get_warnings)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from poselab.types import PersonPose

# 胴体領域を構成するキーポイント index (両肩・両腰)
_TORSO_INDICES = (11, 12, 23, 24)


class PersonTracker:
    """位置予測 + 外見特徴の貪欲マッチングによるトラッカー。

    Parameters
    ----------
    max_distance:
        同一人物とみなす予測位置からのずれの上限 (画像対角長比)。
    max_missed:
        未検出がこのフレーム数続いたらトラックを破棄する。
    appearance_weight:
        マッチングコストに占める外見 (色ヒストグラム) の重み。
        0 にすると位置のみでマッチングする。
    reappear_threshold:
        このフレーム数以上の未検出から復帰した場合に警告を記録する。
    """

    def __init__(
        self,
        max_distance: float = 0.25,
        max_missed: int = 30,
        appearance_weight: float = 0.5,
        velocity_smoothing: float = 0.5,
        reappear_threshold: int = 5,
    ) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.appearance_weight = appearance_weight
        self.velocity_smoothing = velocity_smoothing
        self.reappear_threshold = reappear_threshold
        self._tracks: Dict[int, dict] = {}
        self._next_id = 0
        self._events: List[dict] = []  # 入れ替わりリスク事象 (フレーム単位)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0
        self._events.clear()

    # ------------------------------------------------------------ 特徴量
    @staticmethod
    def _centroid(person: PersonPose) -> np.ndarray:
        pts = [
            (kp.x_px, kp.y_px)
            for kp in person.keypoints
            if kp.visibility >= 0.5
        ]
        if not pts:
            pts = [(kp.x_px, kp.y_px) for kp in person.keypoints]
        return np.asarray(pts, dtype=np.float64).mean(axis=0)

    @staticmethod
    def _size(person: PersonPose) -> float:
        pts = np.asarray(
            [(kp.x_px, kp.y_px) for kp in person.keypoints], dtype=np.float64
        )
        span = pts.max(axis=0) - pts.min(axis=0)
        return float(np.hypot(span[0], span[1]))

    @staticmethod
    def _torso_histogram(
        frame: np.ndarray, person: PersonPose
    ) -> Optional[np.ndarray]:
        """胴体 (両肩・両腰の囲む矩形) の HSV 色ヒストグラム。"""
        h, w = frame.shape[:2]
        pts = [person.keypoints[i] for i in _TORSO_INDICES]
        if any(p.visibility < 0.3 for p in pts):
            return None
        xs = [p.x_px for p in pts]
        ys = [p.y_px for p in pts]
        x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
        y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        patch = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _appearance_cost(
        self, track_hist: Optional[np.ndarray], det_hist: Optional[np.ndarray]
    ) -> float:
        if track_hist is None or det_hist is None:
            return 0.0
        correl = float(cv2.compareHist(track_hist, det_hist, cv2.HISTCMP_CORREL))
        return min(1.0, max(0.0, 1.0 - correl))

    # ------------------------------------------------------------ 割り当て
    def assign(
        self,
        persons: List[PersonPose],
        width: int,
        height: int,
        frame: Optional[np.ndarray] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ) -> List[PersonPose]:
        """persons の person_index を安定 ID に書き換えて返す (ID 順)。"""
        if not persons:
            for track in self._tracks.values():
                track["missed"] += 1
            self._prune()
            return persons

        diag = float(np.hypot(width, height)) or 1.0
        centroids = [self._centroid(p) for p in persons]
        sizes = [self._size(p) for p in persons]
        hists = [
            self._torso_histogram(frame, p) if frame is not None else None
            for p in persons
        ]

        # 等速モデルで現在位置を予測し、コスト (位置 + 外見) を計算
        candidates = []
        for tid, track in self._tracks.items():
            steps = track["missed"] + 1
            predicted = track["centroid"] + track["velocity"] * steps
            # 未検出が続いたトラックはゲートを広げて復帰しやすくする
            gate = self.max_distance * diag * min(2.0, 1.0 + 0.1 * track["missed"])
            for pi, c in enumerate(centroids):
                d = float(np.linalg.norm(predicted - c))
                if d > gate:
                    continue
                cost = d / diag + self.appearance_weight * self._appearance_cost(
                    track["hist"], hists[pi]
                )
                candidates.append((cost, tid, pi))
        candidates.sort(key=lambda t: t[0])

        assigned: Dict[int, int] = {}  # person list index -> track id
        used_tracks = set()
        for _, tid, pi in candidates:
            if tid in used_tracks or pi in assigned:
                continue
            assigned[pi] = tid
            used_tracks.add(tid)

        matched_ids = set()
        for pi, person in enumerate(persons):
            tid = assigned.get(pi)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {
                    "centroid": centroids[pi],
                    "velocity": np.zeros(2),
                    "missed": 0,
                    "hist": hists[pi],
                    "size": sizes[pi],
                }
            else:
                track = self._tracks[tid]
                steps = track["missed"] + 1
                if track["missed"] >= self.reappear_threshold:
                    self._record(
                        "reappear", (tid,), frame_index, timestamp_ms,
                        missed=track["missed"],
                    )
                new_velocity = (centroids[pi] - track["centroid"]) / steps
                track["velocity"] = (
                    self.velocity_smoothing * track["velocity"]
                    + (1.0 - self.velocity_smoothing) * new_velocity
                )
                track["centroid"] = centroids[pi]
                track["size"] = 0.7 * track["size"] + 0.3 * sizes[pi]
                if hists[pi] is not None:
                    track["hist"] = (
                        hists[pi] if track["hist"] is None
                        else 0.9 * track["hist"] + 0.1 * hists[pi]
                    )
                track["missed"] = 0
            person.person_index = tid
            matched_ids.add(tid)

        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track["missed"] += 1
        self._prune()

        # 人物同士の接近 (交差の可能性) を記録
        present = [(p.person_index, self._centroid(p), self._size(p)) for p in persons]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                ida, ca, sa = present[i]
                idb, cb, sb = present[j]
                if np.linalg.norm(ca - cb) < 0.5 * (sa + sb) / 2:
                    self._record(
                        "crossing", tuple(sorted((ida, idb))),
                        frame_index, timestamp_ms,
                    )

        persons.sort(key=lambda p: p.person_index)
        return persons

    def _prune(self) -> None:
        self._tracks = {
            tid: track
            for tid, track in self._tracks.items()
            if track["missed"] <= self.max_missed
        }

    # ------------------------------------------------------------ 警告
    def _record(self, kind, ids, frame_index, timestamp_ms, **extra) -> None:
        self._events.append(
            {
                "type": kind,
                "ids": tuple(ids),
                "frame": frame_index,
                "timestamp_ms": timestamp_ms,
                **extra,
            }
        )

    def get_warnings(self, merge_gap: int = 10) -> List[dict]:
        """入れ替わりリスクのある区間のリストを返す。

        連続するフレームの同種事象は 1 区間にまとめる。
        各要素: type ("crossing" / "reappear"), ids,
        frame_start / frame_end, time_start_s / time_end_s。
        """
        groups: Dict[tuple, List[dict]] = {}
        for ev in self._events:
            groups.setdefault((ev["type"], ev["ids"]), []).append(ev)

        warnings: List[dict] = []
        for (kind, ids), events in groups.items():
            events.sort(key=lambda e: (e["frame"] if e["frame"] is not None else 0))
            run: List[dict] = []
            for ev in events:
                if (
                    run
                    and ev["frame"] is not None
                    and run[-1]["frame"] is not None
                    and ev["frame"] - run[-1]["frame"] > merge_gap
                ):
                    warnings.append(self._merge_run(kind, ids, run))
                    run = []
                run.append(ev)
            if run:
                warnings.append(self._merge_run(kind, ids, run))
        warnings.sort(key=lambda w: w["frame_start"] if w["frame_start"] is not None else 0)
        return warnings

    @staticmethod
    def _merge_run(kind: str, ids: tuple, run: List[dict]) -> dict:
        first, last = run[0], run[-1]

        def sec(ev):
            return (
                round(ev["timestamp_ms"] / 1000.0, 2)
                if ev.get("timestamp_ms") is not None
                else None
            )

        merged = {
            "type": kind,
            "ids": list(ids),
            "frame_start": first["frame"],
            "frame_end": last["frame"],
            "time_start_s": sec(first),
            "time_end_s": sec(last),
        }
        if kind == "reappear":
            merged["missed_frames"] = first.get("missed")
        return merged


def format_warning(w: dict) -> str:
    """get_warnings() の 1 要素を人間向けの文字列にする。"""
    if w["frame_start"] == w["frame_end"]:
        frames = f"フレーム {w['frame_start']}"
    else:
        frames = f"フレーム {w['frame_start']}–{w['frame_end']}"
    if w.get("time_start_s") is not None:
        if w["time_start_s"] == w["time_end_s"]:
            frames += f" (t={w['time_start_s']}s)"
        else:
            frames += f" (t={w['time_start_s']}–{w['time_end_s']}s)"
    ids = " と ".join(f"P{i}" for i in w["ids"])
    if w["type"] == "crossing":
        return f"{frames}: {ids} が接近・交差"
    return (
        f"{frames}: {ids} が {w.get('missed_frames', '?')} フレームの"
        "未検出後に再出現"
    )
