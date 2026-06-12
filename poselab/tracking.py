"""複数人検出時の人物 ID トラッキング。

推定バックエンドはフレームごとの検出順を保証しないため、
重心の最近傍マッチングでフレーム間の人物 ID を安定させる。
ID は出現順に割り当てられ、画面から消えても max_missed
フレームまでは同じ ID として復帰できる。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from poselab.types import PersonPose


class PersonTracker:
    """重心の貪欲最近傍マッチングによる簡易トラッカー。

    Parameters
    ----------
    max_distance:
        同一人物とみなす重心移動の上限 (画像対角長に対する割合)。
    max_missed:
        未検出がこのフレーム数続いたらトラックを破棄する。
    """

    def __init__(self, max_distance: float = 0.25, max_missed: int = 30) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._tracks: Dict[int, dict] = {}  # id -> {"centroid", "missed"}
        self._next_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

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

    def assign(
        self, persons: List[PersonPose], width: int, height: int
    ) -> List[PersonPose]:
        """persons の person_index を安定 ID に書き換えて返す (ID 順)。"""
        if not persons:
            for track in self._tracks.values():
                track["missed"] += 1
            self._prune()
            return persons

        diag = float(np.hypot(width, height)) or 1.0
        centroids = [self._centroid(p) for p in persons]

        # 距離の近い (トラック, 検出) ペアから貪欲にマッチング
        candidates = []
        for tid, track in self._tracks.items():
            for pi, c in enumerate(centroids):
                d = float(np.linalg.norm(track["centroid"] - c))
                if d <= self.max_distance * diag:
                    candidates.append((d, tid, pi))
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
            self._tracks[tid] = {"centroid": centroids[pi], "missed": 0}
            person.person_index = tid
            matched_ids.add(tid)

        for tid, track in self._tracks.items():
            if tid not in matched_ids:
                track["missed"] += 1
        self._prune()

        persons.sort(key=lambda p: p.person_index)
        return persons

    def _prune(self) -> None:
        self._tracks = {
            tid: track
            for tid, track in self._tracks.items()
            if track["missed"] <= self.max_missed
        }
