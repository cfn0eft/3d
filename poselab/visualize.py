"""骨格・軌跡のオーバーレイ描画。"""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from poselab.skeleton import LANDMARK_NAMES, landmark_side
from poselab.types import FrameResult, PersonPose

# BGR
COLOR_LEFT = (80, 175, 255)    # オレンジ系 (左半身)
COLOR_RIGHT = (255, 190, 80)   # 水色系 (右半身)
COLOR_CENTER = (120, 230, 120)  # 緑系 (体幹・顔中心)
COLOR_TEXT = (240, 240, 240)


def _edge_color(name_a: str, name_b: str) -> Tuple[int, int, int]:
    sides = {landmark_side(name_a), landmark_side(name_b)}
    if sides == {"left"}:
        return COLOR_LEFT
    if sides == {"right"}:
        return COLOR_RIGHT
    return COLOR_CENTER


def draw_person(
    frame: np.ndarray,
    person: PersonPose,
    skeleton: Sequence[Tuple[int, int]],
    min_visibility: float = 0.3,
    point_radius: int = 4,
    line_thickness: int = 2,
    draw_labels: bool = False,
) -> None:
    """1 人分の骨格を frame に上書き描画する。"""
    kps = person.keypoints
    for a, b in skeleton:
        if a >= len(kps) or b >= len(kps):
            continue
        ka, kb = kps[a], kps[b]
        if ka.visibility < min_visibility or kb.visibility < min_visibility:
            continue
        cv2.line(
            frame,
            (int(ka.x_px), int(ka.y_px)),
            (int(kb.x_px), int(kb.y_px)),
            _edge_color(ka.name, kb.name),
            line_thickness,
            cv2.LINE_AA,
        )
    for kp in kps:
        if kp.visibility < min_visibility:
            continue
        side = landmark_side(kp.name)
        color = (
            COLOR_LEFT if side == "left"
            else COLOR_RIGHT if side == "right"
            else COLOR_CENTER
        )
        cv2.circle(
            frame, (int(kp.x_px), int(kp.y_px)), point_radius, color, -1, cv2.LINE_AA
        )
        if draw_labels:
            cv2.putText(
                frame,
                kp.name,
                (int(kp.x_px) + 5, int(kp.y_px) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )


def draw_result(
    frame: np.ndarray,
    result: FrameResult,
    skeleton: Sequence[Tuple[int, int]],
    min_visibility: float = 0.3,
    draw_labels: bool = False,
) -> np.ndarray:
    """フレーム結果全体 (複数人) を描画して frame を返す。"""
    for person in result.persons:
        draw_person(
            frame, person, skeleton,
            min_visibility=min_visibility, draw_labels=draw_labels,
        )
    return frame


def _side_color(name: str) -> Tuple[int, int, int]:
    side = landmark_side(name)
    if side == "left":
        return COLOR_LEFT
    if side == "right":
        return COLOR_RIGHT
    return COLOR_CENTER


class TrajectoryOverlay:
    """キーポイント位置の軌跡 (モーショントレイル) を動画上に描画する。

    update() で毎フレームの位置を蓄積し、draw() で直近 length
    フレーム分の軌跡を古いほど暗く・細くフェードさせて描画する。
    検出が途切れた区間は線を繋がない。
    """

    def __init__(
        self,
        keypoint_names: Optional[Sequence[str]] = None,
        length: int = 30,
        min_visibility: float = 0.5,
    ) -> None:
        names = list(keypoint_names) if keypoint_names else ["left_wrist", "right_wrist"]
        if "all" in names:
            names = list(LANDMARK_NAMES)
        unknown = [n for n in names if n not in LANDMARK_NAMES]
        if unknown:
            raise ValueError(f"unknown keypoint name(s): {unknown}")
        self.indices = [LANDMARK_NAMES.index(n) for n in names]
        self.length = max(2, length)
        self.min_visibility = min_visibility
        # (person_index, keypoint_index) -> deque[(x, y) | None]
        self._trails: Dict[Tuple[int, int], deque] = {}

    def reset(self) -> None:
        self._trails.clear()

    def update(self, result: FrameResult) -> None:
        seen = set()
        for person in result.persons:
            for ki in self.indices:
                if ki >= len(person.keypoints):
                    continue
                kp = person.keypoints[ki]
                key = (person.person_index, ki)
                seen.add(key)
                trail = self._trails.setdefault(key, deque(maxlen=self.length))
                if kp.visibility >= self.min_visibility:
                    trail.append((kp.x_px, kp.y_px))
                else:
                    trail.append(None)
        # このフレームで見えなかった人物の軌跡にも切れ目を入れる
        for key, trail in self._trails.items():
            if key not in seen:
                trail.append(None)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for (_, ki), trail in self._trails.items():
            n = len(trail)
            if n < 2:
                continue
            color = np.array(_side_color(LANDMARK_NAMES[ki]), dtype=np.float64)
            points = list(trail)
            for i in range(1, n):
                p0, p1 = points[i - 1], points[i]
                if p0 is None or p1 is None:
                    continue
                fade = 0.25 + 0.75 * i / (n - 1)  # 古い線分ほど暗く
                thickness = 1 + int(round(2 * i / (n - 1)))
                cv2.line(
                    frame,
                    (int(p0[0]), int(p0[1])),
                    (int(p1[0]), int(p1[1])),
                    tuple(int(c) for c in color * fade),
                    thickness,
                    cv2.LINE_AA,
                )
        return frame


def draw_status(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int] = (10, 28),
) -> None:
    """FPS 等のステータス文字列を左上に描画する。"""
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
        0.7, COLOR_TEXT, 1, cv2.LINE_AA,
    )
