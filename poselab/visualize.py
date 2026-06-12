"""骨格のオーバーレイ描画。"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from poselab.skeleton import landmark_side
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
