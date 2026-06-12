"""MediaPipe Pose Landmarker (Tasks API) を用いたバックエンド。

MediaPipe は依存ライブラリとして公開 API 経由でのみ利用しています
(Apache-2.0)。本ファイルのコードは独自実装です。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from poselab.backends.base import PoseBackend
from poselab.models import get_model_path
from poselab.skeleton import LANDMARK_NAMES, SKELETON_EDGES
from poselab.types import Keypoint, PersonPose, WorldKeypoint


class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose Landmarker バックエンド (33 点、複数人対応)。

    Parameters
    ----------
    model:
        "lite" / "full" / "heavy"。精度と速度のトレードオフ。
    num_poses:
        同時に検出する最大人数。
    static_image_mode:
        True なら毎フレーム独立に検出 (静止画向き)。False なら
        トラッキングを併用 (動画・カメラ向き、高速)。
    """

    name = "mediapipe"

    def __init__(
        self,
        model: str = "full",
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
        model_path: "str | None" = None,
    ) -> None:
        path = model_path or str(get_model_path(model))
        self._static = static_image_mode
        mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        # VIDEO モードはタイムスタンプの単調増加が必須なので内部で保証する
        self._last_ts_ms = -1

    @property
    def keypoint_names(self) -> Sequence[str]:
        return LANDMARK_NAMES

    @property
    def skeleton(self) -> Sequence[Tuple[int, int]]:
        return SKELETON_EDGES

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> List[PersonPose]:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static:
            result = self._landmarker.detect(mp_image)
        else:
            ts = int(round(timestamp_ms))
            if ts <= self._last_ts_ms:
                ts = self._last_ts_ms + 1
            self._last_ts_ms = ts
            result = self._landmarker.detect_for_video(mp_image, ts)

        persons: List[PersonPose] = []
        world_lists = result.pose_world_landmarks or []
        for pi, landmarks in enumerate(result.pose_landmarks or []):
            keypoints = [
                Keypoint(
                    index=i,
                    name=LANDMARK_NAMES[i],
                    x_norm=lm.x,
                    y_norm=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                    presence=lm.presence,
                    x_px=lm.x * w,
                    y_px=lm.y * h,
                )
                for i, lm in enumerate(landmarks)
            ]
            world_keypoints = []
            if pi < len(world_lists):
                world_keypoints = [
                    WorldKeypoint(
                        index=i,
                        name=LANDMARK_NAMES[i],
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                        visibility=lm.visibility,
                    )
                    for i, lm in enumerate(world_lists[pi])
                ]
            persons.append(
                PersonPose(
                    person_index=pi,
                    keypoints=keypoints,
                    world_keypoints=world_keypoints,
                )
            )
        return persons

    def close(self) -> None:
        self._landmarker.close()
