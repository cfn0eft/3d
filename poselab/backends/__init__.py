"""推定バックエンド。

バックエンドは PoseBackend を継承して追加できます。
MediaPipe Pose Landmarker (標準) と MMPose (オプション依存、
RTMDet + RTMPose) の実装を同梱しています。
"""

from poselab.backends.base import PoseBackend

__all__ = ["PoseBackend", "create_backend"]


def create_backend(name: str = "mediapipe", **kwargs) -> PoseBackend:
    """名前からバックエンドを生成するファクトリ。"""
    if name == "mediapipe":
        from poselab.backends.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(**kwargs)
    if name == "mmpose":
        from poselab.backends.mmpose_backend import MMPoseBackend

        return MMPoseBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r}")
