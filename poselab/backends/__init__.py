"""推定バックエンド。

バックエンドは PoseBackend を継承して追加できます。
現在は MediaPipe Pose Landmarker 実装を同梱しています。
"""

from poselab.backends.base import PoseBackend

__all__ = ["PoseBackend", "create_backend"]


def create_backend(name: str = "mediapipe", **kwargs) -> PoseBackend:
    """名前からバックエンドを生成するファクトリ。"""
    if name == "mediapipe":
        from poselab.backends.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(**kwargs)
    raise ValueError(f"unknown backend: {name!r}")
