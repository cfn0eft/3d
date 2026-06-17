"""poselab: 研究用ヒト骨格推定ツールキット。

画像・動画・カメラ入力に対応し、CLI / GUI の両方から
2D・3D キーポイント座標の推定とエクスポートができます。
"""

# パッケージ版数の唯一のソース (pyproject.toml が dynamic version で参照)
__version__ = "0.9.4"

from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint

__all__ = [
    "FrameResult",
    "Keypoint",
    "PersonPose",
    "WorldKeypoint",
    "__version__",
]
