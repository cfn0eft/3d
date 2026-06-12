"""MMPose (RTMDet 検出 + RTMPose 2D 推定) を用いたバックエンド。

MMPose / MMDetection (いずれも Apache-2.0) は公開 API 経由でのみ
利用しており、本ファイルのコードは独自実装です。

このバックエンドはオプション依存です。インストール方法:

    pip install -U openmim
    mim install "mmcv>=2.0.1,<2.2" "mmdet>=3.1,<3.3" "mmpose>=1.2,<1.4"

(PyTorch が未導入の場合は https://pytorch.org/ の手順で先に
インストールしてください。GPU 推奨ですが CPU でも動作します)

モデルの重みは初回使用時に MMPose の公式 model zoo から自動
ダウンロードされます (保存先: torch.hub のキャッシュディレクトリ)。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from poselab.backends.base import PoseBackend
from poselab.skeleton import COCO17_EDGES, COCO17_NAMES
from poselab.types import Keypoint, PersonPose

# 既定モデル (MMPose model zoo のコンフィグ名。重みは metafile から自動解決)
DEFAULT_POSE2D = "rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192"
ACCURATE_POSE2D = "rtmpose-l_8xb256-420e_aic-coco-384x288"

_INSTALL_HINT = (
    "MMPose バックエンドには mmpose / mmdet / mmcv が必要です。\n"
    "インストール方法:\n"
    "  pip install -U openmim\n"
    '  mim install "mmcv>=2.0.1,<2.2" "mmdet>=3.1,<3.3" "mmpose>=1.2,<1.4"\n'
    "(PyTorch 未導入の場合は https://pytorch.org/ を参照)"
)


def _load_pose2d_inferencer():
    """Pose2DInferencer クラスを遅延 import する。"""
    try:
        from mmpose.apis.inferencers import Pose2DInferencer
    except ImportError as exc:  # pragma: no cover - 実環境依存
        raise ImportError(_INSTALL_HINT) from exc
    return Pose2DInferencer


def instances_to_persons(
    instances: Sequence[dict],
    width: int,
    height: int,
    max_persons: int = 0,
    keypoint_names: Sequence[str] = COCO17_NAMES,
) -> List[PersonPose]:
    """MMPose の per-instance 辞書を PersonPose に変換する。

    instances は MMPose inferencer の predictions 形式
    (``keypoints``: [K][2], ``keypoint_scores``: [K]) を想定。
    max_persons > 0 のときは bbox/平均スコアの高い順に上位のみ残す。
    """
    scored = []
    for inst in instances:
        kpts = np.asarray(inst.get("keypoints", ()), dtype=float)
        scores = np.asarray(inst.get("keypoint_scores", ()), dtype=float)
        if kpts.ndim != 2 or len(kpts) == 0:
            continue
        if scores.shape[0] != kpts.shape[0]:
            scores = np.ones(kpts.shape[0], dtype=float)
        order_score = inst.get("bbox_score")
        if order_score is None:
            order_score = float(scores.mean())
        scored.append((float(order_score), kpts, scores))

    scored.sort(key=lambda item: item[0], reverse=True)
    if max_persons and max_persons > 0:
        scored = scored[:max_persons]

    persons: List[PersonPose] = []
    for pi, (_, kpts, scores) in enumerate(scored):
        keypoints = []
        for i in range(kpts.shape[0]):
            name = (
                keypoint_names[i] if i < len(keypoint_names) else f"kpt_{i}"
            )
            x = float(kpts[i, 0])
            y = float(kpts[i, 1])
            score = float(scores[i])
            keypoints.append(
                Keypoint(
                    index=i,
                    name=name,
                    x_norm=x / width if width else 0.0,
                    y_norm=y / height if height else 0.0,
                    z=0.0,
                    visibility=score,
                    presence=score,
                    x_px=x,
                    y_px=y,
                )
            )
        persons.append(
            PersonPose(person_index=pi, keypoints=keypoints, world_keypoints=[])
        )
    return persons


class MMPoseBackend(PoseBackend):
    """RTMDet (人物検出) + RTMPose (COCO 17 点 2D) バックエンド。

    Parameters
    ----------
    pose2d:
        MMPose のモデル指定 (コンフィグ名 / エイリアス / コンフィグパス)。
    pose2d_weights:
        チェックポイントのパス。省略時は metafile から自動取得。
    det_model / det_weights:
        人物検出器の指定。省略時は MMPose 既定の RTMDet-M 人物検出器。
    device:
        "cuda:0" / "cpu" など。省略時は自動選択。
    num_poses:
        残す最大人数 (0 で無制限)。
    """

    name = "mmpose"

    def __init__(
        self,
        pose2d: str = DEFAULT_POSE2D,
        pose2d_weights: Optional[str] = None,
        det_model: Optional[str] = None,
        det_weights: Optional[str] = None,
        device: Optional[str] = None,
        num_poses: int = 1,
        bbox_thr: float = 0.3,
        nms_thr: float = 0.3,
        inferencer=None,
    ) -> None:
        self._num_poses = num_poses
        self._bbox_thr = bbox_thr
        self._nms_thr = nms_thr
        if inferencer is None:
            pose2d_cls = _load_pose2d_inferencer()
            inferencer = pose2d_cls(
                model=pose2d,
                weights=pose2d_weights,
                device=device,
                det_model=det_model,
                det_weights=det_weights,
            )
        self._inferencer = inferencer

    @property
    def keypoint_names(self) -> Sequence[str]:
        return COCO17_NAMES

    @property
    def skeleton(self) -> Sequence[Tuple[int, int]]:
        return COCO17_EDGES

    def process(self, frame_bgr: np.ndarray, timestamp_ms: float) -> List[PersonPose]:
        h, w = frame_bgr.shape[:2]
        result = next(
            self._inferencer(
                frame_bgr,
                return_datasamples=False,
                bbox_thr=self._bbox_thr,
                nms_thr=self._nms_thr,
            )
        )
        batches = result.get("predictions", [])
        instances = batches[0] if batches else []
        return instances_to_persons(
            instances, w, h, max_persons=self._num_poses
        )

    def close(self) -> None:
        self._inferencer = None
