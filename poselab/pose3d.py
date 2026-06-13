"""動画からの 3D 骨格推定 (2D 推定 + リフティング)。

MMPose の Pose3DInferencer (人物検出 → RTMPose 2D → VideoPose3D 系
時系列リフティング) を動画単位で実行し、poselab のエクスポータと
MMPose 互換の results JSON (``meta_info`` / ``instance_info``) へ
変換する。Pose3DStudio など MMPose 系ツールの出力と互換であり、
`poselab-viewer` でそのまま再生できる。

MMPose は公開 API 経由でのみ利用しており、本ファイルのコードは
独自実装です。依存のインストール方法は
:mod:`poselab.backends.mmpose_backend` の docstring を参照。

座標系について: リフティング結果は MMPose の 3D 可視化規約
(x: 左右 (反転済み)、y: 奥行き、z: 高さ・床基準) で出力されます。
`poselab-viewer` では座標系オプション「Z 上向き」で表示してください
(MMPose 形式 JSON の読み込み時は自動で選択されます)。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

from poselab.backends.mmpose_backend import DEFAULT_POSE2D, _INSTALL_HINT
from poselab.exporters import Exporter
from poselab.skeleton import H36M17_EDGES, H36M17_NAMES
from poselab.types import FrameResult, Keypoint, PersonPose, WorldKeypoint

# 既定のリフティングモデル (MMPose model zoo のコンフィグ名)
DEFAULT_LIFT = "video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m"
FAST_LIFT = "video-pose-lift_tcn-81frm-supv_8xb128-160e_h36m"


def _load_pose3d_inferencer():
    """Pose3DInferencer クラスを遅延 import する。"""
    try:
        from mmpose.apis.inferencers import Pose3DInferencer
    except ImportError as exc:  # pragma: no cover - 実環境依存
        raise ImportError(_INSTALL_HINT) from exc
    return Pose3DInferencer


def prepare_models(
    *,
    lift_model: str = DEFAULT_LIFT,
    lift_weights: Optional[str] = None,
    pose2d: str = DEFAULT_POSE2D,
    pose2d_weights: Optional[str] = None,
    det_model: Optional[str] = None,
    det_weights: Optional[str] = None,
    device: Optional[str] = None,
    inferencer=None,
):
    """推定モデル一式 (人物検出 + 2D + 3D リフタ) の重みを事前取得する。

    Pose3DInferencer を構築するだけで、検出器・2D・3D の各チェックポイントが
    MMPose の model zoo から (未取得なら) ダウンロードされる。動画は不要。
    GUI の「モデルダウンロード」から事前に呼ぶための入口。
    """
    if inferencer is None:
        pose3d_cls = _load_pose3d_inferencer()
        inferencer = pose3d_cls(
            model=lift_model,
            weights=lift_weights,
            pose2d_model=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            device=device,
        )
    return inferencer


def _jsonable(value):
    """dataset_meta などを JSON 化できる形に再帰変換する。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_meta_info(dataset_meta: dict) -> dict:
    """results JSON 用の meta_info を dataset_meta から構築する。"""
    meta = dataset_meta or {}
    num = int(meta.get("num_keypoints", len(H36M17_NAMES)))
    id2name = meta.get("keypoint_id2name")
    if not id2name:
        names = meta.get("keypoint_names") or H36M17_NAMES
        id2name = {i: n for i, n in enumerate(names[:num])}
    links = meta.get("skeleton_links") or H36M17_EDGES
    info = {
        "dataset_name": meta.get("dataset_name", "h36m"),
        "num_keypoints": num,
        "keypoint_id2name": _jsonable(id2name),
        "skeleton_links": _jsonable(links),
    }
    if meta.get("keypoint_name2id"):
        info["keypoint_name2id"] = _jsonable(meta["keypoint_name2id"])
    return info


def keypoint_names_from_meta(dataset_meta: dict) -> List[str]:
    """dataset_meta からキーポイント名リストを取り出す (なければ H36M)。"""
    meta = dataset_meta or {}
    num = int(meta.get("num_keypoints", len(H36M17_NAMES)))
    id2name = meta.get("keypoint_id2name") or {}
    if id2name:
        return [
            str(id2name.get(i, id2name.get(str(i), f"kpt_{i}")))
            for i in range(num)
        ]
    return list(H36M17_NAMES[:num]) if num <= len(H36M17_NAMES) else [
        f"kpt_{i}" for i in range(num)
    ]


def instances_to_frame_result(
    instances: Sequence[dict],
    frame_index: int,
    timestamp_ms: float,
    width: int,
    height: int,
    names: Sequence[str],
    keypoints_2d: Optional[Sequence[np.ndarray]] = None,
    source: Optional[str] = None,
) -> FrameResult:
    """3D インスタンス辞書を FrameResult に変換する。

    instances は MMPose inferencer の predictions 形式
    (``keypoints``: [K][3], ``keypoint_scores``: [K])。
    keypoints_2d には対応する順序の 2D 座標 (px, [K][2]) を渡せる
    (リフタ入力に使われた 2D 推定値。省略時は 0 を入れる)。
    """
    persons: List[PersonPose] = []
    for pi, inst in enumerate(instances):
        kpts = np.asarray(inst.get("keypoints", ()), dtype=float)
        scores = np.asarray(inst.get("keypoint_scores", ()), dtype=float)
        if kpts.ndim != 2 or len(kpts) == 0:
            continue
        if scores.shape[0] != kpts.shape[0]:
            scores = np.ones(kpts.shape[0], dtype=float)
        px = None
        if keypoints_2d is not None and pi < len(keypoints_2d):
            arr = np.asarray(keypoints_2d[pi], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == kpts.shape[0]:
                px = arr
        keypoints = []
        world = []
        for i in range(kpts.shape[0]):
            name = str(names[i]) if i < len(names) else f"kpt_{i}"
            score = float(scores[i])
            x_px = float(px[i, 0]) if px is not None else 0.0
            y_px = float(px[i, 1]) if px is not None else 0.0
            keypoints.append(
                Keypoint(
                    index=i,
                    name=name,
                    x_norm=x_px / width if width else 0.0,
                    y_norm=y_px / height if height else 0.0,
                    z=0.0,
                    visibility=score,
                    presence=score,
                    x_px=x_px,
                    y_px=y_px,
                )
            )
            world.append(
                WorldKeypoint(
                    index=i,
                    name=name,
                    x=float(kpts[i, 0]),
                    y=float(kpts[i, 1]),
                    z=float(kpts[i, 2]),
                    visibility=score,
                )
            )
        persons.append(
            PersonPose(
                person_index=pi, keypoints=keypoints, world_keypoints=world
            )
        )
    return FrameResult(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        width=width,
        height=height,
        persons=persons,
        source=source,
    )


def _converted_2d_from_buffer(inferencer) -> Optional[List[np.ndarray]]:
    """inferencer 内部バッファから現在フレームの 2D (H36M 変換済み px) を取る。

    予測結果は track_id 昇順に整列されるため、同じ順に並べて返す。
    内部実装に依存するため、取得できない場合は None。
    """
    try:
        frames_list = inferencer._buffer["pose_est_results_list"]
        samples = frames_list[-1]
        entries = []
        for ds in samples:
            track = getattr(ds, "track_id", None)
            if track is None:
                track = ds.get("track_id", 1e4)
            kpts = np.asarray(ds.pred_instances.keypoints, dtype=float)
            if kpts.ndim == 3:
                kpts = kpts[0]
            entries.append((track, kpts[:, :2]))
        entries.sort(key=lambda item: item[0])
        return [k for _, k in entries]
    except Exception:
        return None


def _video_props(video: Path) -> tuple:
    """動画の (フレーム数, fps, 幅, 高さ) を取得する。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return frames, fps, width, height


def collect_vis_output(vis_dir: Path, video: Path, save_video: Path) -> bool:
    """MMPose が vis_dir に書いた可視化動画を save_video へ移動する。"""
    candidate = vis_dir / video.name
    if not candidate.is_file():
        stem_matches = sorted(vis_dir.glob(video.stem + ".*"))
        if not stem_matches:
            return False
        candidate = stem_matches[0]
    save_video.parent.mkdir(parents=True, exist_ok=True)
    if save_video.exists():
        save_video.unlink()
    shutil.move(str(candidate), str(save_video))
    return True


def run_pose3d(
    video: "str | Path",
    *,
    lift_model: str = DEFAULT_LIFT,
    lift_weights: Optional[str] = None,
    pose2d: str = DEFAULT_POSE2D,
    pose2d_weights: Optional[str] = None,
    det_model: Optional[str] = None,
    det_weights: Optional[str] = None,
    device: Optional[str] = None,
    json_path: "Path | None" = None,
    exporters: Sequence[Exporter] = (),
    save_video: "Path | None" = None,
    h264: bool = False,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    quiet: bool = False,
    inferencer=None,
) -> List[FrameResult]:
    """動画 1 本を 3D 推定して結果を書き出す。

    Parameters
    ----------
    json_path:
        MMPose 互換 results JSON (``meta_info`` / ``instance_info``) の
        出力先。`poselab-viewer` でそのまま再生できる。
    exporters:
        poselab のエクスポータ (CSV 等)。H36M 17 点の 3D 座標が
        world_x/y/z に入る。
    save_video:
        2D + 3D の可視化動画の出力先 (MMPose visualizer による描画)。
    inferencer:
        テスト用に構築済み inferencer を注入できる。
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"動画が見つかりません: {video}")

    total_frames, fps, width, height = _video_props(video)
    if fps <= 0:
        fps = 30.0

    if inferencer is None:
        pose3d_cls = _load_pose3d_inferencer()
        inferencer = pose3d_cls(
            model=lift_model,
            weights=lift_weights,
            pose2d_model=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            device=device,
        )

    dataset_meta = getattr(getattr(inferencer, "model", None), "dataset_meta", None) or {}
    names = keypoint_names_from_meta(dataset_meta)

    call_kwargs = {"return_datasamples": False}
    vis_tmp: Optional[Path] = None
    if save_video is not None:
        save_video = Path(save_video)
        vis_tmp = save_video.parent / (save_video.stem + "_mmpose_vis")
        vis_tmp.mkdir(parents=True, exist_ok=True)
        call_kwargs["vis_out_dir"] = str(vis_tmp)
        call_kwargs["num_instances"] = -1  # フレームごとの人数に合わせる

    results: List[FrameResult] = []
    instance_info = []
    try:
        for frame_index, result in enumerate(
            inferencer(str(video), **call_kwargs)
        ):
            batches = result.get("predictions", [])
            instances = batches[0] if batches else []
            instance_info.append(
                {"frame_id": frame_index, "instances": _jsonable(instances)}
            )
            keypoints_2d = _converted_2d_from_buffer(inferencer)
            frame_result = instances_to_frame_result(
                instances,
                frame_index=frame_index,
                timestamp_ms=frame_index * 1000.0 / fps,
                width=width,
                height=height,
                names=names,
                keypoints_2d=keypoints_2d,
                source=str(video),
            )
            results.append(frame_result)
            for exporter in exporters:
                exporter.add(frame_result)
            if progress is not None:
                progress(frame_index + 1, total_frames or None)
    finally:
        for exporter in exporters:
            exporter.close()

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta_info": build_meta_info(dataset_meta),
            "instance_info": instance_info,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    if save_video is not None and vis_tmp is not None:
        moved = collect_vis_output(vis_tmp, video, save_video)
        shutil.rmtree(vis_tmp, ignore_errors=True)
        if moved:
            if h264:
                from poselab.pipeline import reencode_h264

                if not reencode_h264(save_video) and not quiet:
                    print(
                        "注意: ffmpeg が見つからないため H.264 再エンコードを"
                        "スキップしました",
                        file=sys.stderr,
                    )
        elif not quiet:
            print(
                "注意: 可視化動画が生成されませんでした "
                "(検出 0 件の可能性があります)",
                file=sys.stderr,
            )

    return results
