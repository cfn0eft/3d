"""コマンドラインインターフェース。

例:
    # 動画を処理して CSV と注釈付き動画を出力
    poselab --input walk.mp4 --csv walk.csv --save-video walk_annotated.mp4

    # 静止画 (複数可)
    poselab --input photo.jpg --json photo.json --save-image annotated.jpg

    # カメラ 0 番をライブ表示しつつ座標を記録 (q で終了)
    poselab --input camera:0 --show --csv live.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from poselab import __version__
from poselab.exporters import CsvExporter, Exporter, JsonExporter, NpzExporter
from poselab.models import MODEL_VARIANTS
from poselab.pipeline import VideoWriter, run_pipeline
from poselab.skeleton import LANDMARK_NAMES
from poselab.sources import ImageSource, open_source

# --auto-output で選択できる出力形式 (--outputs)
AUTO_OUTPUT_FORMATS = ("long", "wide", "json", "angles", "summary", "video", "image")


def parse_outputs(value: "str | None", parser: argparse.ArgumentParser) -> set:
    """--outputs のカンマ区切りリストを検証して set にする (None = 全形式)。"""
    if value is None:
        return set(AUTO_OUTPUT_FORMATS)
    tokens = {t.strip() for t in str(value).split(",") if t.strip()}
    if not tokens:
        parser.error("--outputs に形式が指定されていません")
    invalid = sorted(tokens - set(AUTO_OUTPUT_FORMATS))
    if invalid:
        parser.error(
            f"--outputs に不明な形式があります: {', '.join(invalid)} "
            f"(指定可能: {', '.join(AUTO_OUTPUT_FORMATS)})"
        )
    return tokens


def auto_output_paths(spec: Path, selected: set, is_image: bool) -> dict:
    """--auto-output の 1 入力分の出力先パスを選択形式に応じて組み立てる。"""
    stem = spec.stem
    out_dir = spec.parent / f"{stem}_poselab"
    return {
        "out_dir": out_dir,
        "csv": out_dir / f"{stem}_long.csv" if "long" in selected else None,
        "wide_csv": out_dir / f"{stem}_wide.csv" if "wide" in selected else None,
        "json": out_dir / f"{stem}.json" if "json" in selected else None,
        "angles_csv": (
            out_dir / f"{stem}_angles.csv" if "angles" in selected else None
        ),
        "summary_json": (
            out_dir / f"{stem}_summary.json" if "summary" in selected else None
        ),
        "save_image": (
            out_dir / f"{stem}_annotated.png"
            if is_image and "image" in selected else None
        ),
        "save_video": (
            out_dir / f"{stem}_annotated.mp4"
            if not is_image and "video" in selected else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poselab",
        description="ヒト骨格推定ツール: 画像・動画・カメラから座標を推定して出力します。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"poselab {__version__}")
    parser.add_argument(
        "--input", "-i", nargs="+",
        help="入力。画像/動画のパス (画像は複数可) または camera:0 のようなカメラ指定",
    )
    parser.add_argument(
        "--list-keypoints", action="store_true",
        help="キーポイント名一覧を表示して終了",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="環境情報 (バージョン・モデルキャッシュ等) を表示して終了",
    )
    parser.add_argument(
        "--list-cameras", action="store_true",
        help="利用可能なカメラを検索して表示して終了",
    )

    g_model = parser.add_argument_group("モデル設定")
    g_model.add_argument(
        "--backend", choices=("mediapipe", "mmpose"), default="mediapipe",
        help="推定バックエンド (mmpose はオプション依存。README 参照)",
    )
    g_model.add_argument(
        "--model", choices=MODEL_VARIANTS, default="full",
        help="mediapipe のモデルサイズ (lite=高速, heavy=高精度)",
    )
    g_model.add_argument("--num-poses", type=int, default=1, help="最大検出人数")
    g_model.add_argument(
        "--device", default=None,
        help="mmpose の実行デバイス (cuda:0 / cpu。省略時は自動選択)",
    )
    g_model.add_argument(
        "--pose2d-model", default=None,
        help="mmpose の 2D モデル (コンフィグ名/パス。既定: RTMPose-M COCO)",
    )
    g_model.add_argument("--pose2d-weights", default=None, help="2D チェックポイント")
    g_model.add_argument(
        "--det-model", default=None,
        help="mmpose の人物検出モデル (既定: RTMDet-M 人物検出器)",
    )
    g_model.add_argument("--det-weights", default=None, help="検出チェックポイント")
    g_model.add_argument(
        "--pose3d", action="store_true",
        help="3D リフティングを実行 (動画入力のみ、mmpose を使用。"
             "--json は MMPose 互換 results JSON になり poselab-viewer で再生可)",
    )
    g_model.add_argument(
        "--prepare-models", action="store_true",
        help="推定モデル一式 (検出 + 2D + 3D) の重みを事前ダウンロードして終了 "
             "(--pose3d と併用、動画不要)",
    )
    g_model.add_argument(
        "--lift-model", default=None,
        help="3D リフティングモデル (既定: VideoPose3D 243frm 教師あり)",
    )
    g_model.add_argument("--lift-weights", default=None, help="リフティングチェックポイント")
    g_model.add_argument(
        "--no-track", action="store_true",
        help="複数人検出時の人物 ID トラッキングを無効化",
    )
    g_model.add_argument(
        "--min-detection-confidence", type=float, default=0.5,
        help="検出の信頼度しきい値",
    )
    g_model.add_argument(
        "--min-tracking-confidence", type=float, default=0.5,
        help="トラッキングの信頼度しきい値",
    )

    g_out = parser.add_argument_group("出力")
    g_out.add_argument("--csv", type=Path, help="座標を CSV (ロング形式) で出力")
    g_out.add_argument(
        "--wide-csv", type=Path,
        help="座標を CSV (ワイド形式、1 行 = 1 フレーム) で出力",
    )
    g_out.add_argument(
        "--auto-output", action="store_true",
        help="動画と同じ場所の <動画名>_poselab/ フォルダへ一括出力 "
             "(複数動画の連続処理に対応。既定は全形式、--outputs で選択)",
    )
    g_out.add_argument(
        "--outputs", metavar="LIST", default=None,
        help="--auto-output で書き出す形式をカンマ区切りで選択。"
             f"指定可能: {', '.join(AUTO_OUTPUT_FORMATS)} "
             "(例: --outputs json,wide。既定: すべて)",
    )
    g_out.add_argument("--json", type=Path, help="座標を JSON で出力")
    g_out.add_argument("--npz", type=Path, help="座標を NumPy .npz で出力")
    g_out.add_argument(
        "--angles-csv", type=Path,
        help="関節角度 (肘・肩・股・膝・足首) を CSV で出力",
    )
    g_out.add_argument(
        "--velocity-csv", type=Path,
        help="キーポイント速度 (px/s, m/s) を CSV で出力",
    )
    g_out.add_argument(
        "--distance", action="append", metavar="A:B",
        help="2 点間距離を計算するペア (例: right_wrist:nose、複数指定可)",
    )
    g_out.add_argument(
        "--distance-csv", type=Path,
        help="--distance で指定した距離の出力先 CSV",
    )
    g_out.add_argument(
        "--summary-json", type=Path,
        help="処理サマリ (検出率等) を JSON で出力",
    )
    g_out.add_argument(
        "--smooth", type=int, default=0, metavar="N",
        help="座標を N フレームの移動平均で平滑化してから出力 (0=無効)",
    )
    g_out.add_argument("--save-video", type=Path, help="骨格を描画した動画を保存")
    g_out.add_argument(
        "--h264", action="store_true",
        help="--save-video の動画を H.264 に再エンコード "
             "(ブラウザ再生可、要 ffmpeg)",
    )
    g_out.add_argument("--save-image", type=Path, help="骨格を描画した画像を保存 (静止画入力時)")

    g_view = parser.add_argument_group("表示・描画")
    g_view.add_argument("--show", action="store_true", help="プレビューウィンドウを表示 (q で終了)")
    g_view.add_argument("--no-draw", action="store_true", help="骨格描画を行わない")
    g_view.add_argument("--draw-labels", action="store_true", help="キーポイント名も描画する")
    g_view.add_argument(
        "--trail", type=int, default=0, metavar="N",
        help="キーポイントの軌跡を直近 N フレーム分描画 (0=無効)",
    )
    g_view.add_argument(
        "--trail-keypoints", default="left_wrist,right_wrist",
        help="軌跡を描画するキーポイント (カンマ区切り、all で全点)",
    )
    g_view.add_argument(
        "--min-visibility", type=float, default=0.3,
        help="描画する visibility のしきい値",
    )

    g_misc = parser.add_argument_group("その他")
    g_misc.add_argument("--max-frames", type=int, help="処理する最大フレーム数")
    g_misc.add_argument("--camera-width", type=int, help="カメラの取得幅")
    g_misc.add_argument("--camera-height", type=int, help="カメラの取得高さ")
    g_misc.add_argument(
        "--camera-mirror", action="store_true",
        help="カメラ映像を左右反転 (鏡像) で処理する",
    )
    g_misc.add_argument("--quiet", "-q", action="store_true", help="進捗表示を抑制")
    return parser


def print_info() -> None:
    """環境診断情報を表示する。"""
    import platform

    import cv2
    import mediapipe
    import numpy

    from poselab.models import MODEL_VARIANTS, default_cache_dir

    print(f"poselab   : {__version__}")
    print(f"Python    : {platform.python_version()} ({platform.platform()})")
    print(f"mediapipe : {mediapipe.__version__}")
    print(f"OpenCV    : {cv2.__version__}")
    print(f"NumPy     : {numpy.__version__}")
    try:
        import tkinter  # noqa: F401

        print("tkinter   : 利用可能 (GUI 起動可)")
    except ImportError:
        print("tkinter   : 見つかりません (GUI は利用不可)")
    try:
        import mmpose

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "不明"
        print(f"mmpose    : {mmpose.__version__} (--backend mmpose 利用可, device={device})")
    except ImportError:
        print("mmpose    : 未導入 (--backend mmpose / --pose3d は利用不可。README 参照)")
    cache = default_cache_dir()
    print(f"モデルキャッシュ: {cache}")
    for variant in MODEL_VARIANTS:
        path = cache / f"pose_landmarker_{variant}.task"
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {variant:5s}: ダウンロード済み ({size_mb:.1f} MB)")
        else:
            print(f"  {variant:5s}: 未取得 (初回使用時に自動ダウンロード)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_keypoints:
        if args.backend == "mmpose":
            from poselab.skeleton import COCO17_NAMES, H36M17_NAMES

            print("# 2D (COCO 17 点)")
            for i, name in enumerate(COCO17_NAMES):
                print(f"{i:2d}  {name}")
            print("# 3D リフティング (--pose3d, Human3.6M 17 点)")
            for i, name in enumerate(H36M17_NAMES):
                print(f"{i:2d}  {name}")
        else:
            for i, name in enumerate(LANDMARK_NAMES):
                print(f"{i:2d}  {name}")
        return 0
    if args.info:
        print_info()
        return 0
    if args.list_cameras:
        from poselab.sources import scan_cameras

        print("カメラを検索中... (数秒かかります)", file=sys.stderr)
        cameras = scan_cameras()
        if not cameras:
            print("利用可能なカメラが見つかりませんでした。")
            print("接続と OS のカメラアクセス許可を確認してください。")
            return 1
        for cam in cameras:
            print(
                f"カメラ {cam['index']}: 利用可能 "
                f"({cam['width']}x{cam['height']}) "
                f"→ poselab --input camera:{cam['index']} --show"
            )
        return 0

    if args.prepare_models:
        return _run_prepare_models(parser, args)

    if not args.input:
        parser.error("--input を指定してください (--list-keypoints 以外では必須)")

    if args.outputs is not None and not args.auto_output:
        parser.error("--outputs は --auto-output と併用してください")

    if args.auto_output:
        return _run_auto_output(parser, args)
    return _run_job(parser, args, args.input)


def _run_auto_output(parser: argparse.ArgumentParser, args) -> int:
    """入力ごとに <名前>_poselab/ フォルダを作って全形式を出力する。"""
    from poselab.sources import IMAGE_EXTENSIONS

    specs = [Path(s) for s in args.input]
    for spec in specs:
        if str(spec).lower().startswith(("camera:", "cam:")):
            parser.error("--auto-output はカメラ入力では使用できません")
        if not spec.exists():
            parser.error(f"入力ファイルが見つかりません: {spec}")

    selected = parse_outputs(args.outputs, parser)
    total = len(specs)
    for i, spec in enumerate(specs):
        job = argparse.Namespace(**vars(args))
        is_image = spec.suffix.lower() in IMAGE_EXTENSIONS
        paths = auto_output_paths(spec, selected, is_image)
        out_dir = paths["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        job.csv = paths["csv"]
        job.wide_csv = paths["wide_csv"]
        job.json = paths["json"]
        job.angles_csv = paths["angles_csv"]
        job.summary_json = paths["summary_json"]
        job.save_image = paths["save_image"]
        job.save_video = paths["save_video"]
        if job.save_video is not None:
            job.h264 = True
        if not args.quiet and total > 1:
            print(f"=== 入力 {i + 1}/{total}: {spec}", file=sys.stderr)
        code = _run_job(parser, job, [str(spec)])
        if code != 0:
            return code
        if not args.quiet:
            print(f"  出力フォルダ: {out_dir}", file=sys.stderr)
    return 0


def _run_prepare_models(parser: argparse.ArgumentParser, args) -> int:
    """--prepare-models: 推定モデルの重みを事前ダウンロードする。"""
    # MediaPipe バックエンド (--pose3d 無し) は Pose Landmarker の .task を取得
    if args.backend == "mediapipe" and not args.pose3d:
        from poselab.models import get_model_path

        try:
            print(
                f"MediaPipe モデル '{args.model}' を準備しています...",
                file=sys.stderr,
            )
            path = get_model_path(args.model)
        except Exception as exc:  # noqa: BLE001 - 取得失敗はそのまま報告
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        print(f"モデルの準備が完了しました: {path}", file=sys.stderr)
        return 0

    from poselab.backends.mmpose_backend import DEFAULT_POSE2D
    from poselab.pose3d import DEFAULT_LIFT, prepare_models

    lift = args.lift_model or DEFAULT_LIFT
    pose2d = args.pose2d_model or DEFAULT_POSE2D
    try:
        print(
            "モデルを準備しています (未取得の重みはここでダウンロードされます)...",
            file=sys.stderr,
        )
        prepare_models(
            lift_model=lift,
            lift_weights=args.lift_weights,
            pose2d=pose2d,
            pose2d_weights=args.pose2d_weights,
            det_model=args.det_model,
            det_weights=args.det_weights,
            device=args.device,
        )
    except ImportError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    print("モデルの準備が完了しました", file=sys.stderr)
    return 0


def _run_pose3d_job(parser: argparse.ArgumentParser, args, specs: List[str]) -> int:
    """--pose3d: 動画 1 本を 2D 推定 + 3D リフティングで処理する。"""
    from poselab.sources import IMAGE_EXTENSIONS

    if len(specs) != 1:
        parser.error("--pose3d は動画 1 本ごとに実行してください")
    spec = specs[0]
    if str(spec).lower().startswith(("camera:", "cam:")):
        parser.error("--pose3d はカメラ入力では使用できません")
    video = Path(spec)
    if video.suffix.lower() in IMAGE_EXTENSIONS:
        parser.error("--pose3d は動画入力専用です")
    if not video.exists():
        parser.error(f"入力ファイルが見つかりません: {video}")
    if args.show:
        parser.error("--pose3d は --show に対応していません")
    for value, name in (
        (args.npz, "--npz"),
        (args.velocity_csv, "--velocity-csv"),
        (args.distance, "--distance"),
    ):
        if value:
            parser.error(f"{name} は --pose3d では未対応です")
    notes = []
    if args.smooth and args.smooth > 1:
        notes.append("--smooth は無視されます")
    if args.max_frames:
        notes.append("--max-frames は無視されます (全フレームを処理)")
    if args.angles_csv:
        notes.append("角度 CSV は出力されません")
    if notes and not args.quiet:
        for note in notes:
            print(f"注意: --pose3d では {note}", file=sys.stderr)

    from poselab.backends.mmpose_backend import DEFAULT_POSE2D
    from poselab.exporters import WideCsvExporter
    from poselab.pose3d import DEFAULT_LIFT, run_pose3d
    from poselab.progress import ProgressReporter
    from poselab.skeleton import H36M17_NAMES

    exporters: List[Exporter] = []
    if args.csv:
        exporters.append(CsvExporter(args.csv))
    if args.wide_csv:
        exporters.append(WideCsvExporter(args.wide_csv, H36M17_NAMES))

    reporter = ProgressReporter(total=None, enabled=not args.quiet)

    def progress(done: int, total) -> None:
        reporter.total = total
        reporter.update(done)

    lift_model = args.lift_model or DEFAULT_LIFT
    pose2d_model = args.pose2d_model or DEFAULT_POSE2D
    try:
        results = run_pose3d(
            video,
            lift_model=lift_model,
            lift_weights=args.lift_weights,
            pose2d=pose2d_model,
            pose2d_weights=args.pose2d_weights,
            det_model=args.det_model,
            det_weights=args.det_weights,
            device=args.device,
            json_path=args.json,
            exporters=exporters,
            save_video=args.save_video,
            h264=args.h264,
            progress=progress,
            quiet=args.quiet,
        )
        reporter.finish()
    except ImportError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    from poselab.analysis import summarize_results

    summary = summarize_results(results)
    if args.summary_json:
        import json

        metadata = {
            "tool": f"poselab {__version__}",
            "backend": "mmpose",
            "model": pose2d_model,
            "lift_model": lift_model,
            "input": [str(video)],
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump({**metadata, **summary}, f, ensure_ascii=False, indent=2)

    if not args.quiet:
        print(
            "完了: {total_frames} フレーム処理、"
            "{detected_frames} フレームで人物を検出 (検出率 {rate:.1f}%)".format(
                rate=summary["detection_rate"] * 100, **summary
            ),
            file=sys.stderr,
        )
        for label, path in (
            ("CSV", args.csv), ("ワイドCSV", args.wide_csv),
            ("3D JSON", args.json), ("サマリ", args.summary_json),
            ("動画", args.save_video),
        ):
            if path:
                print(f"  {label}: {path}", file=sys.stderr)
        if args.json:
            print(
                "  ヒント: 3D JSON は poselab-viewer にドロップすると再生できます",
                file=sys.stderr,
            )
    return 0


def _run_job(parser: argparse.ArgumentParser, args, specs: List[str]) -> int:
    if args.pose3d:
        return _run_pose3d_job(parser, args, specs)

    # 入力ソースの決定
    image_paths = [s for s in specs if Path(s).suffix.lower() in
                   {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
    if len(specs) > 1:
        if len(image_paths) != len(specs):
            parser.error("複数入力に対応しているのは画像ファイルのみです")
        source = ImageSource(specs)
        is_static = True
    else:
        source = open_source(
            specs[0],
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_mirror=args.camera_mirror,
        )
        is_static = isinstance(source, ImageSource)

    # バックエンド (重い import は必要時のみ)
    from poselab.backends import create_backend

    if args.backend == "mmpose":
        from poselab.backends.mmpose_backend import DEFAULT_POSE2D

        model_label = args.pose2d_model or DEFAULT_POSE2D
        backend = create_backend(
            "mmpose",
            pose2d=model_label,
            pose2d_weights=args.pose2d_weights,
            det_model=args.det_model,
            det_weights=args.det_weights,
            device=args.device,
            num_poses=args.num_poses,
        )
    else:
        model_label = args.model
        backend = create_backend(
            "mediapipe",
            model=args.model,
            num_poses=args.num_poses,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            static_image_mode=is_static,
        )
    keypoint_names = list(backend.keypoint_names)

    metadata = {
        "tool": f"poselab {__version__}",
        "backend": backend.name,
        "model": model_label,
        "input": specs,
    }
    if args.smooth > 1:
        metadata["smoothing_window"] = args.smooth

    def build_exporters() -> List[Exporter]:
        exporters: List[Exporter] = []
        if args.csv:
            exporters.append(CsvExporter(args.csv))
        if args.wide_csv:
            from poselab.exporters import WideCsvExporter

            exporters.append(WideCsvExporter(args.wide_csv, keypoint_names))
        if args.json:
            exporters.append(JsonExporter(args.json, keypoint_names, metadata))
        if args.npz:
            exporters.append(NpzExporter(args.npz, keypoint_names, args.num_poses))
        if args.angles_csv:
            from poselab.analysis import AngleCsvExporter

            exporters.append(AngleCsvExporter(args.angles_csv))
        if args.velocity_csv:
            from poselab.analysis import VelocityCsvExporter

            exporters.append(VelocityCsvExporter(args.velocity_csv))
        if args.distance:
            from poselab.features import DistanceCsvExporter, parse_pair

            if not args.distance_csv:
                parser.error("--distance を使う場合は --distance-csv も指定してください")
            try:
                pairs = [parse_pair(spec) for spec in args.distance]
            except ValueError as exc:
                parser.error(str(exc))
            exporters.append(DistanceCsvExporter(args.distance_csv, pairs))
        return exporters

    # 平滑化は全フレームを見てから行うため、有効時は後段でまとめて出力する
    streaming = args.smooth <= 1
    exporters = build_exporters() if streaming else []

    video_writer = None
    if args.save_video:
        video_writer = VideoWriter(args.save_video, fps=source.fps or 30.0)
    if args.save_image and not is_static:
        parser.error("--save-image は静止画入力でのみ使用できます")

    from poselab.progress import ProgressReporter

    reporter = ProgressReporter(
        total=args.max_frames or source.frame_count,
        enabled=not args.quiet,
    )

    def progress(done: int, total: Optional[int]) -> None:
        reporter.update(done)

    tracker = None
    if args.num_poses > 1 and not args.no_track and not is_static:
        from poselab.tracking import PersonTracker

        tracker = PersonTracker()

    trajectory = None
    if args.trail > 0:
        from poselab.visualize import TrajectoryOverlay

        trajectory = TrajectoryOverlay(
            keypoint_names=[
                n.strip() for n in args.trail_keypoints.split(",") if n.strip()
            ],
            length=args.trail,
        )

    try:
        results = run_pipeline(
            source,
            backend,
            exporters=exporters,
            video_writer=video_writer,
            image_output=args.save_image,
            draw=not args.no_draw,
            draw_labels=args.draw_labels,
            trajectory=trajectory,
            tracker=tracker,
            draw_ids=args.num_poses > 1,
            min_visibility=args.min_visibility,
            show=args.show,
            max_frames=args.max_frames,
            progress=progress,
        )
    finally:
        backend.close()
        reporter.finish()

    if not streaming:
        from poselab.exporters import export_results
        from poselab.filters import smooth_results

        results = smooth_results(results, args.smooth)
        export_results(results, build_exporters())

    if args.save_video and args.h264:
        from poselab.pipeline import reencode_h264

        if not reencode_h264(args.save_video):
            print(
                "注意: ffmpeg が見つからないため H.264 再エンコードを"
                "スキップしました (mp4v のままです)",
                file=sys.stderr,
            )

    from poselab.analysis import summarize_results

    summary = summarize_results(results)
    id_warnings = tracker.get_warnings() if tracker is not None else []
    if id_warnings:
        from poselab.tracking import format_warning

        summary["id_warnings"] = id_warnings
        if not args.quiet:
            print(
                "⚠ 人物 ID が入れ替わっている可能性のある区間があります:",
                file=sys.stderr,
            )
            for w in id_warnings:
                print("  - " + format_warning(w), file=sys.stderr)
            print(
                "  座標データを解析する際は該当区間の前後で ID を確認してください。",
                file=sys.stderr,
            )
    if args.summary_json:
        import json

        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {**metadata, **summary}, f, ensure_ascii=False, indent=2
            )

    if not args.quiet:
        print(
            "完了: {total_frames} フレーム処理、"
            "{detected_frames} フレームで人物を検出 (検出率 {rate:.1f}%)".format(
                rate=summary["detection_rate"] * 100, **summary
            ),
            file=sys.stderr,
        )
        for label, path in (
            ("CSV", args.csv), ("ワイドCSV", args.wide_csv),
            ("JSON", args.json), ("NPZ", args.npz),
            ("角度CSV", args.angles_csv), ("速度CSV", args.velocity_csv),
            ("距離CSV", args.distance_csv), ("サマリ", args.summary_json),
            ("動画", args.save_video), ("画像", args.save_image),
        ):
            if path:
                print(f"  {label}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
