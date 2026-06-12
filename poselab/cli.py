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

    g_model = parser.add_argument_group("モデル設定")
    g_model.add_argument(
        "--model", choices=MODEL_VARIANTS, default="full",
        help="モデルサイズ (lite=高速, heavy=高精度)",
    )
    g_model.add_argument("--num-poses", type=int, default=1, help="最大検出人数")
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
    g_out.add_argument("--json", type=Path, help="座標を JSON で出力")
    g_out.add_argument("--npz", type=Path, help="座標を NumPy .npz で出力")
    g_out.add_argument("--save-video", type=Path, help="骨格を描画した動画を保存")
    g_out.add_argument("--save-image", type=Path, help="骨格を描画した画像を保存 (静止画入力時)")

    g_view = parser.add_argument_group("表示・描画")
    g_view.add_argument("--show", action="store_true", help="プレビューウィンドウを表示 (q で終了)")
    g_view.add_argument("--no-draw", action="store_true", help="骨格描画を行わない")
    g_view.add_argument("--draw-labels", action="store_true", help="キーポイント名も描画する")
    g_view.add_argument(
        "--min-visibility", type=float, default=0.3,
        help="描画する visibility のしきい値",
    )

    g_misc = parser.add_argument_group("その他")
    g_misc.add_argument("--max-frames", type=int, help="処理する最大フレーム数")
    g_misc.add_argument("--camera-width", type=int, help="カメラの取得幅")
    g_misc.add_argument("--camera-height", type=int, help="カメラの取得高さ")
    g_misc.add_argument("--quiet", "-q", action="store_true", help="進捗表示を抑制")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_keypoints:
        for i, name in enumerate(LANDMARK_NAMES):
            print(f"{i:2d}  {name}")
        return 0

    if not args.input:
        parser.error("--input を指定してください (--list-keypoints 以外では必須)")

    # 入力ソースの決定
    specs = args.input
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
        )
        is_static = isinstance(source, ImageSource)

    # バックエンド (mediapipe の import は重いので必要時のみ)
    from poselab.backends import create_backend

    backend = create_backend(
        "mediapipe",
        model=args.model,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        static_image_mode=is_static,
    )

    exporters: List[Exporter] = []
    metadata = {
        "tool": f"poselab {__version__}",
        "backend": backend.name,
        "model": args.model,
        "input": specs,
    }
    if args.csv:
        exporters.append(CsvExporter(args.csv))
    if args.json:
        exporters.append(JsonExporter(args.json, LANDMARK_NAMES, metadata))
    if args.npz:
        exporters.append(NpzExporter(args.npz, LANDMARK_NAMES, args.num_poses))

    video_writer = None
    if args.save_video:
        video_writer = VideoWriter(args.save_video, fps=source.fps or 30.0)
    if args.save_image and not is_static:
        parser.error("--save-image は静止画入力でのみ使用できます")

    def progress(done: int, total: Optional[int]) -> None:
        if args.quiet:
            return
        if total:
            print(f"\r処理中 {done}/{total} フレーム", end="", file=sys.stderr)
        elif done % 30 == 0:
            print(f"\r処理中 {done} フレーム", end="", file=sys.stderr)

    try:
        results = run_pipeline(
            source,
            backend,
            exporters=exporters,
            video_writer=video_writer,
            image_output=args.save_image,
            draw=not args.no_draw,
            draw_labels=args.draw_labels,
            min_visibility=args.min_visibility,
            show=args.show,
            max_frames=args.max_frames,
            progress=progress,
        )
    finally:
        backend.close()

    if not args.quiet:
        print(file=sys.stderr)
        n_detected = sum(1 for r in results if r.persons)
        print(
            f"完了: {len(results)} フレーム処理、{n_detected} フレームで人物を検出",
            file=sys.stderr,
        )
        for label, path in (
            ("CSV", args.csv), ("JSON", args.json), ("NPZ", args.npz),
            ("動画", args.save_video), ("画像", args.save_image),
        ):
            if path:
                print(f"  {label}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
