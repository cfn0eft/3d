"""poselab が出力した CSV からグラフを生成する (poselab-plot)。

コーディングなしで時系列グラフ・軌跡プロット・ヒートマップを
作成できる。CSV の種類 (座標 / 関節角度 / 速度 / 距離) はヘッダ
から自動判別する。matplotlib が必要 (pip install matplotlib)。

例:
    poselab-plot coords.csv --kind trajectory --keypoints right_wrist
    poselab-plot coords.csv --kind heatmap --keypoints nose
    poselab-plot angles.csv
    poselab-plot velocity.csv --keypoints right_wrist,left_wrist
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _require_matplotlib(show: bool = False):
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise SystemExit(
            "matplotlib が必要です。次のコマンドでインストールしてください:\n"
            "  pip install matplotlib"
        )


def detect_csv_type(fieldnames: List[str]) -> str:
    """ヘッダから CSV の種類を判別する。"""
    cols = set(fieldnames or [])
    if {"keypoint_name", "x_px", "y_px"} <= cols:
        return "coords"
    if {"angle_name", "angle_deg"} <= cols:
        return "angles"
    if {"keypoint_name", "speed_px_per_s"} <= cols:
        return "velocity"
    if {"pair", "distance_px"} <= cols:
        return "distance"
    raise SystemExit(
        "poselab が出力した CSV (座標 / 角度 / 速度 / 距離) を指定してください"
    )


def _load(path: Path, person: int):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        kind = detect_csv_type(reader.fieldnames)
        rows = [r for r in reader if int(r["person"]) == person]
    if not rows:
        raise SystemExit(f"person={person} のデータが見つかりません: {path}")
    return kind, rows


def _series(rows, name_col: str, value_cols: List[str], names: Optional[List[str]]):
    """名前列ごとに (時刻, 値...) の系列へまとめる。"""
    series: Dict[str, List[tuple]] = defaultdict(list)
    for r in rows:
        name = r[name_col]
        if names and name not in names:
            continue
        try:
            values = [float(r[c]) for c in value_cols]
        except ValueError:
            continue  # 空欄 (未検出など) はスキップ
        series[name].append((float(r["timestamp_ms"]) / 1000.0, *values))
    if not series:
        raise SystemExit(f"指定の {name_col} のデータが見つかりません")
    return series


def plot_csv(
    input_path: "str | Path",
    output_path: "str | Path | None" = None,
    kind: Optional[str] = None,
    keypoints: Optional[List[str]] = None,
    person: int = 0,
    frame: Optional[int] = None,
    show: bool = False,
    dpi: int = 120,
) -> Path:
    """CSV からグラフ画像を生成して出力パスを返す。"""
    plt = _require_matplotlib(show)
    input_path = Path(input_path)
    csv_type, rows = _load(input_path, person)

    if output_path is None:
        suffix = kind or csv_type
        output_path = input_path.with_name(f"{input_path.stem}_{suffix}.png")
    output_path = Path(output_path)

    if csv_type == "coords":
        names = keypoints or ["right_wrist", "left_wrist"]
        series = _series(rows, "keypoint_name", ["x_px", "y_px"], names)
        if kind == "trajectory":
            fig, ax = plt.subplots(figsize=(8, 6))
            for name, pts in series.items():
                t = [p[0] for p in pts]
                ax.scatter(
                    [p[1] for p in pts], [p[2] for p in pts],
                    c=t, s=8, cmap="viridis", label=name,
                )
                ax.plot([p[1] for p in pts], [p[2] for p in pts],
                        alpha=0.3, linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")
            ax.set_title(f"trajectory (person {person}, color = time)")
            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")
        elif kind == "heatmap":
            fig, ax = plt.subplots(figsize=(8, 6))
            xs = [p[1] for pts in series.values() for p in pts]
            ys = [p[2] for pts in series.values() for p in pts]
            h = ax.hist2d(xs, ys, bins=60, cmap="hot")
            fig.colorbar(h[3], ax=ax, label="frames")
            ax.invert_yaxis()
            ax.set_xlabel("x [px]")
            ax.set_ylabel("y [px]")
            ax.set_title(f"position heatmap: {', '.join(series)} (person {person})")
        elif kind == "pose3d":
            fig = _plot_pose3d(plt, rows, person, frame)
        else:  # timeseries (デフォルト)
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for name, pts in series.items():
                t = [p[0] for p in pts]
                axes[0].plot(t, [p[1] for p in pts], label=name)
                axes[1].plot(t, [p[2] for p in pts], label=name)
            axes[0].set_ylabel("x [px]")
            axes[1].set_ylabel("y [px]")
            axes[1].invert_yaxis()
            axes[1].set_xlabel("time [s]")
            axes[0].set_title(f"keypoint coordinates (person {person})")
            axes[0].legend()
            for ax in axes:
                ax.grid(alpha=0.3)

    elif csv_type == "angles":
        series = _series(rows, "angle_name", ["angle_deg"], keypoints)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, pts in series.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("angle [deg]")
        ax.set_title(f"joint angles (person {person})")
        ax.grid(alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    elif csv_type == "velocity":
        names = keypoints or ["right_wrist", "left_wrist"]
        series = _series(rows, "keypoint_name", ["speed_px_per_s"], names)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, pts in series.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("speed [px/s]")
        ax.set_title(f"keypoint speed (person {person})")
        ax.grid(alpha=0.3)
        ax.legend()

    else:  # distance
        series = _series(rows, "pair", ["distance_px"], keypoints)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, pts in series.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=name)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance [px]")
        ax.set_title(f"keypoint distance (person {person})")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def _plot_pose3d(plt, rows, person: int, frame: Optional[int]):
    """ワールド座標 (3D) の骨格を 1 フレーム分描画する。"""
    from poselab.skeleton import SKELETON_EDGES, landmark_side

    frames = sorted({int(r["frame"]) for r in rows})
    if frame is None:
        frame = frames[len(frames) // 2]  # 中央フレーム
    if frame not in frames:
        raise SystemExit(f"フレーム {frame} のデータがありません (0–{frames[-1]})")
    points: Dict[str, tuple] = {}
    ts = 0.0
    for r in rows:
        if int(r["frame"]) != frame:
            continue
        ts = float(r["timestamp_ms"]) / 1000.0
        if r.get("world_x"):
            points[r["keypoint_name"]] = (
                float(r["world_x"]), float(r["world_y"]), float(r["world_z"])
            )
    if not points:
        raise SystemExit(
            "ワールド座標 (world_x/y/z) がありません。座標 CSV を指定してください"
        )

    from poselab.skeleton import LANDMARK_NAMES

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    colors = {"left": "tab:orange", "right": "tab:blue", "center": "tab:green"}
    # 表示は (x, z, -y): MediaPipe ワールド座標は y が下向きのため上下を反転
    for a, b in SKELETON_EDGES:
        na, nb = LANDMARK_NAMES[a], LANDMARK_NAMES[b]
        if na not in points or nb not in points:
            continue
        pa, pb = points[na], points[nb]
        side = landmark_side(na)
        if landmark_side(nb) != side:
            side = "center"
        ax.plot(
            [pa[0], pb[0]], [pa[2], pb[2]], [-pa[1], -pb[1]],
            color=colors[side], linewidth=2,
        )
    xs = [p[0] for p in points.values()]
    ys = [p[2] for p in points.values()]
    zs = [-p[1] for p in points.values()]
    ax.scatter(xs, ys, zs, s=12, c="#444444")
    limit = max(max(map(abs, xs)), max(map(abs, ys)), max(map(abs, zs)), 0.1)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_zlabel("height [m]")
    ax.set_title(f"3D pose (person {person}, frame {frame}, t={ts:.2f}s)")
    ax.view_init(elev=12, azim=-70)
    return fig


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="poselab-plot",
        description="poselab の CSV からグラフ画像を生成します "
                    "(種類はヘッダから自動判別)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="poselab が出力した CSV")
    parser.add_argument("--out", "-o", type=Path,
                        help="出力 PNG パス (省略時は入力名から自動)")
    parser.add_argument(
        "--kind", choices=["timeseries", "trajectory", "heatmap", "pose3d"],
        help="座標 CSV のグラフ種類 (pose3d = 3D 骨格ビュー)",
    )
    parser.add_argument(
        "--frame", type=int,
        help="pose3d で表示するフレーム番号 (省略時は中央フレーム)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="保存後にウィンドウ表示 (pose3d はマウスで回転可能)",
    )
    parser.add_argument(
        "--keypoints", "-k",
        help="対象キーポイント / 角度名 / ペア名 (カンマ区切り)",
    )
    parser.add_argument("--person", "-p", type=int, default=0, help="対象人物 ID")
    parser.add_argument("--dpi", type=int, default=120, help="出力解像度")
    args = parser.parse_args(argv)

    names = (
        [n.strip() for n in args.keypoints.split(",") if n.strip()]
        if args.keypoints else None
    )
    out = plot_csv(
        args.input, args.out, kind=args.kind,
        keypoints=names, person=args.person,
        frame=args.frame, show=args.show, dpi=args.dpi,
    )
    print(f"保存しました: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
