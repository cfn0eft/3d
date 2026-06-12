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


def _require_matplotlib():
    try:
        import matplotlib

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
    dpi: int = 120,
) -> Path:
    """CSV からグラフ画像を生成して出力パスを返す。"""
    plt = _require_matplotlib()
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
    plt.close(fig)
    return output_path


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
        "--kind", choices=["timeseries", "trajectory", "heatmap"],
        help="座標 CSV のグラフ種類",
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
        keypoints=names, person=args.person, dpi=args.dpi,
    )
    print(f"保存しました: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
