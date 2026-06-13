"""PoseLab Studio の Windows アイコン (poselab.ico) を生成する。

GUI のブランドマーク (poselab/studio/gui/index.html の favicon SVG) と
同じ図柄 — 角丸のダークネイビー地に、シアン→パープルのグラデーションで
描いた人型スティックフィギュア — を Pillow でラスタライズし、
マルチ解像度の .ico (256/128/64/48/32/16) として書き出す。

再生成:
    pip install Pillow
    python packaging/make_icon.py

出力: packaging/installer/poselab.ico
ショートカット (install_local.ps1) と Inno Setup (poselab_studio.iss) が参照する。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "installer" / "poselab.ico"

# index.html の favicon と同じ配色 / 図形 (viewBox 0 0 64 を 16 倍で描画)
SCALE = 16
N = 64 * SCALE
BG = (11, 16, 26, 255)          # #0b101a
C0 = (0x22, 0xd3, 0xee)         # cyan  #22d3ee
C1 = (0xa7, 0x8b, 0xfa)         # purple #a78bfa
STROKE = 4 * SCALE              # SVG stroke-width 4
RADIUS = 14 * SCALE             # 角丸 rx=14

# スティックフィギュア (SVG path と同じ座標、64 単位 → SCALE 倍)
HEAD = (32, 15, 6)              # cx, cy, r
SEGMENTS = [
    ((32, 22), (32, 38)),       # 背骨
    ((32, 26), (21, 33)),       # 左腕
    ((32, 26), (43, 33)),       # 右腕
    ((32, 38), (24, 51)),       # 左脚
    ((32, 38), (40, 51)),       # 右脚
]


def _p(x: int, y: int) -> tuple[int, int]:
    return (x * SCALE, y * SCALE)


def _lerp(a: int, b: int, t: float) -> int:
    return round(a + (b - a) * t)


def build_base() -> Image.Image:
    # 角丸の背景
    base = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(base).rounded_rectangle(
        [0, 0, N - 1, N - 1], radius=RADIUS, fill=BG
    )

    # フィギュアのマスク (白=描画) — 線は丸キャップを端点の円で補う
    mask = Image.new("L", (N, N), 0)
    md = ImageDraw.Draw(mask)
    cx, cy, r = HEAD
    hx, hy, hr = cx * SCALE, cy * SCALE, r * SCALE
    md.ellipse(
        [hx - hr, hy - hr, hx + hr, hy + hr], outline=255, width=STROKE
    )
    cap = STROKE // 2
    for (x0, y0), (x1, y1) in SEGMENTS:
        a, b = _p(x0, y0), _p(x1, y1)
        md.line([a, b], fill=255, width=STROKE)
        for px, py in (a, b):  # 丸キャップ / ジョイント
            md.ellipse([px - cap, py - cap, px + cap, py + cap], fill=255)

    # 対角グラデーション (左上 C0 → 右下 C1)。lut[x+y] で 1 回だけ補間
    span = 2 * (N - 1)
    lut = [
        (_lerp(C0[0], C1[0], i / span),
         _lerp(C0[1], C1[1], i / span),
         _lerp(C0[2], C1[2], i / span), 255)
        for i in range(span + 1)
    ]
    grad = Image.new("RGBA", (N, N))
    grad.putdata([lut[x + y] for y in range(N) for x in range(N)])

    base.paste(grad, (0, 0), mask)
    return base


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = build_base().resize((256, 256), Image.LANCZOS)
    base.save(
        OUT,
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
