"""matplotlib 3.10+ 互換シム (mmpose 3D 可視化向け)。

mmpose の 3D 可視化 (``mmpose/visualization/local_visualizer_3d.py``) は
matplotlib 3.8 で非推奨化・3.10 で削除された
``FigureCanvasAgg.tostring_rgb()`` をそのまま呼ぶ。新しい matplotlib では::

    AttributeError: 'FigureCanvasTkAgg' object has no attribute 'tostring_rgb'

となり、``--save-video`` (2D+3D 可視化動画) 付きの 3D 推定が中断する
(``FigureCanvasTkAgg`` は ``FigureCanvasAgg`` を継承するため、対話
バックエンドが選ばれていても同じ症状になる)。

本モジュールは削除された ``tostring_rgb`` を ``buffer_rgba`` から再現して
補い、併せてオフスクリーン描画に適した非対話 Agg バックエンドへ切り替える。
mmpose を公開 API 経由で使う poselab 側の独自実装。
"""

from __future__ import annotations


def _needs_agg_switch(backend: str) -> bool:
    """対話バックエンドから非対話 Agg へ切り替えるべきか判定する。

    非対話の純 Agg (``"agg"``) のときだけ切替不要。``TkAgg`` / ``QtAgg``
    など対話バックエンド名も部分文字列に ``agg`` を含むため、単純な
    ``"agg" in backend`` ではなく完全一致で判定する。
    """
    return (backend or "").strip().lower() != "agg"


def _rgb_bytes_from_canvas(canvas) -> bytes:
    """Agg 系キャンバスの RGBA バッファから RGB バイト列を作る。

    matplotlib 3.10 で削除された ``tostring_rgb`` と同じ並び (行優先・
    アルファ無しの RGB) を返す。呼び出し側 (mmpose) は事前に
    ``canvas.draw()`` 済みであることを前提とする (旧 ``tostring_rgb``
    も同様だった)。
    """
    import numpy as np

    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[..., :3].tobytes()


def ensure_matplotlib_canvas_compat() -> None:
    """mmpose の 3D 可視化が動くよう matplotlib を整える (冪等)。

    - 対話バックエンド (TkAgg 等) が選ばれていれば Agg に切り替える
      (オフスクリーン描画。Tk への依存やウィンドウ表示を避ける)
    - 削除済みの ``FigureCanvasAgg.tostring_rgb`` を補う

    matplotlib 未導入や予期せぬ失敗でも例外は送出しない。可視化を伴わない
    推定はこの調整なしでも成立するため、すべてベストエフォートで行う。
    """
    try:
        import matplotlib
    except Exception:  # pragma: no cover - matplotlib 未導入環境
        return

    # オフスクリーン描画用に非対話 Agg を使う (図を作る前に切り替える)。
    try:
        if _needs_agg_switch(matplotlib.get_backend()):
            matplotlib.use("Agg", force=True)
    except Exception:  # pragma: no cover - 切替不能でも推定は続行
        pass

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except Exception:  # pragma: no cover - 想定外の matplotlib 構成
        return

    # 旧 API が残っている (matplotlib < 3.10) なら何もしない。
    if hasattr(FigureCanvasAgg, "tostring_rgb"):
        return

    def tostring_rgb(self):  # noqa: D401 - 旧 API 互換シム
        return _rgb_bytes_from_canvas(self)

    FigureCanvasAgg.tostring_rgb = tostring_rgb
