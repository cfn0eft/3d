"""Tkinter ベースの GUI。

起動: poselab-gui  (または python -m poselab.gui)

機能:
- 画像 / 動画 / カメラ入力の骨格推定とライブプレビュー
- キーポイント軌跡 (モーショントレイル) の動画上へのプロット
- 再生位置 (%) ・FPS・検出率などの状況表示
- 関節角度のライブ表示
- 推定座標の記録と CSV / JSON / NPZ / 角度 / 速度エクスポート
  (ワンクリックの全形式エクスポート、エクスポート時平滑化対応)
- 動画ファイルの一括処理 (進捗 % と残り時間表示)
- 設定の自動保存・復元
"""

from __future__ import annotations

import copy
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from poselab import __version__
from poselab.config import load_settings, save_settings
from poselab.exporters import (
    CsvExporter,
    JsonExporter,
    NpzExporter,
    export_results,
)
from poselab.models import MODEL_VARIANTS
from poselab.pipeline import VideoWriter, run_pipeline
from poselab.progress import format_duration
from poselab.skeleton import LANDMARK_NAMES
from poselab.sources import CameraSource, ImageSource, VideoSource
from poselab.types import FrameResult

_IMAGE_FILETYPES = [
    ("画像", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
    ("すべて", "*.*"),
]
_VIDEO_FILETYPES = [
    ("動画", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
    ("すべて", "*.*"),
]

# 軌跡対象のプリセット (表示名 -> キーポイント名リスト)
_TRAIL_CHOICES = {
    "両手首": ["left_wrist", "right_wrist"],
    "両足首": ["left_ankle", "right_ankle"],
    "手首と足首": ["left_wrist", "right_wrist", "left_ankle", "right_ankle"],
    "全キーポイント": ["all"],
}

# PoseLab Studio (Web GUI) の配色トークンに合わせたダークテーマ
_BG = "#0a0a0b"            # 背景
_BG_SOFT = "#101013"       # やや明るい背景 (キャンバス等)
_BG_PANEL = "#141417"      # パネル (ツールバー / ステータス / タブ地)
_PANEL_STRONG = "#1c1c21"  # フィールド / ボタン地
_BG_FIELD = "#1c1c21"      # 入力欄
_LINE = "#2a2a31"          # 罫線
_FG = "#ededef"            # 文字
_FG_DIM = "#a1a1ab"        # 淡い文字
_FG_FAINT = "#6b6b76"      # さらに淡い文字
_ACCENT = "#8b93ff"        # アクセント (紫)
_ACCENT_STRONG = "#6e78f0"  # アクセント (濃)
_ACCENT_INK = "#0a0b16"    # アクセント上の文字
_OK = "#57d399"            # 成功
_WARN = "#f0848c"          # 警告
_AMBER = "#f3c45a"         # 注意
_REC_RED = "#f0848c"       # 記録インジケータ


class PoseLabApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title(f"poselab {__version__} — 骨格推定")
        self.root.geometry("1180x780")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 実行状態
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        # Tk 変数はワーカースレッドから読めないため Event にミラーする
        self._record_enabled = threading.Event()
        self._frame_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self._recorded: List[FrameResult] = []
        self._photo = None  # GC 防止のため参照を保持
        self._last_annotated: Optional[np.ndarray] = None
        self._fps = 0.0
        self._last_frame_time: Optional[float] = None
        self._source_info: dict = {}
        self._last_ts_ms = 0.0
        self._scenes: List[tuple] = []  # (label, start_ms, end_ms)
        self._scene_start: Optional[tuple] = None  # (label, start_ms)

        self._settings = load_settings()
        self._apply_theme()
        self._build_ui()
        self._sync_record_flag()
        self.record_var.trace_add("write", lambda *_: self._sync_record_flag())
        self.root.after(30, self._poll_queue)

    def _setting(self, key: str, default):
        value = self._settings.get(key, default)
        return value if isinstance(value, type(default)) else default

    # ------------------------------------------------------------------ テーマ
    def _pick_font(self, prefs: list) -> str:
        """インストール済みフォントから優先順に最初の 1 つを選ぶ。"""
        import tkinter.font as tkfont

        available = set(tkfont.families(self.root))
        for family in prefs:
            if family in available:
                return family
        return prefs[-1]

    def _apply_theme(self) -> None:
        tk, ttk = self.tk, self.ttk
        import tkinter.font as tkfont

        self.root.configure(bg=_BG)

        # フォント: Studio の Inter + Noto Sans JP に相当するものを優先
        ui_family = self._pick_font([
            "Noto Sans JP", "Yu Gothic UI", "Meiryo UI", "Segoe UI",
            "Hiragino Sans", "Helvetica",
        ])
        mono_family = self._pick_font([
            "Cascadia Mono", "Consolas", "SF Mono", "Menlo",
            "DejaVu Sans Mono", "Courier",
        ])
        self._mono_family = mono_family
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=ui_family, size=10)
            except tk.TclError:
                pass
        try:
            tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=10)
        except tk.TclError:
            pass

        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(".", background=_BG, foreground=_FG,
                        fieldbackground=_BG_FIELD, bordercolor=_LINE,
                        lightcolor=_PANEL_STRONG, darkcolor=_BG,
                        troughcolor=_BG_FIELD, focuscolor=_ACCENT)
        style.configure("TFrame", background=_BG)
        style.configure("Panel.TFrame", background=_BG_PANEL)
        style.configure("TLabel", background=_BG, foreground=_FG)
        style.configure("Dim.TLabel", background=_BG, foreground=_FG_DIM)
        style.configure("Faint.TLabel", background=_BG, foreground=_FG_FAINT)
        style.configure("Heading.TLabel", background=_BG, foreground=_FG,
                        font=(ui_family, 12, "bold"))
        style.configure("Badge.TLabel", background=_PANEL_STRONG,
                        foreground=_ACCENT, padding=(8, 2))
        style.configure("Status.TLabel", background=_BG_PANEL, foreground=_FG_DIM,
                        padding=(8, 4))
        style.configure("Rec.TLabel", background=_BG_PANEL, foreground=_REC_RED,
                        padding=(8, 4))
        # ラベルフレームは同じ背景＋細い罫線でカード風の輪郭にする
        style.configure("TLabelframe", background=_BG, bordercolor=_LINE,
                        relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=_BG, foreground=_FG_DIM)
        style.configure("TButton", background=_PANEL_STRONG, foreground=_FG,
                        padding=(10, 6), borderwidth=1, relief="flat")
        style.map("TButton",
                  background=[("active", "#26262d"), ("pressed", "#26262d")],
                  bordercolor=[("focus", _ACCENT)])
        style.configure("Tool.TButton", padding=(12, 7))
        style.configure("Accent.TButton", background=_ACCENT,
                        foreground=_ACCENT_INK, borderwidth=0)
        style.map("Accent.TButton",
                  background=[("active", _ACCENT_STRONG), ("pressed", _ACCENT_STRONG)],
                  foreground=[("active", _ACCENT_INK)])
        style.configure("TCheckbutton", background=_BG, foreground=_FG)
        style.map("TCheckbutton",
                  background=[("active", _BG)],
                  indicatorcolor=[("selected", _ACCENT)])
        style.configure("TSpinbox", arrowcolor=_FG, background=_BG_FIELD,
                        foreground=_FG, insertcolor=_FG, bordercolor=_LINE)
        style.configure("TCombobox", arrowcolor=_FG, background=_BG_FIELD,
                        foreground=_FG, bordercolor=_LINE)
        style.map("TCombobox", fieldbackground=[("readonly", _BG_FIELD)],
                  foreground=[("readonly", _FG)])
        style.configure("TEntry", fieldbackground=_BG_FIELD, foreground=_FG,
                        insertcolor=_FG, bordercolor=_LINE)
        style.configure("TNotebook", background=_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=_BG_PANEL, foreground=_FG_DIM,
                        padding=(14, 7), borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", _BG)],
                  foreground=[("selected", _ACCENT), ("active", _FG)])
        style.configure("Horizontal.TProgressbar", background=_ACCENT,
                        troughcolor=_BG_FIELD, borderwidth=0)
        style.configure("TSeparator", background=_LINE)
        style.configure("TScrollbar", background=_PANEL_STRONG,
                        troughcolor=_BG, bordercolor=_BG, arrowcolor=_FG_DIM)
        # Combobox のドロップダウンリスト (tk オプション)
        self.root.option_add("*TCombobox*Listbox.background", _BG_FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", _FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", _ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", _ACCENT_INK)

    # ---------------------------------------------------------------- UI 構築
    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk

        self._build_menu()

        # ツールバー
        toolbar = ttk.Frame(self.root, style="Panel.TFrame", padding=(6, 6))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="画像を開く", style="Tool.TButton",
                   command=self.open_image).pack(side="left", padx=(0, 4))
        ttk.Button(toolbar, text="動画を開く", style="Tool.TButton",
                   command=self.open_video).pack(side="left", padx=4)
        ttk.Button(toolbar, text="カメラ開始", style="Tool.TButton",
                   command=self.start_camera).pack(side="left", padx=4)
        ttk.Separator(toolbar, orient="vertical").pack(
            side="left", fill="y", padx=8, pady=2)
        self.pause_button = ttk.Button(
            toolbar, text="一時停止", style="Tool.TButton",
            command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=4)
        ttk.Button(toolbar, text="停止", style="Tool.TButton",
                   command=self.stop).pack(side="left", padx=4)
        ttk.Button(toolbar, text="すべて保存", style="Accent.TButton",
                   command=self.export_all).pack(side="right")

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)

        # 左: プレビューキャンバス + 再生位置
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, bg=_BG_SOFT, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        play_row = ttk.Frame(left)
        play_row.pack(fill="x", pady=(6, 0))
        self.play_progress = ttk.Progressbar(play_row, mode="determinate")
        self.play_progress.pack(side="left", fill="x", expand=True)
        self.play_label = ttk.Label(play_row, text="", width=26, anchor="e",
                                    style="Dim.TLabel")
        self.play_label.pack(side="right", padx=(8, 0))

        # 右: タブパネル
        notebook = ttk.Notebook(main, width=270)
        notebook.pack(side="right", fill="y", padx=(8, 0))
        tab_main = ttk.Frame(notebook, padding=8)
        tab_rec = ttk.Frame(notebook, padding=8)
        tab_batch = ttk.Frame(notebook, padding=8)
        tab_angle = ttk.Frame(notebook, padding=8)
        tab_scene = ttk.Frame(notebook, padding=8)
        notebook.add(tab_main, text=" 設定 ")
        notebook.add(tab_rec, text=" 記録・保存 ")
        notebook.add(tab_batch, text=" 一括処理 ")
        notebook.add(tab_angle, text=" 関節角度 ")
        notebook.add(tab_scene, text=" シーンタグ ")

        self._build_tab_main(tab_main)
        self._build_tab_record(tab_rec)
        self._build_tab_batch(tab_batch)
        self._build_tab_angle(tab_angle)
        self._build_tab_scene(tab_scene)

        # ステータスバー
        status_row = ttk.Frame(self.root, style="Panel.TFrame")
        status_row.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="入力を選択してください")
        ttk.Label(status_row, textvariable=self.status, anchor="w",
                  style="Status.TLabel").pack(side="left", fill="x", expand=True)
        self.rec_indicator = ttk.Label(status_row, text="", style="Rec.TLabel")
        self.rec_indicator.pack(side="right")

        # キーボードショートカット
        self.root.bind("<Control-i>", lambda e: self.open_image())
        self.root.bind("<Control-o>", lambda e: self.open_video())
        self.root.bind("<Control-s>", lambda e: self.export_all())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.bind("t", self._on_scene_key)
        self.root.bind("T", self._on_scene_key)

    def _build_tab_main(self, tab) -> None:
        tk, ttk = self.tk, self.ttk

        cam = ttk.LabelFrame(tab, text="カメラ", padding=8)
        cam.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(cam)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="カメラ番号").pack(side="left")
        self.camera_index = tk.IntVar(value=self._setting("camera_index", 0))
        ttk.Spinbox(row, from_=0, to=16, width=5,
                    textvariable=self.camera_index).pack(side="right")
        self.mirror_var = tk.BooleanVar(value=self._setting("mirror", True))
        ttk.Checkbutton(cam, text="ミラー表示 (左右反転)",
                        variable=self.mirror_var).pack(anchor="w", pady=2)
        ttk.Button(cam, text="使えるカメラを検索",
                   command=self.scan_cameras_async).pack(fill="x", pady=2)

        cfg = ttk.LabelFrame(tab, text="モデル (次回開始時に適用)", padding=8)
        cfg.pack(fill="x", pady=8)
        row = ttk.Frame(cfg)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="モデル").pack(side="left")
        self.model_var = tk.StringVar(value=self._setting("model", "lite"))
        ttk.Combobox(row, textvariable=self.model_var,
                     values=list(MODEL_VARIANTS),
                     state="readonly", width=9).pack(side="right")
        row = ttk.Frame(cfg)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="最大人数").pack(side="left")
        self.num_poses = tk.IntVar(value=self._setting("num_poses", 1))
        ttk.Spinbox(row, from_=1, to=10, width=5,
                    textvariable=self.num_poses).pack(side="right")
        row = ttk.Frame(cfg)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="検出しきい値").pack(side="left")
        self.det_conf = tk.DoubleVar(value=self._setting("det_conf", 0.5))
        ttk.Spinbox(row, from_=0.1, to=0.9, increment=0.1, width=5,
                    textvariable=self.det_conf).pack(side="right")

        view = ttk.LabelFrame(tab, text="オーバーレイ表示", padding=8)
        view.pack(fill="x", pady=8)
        self.draw_var = tk.BooleanVar(value=self._setting("draw", True))
        ttk.Checkbutton(view, text="骨格を描画",
                        variable=self.draw_var).pack(anchor="w", pady=1)
        self.labels_var = tk.BooleanVar(value=self._setting("draw_labels", False))
        ttk.Checkbutton(view, text="キーポイント名を表示",
                        variable=self.labels_var).pack(anchor="w", pady=1)
        self.trail_var = tk.BooleanVar(value=self._setting("trail", False))
        ttk.Checkbutton(view, text="軌跡をプロット (モーショントレイル)",
                        variable=self.trail_var).pack(anchor="w", pady=1)
        row = ttk.Frame(view)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="軌跡の対象").pack(side="left")
        self.trail_target = tk.StringVar(
            value=self._setting("trail_target", "両手首"))
        ttk.Combobox(row, textvariable=self.trail_target,
                     values=list(_TRAIL_CHOICES),
                     state="readonly", width=12).pack(side="right")
        row = ttk.Frame(view)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="軌跡の長さ (フレーム)").pack(side="left")
        self.trail_length = tk.IntVar(value=self._setting("trail_length", 30))
        ttk.Spinbox(row, from_=5, to=300, increment=5, width=5,
                    textvariable=self.trail_length).pack(side="right")
        ttk.Button(view, text="現在フレームを画像保存...",
                   command=self.save_current_frame).pack(fill="x", pady=(8, 2))

        ttk.Label(
            tab,
            text="ショートカット:\n"
                 "  Ctrl+I 画像 / Ctrl+O 動画\n"
                 "  Space 一時停止 / Esc 停止\n"
                 "  Ctrl+S すべて保存",
            style="Dim.TLabel", justify="left",
        ).pack(anchor="w", pady=(12, 0))

    def _build_tab_record(self, tab) -> None:
        tk, ttk = self.tk, self.ttk

        rec = ttk.LabelFrame(tab, text="記録", padding=8)
        rec.pack(fill="x", pady=(0, 8))
        self.record_var = tk.BooleanVar(value=self._setting("record", True))
        ttk.Checkbutton(rec, text="座標を記録する",
                        variable=self.record_var).pack(anchor="w")
        self.rec_label = ttk.Label(rec, text="記録: 0 フレーム")
        self.rec_label.pack(anchor="w", pady=2)
        ttk.Button(rec, text="記録をクリア",
                   command=self.clear_recording).pack(fill="x", pady=(6, 0))

        # 処理オプション (Studio と同等)
        proc = ttk.LabelFrame(tab, text="処理オプション", padding=8)
        proc.pack(fill="x", pady=(0, 8))
        r1 = ttk.Frame(proc)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="平滑化手法").pack(side="left")
        self.smooth_method = tk.StringVar(
            value=self._setting("smooth_method", "移動平均"))
        ttk.Combobox(r1, width=14, state="readonly",
                     values=["移動平均", "メディアン", "Butterworth"],
                     textvariable=self.smooth_method).pack(side="right")
        r2 = ttk.Frame(proc)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="窓 (フレーム, 0=なし)").pack(side="left")
        self.smooth_window = tk.IntVar(value=self._setting("smooth_window", 0))
        ttk.Spinbox(r2, from_=0, to=61, width=5,
                    textvariable=self.smooth_window).pack(side="right")
        r3 = ttk.Frame(proc)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="カットオフ Hz (Butterworth)").pack(side="left")
        self.smooth_cutoff = tk.DoubleVar(
            value=self._setting("smooth_cutoff", 0.0))
        ttk.Spinbox(r3, from_=0.0, to=30.0, increment=0.5, width=5,
                    textvariable=self.smooth_cutoff).pack(side="right")
        self.smooth_weighted = tk.BooleanVar(
            value=self._setting("smooth_weighted", False))
        ttk.Checkbutton(proc, text="visibility 加重 (移動平均)",
                        variable=self.smooth_weighted).pack(anchor="w", pady=2)
        r4 = ttk.Frame(proc)
        r4.pack(fill="x", pady=2)
        ttk.Label(r4, text="マスキング閾値 (visibility)").pack(side="left")
        self.mask_visibility = tk.DoubleVar(
            value=self._setting("mask_visibility", 0.0))
        ttk.Spinbox(r4, from_=0.0, to=1.0, increment=0.05, width=5,
                    textvariable=self.mask_visibility).pack(side="right")

        exp = ttk.LabelFrame(tab, text="エクスポート", padding=8)
        exp.pack(fill="x", pady=8)
        ttk.Button(exp, text="すべての形式へ保存...  (Ctrl+S)",
                   style="Accent.TButton",
                   command=self.export_all).pack(fill="x", pady=(0, 6))
        for label, fmt in (
            ("座標 CSV (ロング)...", "csv"),
            ("座標 CSV (ワイド)...", "wide"),
            ("JSON...", "json"),
            ("NumPy NPZ...", "npz"),
            ("関節角度 CSV...", "angles"),
            ("速度・加速度 CSV...", "velocity"),
            ("左右対称性 CSV...", "symmetry"),
        ):
            ttk.Button(exp, text=label,
                       command=lambda f=fmt: self.export_recorded(f)).pack(
                fill="x", pady=2)

    def _build_tab_batch(self, tab) -> None:
        ttk = self.ttk
        ttk.Label(
            tab,
            text="動画ファイル全体を処理して\n座標 (CSV/JSON)・関節角度 CSV・\n"
                 "注釈付き動画 (MP4) を生成します。",
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        ttk.Button(tab, text="動画を選んで一括処理...", style="Accent.TButton",
                   command=self.batch_process).pack(fill="x", pady=4)
        self.progress = ttk.Progressbar(tab, mode="determinate")
        self.progress.pack(fill="x", pady=6)
        self.batch_label = ttk.Label(tab, text="", style="Dim.TLabel")
        self.batch_label.pack(anchor="w")

    def _build_tab_angle(self, tab) -> None:
        ttk = self.ttk
        ttk.Label(tab, text="検出中の関節角度 (度)", style="Heading.TLabel").pack(
            anchor="w", pady=(0, 6))
        self.angle_label = ttk.Label(
            tab, text="(未検出)", font=(self._mono_family, 10), justify="left"
        )
        self.angle_label.pack(anchor="w")
        ttk.Label(
            tab,
            text="? = 信頼度が低い推定値\nワールド座標 (3D) ベースで計算",
            style="Dim.TLabel", justify="left",
        ).pack(anchor="w", pady=(10, 0))

        gait = ttk.LabelFrame(tab, text="歩行リズム (記録データから推定)", padding=8)
        gait.pack(fill="x", pady=(14, 0))
        ttk.Button(gait, text="記録データから推定",
                   command=self._estimate_gait).pack(fill="x")
        self.gait_label = ttk.Label(
            gait, text="(未推定)", style="Dim.TLabel", justify="left"
        )
        self.gait_label.pack(anchor="w", pady=(6, 0))

    def _estimate_gait(self) -> None:
        from tkinter import messagebox

        if not self._recorded:
            messagebox.showinfo("poselab", "記録されたフレームがありません")
            return
        from poselab.kinematics import gait_summary

        summary = gait_summary(self._recorded)
        if not summary:
            self.gait_label.config(text="推定できませんでした (周期性が弱い)")
            return
        lines = []
        for side, info in summary.items():
            jp = "左足首" if side == "left_ankle" else "右足首"
            lines.append(
                f"{jp}: {info['cadence_per_min']} サイクル/分 "
                f"(1 周期 {info['cycle_time_s']}s)"
            )
        self.gait_label.config(text="\n".join(lines))

    def _build_tab_scene(self, tab) -> None:
        """行動観察用のシーンタグ付け (時間区間にラベルを付ける)。"""
        tk, ttk = self.tk, self.ttk

        ttk.Label(
            tab,
            text="再生中の時刻でラベル付き区間を記録します\n(行動コーディング用)。T キーでも開始/終了できます。",
            justify="left", style="Dim.TLabel",
        ).pack(anchor="w", pady=(0, 6))
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="ラベル").pack(side="left")
        self.scene_label_var = tk.StringVar(value="behavior")
        ttk.Entry(row, textvariable=self.scene_label_var, width=18).pack(
            side="right")
        self.scene_button = ttk.Button(
            tab, text="シーン開始  (T)", style="Accent.TButton",
            command=self.toggle_scene)
        self.scene_button.pack(fill="x", pady=4)
        self.scene_list = tk.Listbox(
            tab, height=10, bg=_BG_FIELD, fg=_FG,
            selectbackground=_ACCENT, highlightthickness=0,
        )
        self.scene_list.pack(fill="both", expand=True, pady=4)
        ttk.Button(tab, text="選択した区間を削除",
                   command=self.delete_scene).pack(fill="x", pady=2)
        ttk.Button(tab, text="シーン CSV へ保存...",
                   command=self.export_scenes).pack(fill="x", pady=2)

    def _build_menu(self) -> None:
        tk = self.tk
        menubar = tk.Menu(self.root, bg=_BG_PANEL, fg=_FG,
                          activebackground=_ACCENT, activeforeground="#fff")

        def themed_menu() -> "tk.Menu":
            return tk.Menu(menubar, tearoff=0, bg=_BG_PANEL, fg=_FG,
                           activebackground=_ACCENT, activeforeground="#fff")

        m_file = themed_menu()
        m_file.add_command(label="画像を開く...", accelerator="Ctrl+I",
                           command=self.open_image)
        m_file.add_command(label="動画を開く...", accelerator="Ctrl+O",
                           command=self.open_video)
        m_file.add_command(label="カメラ開始", command=self.start_camera)
        m_file.add_separator()
        m_file.add_command(label="終了", command=self._on_close)
        menubar.add_cascade(label="ファイル", menu=m_file)

        m_export = themed_menu()
        m_export.add_command(label="すべての形式へ保存...", accelerator="Ctrl+S",
                             command=self.export_all)
        m_export.add_separator()
        for label, fmt in (
            ("座標 CSV (ロング)...", "csv"), ("座標 CSV (ワイド)...", "wide"),
            ("JSON...", "json"), ("NumPy NPZ...", "npz"),
            ("関節角度 CSV...", "angles"), ("速度 CSV...", "velocity"),
        ):
            m_export.add_command(
                label=label, command=lambda f=fmt: self.export_recorded(f)
            )
        m_export.add_separator()
        m_export.add_command(label="記録をクリア", command=self.clear_recording)
        menubar.add_cascade(label="エクスポート", menu=m_export)

        m_help = themed_menu()
        m_help.add_command(label="バージョン情報", command=self._show_about)
        menubar.add_cascade(label="ヘルプ", menu=m_help)

        self.root.config(menu=menubar)

    def _show_about(self) -> None:
        from tkinter import messagebox

        messagebox.showinfo(
            "poselab について",
            f"poselab {__version__}\n\n"
            "研究用ヒト骨格推定ツールキット (MIT License)\n"
            "推定エンジン: MediaPipe Pose Landmarker (Apache-2.0)\n\n"
            "33 キーポイントの 2D/3D 座標・関節角度・速度を\n"
            "CSV / JSON / NPZ にエクスポートできます。",
        )

    def _on_space(self, event) -> None:
        # 入力欄へのフォーカス時は通常のキー入力を優先する
        if isinstance(event.widget, (self.tk.Entry, self.tk.Text)):
            return
        self.toggle_pause()

    def _on_scene_key(self, event) -> None:
        if isinstance(event.widget, (self.tk.Entry, self.tk.Text)):
            return
        self.toggle_scene()

    # ------------------------------------------------------------ シーンタグ
    def toggle_scene(self) -> None:
        """現在時刻でシーン区間の開始 / 終了を切り替える。"""
        if self._scene_start is None:
            label = self.scene_label_var.get().strip() or "scene"
            self._scene_start = (label, self._last_ts_ms)
            self.scene_button.config(text=f"シーン終了  (T) — {label} 記録中")
            self.status.set(
                f"シーン '{label}' 開始 (t={self._last_ts_ms / 1000.0:.2f}s)"
            )
        else:
            label, start_ms = self._scene_start
            end_ms = max(self._last_ts_ms, start_ms)
            self._scenes.append((label, start_ms, end_ms))
            self._scene_start = None
            self.scene_button.config(text="シーン開始  (T)")
            self.scene_list.insert(
                "end",
                f"{label}  {start_ms / 1000.0:.2f}s – {end_ms / 1000.0:.2f}s"
                f"  ({(end_ms - start_ms) / 1000.0:.2f}s)",
            )
            self.status.set(f"シーン '{label}' を記録しました")

    def delete_scene(self) -> None:
        selection = self.scene_list.curselection()
        if not selection:
            return
        index = selection[0]
        self.scene_list.delete(index)
        del self._scenes[index]

    def export_scenes(self) -> None:
        from tkinter import filedialog, messagebox

        if not self._scenes:
            messagebox.showinfo("poselab", "記録されたシーンがありません")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        import csv as csv_mod

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.writer(f)
            writer.writerow(
                ["label", "start_ms", "end_ms", "start_s", "end_s", "duration_s"]
            )
            for label, start_ms, end_ms in self._scenes:
                writer.writerow(
                    [
                        label,
                        f"{start_ms:.3f}",
                        f"{end_ms:.3f}",
                        f"{start_ms / 1000.0:.3f}",
                        f"{end_ms / 1000.0:.3f}",
                        f"{(end_ms - start_ms) / 1000.0:.3f}",
                    ]
                )
        self.status.set(f"シーン CSV を保存しました: {path}")

    # ------------------------------------------------------------ 入力ハンドラ
    def open_image(self) -> None:
        from tkinter import filedialog

        paths = filedialog.askopenfilenames(filetypes=_IMAGE_FILETYPES)
        if paths:
            self._start_worker(lambda: ImageSource(list(paths)), static=True)

    def open_video(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=_VIDEO_FILETYPES)
        if path:
            self._start_worker(lambda: VideoSource(path), static=False, pace=True)

    def start_camera(self) -> None:
        index = self.camera_index.get()
        mirror = self.mirror_var.get()
        self._start_worker(
            lambda: CameraSource(index, mirror=mirror), static=False
        )

    def scan_cameras_async(self) -> None:
        """利用可能なカメラ番号をバックグラウンドで検索する。"""
        self.status.set("カメラを検索中... (数秒かかります)")

        def work() -> None:
            from poselab.sources import scan_cameras

            cameras = scan_cameras()
            if cameras:
                desc = ", ".join(
                    f"{c['index']} ({c['width']}x{c['height']})" for c in cameras
                )
                msg = f"使えるカメラ: {desc} — カメラ番号に設定して開始してください"
            else:
                msg = (
                    "カメラが見つかりません。接続と OS のカメラアクセス許可を"
                    "確認してください"
                )
            self._frame_queue.put(("status_msg", msg, None))

        threading.Thread(target=work, daemon=True).start()

    def stop(self) -> None:
        self._stop_flag.set()
        self._pause_flag.clear()
        self.pause_button.config(text="一時停止")

    def toggle_pause(self) -> None:
        if self._pause_flag.is_set():
            self._pause_flag.clear()
            self.pause_button.config(text="一時停止")
            self.status.set("再開しました")
        else:
            self._pause_flag.set()
            self.pause_button.config(text="再開")
            self.status.set("一時停止中")

    def _sync_record_flag(self) -> None:
        if self.record_var.get():
            self._record_enabled.set()
        else:
            self._record_enabled.clear()

    def _make_trajectory(self):
        """設定スナップショットから軌跡オーバーレイを生成 (メインスレッド)。"""
        if not self.trail_var.get():
            return None
        from poselab.visualize import TrajectoryOverlay

        names = _TRAIL_CHOICES.get(self.trail_target.get(), ["left_wrist", "right_wrist"])
        return TrajectoryOverlay(
            keypoint_names=names, length=self.trail_length.get()
        )

    # ------------------------------------------------------------ ワーカー制御
    def _start_worker(self, source_factory, static: bool, pace: bool = False) -> None:
        self.stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        self._stop_flag.clear()
        self._fps = 0.0
        self._last_frame_time = None
        self._source_info = {}
        self.play_progress["value"] = 0
        self.play_label.config(text="")
        self.status.set("モデルを準備中... (初回はダウンロードが入ります)")

        model = self.model_var.get()
        num_poses = self.num_poses.get()
        det_conf = self.det_conf.get()
        draw = self.draw_var.get()
        draw_labels = self.labels_var.get()
        trajectory = self._make_trajectory()
        tracker = None
        if num_poses > 1 and not static:
            from poselab.tracking import PersonTracker

            tracker = PersonTracker()

        def work() -> None:
            try:
                from poselab.backends import create_backend

                try:
                    source = source_factory()
                except (IOError, OSError) as exc:
                    self._frame_queue.put(("user_error", str(exc), None))
                    return
                self._frame_queue.put(
                    (
                        "info",
                        {
                            "total": source.frame_count,
                            "fps": source.fps,
                            "live": source.is_live,
                        },
                        None,
                    )
                )
                backend = create_backend(
                    "mediapipe",
                    model=model,
                    num_poses=num_poses,
                    min_detection_confidence=det_conf,
                    static_image_mode=static,
                )
                fps_interval = 1.0 / source.fps if (pace and source.fps) else 0.0
                last_t = [0.0]
                counts = {"frames": 0, "detected": 0}

                def on_frame(result: FrameResult, annotated: np.ndarray) -> bool:
                    counts["frames"] += 1
                    if result.persons:
                        counts["detected"] += 1
                    if self._record_enabled.is_set():
                        self._recorded.append(result)
                    try:
                        self._frame_queue.put_nowait(("frame", annotated, result))
                    except queue.Full:
                        pass
                    while self._pause_flag.is_set() and not self._stop_flag.is_set():
                        time.sleep(0.05)
                    if fps_interval:
                        now = time.monotonic()
                        wait = fps_interval - (now - last_t[0])
                        if wait > 0:
                            time.sleep(wait)
                        last_t[0] = time.monotonic()
                    return not self._stop_flag.is_set()

                try:
                    run_pipeline(
                        source,
                        backend,
                        draw=draw,
                        draw_labels=draw_labels,
                        trajectory=trajectory,
                        tracker=tracker,
                        draw_ids=num_poses > 1,
                        on_frame=on_frame,
                    )
                finally:
                    backend.close()
                counts["id_warnings"] = (
                    tracker.get_warnings() if tracker is not None else []
                )
                self._frame_queue.put(("done", counts, None))
            except Exception:
                self._frame_queue.put(("error", traceback.format_exc(), None))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ UI 更新ループ
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload, result = self._frame_queue.get_nowait()
                if kind == "info":
                    self._source_info = payload
                elif kind == "frame":
                    self._on_ui_frame(payload, result)
                elif kind == "done":
                    self.rec_indicator.config(text="")
                    if payload and payload.get("frames"):
                        rate = 100.0 * payload["detected"] / payload["frames"]
                        self.status.set(
                            f"処理終了 — {payload['frames']} フレーム中 "
                            f"{payload['detected']} フレームで検出 "
                            f"(検出率 {rate:.1f}%)"
                        )
                    else:
                        self.status.set("処理終了")
                    if payload and payload.get("id_warnings"):
                        self._show_id_warnings(payload["id_warnings"])
                elif kind == "progress":
                    self.progress["value"] = payload["pct"]
                    eta = payload.get("eta_s")
                    self.batch_label.config(
                        text=(
                            f"{payload['pct']:.1f}%  "
                            f"({payload['done']}/{payload['total']})"
                            + (f"  残り {format_duration(eta)}" if eta else "")
                        )
                    )
                elif kind == "batch_done":
                    self.progress["value"] = 100
                    self.batch_label.config(text="100% 完了")
                    self.status.set(
                        f"一括処理が完了しました: {payload['base']}.csv / .json / "
                        f"_angles.csv / .mp4"
                    )
                    if payload.get("id_warnings"):
                        self._show_id_warnings(payload["id_warnings"])
                elif kind == "user_error":
                    from tkinter import messagebox

                    self.rec_indicator.config(text="")
                    first_line = payload.splitlines()[0]
                    self.status.set(first_line)
                    messagebox.showerror("poselab", payload)
                elif kind == "status_msg":
                    self.status.set(payload)
                elif kind == "error":
                    self.rec_indicator.config(text="")
                    self.status.set("エラーが発生しました (詳細はコンソール)")
                    print(payload)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)

    def _on_ui_frame(self, annotated: np.ndarray, result: FrameResult) -> None:
        self._last_annotated = annotated
        self._last_ts_ms = result.timestamp_ms
        self._show_frame(annotated)

        now = time.monotonic()
        if self._last_frame_time is not None:
            dt = now - self._last_frame_time
            if dt > 0:
                inst = 1.0 / dt
                self._fps = (
                    inst if self._fps == 0.0 else 0.85 * self._fps + 0.15 * inst
                )
        self._last_frame_time = now

        # 再生位置 (%) と時刻
        total = self._source_info.get("total")
        fps = self._source_info.get("fps")
        if total and fps:
            pct = 100.0 * (result.frame_index + 1) / total
            self.play_progress["value"] = pct
            self.play_label.config(
                text=(
                    f"{pct:5.1f}%  "
                    f"{format_duration(result.timestamp_ms / 1000.0)}"
                    f" / {format_duration(total / fps)}"
                )
            )
        elif self._source_info.get("live"):
            self.play_label.config(
                text=f"LIVE  経過 {format_duration(result.timestamp_ms / 1000.0)}"
            )

        self.status.set(
            f"フレーム {result.frame_index}  "
            f"検出 {len(result.persons)} 人  "
            f"{self._fps:.1f} fps"
        )
        self.rec_label.config(text=f"記録: {len(self._recorded)} フレーム")
        self.rec_indicator.config(
            text=f"● 記録中 {len(self._recorded)}"
            if self._record_enabled.is_set() else ""
        )
        self._update_angle_panel(result)

    def _show_id_warnings(self, warnings: list) -> None:
        """人物 ID 入れ替わりリスクの警告ダイアログを表示する。"""
        from tkinter import messagebox

        from poselab.tracking import format_warning

        shown = warnings[:8]
        lines = [format_warning(w) for w in shown]
        if len(warnings) > len(shown):
            lines.append(f"... 他 {len(warnings) - len(shown)} 件")
        messagebox.showwarning(
            "人物 ID 入れ替わりの可能性",
            "以下の区間で人物 ID が入れ替わっている可能性があります。\n"
            "座標データを解析する際は前後の ID を確認してください。\n\n"
            + "\n".join(lines),
        )

    def _update_angle_panel(self, result: FrameResult) -> None:
        if not result.persons:
            self.angle_label.config(text="(未検出)")
            return
        from poselab.analysis import compute_person_angles

        person = result.persons[0]
        angles = compute_person_angles(person)
        lines = [f"[P{person.person_index}]"] if len(result.persons) > 1 else []
        for name, (deg, vis, _) in angles.items():
            value = f"{deg:6.1f}" if not np.isnan(deg) else "   ---"
            mark = "" if vis >= 0.5 else " ?"
            lines.append(f"{name:<15s}{value}{mark}")
        self.angle_label.config(text="\n".join(lines))

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        from PIL import Image, ImageTk

        cw = max(self.canvas.winfo_width(), 16)
        ch = max(self.canvas.winfo_height(), 16)
        h, w = frame_bgr.shape[:2]
        scale = min(cw / w, ch / h)
        new_size = (max(int(w * scale), 1), max(int(h * scale), 1))
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        image = image.resize(new_size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def save_current_frame(self) -> None:
        from tkinter import filedialog, messagebox

        if self._last_annotated is None:
            messagebox.showinfo("poselab", "表示中のフレームがありません")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        from poselab.imgio import imwrite

        imwrite(path, self._last_annotated)
        self.status.set(f"保存しました: {path}")

    # ------------------------------------------------------------ エクスポート
    def clear_recording(self) -> None:
        self._recorded = []
        self.rec_label.config(text="記録: 0 フレーム")
        self.rec_indicator.config(text="")

    _SMOOTH_METHODS = {
        "移動平均": "moving", "メディアン": "median", "Butterworth": "butter",
    }

    def _recorded_fps(self) -> float:
        """記録データのタイムスタンプから実効 fps を推定する。"""
        rec = self._recorded
        if len(rec) >= 2:
            dur = (rec[-1].timestamp_ms - rec[0].timestamp_ms) / 1000.0
            if dur > 0:
                return (len(rec) - 1) / dur
        return float(self._source_info.get("fps") or 30.0)

    def _recorded_for_export(self) -> List[FrameResult]:
        """記録データのコピーを返す。平滑化設定があれば適用する。"""
        method = self._SMOOTH_METHODS.get(self.smooth_method.get(), "moving")
        window = self.smooth_window.get()
        cutoff = float(self.smooth_cutoff.get() or 0.0)
        weighted = self.smooth_weighted.get()
        active = (method == "butter" and cutoff > 0) or (
            method in ("moving", "median") and window > 1
        )
        if active:
            from poselab.filters import smooth_results

            return smooth_results(
                copy.deepcopy(self._recorded), window,
                method=method, weighted=weighted,
                cutoff=(cutoff or None), fps=self._recorded_fps(),
            )
        return list(self._recorded)

    def _mask(self) -> float:
        return float(self.mask_visibility.get() or 0.0)

    def _make_exporter(self, fmt: str, path: str):
        mask = self._mask()
        if fmt == "csv":
            return CsvExporter(path, mask)
        if fmt == "wide":
            from poselab.exporters import WideCsvExporter

            return WideCsvExporter(path, LANDMARK_NAMES, mask)
        if fmt == "json":
            return JsonExporter(
                path, LANDMARK_NAMES, {"tool": f"poselab {__version__}"}
            )
        if fmt == "npz":
            max_persons = max(
                (p.person_index + 1 for r in self._recorded for p in r.persons),
                default=1,
            )
            return NpzExporter(path, LANDMARK_NAMES, max_persons, None, mask)
        if fmt == "angles":
            from poselab.analysis import AngleCsvExporter

            return AngleCsvExporter(path)
        if fmt == "velocity":
            from poselab.analysis import VelocityCsvExporter

            return VelocityCsvExporter(path)
        if fmt == "symmetry":
            from poselab.analysis import SymmetryCsvExporter

            return SymmetryCsvExporter(path)
        raise ValueError(fmt)

    def export_recorded(self, fmt: str) -> None:
        from tkinter import filedialog, messagebox

        if not self._recorded:
            messagebox.showinfo("poselab", "記録されたフレームがありません")
            return
        ext = {
            "csv": ".csv", "wide": ".csv", "json": ".json", "npz": ".npz",
            "angles": ".csv", "velocity": ".csv", "symmetry": ".csv",
        }[fmt]
        path = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=[(fmt.upper(), f"*{ext}")]
        )
        if not path:
            return
        export_results(
            self._recorded_for_export(), [self._make_exporter(fmt, path)]
        )
        self.status.set(f"保存しました: {path}")

    def export_all(self) -> None:
        """記録データを全形式 (CSV/JSON/NPZ/角度/速度/対称性) で一括保存する。"""
        from tkinter import filedialog, messagebox

        if not self._recorded:
            messagebox.showinfo("poselab", "記録されたフレームがありません")
            return
        base = filedialog.asksaveasfilename(
            title="ベース名を指定 (各形式のファイルを生成)",
            initialfile="pose_export",
        )
        if not base:
            return
        base_path = Path(base).with_suffix("")
        targets = [
            ("csv", str(base_path) + "_long.csv"),
            ("wide", str(base_path) + "_wide.csv"),
            ("json", str(base_path) + ".json"),
            ("npz", str(base_path) + ".npz"),
            ("angles", str(base_path) + "_angles.csv"),
            ("velocity", str(base_path) + "_velocity.csv"),
            ("symmetry", str(base_path) + "_symmetry.csv"),
        ]
        exporters = [self._make_exporter(fmt, path) for fmt, path in targets]
        export_results(self._recorded_for_export(), exporters)
        self.status.set(f"{len(targets)} 形式で保存しました: {base_path}_*.*")

    # ------------------------------------------------------------ 一括処理
    def batch_process(self) -> None:
        from tkinter import filedialog

        in_path = filedialog.askopenfilename(
            title="処理する動画を選択", filetypes=_VIDEO_FILETYPES
        )
        if not in_path:
            return
        base = filedialog.asksaveasfilename(
            title="出力ファイル名 (拡張子なし)",
            initialfile=Path(in_path).stem + "_pose",
        )
        if not base:
            return
        base_path = Path(base).with_suffix("")
        self.stop()
        self.status.set("一括処理中...")
        model = self.model_var.get()
        num_poses = self.num_poses.get()
        det_conf = self.det_conf.get()
        draw_labels = self.labels_var.get()
        trajectory = self._make_trajectory()
        tracker = None
        if num_poses > 1:
            from poselab.tracking import PersonTracker

            tracker = PersonTracker()

        def work() -> None:
            try:
                from poselab.analysis import (
                    AngleCsvExporter,
                    SymmetryCsvExporter,
                    VelocityCsvExporter,
                )
                from poselab.backends import create_backend

                source = VideoSource(in_path)
                backend = create_backend(
                    "mediapipe", model=model, num_poses=num_poses,
                    min_detection_confidence=det_conf,
                )
                mask = self._mask()
                exporters = [
                    CsvExporter(str(base_path) + ".csv", mask),
                    JsonExporter(
                        str(base_path) + ".json", LANDMARK_NAMES,
                        {"tool": f"poselab {__version__}", "input": str(in_path)},
                    ),
                    AngleCsvExporter(str(base_path) + "_angles.csv"),
                    VelocityCsvExporter(str(base_path) + "_velocity.csv"),
                    SymmetryCsvExporter(str(base_path) + "_symmetry.csv"),
                ]
                writer = VideoWriter(
                    str(base_path) + ".mp4", fps=source.fps or 30.0
                )
                total = source.frame_count
                start = time.monotonic()

                def progress(done: int, _total) -> None:
                    if not total:
                        return
                    elapsed = time.monotonic() - start
                    fps = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / fps if fps > 0 else None
                    self._frame_queue.put(
                        (
                            "progress",
                            {
                                "pct": 100.0 * done / total,
                                "done": done,
                                "total": total,
                                "eta_s": eta,
                            },
                            None,
                        )
                    )

                try:
                    run_pipeline(
                        source, backend, exporters=exporters,
                        video_writer=writer, draw_labels=draw_labels,
                        trajectory=trajectory, tracker=tracker,
                        draw_ids=num_poses > 1, progress=progress,
                    )
                finally:
                    backend.close()
                self._frame_queue.put(
                    (
                        "batch_done",
                        {
                            "base": str(base_path),
                            "id_warnings": (
                                tracker.get_warnings()
                                if tracker is not None else []
                            ),
                        },
                        None,
                    )
                )
            except Exception:
                self._frame_queue.put(("error", traceback.format_exc(), None))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ 終了処理
    def _save_settings(self) -> None:
        save_settings(
            {
                "model": self.model_var.get(),
                "num_poses": self.num_poses.get(),
                "det_conf": self.det_conf.get(),
                "camera_index": self.camera_index.get(),
                "mirror": self.mirror_var.get(),
                "draw": self.draw_var.get(),
                "draw_labels": self.labels_var.get(),
                "trail": self.trail_var.get(),
                "trail_target": self.trail_target.get(),
                "trail_length": self.trail_length.get(),
                "record": self.record_var.get(),
                "smooth_window": self.smooth_window.get(),
                "smooth_method": self.smooth_method.get(),
                "smooth_cutoff": self.smooth_cutoff.get(),
                "smooth_weighted": self.smooth_weighted.get(),
                "mask_visibility": self.mask_visibility.get(),
            }
        )

    def _on_close(self) -> None:
        self.stop()
        self._save_settings()
        self.root.after(100, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "tkinter が見つかりません。OS のパッケージとしてインストールしてください\n"
            "  (例: sudo apt install python3-tk)"
        )
        return 1
    PoseLabApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
