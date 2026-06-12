"""Tkinter ベースの GUI。

起動: poselab-gui  (または python -m poselab.gui)

機能:
- 画像 / 動画 / カメラ入力の骨格推定とライブプレビュー
- 推定座標の記録と CSV / JSON / NPZ エクスポート
- 動画ファイルの一括処理 (座標 + 注釈付き動画の書き出し)
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from poselab import __version__
from poselab.exporters import (
    CsvExporter,
    JsonExporter,
    NpzExporter,
    export_results,
)
from poselab.models import MODEL_VARIANTS
from poselab.pipeline import VideoWriter, run_pipeline
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


class PoseLabApp:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title(f"poselab {__version__} - 骨格推定")
        self.root.geometry("1100x700")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # 実行状態
        self._worker: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        # Tk 変数はワーカースレッドから読めないため Event にミラーする
        self._record_enabled = threading.Event()
        self._frame_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self._recorded: List[FrameResult] = []
        self._photo = None  # GC 防止のため参照を保持

        self._build_ui()
        self._sync_record_flag()
        self.record_var.trace_add("write", lambda *_: self._sync_record_flag())
        self.root.after(30, self._poll_queue)

    # ---------------------------------------------------------------- UI 構築
    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk

        main = ttk.Frame(self.root, padding=4)
        main.pack(fill="both", expand=True)

        # 左: プレビューキャンバス
        self.canvas = tk.Canvas(main, bg="#202020", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        # 右: 操作パネル
        panel = ttk.Frame(main, padding=(8, 0))
        panel.pack(side="right", fill="y")

        src = ttk.LabelFrame(panel, text="入力", padding=6)
        src.pack(fill="x", pady=(0, 6))
        ttk.Button(src, text="画像を開く...", command=self.open_image).pack(fill="x", pady=2)
        ttk.Button(src, text="動画を開く...", command=self.open_video).pack(fill="x", pady=2)
        cam_row = ttk.Frame(src)
        cam_row.pack(fill="x", pady=2)
        ttk.Label(cam_row, text="カメラ番号").pack(side="left")
        self.camera_index = tk.IntVar(value=0)
        ttk.Spinbox(cam_row, from_=0, to=16, width=4,
                    textvariable=self.camera_index).pack(side="left", padx=4)
        ttk.Button(src, text="カメラ開始", command=self.start_camera).pack(fill="x", pady=2)
        ttk.Button(src, text="停止", command=self.stop).pack(fill="x", pady=2)

        cfg = ttk.LabelFrame(panel, text="モデル設定 (次回開始時に適用)", padding=6)
        cfg.pack(fill="x", pady=6)
        row = ttk.Frame(cfg); row.pack(fill="x", pady=2)
        ttk.Label(row, text="モデル").pack(side="left")
        self.model_var = tk.StringVar(value="lite")
        ttk.Combobox(row, textvariable=self.model_var, values=list(MODEL_VARIANTS),
                     state="readonly", width=8).pack(side="right")
        row = ttk.Frame(cfg); row.pack(fill="x", pady=2)
        ttk.Label(row, text="最大人数").pack(side="left")
        self.num_poses = tk.IntVar(value=1)
        ttk.Spinbox(row, from_=1, to=10, width=4,
                    textvariable=self.num_poses).pack(side="right")
        row = ttk.Frame(cfg); row.pack(fill="x", pady=2)
        ttk.Label(row, text="検出しきい値").pack(side="left")
        self.det_conf = tk.DoubleVar(value=0.5)
        ttk.Spinbox(row, from_=0.1, to=0.9, increment=0.1, width=4,
                    textvariable=self.det_conf).pack(side="right")

        view = ttk.LabelFrame(panel, text="表示", padding=6)
        view.pack(fill="x", pady=6)
        self.draw_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view, text="骨格を描画", variable=self.draw_var).pack(anchor="w")
        self.labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view, text="キーポイント名を表示",
                        variable=self.labels_var).pack(anchor="w")

        rec = ttk.LabelFrame(panel, text="記録・エクスポート", padding=6)
        rec.pack(fill="x", pady=6)
        self.record_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rec, text="座標を記録する", variable=self.record_var).pack(anchor="w")
        self.rec_label = ttk.Label(rec, text="記録: 0 フレーム")
        self.rec_label.pack(anchor="w", pady=2)
        ttk.Button(rec, text="記録をクリア", command=self.clear_recording).pack(fill="x", pady=2)
        ttk.Button(rec, text="CSV へ保存...",
                   command=lambda: self.export_recorded("csv")).pack(fill="x", pady=2)
        ttk.Button(rec, text="JSON へ保存...",
                   command=lambda: self.export_recorded("json")).pack(fill="x", pady=2)
        ttk.Button(rec, text="NPZ へ保存...",
                   command=lambda: self.export_recorded("npz")).pack(fill="x", pady=2)

        batch = ttk.LabelFrame(panel, text="一括処理", padding=6)
        batch.pack(fill="x", pady=6)
        ttk.Button(batch, text="動画を一括処理...",
                   command=self.batch_process).pack(fill="x", pady=2)
        self.progress = ttk.Progressbar(batch, mode="determinate")
        self.progress.pack(fill="x", pady=2)

        self.status = tk.StringVar(value="入力を選択してください")
        ttk.Label(self.root, textvariable=self.status, anchor="w",
                  padding=(6, 2)).pack(fill="x", side="bottom")

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
        self._start_worker(lambda: CameraSource(index), static=False)

    def stop(self) -> None:
        self._stop_flag.set()

    def _sync_record_flag(self) -> None:
        if self.record_var.get():
            self._record_enabled.set()
        else:
            self._record_enabled.clear()

    # ------------------------------------------------------------ ワーカー制御
    def _start_worker(self, source_factory, static: bool, pace: bool = False) -> None:
        self.stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=5.0)
        self._stop_flag.clear()
        self.status.set("モデルを準備中... (初回はダウンロードが入ります)")

        model = self.model_var.get()
        num_poses = self.num_poses.get()
        det_conf = self.det_conf.get()
        draw = self.draw_var.get()
        draw_labels = self.labels_var.get()

        def work() -> None:
            try:
                from poselab.backends import create_backend

                source = source_factory()
                backend = create_backend(
                    "mediapipe",
                    model=model,
                    num_poses=num_poses,
                    min_detection_confidence=det_conf,
                    static_image_mode=static,
                )
                fps_interval = 1.0 / source.fps if (pace and source.fps) else 0.0
                last_t = [0.0]

                def on_frame(result: FrameResult, annotated: np.ndarray) -> bool:
                    if self._record_enabled.is_set():
                        self._recorded.append(result)
                    try:
                        self._frame_queue.put_nowait(("frame", annotated, result))
                    except queue.Full:
                        pass
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
                        on_frame=on_frame,
                    )
                finally:
                    backend.close()
                self._frame_queue.put(("done", None, None))
            except Exception:
                self._frame_queue.put(("error", traceback.format_exc(), None))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------ UI 更新ループ
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload, result = self._frame_queue.get_nowait()
                if kind == "frame":
                    self._show_frame(payload)
                    self.status.set(
                        f"フレーム {result.frame_index}  "
                        f"検出 {len(result.persons)} 人  "
                        f"t={result.timestamp_ms / 1000.0:.2f}s"
                    )
                    self.rec_label.config(text=f"記録: {len(self._recorded)} フレーム")
                elif kind == "done":
                    self.status.set("処理終了")
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "batch_done":
                    self.progress["value"] = 100
                    self.status.set(
                        f"一括処理が完了しました: {payload}.csv / .json / .mp4"
                    )
                elif kind == "error":
                    self.status.set("エラーが発生しました (詳細はコンソール)")
                    print(payload)
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)

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

    # ------------------------------------------------------------ エクスポート
    def clear_recording(self) -> None:
        self._recorded = []
        self.rec_label.config(text="記録: 0 フレーム")

    def export_recorded(self, fmt: str) -> None:
        from tkinter import filedialog, messagebox

        if not self._recorded:
            messagebox.showinfo("poselab", "記録されたフレームがありません")
            return
        ext = {"csv": ".csv", "json": ".json", "npz": ".npz"}[fmt]
        path = filedialog.asksaveasfilename(
            defaultextension=ext, filetypes=[(fmt.upper(), f"*{ext}")]
        )
        if not path:
            return
        if fmt == "csv":
            exporter = CsvExporter(path)
        elif fmt == "json":
            exporter = JsonExporter(
                path, LANDMARK_NAMES, {"tool": f"poselab {__version__}"}
            )
        else:
            max_persons = max(
                (len(r.persons) for r in self._recorded), default=1
            ) or 1
            exporter = NpzExporter(path, LANDMARK_NAMES, max_persons)
        export_results(self._recorded, [exporter])
        self.status.set(f"保存しました: {path}")

    # ------------------------------------------------------------ 一括処理
    def batch_process(self) -> None:
        from tkinter import filedialog, messagebox

        in_path = filedialog.askopenfilename(
            title="処理する動画を選択", filetypes=_VIDEO_FILETYPES
        )
        if not in_path:
            return
        base = filedialog.asksaveasfilename(
            title="出力ファイル名 (拡張子なし。.csv / .json / .mp4 を生成)",
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

        def work() -> None:
            try:
                from poselab.backends import create_backend

                source = VideoSource(in_path)
                backend = create_backend(
                    "mediapipe", model=model, num_poses=num_poses,
                    min_detection_confidence=det_conf,
                )
                exporters = [
                    CsvExporter(base_path.with_suffix(".csv")),
                    JsonExporter(
                        base_path.with_suffix(".json"), LANDMARK_NAMES,
                        {"tool": f"poselab {__version__}", "input": str(in_path)},
                    ),
                ]
                writer = VideoWriter(
                    base_path.with_suffix(".mp4"), fps=source.fps or 30.0
                )
                total = source.frame_count

                def progress(done: int, _total) -> None:
                    if total:
                        self._frame_queue.put(
                            ("progress", 100.0 * done / total, None)
                        )

                try:
                    run_pipeline(
                        source, backend, exporters=exporters,
                        video_writer=writer, progress=progress,
                    )
                finally:
                    backend.close()
                self._frame_queue.put(
                    ("batch_done", str(base_path), None)
                )
            except Exception:
                self._frame_queue.put(("error", traceback.format_exc(), None))

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------ 終了処理
    def _on_close(self) -> None:
        self.stop()
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
