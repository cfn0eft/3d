"""Pose3DStudio 後継のローカル Web GUI サーバー (独自実装)。

`poselab/studio/gui/` の Web GUI (旧 Pose3DStudio.exe と同じ画面) を
poselab 自身のパイプラインへ接続する。推定は poselab CLI を
サブプロセスとして実行し (バックエンドは選択可:
``--pose3d`` = mmpose 2D + 3D リフティング、または
``--backend mediapipe`` = GPU 不要の MediaPipe 2D/3D)、
進捗・ログ・出力を Server-Sent Events でブラウザへ流す。

起動: poselab-studio serve  (または配布版 exe を起動)

GUI が要求する API (旧 exe と同じ契約):
- 静的:  GET /  /app.css  /app.js (エンジン + GUI を連結して動的生成)
- 状態:  GET /status  /events (SSE)  /gpu
- 実行:  POST /run  /enqueue  /queue-move  /clear-queue  /cancel
- 補助:  POST /preflight  /pick-video  /pick-videos  /pick-folder
         GET /file?path=  /summary?path=  /open?path=

本ファイルのコードはすべて独自実装で、旧 exe のコードは含まない。
"""

from __future__ import annotations

import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import parse_qs, urlparse

VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg",
)

DEFAULT_PORT = 7860


def force_utf8_stdio() -> None:
    """stdout / stderr を UTF-8 にする (日本語ログが cp1252 で落ちるのを防ぐ)。

    Windows のコンソール既定 (cp1252 等) では日本語を含む print が
    UnicodeEncodeError になる。frozen exe / cmd.exe いずれでも安全なよう、
    エンコード不能文字は置換する。
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

# GUI の MODEL_PROFILES / DETECTOR_PROFILES は旧 exe 内のコンフィグパスを
# 送ってくる。パス末尾のコンフィグ名は MMPose / MMDetection model zoo の
# 名前なので、そのままモデル名として解決できる (例外はここで読み替える)。
_CONFIG_ALIASES = {
    # mmpose 同梱のデモ用コンフィグ名 → mmdet model zoo の正式名
    "faster_rcnn_r50_fpn_coco": "faster-rcnn_r50_fpn_1x_coco",
}


def config_to_model_name(value: "str | None") -> Optional[str]:
    """旧 GUI が送る config パスを model zoo のモデル名へ読み替える。"""
    if not value:
        return None
    stem = Path(str(value)).stem
    return _CONFIG_ALIASES.get(stem, stem)


def worker_command_prefix() -> List[str]:
    """ジョブ実行用 poselab CLI の起動コマンド前置部。

    PyInstaller で固めた exe では sys.executable が exe 自身を指すため、
    エントリスクリプト (packaging/studio_entry.py) が解釈する
    ``--cli`` ディスパッチを使う。通常の Python 環境では -m 起動。
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "--cli"]
    return [sys.executable, "-m", "poselab.cli"]


def normalize_backend(value: "str | None") -> str:
    """バックエンド名を正規化する (既定は mmpose、未知の値も mmpose)。"""
    backend = str(value or "mmpose").strip().lower()
    return backend if backend in ("mmpose", "mediapipe") else "mmpose"


def normalize_job(payload: dict) -> dict:
    """GUI の payload を 1 ジョブの正規形に変換する。"""
    input_path = str(payload.get("input") or "").strip()
    output_root = str(payload.get("output_root") or "").strip()
    if input_path and not output_root:
        p = Path(input_path)
        output_root = str(p.parent / p.stem)
    try:
        num_poses = max(1, int(payload.get("num_poses") or 1))
    except (TypeError, ValueError):
        num_poses = 1
    return {
        "input": input_path,
        "output_root": output_root,
        "backend": normalize_backend(payload.get("backend")),
        "csv_format": payload.get("csv_format") or "both",
        "reencode": bool(payload.get("reencode", True)),
        "progress": bool(payload.get("progress", True)),
        "center_root": bool(payload.get("center_root", False)),
        "normalize_scale": bool(payload.get("normalize_scale", False)),
        # MediaPipe バックエンド用 (モデルサイズ / 最大検出人数)
        "model": str(payload.get("model") or "full"),
        "num_poses": num_poses,
        # MMPose バックエンド用 (人物検出 + 2D + 3D リフティングのモデル)
        "pose2d_model": config_to_model_name(payload.get("pose2d_config")),
        "lift_model": config_to_model_name(payload.get("pose3d_config")),
        "det_model": config_to_model_name(payload.get("det_config")),
        "device": payload.get("device") or None,
    }


def job_output_paths(job: dict) -> Dict[str, Path]:
    """ジョブの出力ファイル一式 (output_root フォルダ内) を決める。"""
    out_dir = Path(job["output_root"])
    stem = Path(job["input"]).stem or "output"
    csv_format = job.get("csv_format", "both")
    paths: Dict[str, Path] = {
        "out_dir": out_dir,
        "json": out_dir / f"results_{stem}.json",
        "summary": out_dir / f"{stem}_summary.json",
        "video": out_dir / f"{stem}_2d3d.mp4",
    }
    if csv_format in ("both", "wide"):
        paths["wide_csv"] = out_dir / f"{stem}_wide.csv"
    if csv_format in ("both", "long"):
        paths["long_csv"] = out_dir / f"{stem}_long.csv"
    return paths


def _append_output_flags(cmd: List[str], job: dict, paths: Dict[str, Path]) -> None:
    """両バックエンド共通の出力フラグ (H.264 / CSV / quiet) を付ける。"""
    if job.get("reencode"):
        cmd.append("--h264")
    if "wide_csv" in paths:
        cmd.extend(["--wide-csv", str(paths["wide_csv"])])
    if "long_csv" in paths:
        cmd.extend(["--csv", str(paths["long_csv"])])
    if not job.get("progress", True):
        cmd.append("--quiet")


def build_command(job: dict) -> List[str]:
    """ジョブを実行する poselab CLI コマンドを組み立てる。"""
    paths = job_output_paths(job)
    if job.get("backend") == "mediapipe":
        # MediaPipe: GPU 不要の 2D/3D 推定 (--pose3d は使わない)。
        # 出力 JSON は poselab 形式 (world_keypoints 入り、ビューアが再生可)。
        cmd = [
            *worker_command_prefix(),
            "--input", job["input"],
            "--backend", "mediapipe",
            "--model", str(job.get("model") or "full"),
            "--json", str(paths["json"]),
            "--summary-json", str(paths["summary"]),
            "--save-video", str(paths["video"]),
        ]
        num_poses = int(job.get("num_poses") or 1)
        if num_poses > 1:
            cmd.extend(["--num-poses", str(num_poses)])
        _append_output_flags(cmd, job, paths)
        return cmd

    # MMPose: RTMDet + RTMPose 2D → VideoPose3D 3D リフティング。
    cmd = [
        *worker_command_prefix(),
        "--input", job["input"],
        "--pose3d",
        "--json", str(paths["json"]),
        "--summary-json", str(paths["summary"]),
        "--save-video", str(paths["video"]),
    ]
    _append_output_flags(cmd, job, paths)
    for key, flag in (
        ("pose2d_model", "--pose2d-model"),
        ("lift_model", "--lift-model"),
        ("det_model", "--det-model"),
        ("device", "--device"),
    ):
        if job.get(key):
            cmd.extend([flag, str(job[key])])
    return cmd


def prepare_models_command(job: dict) -> List[str]:
    """推定モデルを事前ダウンロードするコマンド。

    MMPose は検出 + 2D + 3D の重みを、MediaPipe は Pose Landmarker の
    .task モデルを取得する。
    """
    if job.get("backend") == "mediapipe":
        return [
            *worker_command_prefix(),
            "--backend", "mediapipe",
            "--model", str(job.get("model") or "full"),
            "--prepare-models",
        ]
    cmd = [*worker_command_prefix(), "--pose3d", "--prepare-models"]
    for key, flag in (
        ("pose2d_model", "--pose2d-model"),
        ("lift_model", "--lift-model"),
        ("det_model", "--det-model"),
        ("device", "--device"),
    ):
        if job.get(key):
            cmd.extend([flag, str(job[key])])
    return cmd


def model_download_entries(job: dict, status: str) -> List[dict]:
    """「モデルダウンロード」パネル用のモデル一覧。

    MMPose は 3 モデル (検出 / 2D / 3D)、MediaPipe は 1 モデル
    (Pose Landmarker) を返す。
    """
    if job.get("backend") == "mediapipe":
        size = job.get("model") or "full"
        return [
            {"name": f"MediaPipe Pose ({size})", "status": status},
        ]
    det = job.get("det_model") or "RTMDet (既定)"
    pose2d = job.get("pose2d_model") or "RTMPose (既定)"
    lift = job.get("lift_model") or "VideoPose3D (既定)"
    return [
        {"name": f"人物検出: {det}", "status": status},
        {"name": f"2D 推定: {pose2d}", "status": status},
        {"name": f"3D リフティング: {lift}", "status": status},
    ]


# 既定モデルを一度取得したことを示すマーカー (再起動後も ready 表示にする)
MODEL_READY_MARKER = Path.home() / ".cache" / "poselab" / ".studio_models_ready"


# ProgressReporter の出力 ("[####----]  60.0%  18/30 ...") から % を拾う
_PROGRESS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def gpu_info() -> dict:
    """GPU の利用可否を調べる (torch 優先、なければ nvidia-smi)。"""
    try:
        import torch

        available = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if available else None
        return {"available": available, "name": name, "via": "torch"}
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            proc = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            names = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
            if proc.returncode == 0 and names:
                return {"available": True, "name": names[0], "via": "nvidia-smi"}
        except Exception:
            pass
    return {"available": False, "name": None, "via": None}


def video_info(path: Path) -> dict:
    """動画のフレーム数 / fps / 長さを取得する (失敗時は空 dict)。"""
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        try:
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if frames <= 0:
            return {}
        info = {"frames": frames}
        if fps > 0:
            info["fps"] = fps
            info["duration"] = frames / fps
        return info
    except Exception:
        return {}


def preflight(payload: dict) -> dict:
    """実行前チェック。GUI の「Checks」パネルに出す警告と動画情報。"""
    job = normalize_job(payload)
    warnings: List[str] = []
    info: dict = {}

    if not job["input"]:
        warnings.append("入力動画が設定されていません。")
    else:
        path = Path(job["input"])
        if not path.is_file():
            warnings.append(f"入力が見つかりません: {path}")
        elif path.suffix.lower() not in VIDEO_EXTENSIONS:
            warnings.append(f"動画ファイルではない可能性があります: {path.name}")
        else:
            info = video_info(path)

    if job["output_root"]:
        parent = Path(job["output_root"]).parent
        if not parent.exists():
            warnings.append(f"出力先フォルダが存在しません: {parent}")

    if job["backend"] == "mmpose":
        try:
            import importlib.util

            if importlib.util.find_spec("mmpose") is None:
                warnings.append(
                    "mmpose がインストールされていません: 3D パイプラインを"
                    "実行できません (README の MMPose バックエンド参照)。"
                )
        except Exception:
            pass

    if job["reencode"] and shutil.which("ffmpeg") is None:
        warnings.append("ffmpeg が見つかりません: H.264 再エンコードはスキップされます。")

    return {"ok": True, "warnings": warnings, "info": info}


def _summary_from_counts(
    frames: int, counts: List[int], scores: List[float]
) -> dict:
    avg_instances = round(sum(counts) / frames, 2) if frames else 0
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0
    return {
        "frames": frames,
        "avg_instances": avg_instances,
        "avg_score": avg_score,
    }


def summarize_results_json(path: Path) -> dict:
    """results JSON から GUI 用の簡易サマリを作る。

    MMPose 形式 (instance_info / keypoint_scores) と poselab 形式
    (frames / persons / keypoints[].visibility) の両方に対応する。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    counts: List[int] = []
    scores: List[float] = []

    if "instance_info" in data:  # MMPose 形式
        seq = data.get("instance_info") or []
        for entry in seq:
            instances = entry.get("instances") or []
            counts.append(len(instances))
            for inst in instances:
                for s in inst.get("keypoint_scores") or []:
                    try:
                        scores.append(float(s))
                    except (TypeError, ValueError):
                        continue
        return _summary_from_counts(len(seq), counts, scores)

    # poselab 形式 ({metadata, frames}): MediaPipe バックエンドの出力。
    frames_seq = data.get("frames") or []
    for entry in frames_seq:
        persons = entry.get("persons") or []
        counts.append(len(persons))
        for person in persons:
            for kp in person.get("keypoints") or []:
                v = kp.get("visibility")
                if v is None:
                    continue
                try:
                    scores.append(float(v))
                except (TypeError, ValueError):
                    continue
    return _summary_from_counts(len(frames_seq), counts, scores)


def open_in_explorer(path: Path) -> bool:
    """OS のファイラ / 既定アプリでパスを開く。"""
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # noqa: S606 - ローカル GUI の明示操作
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True
    except Exception:
        return False


class FileDialogs:
    """tkinter のネイティブファイルダイアログ (1 度に 1 つだけ)。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _ask(self, kind: str) -> dict:
        if not self._lock.acquire(blocking=False):
            return {"ok": False, "error": "別のダイアログが開いています。"}
        try:
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                return {
                    "ok": False,
                    "error": "tkinter が利用できないためダイアログを開けません。"
                             "パスを直接入力してください。",
                }
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                filetypes = [
                    ("動画ファイル", " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)),
                    ("すべてのファイル", "*.*"),
                ]
                if kind == "video":
                    path = filedialog.askopenfilename(filetypes=filetypes)
                    return {"ok": True, "path": path or ""}
                if kind == "videos":
                    paths = filedialog.askopenfilenames(filetypes=filetypes)
                    return {"ok": True, "paths": list(paths)}
                folder = filedialog.askdirectory()
                if not folder:
                    return {"ok": True, "paths": []}
                found = sorted(
                    str(p) for p in Path(folder).iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                )
                return {"ok": True, "paths": found}
            finally:
                root.destroy()
        except Exception as exc:  # tk はヘッドレス環境で TclError を出す
            return {"ok": False, "error": f"ダイアログを開けません: {exc}"}
        finally:
            self._lock.release()

    def pick_video(self) -> dict:
        return self._ask("video")

    def pick_videos(self) -> dict:
        return self._ask("videos")

    def pick_folder(self) -> dict:
        return self._ask("folder")


class JobManager:
    """ジョブキューの実行と状態・イベント配信。

    ジョブは poselab CLI のサブプロセスとして 1 件ずつ実行する
    (キャンセル時はプロセスを終了、GPU メモリもジョブごとに解放される)。
    command_builder はテストで差し替え可能。
    """

    def __init__(
        self,
        command_builder: Callable[[dict], List[str]] = build_command,
        download_command_builder: Callable[[dict], List[str]] = prepare_models_command,
    ) -> None:
        self._command_builder = command_builder
        self._download_command_builder = download_command_builder
        self._lock = threading.Lock()
        self._queue: List[dict] = []
        self._completed: List[dict] = []
        self._current: Optional[dict] = None
        self._current_index = 0
        self._current_total = 0
        self._progress = 0
        self._running = False
        self._last_return_code: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None
        self._cancel = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._subscribers: List["queue.Queue[str]"] = []
        # モデルダウンロード状態 (前回取得済みなら ready で開始)
        ready = MODEL_READY_MARKER.exists()
        self._downloads: List[dict] = model_download_entries(
            {}, "ready" if ready else "pending"
        )
        self._downloading = False
        self._download_thread: Optional[threading.Thread] = None

    # ---- イベント配信 (SSE) ----

    def subscribe(self) -> "queue.Queue[str]":
        q: "queue.Queue[str]" = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event: dict) -> None:
        data = json.dumps(event, ensure_ascii=False)
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass

    def emit_log(self, message: str) -> None:
        self._emit({"type": "log", "message": message})

    def _emit_status(self) -> None:
        self._emit({"type": "status", **self.status()})

    # ---- 状態 ----

    def status(self) -> dict:
        with self._lock:
            current = None
            if self._current is not None:
                current = {
                    "input": self._current.get("input"),
                    "index": self._current_index,
                    "total": self._current_total,
                }
            return {
                "running": self._running,
                "progress": self._progress,
                "current_job": current,
                "queue": [
                    {"input": job.get("input")} for job in self._queue
                ],
                "completed": list(self._completed),
                "downloads": list(self._downloads),
                "return_code": self._last_return_code,
            }

    # ---- キュー操作 ----

    def enqueue(self, payload: dict) -> dict:
        inputs = payload.get("inputs")
        if isinstance(inputs, (list, tuple)) and inputs:
            jobs = []
            for path in inputs:
                item = dict(payload)
                item["input"] = path
                item["output_root"] = ""
                jobs.append(normalize_job(item))
        else:
            jobs = [normalize_job(payload)]
        for job in jobs:
            if not job["input"]:
                return {"ok": False, "error": "入力動画を指定してください。"}
            if not Path(job["input"]).is_file():
                return {"ok": False, "error": f"入力が見つかりません: {job['input']}"}
        with self._lock:
            self._queue.extend(jobs)
        self._emit_status()
        return {"ok": True, "queued": len(jobs)}

    def move(self, index: int, offset: int) -> dict:
        with self._lock:
            target = index + offset
            if not (0 <= index < len(self._queue)):
                return {"ok": False, "error": "キューのインデックスが不正です。"}
            target = max(0, min(len(self._queue) - 1, target))
            job = self._queue.pop(index)
            self._queue.insert(target, job)
        self._emit_status()
        return {"ok": True}

    def clear(self) -> dict:
        with self._lock:
            self._queue.clear()
        self._emit_status()
        return {"ok": True}

    def cancel(self) -> dict:
        self._cancel.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self.emit_log("Cancel requested.")
        return {"ok": True}

    # ---- モデルダウンロード ----

    def download_models(self, payload: Optional[dict] = None) -> dict:
        """推定モデル一式 (検出 + 2D + 3D) を事前ダウンロードする。"""
        with self._lock:
            if self._downloading:
                return {"ok": False, "error": "すでにモデルをダウンロード中です。"}
            if self._running:
                return {"ok": False, "error": "ジョブの実行中です。"}
            self._downloading = True
        self._download_thread = threading.Thread(
            target=self._download_worker, args=(payload or {},),
            name="studio-download", daemon=True,
        )
        self._download_thread.start()
        return {"ok": True}

    def wait_download(self, timeout: Optional[float] = None) -> None:
        """テスト用: ダウンロードスレッドの終了を待つ。"""
        thread = self._download_thread
        if thread is not None:
            thread.join(timeout)

    def _download_worker(self, payload: dict) -> None:
        job = normalize_job({**payload, "input": "_prepare_"})
        command = self._download_command_builder(job)
        try:
            with self._lock:
                self._downloads = model_download_entries(job, "downloading")
            self._emit_status()
            self.emit_log("Preparing models: " + " ".join(command))
            code = self._run_streamed(command)
            if code == 0:
                try:
                    MODEL_READY_MARKER.parent.mkdir(parents=True, exist_ok=True)
                    MODEL_READY_MARKER.write_text("ok", encoding="utf-8")
                except OSError:
                    pass
            else:
                self.emit_log(f"Model download failed (code {code}).")
            with self._lock:
                self._downloads = model_download_entries(
                    job, "ready" if code == 0 else "failed"
                )
        finally:
            with self._lock:
                self._downloading = False
            self._emit_status()

    def _run_streamed(self, command: List[str]) -> int:
        """サブプロセスを実行し、出力を行単位で配信して終了コードを返す。"""
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONWARNINGS", "ignore")
        try:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, env=env,
            )
        except OSError as exc:
            self.emit_log(f"Error: failed to start: {exc}")
            return 1
        self._stream_process_output(process)
        process.wait()
        return int(process.returncode or 0)

    # ---- 実行 ----

    def run(self, payload: Optional[dict] = None) -> dict:
        """ジョブ (または積んであるキュー) の実行を開始する。"""
        if payload and payload.get("input"):
            result = self.enqueue(payload)
            if not result.get("ok"):
                return result
        with self._lock:
            if self._running:
                return {"ok": False, "error": "すでにジョブを実行中です。"}
            if not self._queue:
                return {"ok": False, "error": "キューが空です。"}
            self._running = True
            self._cancel.clear()
            self._worker = threading.Thread(
                target=self._run_queue, name="studio-jobs", daemon=True
            )
        self._worker.start()
        return {"ok": True}

    def wait(self, timeout: Optional[float] = None) -> None:
        """テスト用: 実行スレッドの終了を待つ。"""
        worker = self._worker
        if worker is not None:
            worker.join(timeout)

    def _run_queue(self) -> None:
        try:
            with self._lock:
                total = len(self._queue)
            index = 0
            while not self._cancel.is_set():
                with self._lock:
                    if not self._queue:
                        break
                    job = self._queue.pop(0)
                    index += 1
                    self._current = job
                    self._current_index = index
                    self._current_total = max(total, index)
                    self._progress = 0
                self._emit_status()
                code = self._run_one(job)
                with self._lock:
                    self._completed.append(
                        {
                            "input": job.get("input"),
                            "return_code": code,
                            "timestamp": time.time(),
                        }
                    )
                    self._completed = self._completed[-50:]
                    self._last_return_code = code
                    self._current = None
                if code == 0:
                    self._emit_outputs(job)
                self._emit_status()
        finally:
            with self._lock:
                self._running = False
                self._current = None
                self._progress = 0
            self._emit_status()

    def _run_one(self, job: dict) -> int:
        """1 ジョブをサブプロセスで実行してログ / 進捗を流す。"""
        try:
            Path(job["output_root"]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.emit_log(f"Error: cannot create output folder: {exc}")
            return 1
        command = self._command_builder(job)
        self.emit_log("Run: " + " ".join(command))
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONWARNINGS", "ignore")  # mmengine の pkg_resources 警告など抑制
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except OSError as exc:
            self.emit_log(f"Error: failed to start job: {exc}")
            return 1
        with self._lock:
            self._process = process
        try:
            self._stream_process_output(process)
            process.wait()
            if self._cancel.is_set() and process.poll() is None:
                process.kill()
            return int(process.returncode or 0)
        finally:
            with self._lock:
                self._process = None

    def _stream_process_output(self, process: subprocess.Popen) -> None:
        """stdout(+stderr) を行単位 (\\r も区切り) で読んで配信する。"""
        assert process.stdout is not None
        buffer = b""
        while True:
            chunk = process.stdout.read(1)
            if not chunk:
                break
            if chunk in (b"\n", b"\r"):
                if buffer:
                    self._handle_line(buffer.decode("utf-8", "replace"))
                    buffer = b""
                continue
            buffer += chunk
        if buffer:
            self._handle_line(buffer.decode("utf-8", "replace"))

    def _handle_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        match = _PROGRESS_RE.search(line)
        if match and ("[" in line or "fps" in line):
            percent = max(0, min(100, int(float(match.group(1)))))
            with self._lock:
                self._progress = percent
            self._emit({"type": "progress", "percent": percent, "text": line})
            return
        self.emit_log(line)

    def _emit_outputs(self, job: dict) -> None:
        paths = job_output_paths(job)
        if job.get("center_root") or job.get("normalize_scale"):
            try:
                changed = postprocess_csv_outputs(job, paths)
                if changed:
                    self.emit_log(
                        "Applied export transform: "
                        + ", ".join(sorted(changed))
                    )
            except Exception as exc:
                self.emit_log(f"Warning: export transform failed: {exc}")
        kinds = (
            ("json", "json"),
            ("wide_csv", "csv"),
            ("long_csv", "csv"),
            ("summary", "summary"),
            ("video", "video"),
        )
        for key, kind in kinds:
            path = paths.get(key)
            if path is not None and Path(path).is_file():
                self._emit({"type": "output", "path": str(path), "kind": kind})


def _transform_groups(rows, get, set_, center: bool, normalize: bool) -> None:
    """(フレーム, 人物) ごとに world 座標へセンタリング / 正規化を適用する。

    get(row, j) -> (x, y, z) | None, set_(row, j, xyz) のアクセサで
    ワイド / ロング両形式に対応する。j は関節インデックスの iterable。
    """
    for row_group, joints in rows:
        pts = {j: get(row_group, j) for j in joints}
        valid = [p for p in pts.values() if p is not None]
        if not valid:
            continue
        if center:
            root = pts.get(0)
            if root is None:
                root = tuple(
                    sum(c) / len(valid) for c in zip(*valid)
                )
            pts = {
                j: (None if p is None else tuple(a - b for a, b in zip(p, root)))
                for j, p in pts.items()
            }
        if normalize:
            scale = max(
                (sum(c * c for c in p) ** 0.5 for p in pts.values() if p),
                default=0.0,
            )
            if scale > 1e-9:
                pts = {
                    j: (None if p is None else tuple(c / scale for c in p))
                    for j, p in pts.items()
                }
        for j, p in pts.items():
            if p is not None:
                set_(row_group, j, p)


def postprocess_csv_outputs(job: dict, paths: Dict[str, Path]) -> List[str]:
    """center_root / normalize_scale を CSV の world 座標へ適用する。

    センタリングは各 (フレーム, 人物) の関節 0 (H36M の root。無効なら
    重心) を原点へ移動する。正規化は原点からの最大距離が 1 になるように
    スケールする (体格差の吸収用)。results JSON は生のまま残す。
    """
    import csv as _csv

    center = bool(job.get("center_root"))
    normalize = bool(job.get("normalize_scale"))
    if not (center or normalize):
        return []
    changed: List[str] = []

    wide = paths.get("wide_csv")
    if wide is not None and Path(wide).is_file():
        with open(wide, "r", encoding="utf-8", newline="") as f:
            reader = list(_csv.reader(f))
        if reader:
            header, *rows = reader
            cols = {}
            for j_name in {c[: -len("_world_x")] for c in header
                           if c.endswith("_world_x")}:
                try:
                    cols[j_name] = tuple(
                        header.index(f"{j_name}_world_{ax}") for ax in "xyz"
                    )
                except ValueError:
                    continue

            def get_w(row, name):
                try:
                    return tuple(float(row[i]) for i in cols[name])
                except (ValueError, IndexError):
                    return None

            def set_w(row, name, p):
                for i, v in zip(cols[name], p):
                    row[i] = f"{v:.6f}"

            # 関節 0 = ヘッダ出現順の先頭列セット (WideCsvExporter は
            # キーポイント順に列を並べる)
            order = sorted(cols, key=lambda n: cols[n][0])
            idx_of = {j: name for j, name in enumerate(order)}
            grouped = [
                (row, list(idx_of)) for row in rows if len(row) >= 3
            ]
            _transform_groups(
                grouped,
                lambda row, j: get_w(row, idx_of[j]),
                lambda row, j, p: set_w(row, idx_of[j], p),
                center, normalize,
            )
            with open(wide, "w", encoding="utf-8", newline="") as f:
                writer = _csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            changed.append("wide CSV")

    long_csv = paths.get("long_csv")
    if long_csv is not None and Path(long_csv).is_file():
        with open(long_csv, "r", encoding="utf-8", newline="") as f:
            reader = list(_csv.reader(f))
        if reader:
            header, *rows = reader
            try:
                i_frame = header.index("frame")
                i_person = header.index("person")
                i_kid = header.index("keypoint_id")
                i_xyz = tuple(header.index(f"world_{ax}") for ax in "xyz")
            except ValueError:
                i_xyz = ()
            if i_xyz:
                groups: Dict[tuple, Dict[int, list]] = {}
                for row in rows:
                    if len(row) <= max(i_xyz):
                        continue
                    key = (row[i_frame], row[i_person])
                    try:
                        kid = int(row[i_kid])
                    except ValueError:
                        continue
                    groups.setdefault(key, {})[kid] = row

                def get_l(group, j):
                    row = group.get(j)
                    if row is None:
                        return None
                    try:
                        return tuple(float(row[i]) for i in i_xyz)
                    except ValueError:
                        return None

                def set_l(group, j, p):
                    row = group.get(j)
                    if row is not None:
                        for i, v in zip(i_xyz, p):
                            row[i] = f"{v:.6f}"

                grouped = [
                    (group, sorted(group)) for group in groups.values()
                ]
                _transform_groups(grouped, get_l, set_l, center, normalize)
                with open(long_csv, "w", encoding="utf-8", newline="") as f:
                    writer = _csv.writer(f)
                    writer.writerow(header)
                    writer.writerows(rows)
                changed.append("long CSV")

    return changed


class StudioHandler(BaseHTTPRequestHandler):
    """GUI / API を配信するハンドラ。serve() がクラス属性を束ねる。"""

    manager: JobManager
    dialogs: FileDialogs
    quiet = True
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: D102 - 静かに
        if not self.quiet:
            super().log_message(fmt, *args)

    # ---- 共通レスポンス ----

    def _send_bytes(
        self, body: bytes, content_type: str, status: int = HTTPStatus.OK,
        extra: Optional[dict] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _query(self) -> dict:
        return {
            k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()
        }

    # ---- ルーティング ----

    def do_GET(self) -> None:  # noqa: N802 (http.server の規約)
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_gui("index.html", "text/html; charset=utf-8")
        if path == "/app.css":
            return self._serve_gui("app.css", "text/css; charset=utf-8")
        if path == "/app.js":
            from poselab.studio import build_app_js

            body = build_app_js().encode("utf-8")
            return self._send_bytes(body, "text/javascript; charset=utf-8")
        if path == "/status":
            return self._send_json(self.manager.status())
        if path == "/gpu":
            return self._send_json(gpu_info())
        if path == "/events":
            return self._serve_events()
        if path == "/file":
            return self._serve_file()
        if path == "/summary":
            return self._serve_summary()
        if path == "/open":
            target = self._query().get("path", "")
            ok = bool(target) and open_in_explorer(Path(target))
            return self._send_json({"ok": ok})
        return self._send_json(
            {"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND
        )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/run":
            return self._send_json(self.manager.run(self._read_json()))
        if path == "/enqueue":
            return self._send_json(self.manager.enqueue(self._read_json()))
        if path == "/queue-move":
            body = self._read_json()
            return self._send_json(
                self.manager.move(
                    int(body.get("index", -1)), int(body.get("offset", 0))
                )
            )
        if path == "/clear-queue":
            return self._send_json(self.manager.clear())
        if path == "/cancel":
            return self._send_json(self.manager.cancel())
        if path == "/download-models":
            return self._send_json(self.manager.download_models(self._read_json()))
        if path == "/preflight":
            return self._send_json(preflight(self._read_json()))
        if path == "/pick-video":
            return self._send_json(self.dialogs.pick_video())
        if path == "/pick-videos":
            return self._send_json(self.dialogs.pick_videos())
        if path == "/pick-folder":
            return self._send_json(self.dialogs.pick_folder())
        return self._send_json(
            {"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND
        )

    # ---- 各ルートの実装 ----

    def _serve_gui(self, name: str, content_type: str) -> None:
        from poselab.studio import GUI_DIR

        body = (GUI_DIR / name).read_text(encoding="utf-8").encode("utf-8")
        self._send_bytes(body, content_type)

    def _serve_events(self) -> None:
        q = self.manager.subscribe()
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            first = json.dumps(
                {"type": "status", **self.manager.status()},
                ensure_ascii=False,
            )
            self.wfile.write(f"data: {first}\n\n".encode("utf-8"))
            self.wfile.flush()
            while True:
                try:
                    data = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.manager.unsubscribe(q)

    def _serve_file(self) -> None:
        target = self._query().get("path", "")
        path = Path(target)
        if not target or not path.is_file():
            return self._send_json(
                {"ok": False, "error": "file not found"}, HTTPStatus.NOT_FOUND
            )
        mime = {
            ".json": "application/json; charset=utf-8",
            ".csv": "text/csv; charset=utf-8",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
        }.get(path.suffix.lower(), "application/octet-stream")
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    if match.group(2):
                        end = min(int(match.group(2)), size - 1)
                else:  # bytes=-N (末尾 N バイト)
                    start = max(0, size - int(match.group(2)))
                if start > end or start >= size:
                    self.send_response(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
                    )
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = HTTPStatus.PARTIAL_CONTENT
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def _serve_summary(self) -> None:
        target = self._query().get("path", "")
        path = Path(target)
        if not target or not path.is_file():
            return self._send_json({"ok": False, "error": "file not found"})
        try:
            summary = summarize_results_json(path)
        except (ValueError, OSError) as exc:
            return self._send_json({"ok": False, "error": str(exc)})
        return self._send_json({"ok": True, "summary": summary})


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    manager: Optional[JobManager] = None,
    dialogs: Optional[FileDialogs] = None,
    quiet: bool = False,
) -> ThreadingHTTPServer:
    """サーバーを構築して返す (固定ポートが塞がっていれば +1 で探す)。"""
    handler = type(
        "_BoundStudioHandler",
        (StudioHandler,),
        {
            "manager": manager or JobManager(),
            "dialogs": dialogs or FileDialogs(),
            "quiet": quiet,
        },
    )
    last_error: Optional[OSError] = None
    for candidate in ([port] if port == 0 else range(port, port + 20)):
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
            server.daemon_threads = True
            return server
        except OSError as exc:
            last_error = exc
    raise last_error or OSError("空きポートが見つかりません")


def main_serve(argv: Optional[Sequence[str]] = None) -> int:
    """`poselab-studio serve` の本体。"""
    import argparse

    force_utf8_stdio()

    parser = argparse.ArgumentParser(
        prog="poselab-studio serve",
        description="Pose3DStudio 後継の Web GUI をローカルで起動します。",
    )
    parser.add_argument("--host", default="127.0.0.1", help="バインド先")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"ポート (既定: {DEFAULT_PORT}。塞がっていれば自動で +1)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="ブラウザを自動で開かない"
    )
    args = parser.parse_args(argv)

    server = serve(host=args.host, port=args.port)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    gpu = gpu_info()
    print(f"poselab studio: {url}  (Ctrl+C で終了)")
    print(
        "GPU: " + (f"{gpu['name']} を検出" if gpu["available"] else "未検出 (CPU 実行)")
    )
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
