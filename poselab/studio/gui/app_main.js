/* ================================================================
   Pose3D Studio GUI 本体
   (上の IIFE = PoseLab3D エンジン。ここからアプリ配線)
   ================================================================ */

const $ = (id) => document.getElementById(id);

const inputPath = $("input-path");
const outputRoot = $("output-root");
const reencode = $("reencode");
const progressToggle = $("progress");
const csvFormat = $("csv-format");
const runBtn = $("run-btn");
const logOutput = $("log-output");
const progressFill = $("progress-fill");
const progressText = $("progress-text");
const outputList = $("output-list");
const browseInput = $("browse-input");
const browseMulti = $("browse-multi");
const browseFolder = $("browse-folder");
const addQueueBtn = $("add-queue");
const clearQueueBtn = $("clear-queue");
const queueList = $("queue-list");
const queueMeta = $("queue-meta");
const preset = $("preset");
const backendSelect = $("backend");
const modelProfile = $("model-profile");
const detectorProfile = $("detector-profile");
const mmposeOnly = $("mmpose-only");
const mediapipeOnly = $("mediapipe-only");
const mpModel = $("mp-model");
const mpNumPoses = $("mp-num-poses");
const runHint = $("run-hint");
const centerRoot = $("center-root");
const normalizeScale = $("normalize-scale");
const gpuStatus = $("gpu-status");
const serverStatus = $("server-status");
const clearLog = $("clear-log");
const openOutput = $("open-output");
const refreshChecks = $("refresh-checks");
const checkList = $("check-list");
const videoInfo = $("video-info");
const downloadsList = $("downloads-list");
const downloadsMeta = $("downloads-meta");
const downloadModels = $("download-models");
const cancelBtn = $("cancel-btn");
const previewVideo = $("preview-video");
const previewSummary = $("preview-summary");
const openVideo = $("open-video");
const historyList = $("history-list");
const viewerCanvas = $("viewer-canvas");
const viewerStage = $("viewer-stage");
const viewerPlay = $("viewer-play");
const viewerFrame = $("viewer-frame");
const viewerFrameLabel = $("viewer-frame-label");
const viewerFps = $("viewer-fps");
const viewerAxis = $("viewer-axis");
const viewerScale = $("viewer-scale");
const viewerYaw = $("viewer-yaw");
const viewerPitch = $("viewer-pitch");
const viewerInstance = $("viewer-instance");
const viewerJoint = $("viewer-joint");
const viewerCenter = $("viewer-center");
const viewerNormalize = $("viewer-normalize");
const viewerFit = $("viewer-fit");
const viewerReset = $("viewer-reset");
const viewerOpen = $("viewer-open");
const viewerDemo = $("viewer-demo");
const viewerFile = $("viewer-file");

let lastOutputRoot = "";
let outputAuto = true;
let lastRemaining = "";
let lastVideoPath = "";
let lastJsonPath = "";
let queueState = [];
let currentJob = null;
let preflightTimer = null;

const PRESETS = {
  research: {
    reencode: true,
    progress: true,
    csv: "both",
    centerRoot: true,
    normalizeScale: true,
  },
  fast: {
    reencode: false,
    progress: true,
    csv: "wide",
    centerRoot: false,
    normalizeScale: false,
  },
  preview: {
    reencode: true,
    progress: true,
    csv: "wide",
    centerRoot: true,
    normalizeScale: false,
  },
};

const MODEL_PROFILES = {
  balanced: {},
  accuracy: {
    pose2d_config:
      "configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_aic-coco-384x288.py",
    pose2d_checkpoint:
      "checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth",
    pose3d_config:
      "configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py",
    pose3d_checkpoint:
      "checkpoints/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth",
  },
};

const DETECTOR_PROFILES = {
  balanced: {
    det_config: "demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py",
    det_checkpoint:
      "checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
  },
  accuracy: {
    det_config: "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    det_checkpoint:
      "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
  },
};

function appendLog(line) {
  logOutput.textContent += line + "\n";
  logOutput.scrollTop = logOutput.scrollHeight;
}

function extractRemaining(text) {
  if (!text) return "";
  const match = text.match(/\[(?:[^<]*)<([^,\]]+)/);
  return match ? match[1].trim() : "";
}

function updateProgress(percent, text) {
  progressFill.style.width = `${percent}%`;
  const remaining = extractRemaining(text);
  if (remaining) {
    lastRemaining = remaining;
  }
  let label = `${percent}%`;
  const displayRemaining = remaining || lastRemaining;
  if (displayRemaining && percent > 0) {
    label = `${percent}% ${displayRemaining}`;
  } else if (text && !remaining && percent === 0) {
    label = text;
  }
  progressText.textContent = label;
}

function applyPreset(name) {
  const presetConfig = PRESETS[name];
  if (!presetConfig) return;
  reencode.checked = presetConfig.reencode;
  progressToggle.checked = presetConfig.progress;
  csvFormat.value = presetConfig.csv;
  centerRoot.checked = presetConfig.centerRoot;
  normalizeScale.checked = presetConfig.normalizeScale;
  schedulePreflight();
}

function sanitize(value) {
  return value.replace(/^["']+|["']+$/g, "").trim();
}

function currentBackend() {
  return backendSelect ? backendSelect.value : "mmpose";
}

function buildPayload(options = {}) {
  const backend = currentBackend();
  const payload = {
    input: sanitize(inputPath.value),
    output_root: sanitize(outputRoot.value),
    backend,
    csv_format: csvFormat.value,
    reencode: reencode.checked,
    progress: progressToggle.checked,
    center_root: centerRoot.checked,
    normalize_scale: normalizeScale.checked,
    // 分析出力 / 平滑化 / マスキング
    angles: $("out-angles").checked,
    velocity: $("out-velocity").checked,
    symmetry: $("out-symmetry").checked,
    smooth_method: $("smooth-method").value,
    smooth_window: parseInt($("smooth-window").value, 10) || 0,
    smooth_cutoff: parseFloat($("smooth-cutoff").value) || 0,
    smooth_weighted: $("smooth-weighted").checked,
    mask_visibility: parseFloat($("mask-visibility").value) || 0,
    ...options,
  };
  if (backend === "mediapipe") {
    payload.model = mpModel ? mpModel.value : "full";
    payload.num_poses = mpNumPoses
      ? Math.max(1, parseInt(mpNumPoses.value, 10) || 1)
      : 1;
    return payload;
  }
  const profile = MODEL_PROFILES[modelProfile.value] || {};
  Object.keys(profile).forEach((key) => {
    if (profile[key]) {
      payload[key] = profile[key];
    }
  });
  const detProfile = DETECTOR_PROFILES[detectorProfile.value] || {};
  Object.keys(detProfile).forEach((key) => {
    if (detProfile[key]) {
      payload[key] = detProfile[key];
    }
  });
  return payload;
}

function applyBackend() {
  const isMediapipe = currentBackend() === "mediapipe";
  if (mmposeOnly) mmposeOnly.hidden = isMediapipe;
  if (mediapipeOnly) mediapipeOnly.hidden = !isMediapipe;
  if (runHint) runHint.textContent = isMediapipe ? "CPU 可" : "GPU 必須";
  schedulePreflight();
}

function deriveOutputRoot(input) {
  const clean = input.replace(/^["']+|["']+$/g, "").trim();
  if (!clean) return "";
  const parts = clean.split(/[/\\\\]/);
  const file = parts.pop();
  if (!file) return "";
  const stem = file.replace(/\.[^/.\\\\]+$/, "");
  const dir = parts.join("\\");
  return dir ? `${dir}\\${stem}` : stem;
}

function applyAutoOutput() {
  if (!outputAuto) return;
  const derived = deriveOutputRoot(inputPath.value);
  if (derived) {
    outputRoot.value = derived;
  }
}

function renderQueue(queue, current) {
  queueState = queue || [];
  currentJob = current || null;
  queueList.innerHTML = "";
  let count = queueState.length;
  if (currentJob && currentJob.input) {
    const item = document.createElement("li");
    const name = currentJob.input.split(/[/\\\\]/).pop();
    item.textContent = `実行中: ${name}`;
    queueList.appendChild(item);
    count += 1;
  }
  queueState.forEach((job) => {
    const item = document.createElement("li");
    const name = job.input ? job.input.split(/[/\\\\]/).pop() : "(不明)";
    item.textContent = `待機中: ${name}`;
    const controls = document.createElement("div");
    controls.className = "queue-controls";
    const up = document.createElement("button");
    up.textContent = "上へ";
    up.addEventListener("click", () => moveQueue(job, -1));
    const down = document.createElement("button");
    down.textContent = "下へ";
    down.addEventListener("click", () => moveQueue(job, 1));
    controls.appendChild(up);
    controls.appendChild(down);
    item.appendChild(controls);
    queueList.appendChild(item);
  });
  queueMeta.textContent = `${count} 件`;
  updateRunLabel();
}

async function moveQueue(job, offset) {
  const index = queueState.indexOf(job);
  if (index < 0) return;
  await fetchJSON("/queue-move", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index, offset }),
  });
}

function updateRunLabel() {
  const hasQueue = queueState.length > 0;
  const label = hasQueue ? `キュー実行 (${queueState.length})` : "実行";
  runBtn.querySelector("span").textContent = label;
}

function updateChecks(warnings, info) {
  checkList.innerHTML = "";
  if (!warnings || warnings.length === 0) {
    const ok = document.createElement("li");
    ok.className = "ok";
    ok.textContent = "問題は見つかりませんでした。";
    checkList.appendChild(ok);
  } else {
    warnings.forEach((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      checkList.appendChild(item);
    });
  }
  if (info && (info.frames || info.fps || info.duration)) {
    const parts = [];
    if (info.frames) parts.push(`フレーム数: ${info.frames}`);
    if (info.fps) parts.push(`FPS: ${info.fps.toFixed(2)}`);
    if (info.duration) parts.push(`長さ: ${info.duration.toFixed(1)}秒`);
    videoInfo.textContent = parts.join(" | ");
  } else {
    videoInfo.textContent = "";
  }
}

function formatBytes(value) {
  if (value === null || value === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = Number(value);
  let idx = 0;
  while (size >= 1024 && idx < units.length - 1) {
    size /= 1024;
    idx += 1;
  }
  const precision = size >= 10 || idx === 0 ? 0 : 1;
  return `${size.toFixed(precision)} ${units[idx]}`;
}

function renderDownloads(items) {
  if (!downloadsList || !downloadsMeta) return;
  downloadsList.innerHTML = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.className = "download empty";
    item.textContent = "ダウンロードは不要です。";
    downloadsList.appendChild(item);
    downloadsMeta.textContent = "準備完了";
    return;
  }

  const counts = { ready: 0, downloading: 0, pending: 0, failed: 0 };
  const labels = {
    ready: "完了",
    downloading: "取得中",
    pending: "待機",
    failed: "失敗",
  };

  items.forEach((entry) => {
    const status = entry.status || "pending";
    if (counts[status] !== undefined) counts[status] += 1;
    const row = document.createElement("div");
    row.className = "download-row";

    const name = document.createElement("div");
    name.className = "download-name";
    name.textContent = entry.name || "(不明)";

    const badge = document.createElement("span");
    badge.className = `download-status ${status}`;
    badge.textContent = labels[status] || status;
    if (entry.error) {
      badge.title = entry.error;
    }

    row.appendChild(name);
    row.appendChild(badge);

    const meta = document.createElement("div");
    meta.className = "download-meta";
    if (status === "downloading") {
      const parts = [];
      if (typeof entry.progress === "number") {
        parts.push(`${entry.progress}%`);
      }
      if (entry.downloaded !== null && entry.downloaded !== undefined) {
        if (entry.total !== null && entry.total !== undefined) {
          parts.push(`${formatBytes(entry.downloaded)} / ${formatBytes(entry.total)}`);
        } else {
          parts.push(formatBytes(entry.downloaded));
        }
      }
      meta.textContent = parts.join(" ");
    } else if (status === "failed") {
      meta.textContent = "詳細はログを確認してください。";
    } else if (status === "ready") {
      meta.textContent = "取得済み。";
    } else {
      meta.textContent = "ダウンロード待ち。";
    }

    const item = document.createElement("li");
    item.className = `download ${status}`;
    item.appendChild(row);
    item.appendChild(meta);

    if (status === "downloading") {
      const bar = document.createElement("div");
      bar.className = "download-bar";
      if (typeof entry.progress !== "number") {
        bar.classList.add("indeterminate");
      }
      const fill = document.createElement("span");
      if (typeof entry.progress === "number") {
        fill.style.width = `${entry.progress}%`;
      }
      bar.appendChild(fill);
      item.appendChild(bar);
    }

    downloadsList.appendChild(item);
  });

  const parts = [];
  if (counts.downloading) parts.push(`${counts.downloading} 件取得中`);
  if (counts.pending) parts.push(`${counts.pending} 件待機`);
  if (counts.failed) parts.push(`${counts.failed} 件失敗`);
  if (parts.length === 0) parts.push(`${counts.ready} 件完了`);
  downloadsMeta.textContent = parts.join(" • ");
  if (downloadModels) {
    const busy = counts.downloading > 0;
    downloadModels.disabled = busy;
    downloadModels.textContent = busy ? "取得中..." : "ダウンロード";
  }
}

function setPreviewVideo(path) {
  if (!path) return;
  lastVideoPath = path;
  previewVideo.src = `/file?path=${encodeURIComponent(path)}`;
  previewVideo.load();
}

/* ================================================================
   3D ビューア (PoseLab3D エンジン)
   ================================================================ */

const stage3d = new PoseLab3D.PoseStage(viewerCanvas);
stage3d.opts.highlight = -1;
stage3d.opts.trailLen = 70;

let viewerModel = null;
let viewerPath = "";
let viewerPlaying = false;
let viewerFrameFloat = 0;
let viewerLastTick = null;

// エンジン側のカメラ操作 (ドラッグ/ホイール) をスライダーへ反映
stage3d.onCameraChange = () => {
  let yaw = stage3d.cam.yaw % 360;
  if (yaw > 180) yaw -= 360;
  if (yaw < -180) yaw += 360;
  viewerYaw.value = String(Math.round(yaw));
  viewerPitch.value = String(Math.round(stage3d.cam.pitch));
  if (stage3d.home.dist > 0) {
    const s = stage3d.home.dist / stage3d.cam.dist;
    viewerScale.value = String(Math.max(0.4, Math.min(2.5, s)));
  }
};

function setViewerPlaying(value) {
  viewerPlaying = value && !!viewerModel;
  viewerPlay.textContent = viewerPlaying ? "❚❚" : "▶";
  viewerLastTick = null;
}

function updateViewerFrameLabel() {
  const total = viewerModel ? viewerModel.frames.length : 0;
  viewerFrameLabel.textContent = `フレーム ${total ? stage3d.frame + 1 : 0} / ${total}`;
}

function setViewerFrame(i, fromSlider = false) {
  if (!viewerModel) return;
  const max = viewerModel.frames.length - 1;
  const idx = Math.max(0, Math.min(max, Math.round(i)));
  stage3d.frame = idx;
  viewerFrameFloat = idx;
  stage3d.dirty = true;
  if (!fromSlider) viewerFrame.value = String(idx);
  updateViewerFrameLabel();
}

function loadViewerModel(model, sourceLabel) {
  viewerModel = model;
  const axis = model.defaultAxis || "ydown";
  viewerAxis.value = axis;
  stage3d.opts.axis = axis;
  stage3d.setModel(model);

  viewerFrame.max = String(Math.max(0, model.frames.length - 1));
  viewerFrame.value = "0";
  viewerFrameFloat = 0;
  if (model.fps) {
    viewerFps.value = String(Math.round(model.fps));
  }

  viewerInstance.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = `全員 (${model.personIds.length})`;
  viewerInstance.appendChild(all);
  model.personIds.forEach((id) => {
    const option = document.createElement("option");
    option.value = String(id);
    option.textContent = `人物 ${id}`;
    viewerInstance.appendChild(option);
  });

  viewerJoint.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "なし";
  viewerJoint.appendChild(none);
  model.names.forEach((name, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = name;
    viewerJoint.appendChild(option);
  });
  stage3d.opts.highlight = -1;

  // エクスポートパネル
  const exportJoints = $("viewer-export-joints");
  exportJoints.innerHTML = "";
  model.names.forEach((n, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = `${idx}: ${n}`;
    option.selected = true;
    exportJoints.appendChild(option);
  });
  $("viewer-export-start").value = "1";
  $("viewer-export-start").max = String(model.frames.length);
  $("viewer-export-end").value = String(model.frames.length);
  $("viewer-export-end").max = String(model.frames.length);

  updateViewerFrameLabel();
  setViewerPlaying(true);
  if (sourceLabel) {
    appendLog(`ビューア: ${sourceLabel} (${model.formatLabel}, ${model.frames.length} フレーム)`);
  }
}

function setViewerSource(path) {
  if (!path || path === viewerPath) return;
  viewerPath = path;
  fetch(`/file?path=${encodeURIComponent(path)}`)
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.text();
    })
    .then((text) => {
      const name = path.split(/[/\\]/).pop() || path;
      loadViewerModel(PoseLab3D.parseAny(text, name), name);
    })
    .catch(() => {
      /* ビューアは読めなくてもパイプラインに影響させない */
    });
}

function viewerTick(ts) {
  if (viewerPlaying && viewerModel) {
    if (viewerLastTick != null) {
      const dt = (ts - viewerLastTick) / 1000;
      const fps = Number(viewerFps.value) || 30;
      viewerFrameFloat += dt * fps;
      const max = viewerModel.frames.length - 1;
      if (viewerFrameFloat > max) viewerFrameFloat = 0;
      const idx = Math.round(viewerFrameFloat);
      if (idx !== stage3d.frame) {
        stage3d.frame = Math.max(0, Math.min(max, idx));
        viewerFrame.value = String(stage3d.frame);
        updateViewerFrameLabel();
        stage3d.dirty = true;
      }
    }
    viewerLastTick = ts;
  }
  if (stage3d.dirty) {
    stage3d.dirty = false;
    stage3d.render();
  }
  requestAnimationFrame(viewerTick);
}
requestAnimationFrame(viewerTick);

async function loadViewerFiles(files) {
  const list = [...files].filter((f) => /\.(json|csv)$/i.test(f.name));
  list.sort((a, b) => Number(/\.csv$/i.test(a.name)) - Number(/\.csv$/i.test(b.name)));
  for (const file of list) {
    try {
      const text = await file.text();
      viewerPath = `(local) ${file.name}`;
      loadViewerModel(PoseLab3D.parseAny(text, file.name), file.name);
      return;
    } catch (err) {
      appendLog(`ビューア: ${file.name} を読み込めませんでした (${err.message})`);
    }
  }
}

async function updatePreviewSummary(path) {
  if (!path) return;
  lastJsonPath = path;
  setViewerSource(path);
  try {
    const res = await fetchJSON(`/summary?path=${encodeURIComponent(path)}`);
    if (res.ok && res.summary) {
      const { frames, avg_instances, avg_score } = res.summary;
      previewSummary.textContent = `フレーム数: ${frames} | 平均人数: ${avg_instances} | 平均スコア: ${avg_score}`;
    }
  } catch {
    previewSummary.textContent = "";
  }
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "履歴はまだありません。";
    historyList.appendChild(item);
    return;
  }
  items.slice().reverse().forEach((entry) => {
    const item = document.createElement("li");
    const name = entry.input ? entry.input.split(/[/\\\\]/).pop() : "(不明)";
    const status = entry.return_code === 0 ? "成功" : "失敗";
    const time = entry.timestamp ? new Date(entry.timestamp * 1000) : null;
    const label = time ? time.toLocaleTimeString() : "";
    item.innerHTML = `<span>${status}</span><div>${name} ${label}</div>`;
    if (entry.return_code !== 0) {
      const badge = item.querySelector("span");
      badge.style.color = "var(--warn)";
      badge.style.borderColor = "rgba(251, 113, 133, 0.45)";
      badge.style.background = "rgba(251, 113, 133, 0.07)";
    }
    historyList.appendChild(item);
  });
}

async function schedulePreflight() {
  if (preflightTimer) {
    clearTimeout(preflightTimer);
  }
  preflightTimer = setTimeout(async () => {
    const payload = buildPayload();
    try {
      const res = await fetchJSON("/preflight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      updateChecks(res.warnings || [], res.info || {});
    } catch {
      updateChecks(["事前チェックに失敗しました。"], {});
    }
  }, 300);
}

function addOutput(path, kind) {
  const item = document.createElement("li");
  item.innerHTML = `<span>${kind}</span><div>${path}</div>`;
  outputList.appendChild(item);
  if (path) {
    const dir = path.replace(/[/\\\\][^/\\\\]+$/, "");
    if (dir) {
      lastOutputRoot = dir;
    }
  }
  if (kind === "video") {
    setPreviewVideo(path);
  }
  if (kind === "json") {
    updatePreviewSummary(path);
  }
}

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  return await res.json();
}

async function checkGPU() {
  try {
    const data = await fetchJSON("/gpu");
    gpuStatus.textContent = data.available ? "検出" : "未検出";
    gpuStatus.style.color = data.available ? "var(--ok)" : "var(--warn)";
  } catch (err) {
    gpuStatus.textContent = "利用不可";
    gpuStatus.style.color = "var(--warn)";
    serverStatus.textContent = "切断";
  }
}

async function enqueueInputs(paths) {
  const payload = buildPayload({ inputs: paths, output_root: "" });
  const res = await fetchJSON("/enqueue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const msg = res.error || "ジョブをキューに追加できませんでした。";
    alert(msg);
    appendLog(`エラー: ${msg}`);
  }
}

async function enqueueCurrent() {
  const payload = buildPayload();
  if (!payload.input) {
    alert("入力動画を指定してください。");
    return;
  }
  if (!payload.output_root) {
    payload.output_root = deriveOutputRoot(payload.input);
  }
  const res = await fetchJSON("/enqueue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const msg = res.error || "ジョブをキューに追加できませんでした。";
    alert(msg);
    appendLog(`エラー: ${msg}`);
  }
}

async function runPipeline() {
  if (queueState.length > 0) {
    outputList.innerHTML = "";
    logOutput.textContent = "";
    updateProgress(0, "キューを開始中...");
    serverStatus.textContent = "実行中";
    const res = await fetchJSON("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const msg = res.error || "キューを開始できませんでした。";
      alert(msg);
      appendLog(`エラー: ${msg}`);
    }
    return;
  }

  const payload = buildPayload();
  if (!payload.input) {
    alert("入力動画を指定してください。");
    return;
  }
  if (!payload.output_root) {
    payload.output_root = deriveOutputRoot(payload.input);
  }

  lastOutputRoot = payload.output_root;
  outputList.innerHTML = "";
  logOutput.textContent = "";
  updateProgress(0, "開始中...");
  serverStatus.textContent = "実行中";

  const res = await fetchJSON("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const msg = res.error || "ジョブを開始できませんでした。";
    alert(msg);
    appendLog(`エラー: ${msg}`);
  }
}

function startEvents() {
  const source = new EventSource("/events");
  source.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }
    if (!data.type && typeof data.running === "boolean") {
      data.type = "status";
    }
    if (data.type === "log") {
      appendLog(data.message);
    }
    if (data.type === "progress") {
      updateProgress(data.percent, data.text);
    }
    if (data.type === "output") {
      addOutput(data.path, data.kind);
    }
    if (data.type === "status") {
      if (typeof data.progress === "number") {
        if (data.progress > 0) {
          updateProgress(data.progress, `${data.progress}%`);
        } else if (data.running) {
          updateProgress(0, "実行中...");
        } else {
          updateProgress(0, "待機中");
        }
      }
      if (data.running && data.current_job) {
        serverStatus.textContent = `実行中 ${data.current_job.index}/${data.current_job.total}`;
      } else {
        serverStatus.textContent = data.running ? "実行中" : "準備完了";
      }
      runBtn.disabled = !!data.running;
      renderQueue(data.queue || [], data.current_job);
      renderHistory(data.completed || []);
      if (Array.isArray(data.downloads)) {
        renderDownloads(data.downloads);
      }
      if (data.running === false && data.return_code !== null) {
        appendLog(`プロセスがコード ${data.return_code} で終了しました`);
      }
    }
  };
  source.onerror = () => {
    serverStatus.textContent = "切断";
  };
}

async function pollStatus() {
  try {
    const data = await fetchJSON("/status");
    if (typeof data.progress === "number") {
      if (data.progress > 0) {
        updateProgress(data.progress, `${data.progress}%`);
      } else if (data.running) {
        updateProgress(0, "実行中...");
      } else {
        updateProgress(0, "待機中");
      }
    }
    if (data.running && data.current_job) {
      serverStatus.textContent = `実行中 ${data.current_job.index}/${data.current_job.total}`;
    } else {
      serverStatus.textContent = data.running ? "実行中" : "準備完了";
    }
    runBtn.disabled = !!data.running;
    renderQueue(data.queue || [], data.current_job);
    renderHistory(data.completed || []);
    if (Array.isArray(data.downloads)) {
      renderDownloads(data.downloads);
    }
  } catch {
    serverStatus.textContent = "切断";
  }
}

/* ----- パイプライン操作の配線 ----- */

runBtn.addEventListener("click", runPipeline);
clearLog.addEventListener("click", () => (logOutput.textContent = ""));
inputPath.addEventListener("input", () => {
  applyAutoOutput();
  schedulePreflight();
});
outputRoot.addEventListener("input", () => {
  outputAuto = outputRoot.value.trim().length === 0;
  schedulePreflight();
});
browseInput.addEventListener("click", async () => {
  const res = await fetchJSON("/pick-video", { method: "POST" });
  if (res.ok && res.path) {
    inputPath.value = res.path;
    outputAuto = true;
    applyAutoOutput();
    schedulePreflight();
  } else if (res.error) {
    alert(res.error);
  }
});
addQueueBtn.addEventListener("click", enqueueCurrent);
browseMulti.addEventListener("click", async () => {
  const res = await fetchJSON("/pick-videos", { method: "POST" });
  if (res.ok && res.paths && res.paths.length > 0) {
    inputPath.value = res.paths[0];
    outputAuto = true;
    applyAutoOutput();
    schedulePreflight();
    await enqueueInputs(res.paths);
  } else if (res.error) {
    alert(res.error);
  }
});
browseFolder.addEventListener("click", async () => {
  const res = await fetchJSON("/pick-folder", { method: "POST" });
  if (res.ok && res.paths && res.paths.length > 0) {
    inputPath.value = res.paths[0];
    outputAuto = true;
    applyAutoOutput();
    schedulePreflight();
    await enqueueInputs(res.paths);
  } else if (res.error) {
    alert(res.error);
  } else {
    alert("フォルダ内に動画ファイルが見つかりません。");
  }
});
clearQueueBtn.addEventListener("click", async () => {
  await fetchJSON("/clear-queue", { method: "POST" });
});
openOutput.addEventListener("click", async () => {
  if (!lastOutputRoot) return;
  await fetchJSON(`/open?path=${encodeURIComponent(lastOutputRoot)}`);
});
openVideo.addEventListener("click", async () => {
  if (!lastVideoPath) return;
  await fetchJSON(`/open?path=${encodeURIComponent(lastVideoPath)}`);
});
cancelBtn.addEventListener("click", async () => {
  await fetchJSON("/cancel", { method: "POST" });
});
downloadModels.addEventListener("click", async () => {
  const res = await fetchJSON("/download-models", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPayload()),
  });
  if (res && res.ok === false && res.error) {
    alert(res.error);
    appendLog(`エラー: ${res.error}`);
  }
});
refreshChecks.addEventListener("click", () => schedulePreflight());
reencode.addEventListener("change", schedulePreflight);
progressToggle.addEventListener("change", schedulePreflight);
csvFormat.addEventListener("change", schedulePreflight);
centerRoot.addEventListener("change", schedulePreflight);
normalizeScale.addEventListener("change", schedulePreflight);
preset.addEventListener("change", () => applyPreset(preset.value));
modelProfile.addEventListener("change", () => schedulePreflight());
detectorProfile.addEventListener("change", () => schedulePreflight());
if (backendSelect) backendSelect.addEventListener("change", applyBackend);
if (mpModel) mpModel.addEventListener("change", schedulePreflight);
if (mpNumPoses) mpNumPoses.addEventListener("change", schedulePreflight);

/* ----- ビューア操作の配線 ----- */

viewerPlay.addEventListener("click", () => setViewerPlaying(!viewerPlaying));
viewerFrame.addEventListener("input", () => {
  setViewerPlaying(false);
  setViewerFrame(Number(viewerFrame.value), true);
});
viewerFps.addEventListener("change", () => updateViewerFrameLabel());
viewerAxis.addEventListener("change", () => {
  stage3d.opts.axis = viewerAxis.value;
  stage3d.fit();
});
viewerScale.addEventListener("input", () => {
  const s = Number(viewerScale.value) || 1;
  if (stage3d.home.dist > 0) {
    stage3d.cam.dist = stage3d.home.dist / s;
    stage3d.dirty = true;
  }
});
viewerYaw.addEventListener("input", () => {
  stage3d.cam.yaw = Number(viewerYaw.value) || 0;
  stage3d.dirty = true;
});
viewerPitch.addEventListener("input", () => {
  stage3d.cam.pitch = Number(viewerPitch.value) || 0;
  stage3d.dirty = true;
});
viewerInstance.addEventListener("change", () => {
  if (!viewerModel) return;
  if (viewerInstance.value === "") {
    stage3d.visible = new Set(viewerModel.personIds);
  } else {
    stage3d.visible = new Set([Number(viewerInstance.value)]);
  }
  stage3d.dirty = true;
});
viewerJoint.addEventListener("change", () => {
  stage3d.opts.highlight =
    viewerJoint.value === "" ? -1 : Number(viewerJoint.value);
  stage3d.dirty = true;
});
viewerCenter.addEventListener("change", () => {
  stage3d.opts.center = viewerCenter.checked;
  stage3d.fit();
});
viewerNormalize.addEventListener("change", () => {
  stage3d.opts.normalize = viewerNormalize.checked;
  stage3d.fit();
});
viewerFit.addEventListener("click", () => stage3d.fit());
viewerReset.addEventListener("click", () => {
  viewerAxis.value = "ydown";
  viewerCenter.checked = true;
  viewerNormalize.checked = false;
  viewerInstance.value = "";
  viewerJoint.value = "";
  viewerFps.value = viewerModel && viewerModel.fps
    ? String(Math.round(viewerModel.fps)) : "30";
  stage3d.opts.axis = "ydown";
  stage3d.opts.center = true;
  stage3d.opts.normalize = false;
  stage3d.opts.highlight = -1;
  if (viewerModel) {
    stage3d.visible = new Set(viewerModel.personIds);
  }
  setViewerFrame(0);
  stage3d.fit();
  stage3d.resetView();
});
document.querySelectorAll(".viewer-top-actions [data-view]").forEach((btn) => {
  btn.addEventListener("click", () => stage3d.setView(btn.dataset.view));
});
viewerOpen.addEventListener("click", () => viewerFile.click());
viewerFile.addEventListener("change", () => {
  if (viewerFile.files.length) loadViewerFiles(viewerFile.files);
  viewerFile.value = "";
});
viewerDemo.addEventListener("click", () => {
  viewerPath = "(demo)";
  loadViewerModel(PoseLab3D.demoModel(), "デモ (合成歩行)");
});

["dragenter", "dragover"].forEach((type) => {
  viewerStage.addEventListener(type, (e) => {
    e.preventDefault();
    e.stopPropagation();
    viewerStage.classList.add("droptarget");
  });
});
["dragleave", "drop"].forEach((type) => {
  viewerStage.addEventListener(type, (e) => {
    e.preventDefault();
    e.stopPropagation();
    viewerStage.classList.remove("droptarget");
  });
});
viewerStage.addEventListener("drop", (e) => {
  if (e.dataTransfer && e.dataTransfer.files.length) {
    loadViewerFiles(e.dataTransfer.files);
  }
});

/* ----- ビューアからのエクスポート ----- */

function viewerExportOptions() {
  const personsMode = $("viewer-export-persons").value;
  const persons = personsMode === "visible"
    ? new Set([...stage3d.visible])
    : new Set(viewerModel.personIds);
  const joints = [...$("viewer-export-joints").selectedOptions]
    .map((o) => Number(o.value));
  const start = (Number($("viewer-export-start").value) || 1) - 1;
  const end = (Number($("viewer-export-end").value) || viewerModel.frames.length) - 1;
  return {
    persons,
    joints,
    start,
    end,
    fps: Number(viewerFps.value) || viewerModel.fps || 30,
    applyView: $("viewer-export-transform").checked,
  };
}

$("viewer-export-joints-all").addEventListener("click", () => {
  [...$("viewer-export-joints").options].forEach((o) => { o.selected = true; });
});
$("viewer-export-joints-none").addEventListener("click", () => {
  [...$("viewer-export-joints").options].forEach((o) => { o.selected = false; });
});
$("viewer-export-btn").addEventListener("click", () => {
  if (!viewerModel) {
    alert("先にビューアへデータを読み込んでください。");
    return;
  }
  const o = viewerExportOptions();
  if (!o.joints.length) {
    alert("関節が選択されていません (「全選択」で戻せます)。");
    return;
  }
  if (!o.persons.size) {
    alert("人物が選択されていません。");
    return;
  }
  const spec = PoseLab3D.EXPORT_FORMATS[$("viewer-export-format").value];
  const col = PoseLab3D.collectExport(viewerModel, {
    ...o,
    points: o.applyView ? (frame) => stage3d.framePoints(frame) : null,
  });
  const note = o.applyView
    ? `display (axis=${stage3d.opts.axis}, y-up)` : "raw";
  // ↑ note はエクスポートデータの座標系メタ情報 (ファイル内に記録される技術値)
  const text = spec.build(col, viewerModel, note);
  const base = (viewerModel.name || "pose")
    .replace(/\.[^.]+$/, "")
    .replace(/[^\w\-一-龠ぁ-んァ-ヶ]+/g, "_") || "pose";
  const blob = new Blob([text], { type: `${spec.mime};charset=utf-8` });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${base}_export.${spec.ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  appendLog(
    `ビューアエクスポート: ${spec.label} (${col.records.length} フレーム × ${col.names.length} 関節)`);
});

document.addEventListener("DOMContentLoaded", () => {
  checkGPU();
  startEvents();
  applyAutoOutput();
  applyPreset(preset.value);
  applyBackend();
  schedulePreflight();
  setInterval(pollStatus, 2000);
});
