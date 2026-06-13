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
  if (runHint) runHint.textContent = isMediapipe ? "CPU 可" : "GPU required";
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
    item.textContent = `RUNNING: ${name}`;
    queueList.appendChild(item);
    count += 1;
  }
  queueState.forEach((job) => {
    const item = document.createElement("li");
    const name = job.input ? job.input.split(/[/\\\\]/).pop() : "unknown";
    item.textContent = `PENDING: ${name}`;
    const controls = document.createElement("div");
    controls.className = "queue-controls";
    const up = document.createElement("button");
    up.textContent = "Up";
    up.addEventListener("click", () => moveQueue(job, -1));
    const down = document.createElement("button");
    down.textContent = "Down";
    down.addEventListener("click", () => moveQueue(job, 1));
    controls.appendChild(up);
    controls.appendChild(down);
    item.appendChild(controls);
    queueList.appendChild(item);
  });
  queueMeta.textContent = `${count} items`;
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
  const label = hasQueue ? `Run Queue (${queueState.length})` : "Run Pipeline";
  runBtn.querySelector("span").textContent = label;
}

function updateChecks(warnings, info) {
  checkList.innerHTML = "";
  if (!warnings || warnings.length === 0) {
    const ok = document.createElement("li");
    ok.className = "ok";
    ok.textContent = "All checks passed.";
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
    if (info.frames) parts.push(`Frames: ${info.frames}`);
    if (info.fps) parts.push(`FPS: ${info.fps.toFixed(2)}`);
    if (info.duration) parts.push(`Duration: ${info.duration.toFixed(1)}s`);
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
    item.textContent = "No downloads required.";
    downloadsList.appendChild(item);
    downloadsMeta.textContent = "Ready";
    return;
  }

  const counts = { ready: 0, downloading: 0, pending: 0, failed: 0 };
  const labels = {
    ready: "Ready",
    downloading: "Downloading",
    pending: "Queued",
    failed: "Failed",
  };

  items.forEach((entry) => {
    const status = entry.status || "pending";
    if (counts[status] !== undefined) counts[status] += 1;
    const row = document.createElement("div");
    row.className = "download-row";

    const name = document.createElement("div");
    name.className = "download-name";
    name.textContent = entry.name || "unknown";

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
      meta.textContent = "Check logs for details.";
    } else if (status === "ready") {
      meta.textContent = "Available.";
    } else {
      meta.textContent = "Waiting for download.";
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
  if (counts.downloading) parts.push(`${counts.downloading} active`);
  if (counts.pending) parts.push(`${counts.pending} queued`);
  if (counts.failed) parts.push(`${counts.failed} failed`);
  if (parts.length === 0) parts.push(`${counts.ready} ready`);
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
  viewerFrameLabel.textContent = `Frame ${total ? stage3d.frame + 1 : 0} / ${total}`;
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
    option.textContent = `Person ${id}`;
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
    appendLog(`Viewer: ${sourceLabel} (${model.formatLabel}, ${model.frames.length} frames)`);
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
      appendLog(`Viewer: ${file.name} を読み込めませんでした (${err.message})`);
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
      previewSummary.textContent = `Frames: ${frames} | Avg persons: ${avg_instances} | Avg score: ${avg_score}`;
    }
  } catch {
    previewSummary.textContent = "";
  }
}

function renderHistory(items) {
  historyList.innerHTML = "";
  if (!items || items.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No history yet.";
    historyList.appendChild(item);
    return;
  }
  items.slice().reverse().forEach((entry) => {
    const item = document.createElement("li");
    const name = entry.input ? entry.input.split(/[/\\\\]/).pop() : "unknown";
    const status = entry.return_code === 0 ? "SUCCESS" : "FAILED";
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
      updateChecks(["Preflight check failed."], {});
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
    gpuStatus.textContent = data.available ? "Detected" : "Not Found";
    gpuStatus.style.color = data.available ? "var(--ok)" : "var(--warn)";
  } catch (err) {
    gpuStatus.textContent = "Unavailable";
    gpuStatus.style.color = "var(--warn)";
    serverStatus.textContent = "Disconnected";
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
    const msg = res.error || "Failed to enqueue jobs.";
    alert(msg);
    appendLog(`Error: ${msg}`);
  }
}

async function enqueueCurrent() {
  const payload = buildPayload();
  if (!payload.input) {
    alert("Input is required.");
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
    const msg = res.error || "Failed to enqueue job.";
    alert(msg);
    appendLog(`Error: ${msg}`);
  }
}

async function runPipeline() {
  if (queueState.length > 0) {
    outputList.innerHTML = "";
    logOutput.textContent = "";
    updateProgress(0, "Starting queue...");
    serverStatus.textContent = "Running";
    const res = await fetchJSON("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) {
      const msg = res.error || "Failed to start queue.";
      alert(msg);
      appendLog(`Error: ${msg}`);
    }
    return;
  }

  const payload = buildPayload();
  if (!payload.input) {
    alert("Input is required.");
    return;
  }
  if (!payload.output_root) {
    payload.output_root = deriveOutputRoot(payload.input);
  }

  lastOutputRoot = payload.output_root;
  outputList.innerHTML = "";
  logOutput.textContent = "";
  updateProgress(0, "Starting...");
  serverStatus.textContent = "Running";

  const res = await fetchJSON("/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const msg = res.error || "Failed to start job.";
    alert(msg);
    appendLog(`Error: ${msg}`);
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
          updateProgress(0, "Running...");
        } else {
          updateProgress(0, "Idle");
        }
      }
      if (data.running && data.current_job) {
        serverStatus.textContent = `Running ${data.current_job.index}/${data.current_job.total}`;
      } else {
        serverStatus.textContent = data.running ? "Running" : "Ready";
      }
      runBtn.disabled = !!data.running;
      renderQueue(data.queue || [], data.current_job);
      renderHistory(data.completed || []);
      if (Array.isArray(data.downloads)) {
        renderDownloads(data.downloads);
      }
      if (data.running === false && data.return_code !== null) {
        appendLog(`Process exited with code ${data.return_code}`);
      }
    }
  };
  source.onerror = () => {
    serverStatus.textContent = "Disconnected";
  };
}

async function pollStatus() {
  try {
    const data = await fetchJSON("/status");
    if (typeof data.progress === "number") {
      if (data.progress > 0) {
        updateProgress(data.progress, `${data.progress}%`);
      } else if (data.running) {
        updateProgress(0, "Running...");
      } else {
        updateProgress(0, "Idle");
      }
    }
    if (data.running && data.current_job) {
      serverStatus.textContent = `Running ${data.current_job.index}/${data.current_job.total}`;
    } else {
      serverStatus.textContent = data.running ? "Running" : "Ready";
    }
    runBtn.disabled = !!data.running;
    renderQueue(data.queue || [], data.current_job);
    renderHistory(data.completed || []);
    if (Array.isArray(data.downloads)) {
      renderDownloads(data.downloads);
    }
  } catch {
    serverStatus.textContent = "Disconnected";
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
    alert("No video files found in the folder.");
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
    appendLog(`Error: ${res.error}`);
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
    `Viewer export: ${spec.label} (${col.records.length} frames x ${col.names.length} joints)`);
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
