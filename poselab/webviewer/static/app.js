/* poselab 3D Viewer — アプリ配線 (UI / DOM)
   3D エンジン本体は engine.js (骨格カタログ / parseAny / PoseStage /
   エクスポート関数群)。index.html が engine.js → app.js の順に読み込む
   ため、エンジンのトップレベル宣言 (PoseStage, parseAny, clamp など) を
   ここからそのまま参照できる。
*/
"use strict";

const $ = (id) => document.getElementById(id);

const stage = new PoseStage($("stage-canvas"));
let model = null;
let playing = false;
let speed = 1;
let frameFloat = 0;
let lastTick = null;

const ui = {
  empty: $("empty-state"),
  drop: $("drop-overlay"),
  toast: $("toast"),
  fileInput: $("file-input"),
  formatBadge: $("format-badge"),
  fileName: $("file-name"),
  statFrames: $("stat-frames"),
  statPersons: $("stat-persons"),
  statJoints: $("stat-joints"),
  statDuration: $("stat-duration"),
  personChips: $("person-chips"),
  jointSelect: $("joint-select"),
  frameSlider: $("frame-slider"),
  frameLabel: $("frame-label"),
  fpsInput: $("fps-input"),
  speedSelect: $("speed-select"),
  btnPlay: $("btn-play"),
  iconPlay: $("icon-play"),
  iconPause: $("icon-pause"),
  btnLoop: $("btn-loop"),
  sidebar: $("sidebar"),
  brandTag: $("brand-tag"),
};

let toastTimer = null;
function toast(message, isError = false) {
  ui.toast.textContent = message;
  ui.toast.classList.toggle("error", isError);
  ui.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), isError ? 5200 : 2800);
}

function setPlaying(value) {
  playing = value && !!model;
  ui.iconPlay.hidden = playing;
  ui.iconPause.hidden = !playing;
  lastTick = null;
}

function setFrame(i, fromSlider = false) {
  if (!model) return;
  const max = model.frames.length - 1;
  const idx = clamp(Math.round(i), 0, max);
  stage.frame = idx;
  frameFloat = idx;
  stage.dirty = true;
  if (!fromSlider) ui.frameSlider.value = String(idx);
  const fps = Number(ui.fpsInput.value) || 30;
  const sec = (idx / fps).toFixed(2);
  ui.frameLabel.textContent = `${idx + 1} / ${max + 1} · ${sec}s`;
}

function applyModel(m) {
  model = m;
  const axis = m.defaultAxis || "ydown";
  $("axis-select").value = axis;
  stage.opts.axis = axis;
  stage.setModel(m);
  ui.empty.classList.add("hidden");
  ui.formatBadge.textContent = m.formatLabel;
  ui.fileName.textContent = m.name;
  ui.statFrames.textContent = String(m.frames.length);
  ui.statPersons.textContent = String(m.personIds.length);
  ui.statJoints.textContent = String(m.names.length);
  const fps = m.fps || 30;
  ui.fpsInput.value = String(Math.round(fps));
  ui.statDuration.textContent = `${(m.frames.length / fps).toFixed(1)}s`;

  // 人物チップ
  ui.personChips.innerHTML = "";
  m.personIds.forEach((id) => {
    const hue = PERSON_HUES[id % PERSON_HUES.length];
    const chip = document.createElement("button");
    chip.className = "chip on";
    chip.textContent = `P${id}`;
    chip.style.setProperty("--chip-color", `hsl(${hue}, 85%, 62%)`);
    chip.addEventListener("click", () => {
      if (stage.visible.has(id)) stage.visible.delete(id);
      else stage.visible.add(id);
      chip.classList.toggle("on", stage.visible.has(id));
      stage.dirty = true;
      updateExportCount();
    });
    ui.personChips.appendChild(chip);
  });

  // 関節セレクト
  ui.jointSelect.innerHTML = '<option value="">なし</option>';
  m.names.forEach((n, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = n;
    ui.jointSelect.appendChild(option);
  });
  const wrist = findJoint(m.names, ["right_wrist", "left_wrist"]);
  ui.jointSelect.value = wrist >= 0 ? String(wrist) : "";
  stage.opts.highlight = wrist;

  // エクスポートパネル
  const exportJoints = $("export-joints");
  exportJoints.innerHTML = "";
  m.names.forEach((n, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${i}: ${n}`;
    option.selected = true;
    exportJoints.appendChild(option);
  });
  $("export-start").value = "1";
  $("export-start").max = String(m.frames.length);
  $("export-end").value = String(m.frames.length);
  $("export-end").max = String(m.frames.length);
  updateExportCount();

  ui.frameSlider.max = String(m.frames.length - 1);
  setFrame(0);
  setPlaying(true);
  toast(`読み込み: ${m.name} — ${m.formatLabel} (${m.frames.length} フレーム)`);
}

/* ----- ファイル読み込み ----- */

async function loadFiles(files) {
  const list = [...files].filter((f) =>
    /\.(json|csv)$/i.test(f.name) || /json|csv/.test(f.type));
  if (!list.length) {
    toast("JSON / CSV ファイルをドロップしてください", true);
    return;
  }
  // JSON を優先して最初に読めたものを表示
  list.sort((a, b) => Number(/\.csv$/i.test(a.name)) - Number(/\.csv$/i.test(b.name)));
  const errors = [];
  for (const file of list) {
    try {
      const text = await file.text();
      applyModel(parseAny(text, file.name));
      if (list.length > 1) {
        toast(`${file.name} を表示中 (${list.length} ファイル中、最初に読めたもの)`);
      }
      return;
    } catch (err) {
      errors.push(`${file.name}: ${err.message}`);
    }
  }
  toast(`読み込めませんでした — ${errors[0]}`, true);
}

async function loadPreloads() {
  try {
    const res = await fetch("manifest.json", { cache: "no-store" });
    if (!res.ok) return false;
    const manifest = await res.json();
    if (!manifest.files || !manifest.files.length) {
      ui.brandTag.textContent = "local server";
      return false;
    }
    ui.brandTag.textContent = "local server";
    const sorted = [...manifest.files].sort(
      (a, b) => Number(/\.csv$/i.test(a.name)) - Number(/\.csv$/i.test(b.name)));
    for (const entry of sorted) {
      try {
        const r = await fetch(`data/${entry.index}`, { cache: "no-store" });
        if (!r.ok) continue;
        applyModel(parseAny(await r.text(), entry.name));
        return true;
      } catch { /* 次のファイルへ */ }
    }
  } catch { /* 静的ホスティング時は manifest なし */ }
  return false;
}

/* ----- UI 配線 ----- */

["btn-open", "btn-open-2"].forEach((id) => {
  $(id).addEventListener("click", () => ui.fileInput.click());
});
ui.fileInput.addEventListener("change", () => {
  if (ui.fileInput.files.length) loadFiles(ui.fileInput.files);
  ui.fileInput.value = "";
});
["btn-demo", "btn-demo-2"].forEach((id) => {
  $(id).addEventListener("click", () => applyModel(demoModel()));
});
$("btn-snapshot").addEventListener("click", () => {
  if (!model) { toast("先にデータを読み込んでください", true); return; }
  const a = document.createElement("a");
  a.href = stage.snapshot();
  a.download = `poselab_frame_${String(stage.frame + 1).padStart(4, "0")}.png`;
  a.click();
  toast("PNG を保存しました");
});
$("btn-sidebar").addEventListener("click", () => ui.sidebar.classList.toggle("open"));

document.querySelectorAll(".hud-views [data-view]").forEach((btn) => {
  btn.addEventListener("click", () => stage.setView(btn.dataset.view));
});
$("btn-fit").addEventListener("click", () => stage.fit());
$("btn-reset-view").addEventListener("click", () => stage.resetView());

$("axis-select").addEventListener("change", (e) => {
  stage.opts.axis = e.target.value;
  stage.fit();
});
[["opt-center", "center"], ["opt-normalize", "normalize"], ["opt-grid", "grid"],
 ["opt-labels", "labels"], ["opt-trail", "trail"]].forEach(([id, key]) => {
  $(id).addEventListener("change", (e) => {
    stage.opts[key] = e.target.checked;
    if (key === "center" || key === "normalize") stage.fit();
    stage.dirty = true;
  });
});
$("opt-bone").addEventListener("input", (e) => {
  stage.opts.bone = Number(e.target.value);
  stage.dirty = true;
});
$("opt-trail-len").addEventListener("input", (e) => {
  stage.opts.trailLen = Number(e.target.value);
  stage.dirty = true;
});
ui.jointSelect.addEventListener("change", () => {
  stage.opts.highlight = ui.jointSelect.value === "" ? -1 : Number(ui.jointSelect.value);
  stage.dirty = true;
});

/* ----- エクスポート ----- */

function exportSelectionOptions() {
  const personsMode = $("export-persons").value;
  const persons = personsMode === "visible"
    ? new Set([...stage.visible])
    : new Set(model.personIds);
  const joints = [...$("export-joints").selectedOptions].map((o) => Number(o.value));
  const start = (Number($("export-start").value) || 1) - 1;
  const end = (Number($("export-end").value) || model.frames.length) - 1;
  return {
    persons,
    joints,
    start,
    end,
    fps: Number(ui.fpsInput.value) || model.fps || 30,
    applyView: $("export-transform").checked,
  };
}

function updateExportCount() {
  const badge = $("export-count");
  if (!model) {
    badge.textContent = "—";
    return;
  }
  const o = exportSelectionOptions();
  const last = model.frames.length - 1;
  const frames = clamp(o.end, 0, last) - clamp(o.start, 0, last) + 1;
  badge.textContent = `${Math.max(0, frames)}f × ${o.persons.size}人 × ${o.joints.length}関節`;
}

["export-format", "export-persons", "export-start", "export-end", "export-joints"]
  .forEach((id) => $(id).addEventListener("change", updateExportCount));
$("export-joints-all").addEventListener("click", () => {
  [...$("export-joints").options].forEach((o) => { o.selected = true; });
  updateExportCount();
});
$("export-joints-none").addEventListener("click", () => {
  [...$("export-joints").options].forEach((o) => { o.selected = false; });
  updateExportCount();
});
$("btn-export").addEventListener("click", () => {
  if (!model) {
    toast("先にデータを読み込んでください", true);
    return;
  }
  const o = exportSelectionOptions();
  if (!o.joints.length) {
    toast("関節が選択されていません (「全選択」で戻せます)", true);
    return;
  }
  if (!o.persons.size) {
    toast("人物が選択されていません", true);
    return;
  }
  const spec = EXPORT_FORMATS[$("export-format").value];
  const col = collectExport(model, {
    ...o,
    points: o.applyView ? (frame) => stage.framePoints(frame) : null,
  });
  const note = o.applyView ? `display (axis=${stage.opts.axis}, y-up)` : "raw";
  const text = spec.build(col, model, note);
  const base = (model.name || "pose")
    .replace(/\.[^.]+$/, "")
    .replace(/[^\w\-一-龠ぁ-んァ-ヶ]+/g, "_") || "pose";
  const blob = new Blob([text], { type: `${spec.mime};charset=utf-8` });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${base}_export.${spec.ext}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  toast(`${spec.label} を書き出しました (${col.records.length} フレーム × ${col.names.length} 関節)`);
});

ui.btnPlay.addEventListener("click", () => setPlaying(!playing));
ui.frameSlider.addEventListener("input", () => {
  setPlaying(false);
  setFrame(Number(ui.frameSlider.value), true);
});
ui.speedSelect.addEventListener("change", () => { speed = Number(ui.speedSelect.value); });
ui.fpsInput.addEventListener("change", () => setFrame(stage.frame));
ui.btnLoop.addEventListener("click", () => ui.btnLoop.classList.toggle("on"));

/* ドラッグ & ドロップ */
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  e.preventDefault();
  dragDepth += 1;
  ui.drop.classList.add("active");
});
window.addEventListener("dragleave", (e) => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) ui.drop.classList.remove("active");
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  ui.drop.classList.remove("active");
  if (e.dataTransfer && e.dataTransfer.files.length) loadFiles(e.dataTransfer.files);
});

/* キーボード */
window.addEventListener("keydown", (e) => {
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const step = e.shiftKey ? 10 : 1;
  switch (e.key) {
    case " ": e.preventDefault(); setPlaying(!playing); break;
    case "ArrowLeft": setPlaying(false); setFrame(stage.frame - step); break;
    case "ArrowRight": setPlaying(false); setFrame(stage.frame + step); break;
    case "Home": setPlaying(false); setFrame(0); break;
    case "End": setPlaying(false); setFrame(model ? model.frames.length - 1 : 0); break;
    case "f": case "F": stage.fit(); break;
    case "r": case "R": stage.resetView(); break;
    case "g": case "G": {
      const el = $("opt-grid");
      el.checked = !el.checked;
      stage.opts.grid = el.checked;
      stage.dirty = true;
      break;
    }
    case "l": case "L": {
      const el = $("opt-labels");
      el.checked = !el.checked;
      stage.opts.labels = el.checked;
      stage.dirty = true;
      break;
    }
    case "1": stage.setView("front"); break;
    case "2": stage.setView("side"); break;
    case "3": stage.setView("top"); break;
    default: return;
  }
});

/* ----- メインループ ----- */
function tick(ts) {
  if (playing && model) {
    if (lastTick != null) {
      const dt = (ts - lastTick) / 1000;
      const fps = Number(ui.fpsInput.value) || 30;
      frameFloat += dt * fps * speed;
      const max = model.frames.length - 1;
      if (frameFloat > max) {
        if (ui.btnLoop.classList.contains("on")) frameFloat = 0;
        else { frameFloat = max; setPlaying(false); }
      }
      if (Math.round(frameFloat) !== stage.frame) {
        const idx = Math.round(frameFloat);
        stage.frame = idx;
        ui.frameSlider.value = String(idx);
        const fps2 = Number(ui.fpsInput.value) || 30;
        ui.frameLabel.textContent = `${idx + 1} / ${max + 1} · ${(idx / fps2).toFixed(2)}s`;
        stage.dirty = true;
      }
    }
    lastTick = ts;
  }
  if (stage.dirty) {
    stage.dirty = false;
    stage.render();
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* ----- 起動 ----- */
(async () => {
  const params = new URLSearchParams(location.search);
  const preloaded = await loadPreloads();
  if (!preloaded && params.has("demo")) {
    applyModel(demoModel());
  }
})();
