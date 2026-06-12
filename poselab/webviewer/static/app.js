/* poselab 3D Viewer
   依存ライブラリゼロの Canvas 2D ベース 3D ポーズビューア。
   対応形式:
   - poselab JSON   ({metadata, frames})
   - poselab CSV    (ロング / ワイド)
   - MMPose 系 JSON ({meta_info, instance_info}) — Pose3DStudio 等
   - 汎用ワイド/ロング CSV (名前_x/_y/_z 列、frame/keypoint 列)
*/
"use strict";

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* ================================================================
   骨格カタログ
   ================================================================ */

// MediaPipe Pose 33 ランドマーク (poselab 標準)
const MP_NAMES = [
  "nose", "left_eye_inner", "left_eye", "left_eye_outer",
  "right_eye_inner", "right_eye", "right_eye_outer",
  "left_ear", "right_ear", "mouth_left", "mouth_right",
  "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
  "left_wrist", "right_wrist", "left_pinky", "right_pinky",
  "left_index", "right_index", "left_thumb", "right_thumb",
  "left_hip", "right_hip", "left_knee", "right_knee",
  "left_ankle", "right_ankle", "left_heel", "right_heel",
  "left_foot_index", "right_foot_index",
];
const MP_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [23, 25], [25, 27], [27, 29], [29, 31], [27, 31],
  [24, 26], [26, 28], [28, 30], [30, 32], [28, 32],
];

// Human3.6M 17 関節 (MMPose 3D リフタ標準) — リンク欠落時のフォールバック
const H36M_EDGES_BY_NAME = [
  ["root", "right_hip"], ["right_hip", "right_knee"], ["right_knee", "right_foot"],
  ["root", "left_hip"], ["left_hip", "left_knee"], ["left_knee", "left_foot"],
  ["root", "spine"], ["spine", "thorax"], ["thorax", "neck_base"], ["neck_base", "head"],
  ["thorax", "left_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
  ["thorax", "right_shoulder"], ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
];

function sideOf(name) {
  const n = String(name).toLowerCase();
  if (n.startsWith("left") || n.startsWith("l_") || n.endsWith("_left") || /^l[A-Z]/.test(name)) return "left";
  if (n.startsWith("right") || n.startsWith("r_") || n.endsWith("_right") || /^r[A-Z]/.test(name)) return "right";
  return "center";
}

function edgesForNames(names) {
  const lower = names.map((n) => String(n).toLowerCase());
  const index = new Map(lower.map((n, i) => [n, i]));
  // MediaPipe 33 のサブセットなら名前ベースで張る
  const mpHits = lower.filter((n) => MP_NAMES.includes(n)).length;
  if (mpHits >= Math.min(10, names.length)) {
    const edges = [];
    MP_EDGES.forEach(([a, b]) => {
      const ia = index.get(MP_NAMES[a]);
      const ib = index.get(MP_NAMES[b]);
      if (ia !== undefined && ib !== undefined) edges.push([ia, ib]);
    });
    if (edges.length) return edges;
  }
  // H36M 名
  const edges = [];
  H36M_EDGES_BY_NAME.forEach(([a, b]) => {
    const ia = index.get(a);
    const ib = index.get(b);
    if (ia !== undefined && ib !== undefined) edges.push([ia, ib]);
  });
  return edges;
}

function findJoint(names, candidates) {
  const lower = names.map((n) => String(n).toLowerCase());
  for (const c of candidates) {
    const i = lower.indexOf(c);
    if (i >= 0) return i;
  }
  return -1;
}

/* ================================================================
   パーサ — すべて統一モデルへ変換
   model = { name, formatLabel, names[], edges[][], fps|null,
             frames: [{ t|null, persons: [{id, pts:Float64Array(J*3), ok:Uint8Array(J)}] }],
             personIds[] }
   ================================================================ */

function finalizeModel(model) {
  const ids = new Set();
  model.frames.forEach((f) => f.persons.forEach((p) => ids.add(p.id)));
  model.personIds = [...ids].sort((a, b) => a - b);
  if (!model.frames.length) throw new Error("フレームが 1 つもありません");
  if (!model.names.length) throw new Error("キーポイントがありません");
  if (model.fps == null) {
    const ts = model.frames.map((f) => f.t).filter((t) => t != null);
    if (ts.length > 5) {
      const diffs = [];
      for (let i = 1; i < ts.length; i += 1) {
        const d = ts[i] - ts[i - 1];
        if (d > 1e-6) diffs.push(d);
      }
      diffs.sort((a, b) => a - b);
      const med = diffs[Math.floor(diffs.length / 2)];
      if (med) model.fps = Math.round(clamp(1 / med, 1, 240) * 10) / 10;
    }
  }
  return model;
}

function parsePoselabJson(obj, name) {
  const meta = obj.metadata || {};
  const rawFrames = obj.frames || [];
  let names = meta.keypoint_names || null;
  // world 座標があるか先に調べる
  let hasWorld = false;
  for (const f of rawFrames) {
    for (const p of f.persons || []) {
      if (p.world_keypoints && p.world_keypoints.length) { hasWorld = true; break; }
    }
    if (hasWorld) break;
  }
  if (!names) {
    const f0 = rawFrames.find((f) => (f.persons || []).length);
    const kps = f0 ? (hasWorld ? f0.persons[0].world_keypoints : f0.persons[0].keypoints) : [];
    names = kps.map((k, i) => k.name || `kpt_${i}`);
  }
  const J = names.length;
  const frames = rawFrames.map((f) => ({
    t: f.timestamp_ms != null ? f.timestamp_ms / 1000 : null,
    persons: (f.persons || []).map((p) => {
      const pts = new Float64Array(J * 3);
      const ok = new Uint8Array(J);
      const src = hasWorld ? p.world_keypoints : p.keypoints;
      (src || []).forEach((k) => {
        const i = k.id != null ? k.id : k.index;
        if (i == null || i < 0 || i >= J) return;
        if (hasWorld) {
          pts[i * 3] = k.x; pts[i * 3 + 1] = k.y; pts[i * 3 + 2] = k.z;
        } else {
          pts[i * 3] = k.x_px; pts[i * 3 + 1] = k.y_px; pts[i * 3 + 2] = 0;
        }
        ok[i] = (k.visibility == null || k.visibility > 0.2) ? 1 : 0;
      });
      return { id: p.person ?? 0, pts, ok };
    }),
  }));
  return finalizeModel({
    name,
    formatLabel: hasWorld ? "poselab JSON (world)" : "poselab JSON (2D px)",
    names: names.map(String),
    edges: edgesForNames(names),
    fps: null,
    frames,
  });
}

function parseMmposeJson(obj, name) {
  const meta = obj.meta_info || {};
  const seq = obj.instance_info || [];
  const J = meta.num_keypoints || 0;
  const id2name = meta.keypoint_id2name || {};
  const names = [];
  for (let i = 0; i < J; i += 1) names.push(String(id2name[i] ?? id2name[String(i)] ?? `kpt_${i}`));
  const edges = (meta.skeleton_links || []).map(([a, b]) => [a, b]);
  const frames = seq.map((entry) => ({
    t: null,
    persons: (entry.instances || []).map((inst, idx) => {
      const kps = inst.keypoints || [];
      const scores = inst.keypoint_scores || [];
      const n = Math.max(J, kps.length);
      const pts = new Float64Array(n * 3);
      const ok = new Uint8Array(n);
      kps.forEach((k, i) => {
        pts[i * 3] = k[0]; pts[i * 3 + 1] = k[1]; pts[i * 3 + 2] = k[2] ?? 0;
        ok[i] = scores[i] == null || scores[i] > 0.2 ? 1 : 0;
      });
      return { id: idx, pts, ok };
    }),
  }));
  const isH36m = names.some((n) => String(n).toLowerCase() === "root");
  const label = names.length === 17 ? "MMPose JSON (H36M 3D)" : "MMPose JSON";
  return finalizeModel({
    name,
    formatLabel: label,
    names: names.length ? names : (frames[0]?.persons[0] ? frames[0].persons[0].pts.length / 3 : 0),
    edges: edges.length ? edges : edgesForNames(names),
    fps: null,
    frames,
    // MMPose の 3D リフタ出力は z-up (x 反転・床基準) 規約
    defaultAxis: isH36m ? "zup" : "ydown",
  });
}

/* ---------- CSV ---------- */

function splitCsvLine(line) {
  if (!line.includes('"')) return line.split(",");
  const out = [];
  let cur = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i += 1; } else quoted = false;
      } else cur += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

function parseCsv(text, name) {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim().length);
  if (lines.length < 2) throw new Error("CSV にデータ行がありません");
  const header = splitCsvLine(lines[0]).map((h) => h.trim());
  const col = new Map(header.map((h, i) => [h.toLowerCase(), i]));
  const rows = lines.slice(1).map(splitCsvLine);

  if (col.has("keypoint_name") && col.has("world_x")) {
    return parseLongCsv(header, rows, name, "poselab CSV (ロング)");
  }
  if (header.some((h) => h.toLowerCase().endsWith("_world_x"))) {
    return parseWideCsv(header, rows, name, true, "poselab CSV (ワイド)");
  }
  const jointCol = ["keypoint_name", "keypoint", "joint", "name", "kpt"].find((c) => col.has(c));
  if (jointCol && (col.has("x") || col.has("x_px"))) {
    return parseLongCsv(header, rows, name, "ロング CSV");
  }
  if (header.some((h) => /_x$/i.test(h)) && header.some((h) => /_y$/i.test(h))) {
    return parseWideCsv(header, rows, name, false, "ワイド CSV");
  }
  throw new Error("CSV の列構成を認識できませんでした");
}

function parseLongCsv(header, rows, name, label) {
  const col = new Map(header.map((h, i) => [h.toLowerCase(), i]));
  const get = (row, key) => {
    const i = col.get(key);
    return i == null ? "" : row[i];
  };
  const jointKey = ["keypoint_name", "keypoint", "joint", "name", "kpt"].find((c) => col.has(c));
  const hasWorld = col.has("world_x");
  const useWorld = hasWorld && rows.some((r) => get(r, "world_x") !== "");
  const xKey = useWorld ? "world_x" : (col.has("x_px") ? "x_px" : "x");
  const yKey = useWorld ? "world_y" : (col.has("y_px") ? "y_px" : "y");
  const zKey = useWorld ? "world_z" : (col.has("z") ? "z" : null);

  const namesSet = [];
  const nameIndex = new Map();
  const frameMap = new Map();
  rows.forEach((row) => {
    const jname = get(row, jointKey);
    if (!jname) return;
    if (!nameIndex.has(jname)) {
      nameIndex.set(jname, namesSet.length);
      namesSet.push(jname);
    }
    const f = Number(get(row, "frame") || 0);
    const pid = Number(get(row, "person") || 0);
    const key = f;
    if (!frameMap.has(key)) frameMap.set(key, { t: null, persons: new Map() });
    const frame = frameMap.get(key);
    const ts = get(row, "timestamp_ms");
    if (ts !== "") frame.t = Number(ts) / 1000;
    if (!frame.persons.has(pid)) frame.persons.set(pid, []);
    frame.persons.get(pid).push({
      j: nameIndex.get(jname),
      x: Number(get(row, xKey)),
      y: Number(get(row, yKey)),
      z: zKey ? Number(get(row, zKey)) : 0,
    });
  });

  const J = namesSet.length;
  const frameKeys = [...frameMap.keys()].sort((a, b) => a - b);
  // 2D px フォールバック時の z スケール推定 (x の広がりに合わせる)
  const frames = frameKeys.map((k) => {
    const src = frameMap.get(k);
    return {
      t: src.t,
      persons: [...src.persons.entries()].map(([pid, list]) => {
        const pts = new Float64Array(J * 3);
        const ok = new Uint8Array(J);
        list.forEach((e) => {
          if (Number.isFinite(e.x) && Number.isFinite(e.y)) {
            pts[e.j * 3] = e.x; pts[e.j * 3 + 1] = e.y; pts[e.j * 3 + 2] = Number.isFinite(e.z) ? e.z : 0;
            ok[e.j] = 1;
          }
        });
        return { id: pid, pts, ok };
      }),
    };
  });
  return finalizeModel({
    name,
    formatLabel: label + (useWorld ? " (world)" : ""),
    names: namesSet,
    edges: edgesForNames(namesSet),
    fps: null,
    frames,
  });
}

function parseWideCsv(header, rows, name, poselabWorld, label) {
  const lower = header.map((h) => h.toLowerCase());
  const col = new Map(lower.map((h, i) => [h, i]));
  // 関節名の抽出
  const joints = [];
  const seen = new Set();
  if (poselabWorld) {
    lower.forEach((h) => {
      const m = h.match(/^(.+)_world_x$/);
      if (m && !seen.has(m[1])) { seen.add(m[1]); joints.push(m[1]); }
    });
  } else {
    lower.forEach((h) => {
      const m = h.match(/^(.+)_x$/);
      if (!m) return;
      const base = m[1];
      if (base.endsWith("_world")) return;
      if (!col.has(`${base}_y`)) return;
      if (seen.has(base)) return;
      seen.add(base);
      joints.push(base);
    });
  }
  if (!joints.length) throw new Error("ワイド CSV から関節列を見つけられませんでした");
  const J = joints.length;
  const cx = joints.map((j) => col.get(poselabWorld ? `${j}_world_x` : `${j}_x`));
  const cy = joints.map((j) => col.get(poselabWorld ? `${j}_world_y` : `${j}_y`));
  const cz = joints.map((j) => {
    const key = poselabWorld ? `${j}_world_z` : `${j}_z`;
    return col.has(key) ? col.get(key) : -1;
  });
  const fCol = col.has("frame") ? col.get("frame") : -1;
  const pCol = col.has("person") ? col.get("person") : -1;
  const tCol = col.has("timestamp_ms") ? col.get("timestamp_ms") : -1;

  const frameMap = new Map();
  rows.forEach((row, ri) => {
    const f = fCol >= 0 ? Number(row[fCol]) : ri;
    const pid = pCol >= 0 ? Number(row[pCol] || 0) : 0;
    if (!frameMap.has(f)) frameMap.set(f, { t: null, persons: [] });
    const frame = frameMap.get(f);
    if (tCol >= 0 && row[tCol] !== "") frame.t = Number(row[tCol]) / 1000;
    const pts = new Float64Array(J * 3);
    const ok = new Uint8Array(J);
    for (let j = 0; j < J; j += 1) {
      const x = Number(row[cx[j]]);
      const y = Number(row[cy[j]]);
      const z = cz[j] >= 0 ? Number(row[cz[j]]) : 0;
      if (Number.isFinite(x) && Number.isFinite(y) && row[cx[j]] !== "") {
        pts[j * 3] = x; pts[j * 3 + 1] = y; pts[j * 3 + 2] = Number.isFinite(z) ? z : 0;
        ok[j] = 1;
      }
    }
    frame.persons.push({ id: pid, pts, ok });
  });
  const frameKeys = [...frameMap.keys()].sort((a, b) => a - b);
  const frames = frameKeys.map((k) => frameMap.get(k));
  return finalizeModel({
    name,
    formatLabel: label,
    names: joints,
    edges: edgesForNames(joints),
    fps: null,
    frames,
  });
}

function parseAny(text, fileName) {
  const trimmed = text.replace(/^﻿/, "").trimStart();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    const obj = JSON.parse(trimmed);
    if (obj && obj.meta_info && obj.instance_info) return parseMmposeJson(obj, fileName);
    if (obj && obj.frames) return parsePoselabJson(obj, fileName);
    throw new Error("JSON の形式を認識できませんでした (meta_info / frames がありません)");
  }
  return parseCsv(text, fileName);
}

/* ================================================================
   デモデータ — 円周を歩く 2 人 (MediaPipe 33 関節、手続き生成)
   ================================================================ */

function demoModel() {
  const J = MP_NAMES.length;
  const idx = new Map(MP_NAMES.map((n, i) => [n, i]));
  const FRAMES = 360;
  const FPS = 30;

  function walker(tNorm, phase0, radius, dir) {
    const pts = new Float64Array(J * 3);
    const ok = new Uint8Array(J).fill(1);
    const set = (name, x, y, z) => {
      const i = idx.get(name);
      pts[i * 3] = x; pts[i * 3 + 1] = -y; pts[i * 3 + 2] = -z; // y-down 形式で格納
    };
    const ang = dir * tNorm * Math.PI * 2 + phase0;
    const cx = Math.cos(ang) * radius;
    const cz = Math.sin(ang) * radius;
    const heading = ang + (dir > 0 ? Math.PI / 2 : -Math.PI / 2);
    const hx = Math.cos(heading);
    const hz = Math.sin(heading);
    // 横方向 (体の左方向)
    const lx = -hz;
    const lz = hx;
    const phi = tNorm * Math.PI * 2 * 14 + phase0 * 3; // 歩行位相
    const bob = 0.035 * Math.sin(phi * 2);
    const sway = 0.02 * Math.sin(phi);
    const hipY = 0.94 + bob;

    const pos = (fwd, side, y) => [
      cx + hx * fwd + lx * (side + sway), y, cz + hz * fwd + lz * (side + sway),
    ];

    // 脚: 矢状面チェーン
    function leg(sign, phase) {
      const swing = Math.sin(phase) * 0.42;
      const lift = Math.max(0, Math.sin(phase + Math.PI / 2)) * 0.18;
      const hip = pos(0, sign * 0.105, hipY);
      const kneeFwd = Math.sin(phase) * 0.24;
      const kneeY = hipY - 0.42 + lift * 0.4;
      const knee = pos(kneeFwd, sign * 0.108, kneeY);
      const ankFwd = swing * 0.62;
      const ankY = 0.08 + lift;
      const ankle = pos(ankFwd, sign * 0.11, ankY);
      const heel = pos(ankFwd - 0.06, sign * 0.11, ankY - 0.045 + 0.02);
      const toe = pos(ankFwd + 0.13, sign * 0.105, Math.max(0.02, ankY - 0.055));
      return { hip, knee, ankle, heel, toe };
    }
    const L = leg(+1, phi);
    const R = leg(-1, phi + Math.PI);

    // 体幹・頭
    const sh = (sign, phase) => {
      const armFwd = Math.sin(phase) * 0.16;
      const shoulder = pos(0, sign * 0.18, 1.43);
      const elbow = pos(armFwd * 0.8, sign * 0.21, 1.16);
      const wristFwd = armFwd * 1.7;
      const wrist = pos(wristFwd, sign * 0.21, 0.93 + Math.abs(armFwd) * 0.12);
      return { shoulder, elbow, wrist, wristFwd, sign };
    };
    const SL = sh(+1, phi + Math.PI); // 腕は脚と逆位相
    const SR = sh(-1, phi);

    set("left_hip", ...L.hip); set("right_hip", ...R.hip);
    set("left_knee", ...L.knee); set("right_knee", ...R.knee);
    set("left_ankle", ...L.ankle); set("right_ankle", ...R.ankle);
    set("left_heel", ...L.heel); set("right_heel", ...R.heel);
    set("left_foot_index", ...L.toe); set("right_foot_index", ...R.toe);
    set("left_shoulder", ...SL.shoulder); set("right_shoulder", ...SR.shoulder);
    set("left_elbow", ...SL.elbow); set("right_elbow", ...SR.elbow);
    set("left_wrist", ...SL.wrist); set("right_wrist", ...SR.wrist);
    [["pinky", 0.085], ["index", 0.1], ["thumb", 0.05]].forEach(([part, len]) => {
      [SL, SR].forEach((S) => {
        const side = S === SL ? "left" : "right";
        const w = S.wrist;
        const p = pos(S.wristFwd + len, S.sign * 0.215, w[1] - 0.03);
        set(`${side}_${part}`, ...p);
      });
    });
    const headBase = pos(0.01, 0, 1.62);
    set("nose", ...pos(0.085, 0, 1.66));
    set("left_ear", ...pos(0, 0.085, 1.65));
    set("right_ear", ...pos(0, -0.085, 1.65));
    set("left_eye", ...pos(0.07, 0.035, 1.69));
    set("right_eye", ...pos(0.07, -0.035, 1.69));
    set("left_eye_inner", ...pos(0.075, 0.018, 1.69));
    set("right_eye_inner", ...pos(0.075, -0.018, 1.69));
    set("left_eye_outer", ...pos(0.06, 0.052, 1.685));
    set("right_eye_outer", ...pos(0.06, -0.052, 1.685));
    set("mouth_left", ...pos(0.075, 0.024, 1.6));
    set("mouth_right", ...pos(0.075, -0.024, 1.6));
    void headBase;
    return { pts, ok };
  }

  const frames = [];
  for (let f = 0; f < FRAMES; f += 1) {
    const tNorm = f / FRAMES;
    frames.push({
      t: f / FPS,
      persons: [
        { id: 0, ...walker(tNorm, 0, 1.15, +1) },
        { id: 1, ...walker(tNorm, Math.PI, 1.15, +1) },
      ],
    });
  }
  return finalizeModel({
    name: "demo_walkers (合成データ)",
    formatLabel: "デモ (合成歩行)",
    names: [...MP_NAMES],
    edges: MP_EDGES.map((e) => [...e]),
    fps: FPS,
    frames,
  });
}

/* ================================================================
   レンダラ
   ================================================================ */

const PERSON_HUES = [187, 268, 22, 132, 330, 56, 200, 290];

class PoseStage {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.model = null;
    this.frame = 0;
    this.visible = new Set();
    this.cam = { yaw: -32, pitch: 12, dist: 6, tx: 0, ty: 0.0, tz: 0 };
    this.home = { ...this.cam };
    this.opts = {
      axis: "ydown",
      center: true,
      normalize: false,
      grid: true,
      labels: false,
      bone: 3.5,
      highlight: -1,
      trail: true,
      trailLen: 90,
    };
    this.floorY = 0;
    this.gridStep = 0.5;
    this.dirty = true;
    this.onCameraChange = null; // 外部 UI (スライダー等) との同期用フック
    this._bindPointer();
    const ro = new ResizeObserver(() => this._resize());
    ro.observe(canvas.parentElement);
    this._resize();
  }

  /* ----- データ ----- */
  setModel(model) {
    this.model = model;
    this.frame = 0;
    this.visible = new Set(model.personIds);
    this._roots = this._findRoots();
    this.fit();
  }

  _findRoots() {
    if (!this.model) return { root: -1, upper: -1 };
    const root = findJoint(this.model.names, ["root", "pelvis", "hip", "mid_hip"]);
    const upper = findJoint(this.model.names, ["thorax", "neck", "neck_base", "spine", "chest"]);
    return {
      root,
      upper,
      lhip: findJoint(this.model.names, ["left_hip"]),
      rhip: findJoint(this.model.names, ["right_hip"]),
      lsho: findJoint(this.model.names, ["left_shoulder"]),
      rsho: findJoint(this.model.names, ["right_shoulder"]),
    };
  }

  /* raw → 表示座標 (y-up) */
  _disp(x, y, z) {
    if (this.opts.axis === "ydown") return [x, -y, -z];
    if (this.opts.axis === "zup") return [x, z, y]; // MMPose 3D (z=高さ, y=奥行き)
    return [x, y, z];
  }

  // 1 人物分の腰位置 (表示座標系)。なければ有効点の重心
  _rootOf(out, ok) {
    const J = this.model.names.length;
    const R = this._roots;
    const mid = (a, b) => [
      (out[a * 3] + out[b * 3]) / 2,
      (out[a * 3 + 1] + out[b * 3 + 1]) / 2,
      (out[a * 3 + 2] + out[b * 3 + 2]) / 2,
    ];
    if (R.root >= 0 && ok[R.root]) {
      return [out[R.root * 3], out[R.root * 3 + 1], out[R.root * 3 + 2]];
    }
    if (R.lhip >= 0 && R.rhip >= 0 && ok[R.lhip] && ok[R.rhip]) {
      return mid(R.lhip, R.rhip);
    }
    let x = 0, y = 0, z = 0, n = 0;
    for (let j = 0; j < J; j += 1) {
      if (!ok[j]) continue;
      x += out[j * 3]; y += out[j * 3 + 1]; z += out[j * 3 + 2]; n += 1;
    }
    return n ? [x / n, y / n, z / n] : null;
  }

  // 1 フレーム全員分の表示座標 (center/normalize はシーン単位で適用 —
  // 複数人のときに相対位置を保ったまま全体を原点へ寄せる)
  framePoints(frame) {
    const J = this.model.names.length;
    const R = this._roots;
    const list = frame.persons.map((person) => {
      const out = new Float64Array(J * 3);
      for (let j = 0; j < J; j += 1) {
        const [x, y, z] = this._disp(
          person.pts[j * 3], person.pts[j * 3 + 1], person.pts[j * 3 + 2]);
        out[j * 3] = x; out[j * 3 + 1] = y; out[j * 3 + 2] = z;
      }
      return { person, out, root: this._rootOf(out, person.ok) };
    });
    if (!list.length) return list;

    // シーン基準点 = 各人物の腰の平均
    let bx = 0, by = 0, bz = 0, n = 0;
    list.forEach(({ root }) => {
      if (root) { bx += root[0]; by += root[1]; bz += root[2]; n += 1; }
    });
    if (!n) return list;
    bx /= n; by /= n; bz /= n;

    if (this.opts.center) {
      list.forEach(({ out }) => {
        for (let j = 0; j < J; j += 1) {
          out[j * 3] -= bx; out[j * 3 + 1] -= by; out[j * 3 + 2] -= bz;
        }
      });
    }
    if (this.opts.normalize) {
      // スケール = 最初の人物の 腰→胸 距離 (シーン全体へ適用)
      let s = 0;
      for (const { person, out, root } of list) {
        if (!root) continue;
        const ok = person.ok;
        let u = null;
        if (R.upper >= 0 && ok[R.upper]) {
          u = [out[R.upper * 3], out[R.upper * 3 + 1], out[R.upper * 3 + 2]];
        } else if (R.lsho >= 0 && R.rsho >= 0 && ok[R.lsho] && ok[R.rsho]) {
          u = [
            (out[R.lsho * 3] + out[R.rsho * 3]) / 2,
            (out[R.lsho * 3 + 1] + out[R.rsho * 3 + 1]) / 2,
            (out[R.lsho * 3 + 2] + out[R.rsho * 3 + 2]) / 2,
          ];
        }
        if (u) {
          const r = this.opts.center
            ? [root[0] - bx, root[1] - by, root[2] - bz] : root;
          const uu = this.opts.center
            ? [u[0] - bx, u[1] - by, u[2] - bz] : u;
          s = Math.hypot(uu[0] - r[0], uu[1] - r[1], uu[2] - r[2]);
          if (s > 1e-9) break;
        }
      }
      if (s > 1e-9) {
        const ox = this.opts.center ? 0 : bx;
        const oy = this.opts.center ? 0 : by;
        const oz = this.opts.center ? 0 : bz;
        list.forEach(({ out }) => {
          for (let j = 0; j < J; j += 1) {
            out[j * 3] = ox + (out[j * 3] - ox) / s;
            out[j * 3 + 1] = oy + (out[j * 3 + 1] - oy) / s;
            out[j * 3 + 2] = oz + (out[j * 3 + 2] - oz) / s;
          }
        });
      }
    }
    return list;
  }

  /* ----- カメラ ----- */
  _viewMatrix() {
    const yaw = (this.cam.yaw * Math.PI) / 180;
    const pitch = (this.cam.pitch * Math.PI) / 180;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    // R = Rx(pitch) * Ry(yaw)
    return { cy, sy, cp, sp };
  }

  _project(M, x, y, z, W, H, f) {
    const dx = x - this.cam.tx;
    const dy = y - this.cam.ty;
    const dz = z - this.cam.tz;
    const x1 = dx * M.cy + dz * M.sy;
    const z1 = -dx * M.sy + dz * M.cy;
    const y2 = dy * M.cp - z1 * M.sp;
    const z2 = dy * M.sp + z1 * M.cp;
    const zc = z2 + this.cam.dist;
    if (zc <= this.cam.dist * 0.05) return null;
    const k = f / zc;
    return [W / 2 + x1 * k, H / 2 - y2 * k, zc, k];
  }

  fit() {
    if (!this.model) return;
    const frames = this.model.frames;
    const step = Math.max(1, Math.floor(frames.length / 24));
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < frames.length; i += step) {
      for (const { person, out } of this.framePoints(frames[i])) {
        if (!this.visible.has(person.id)) continue;
        for (let j = 0; j < this.model.names.length; j += 1) {
          if (!person.ok[j]) continue;
          const x = out[j * 3], y = out[j * 3 + 1], z = out[j * 3 + 2];
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
          minX = Math.min(minX, x); maxX = Math.max(maxX, x);
          minY = Math.min(minY, y); maxY = Math.max(maxY, y);
          minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }
      }
    }
    if (!Number.isFinite(minX)) return;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
    const radius = Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) / 2, 1e-6);
    this.cam.tx = cx; this.cam.ty = cy; this.cam.tz = cz;
    this.cam.dist = radius * 2.9;
    this.floorY = minY;
    const raw = radius / 2.4;
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    this.gridStep = [1, 2, 5].map((m) => m * pow).reduce(
      (best, s) => (Math.abs(s - raw) < Math.abs(best - raw) ? s : best), pow,
    );
    this.home = { ...this.cam };
    this.dirty = true;
    if (this.onCameraChange) this.onCameraChange();
  }

  setView(name) {
    if (name === "front") { this.cam.yaw = 0; this.cam.pitch = 8; }
    else if (name === "side") { this.cam.yaw = 90; this.cam.pitch = 8; }
    else if (name === "top") { this.cam.yaw = 0; this.cam.pitch = 88; }
    this.dirty = true;
    if (this.onCameraChange) this.onCameraChange();
  }

  resetView() {
    this.cam = { ...this.home, yaw: -32, pitch: 12 };
    this.dirty = true;
    if (this.onCameraChange) this.onCameraChange();
  }

  /* ----- 入力 ----- */
  _bindPointer() {
    const c = this.canvas;
    const pointers = new Map();
    let lastPinch = 0;
    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, button: e.button, shift: e.shiftKey });
      c.classList.add("dragging");
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        lastPinch = Math.hypot(a.x - b.x, a.y - b.y);
      }
    });
    c.addEventListener("pointermove", (e) => {
      const p = pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x;
      const dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const pinch = Math.hypot(a.x - b.x, a.y - b.y);
        if (lastPinch > 0) {
          this.cam.dist = clamp(this.cam.dist * (lastPinch / pinch), 1e-4, 1e7);
        }
        lastPinch = pinch;
        this._pan(dx / 2, dy / 2);
      } else if (p.shift || p.button === 1 || p.button === 2) {
        this._pan(dx, dy);
      } else {
        this.cam.yaw = (this.cam.yaw + dx * 0.45) % 360;
        this.cam.pitch = clamp(this.cam.pitch + dy * 0.4, -89, 89);
      }
      this.dirty = true;
      if (this.onCameraChange) this.onCameraChange();
    });
    const up = (e) => {
      pointers.delete(e.pointerId);
      if (!pointers.size) c.classList.remove("dragging");
      lastPinch = 0;
    };
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.dist = clamp(this.cam.dist * Math.exp(e.deltaY * 0.0011), 1e-4, 1e7);
      this.dirty = true;
      if (this.onCameraChange) this.onCameraChange();
    }, { passive: false });
    c.addEventListener("dblclick", () => this.fit());
    c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _pan(dx, dy) {
    const M = this._viewMatrix();
    const f = this._focal();
    const k = this.cam.dist / f;
    // カメラの右方向・上方向ベクトル (ワールド系)
    const rx = M.cy, ry = 0, rz = -M.sy;
    const ux = M.sy * M.sp, uy = M.cp, uz = M.cy * M.sp;
    this.cam.tx -= (rx * dx - ux * dy) * k;
    this.cam.ty -= (ry * dx - uy * dy) * k;
    this.cam.tz -= (rz * dx - uz * dy) * k;
  }

  _focal() {
    return Math.min(this.canvas.clientWidth, this.canvas.clientHeight) * 1.15;
  }

  _resize() {
    const parent = this.canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(64, parent.clientWidth);
    const h = Math.max(64, parent.clientHeight);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.dpr = dpr;
    this.dirty = true;
  }

  snapshot() {
    const out = document.createElement("canvas");
    out.width = this.canvas.width;
    out.height = this.canvas.height;
    const ctx = out.getContext("2d");
    ctx.fillStyle = "#070a12";
    ctx.fillRect(0, 0, out.width, out.height);
    ctx.drawImage(this.canvas, 0, 0);
    return out.toDataURL("image/png");
  }

  /* ----- 描画 ----- */
  render() {
    const ctx = this.ctx;
    const dpr = this.dpr || 1;
    const W = this.canvas.width / dpr;
    const H = this.canvas.height / dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!this.model) return;
    const M = this._viewMatrix();
    const f = this._focal();
    const frame = this.model.frames[this.frame];
    if (!frame) return;

    // スポットライト風の薄い背景
    const spot = this._project(M, this.cam.tx, this.cam.ty, this.cam.tz, W, H, f);
    if (spot) {
      const g = ctx.createRadialGradient(spot[0], spot[1], 0, spot[0], spot[1], Math.min(W, H) * 0.6);
      g.addColorStop(0, "rgba(56, 130, 190, 0.08)");
      g.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    }

    if (this.opts.grid) this._drawGrid(ctx, M, W, H, f);

    const entries = this.framePoints(frame)
      .filter(({ person }) => this.visible.has(person.id));
    if (this.opts.trail && this.opts.highlight >= 0) {
      this._drawTrails(ctx, M, W, H, f);
    }

    // 全人物のボーンを集めて奥→手前に描画
    const segs = [];
    const dots = [];
    entries.forEach(({ person, out }) => {
      const hue = PERSON_HUES[person.id % PERSON_HUES.length];
      const proj = [];
      for (let j = 0; j < this.model.names.length; j += 1) {
        proj.push(person.ok[j]
          ? this._project(M, out[j * 3], out[j * 3 + 1], out[j * 3 + 2], W, H, f)
          : null);
      }
      this.model.edges.forEach(([a, b]) => {
        const pa = proj[a], pb = proj[b];
        if (!pa || !pb) return;
        const sa = sideOf(this.model.names[a]);
        const sb = sideOf(this.model.names[b]);
        const side = sa === sb ? sa : "center";
        segs.push({ pa, pb, z: (pa[2] + pb[2]) / 2, hue, side });
      });
      proj.forEach((p, j) => {
        if (p) dots.push({ p, j, hue, z: p[2] });
      });
    });

    const zs = segs.map((s) => s.z).concat(dots.map((d) => d.z));
    const zMin = Math.min(...zs, this.cam.dist * 0.5);
    const zMax = Math.max(...zs, this.cam.dist * 1.5);
    const depth01 = (z) => clamp((z - zMin) / Math.max(zMax - zMin, 1e-9), 0, 1);

    segs.sort((a, b) => b.z - a.z);
    ctx.lineCap = "round";
    segs.forEach((s) => {
      const d = depth01(s.z);
      const alpha = 1 - d * 0.55;
      const width = this.opts.bone * (1.35 - d * 0.7);
      let hue = s.hue;
      let sat = 85, lit = 62;
      if (s.side === "left") { hue = (s.hue + 38) % 360; lit = 66; }
      else if (s.side === "right") { hue = (s.hue - 30 + 360) % 360; lit = 58; }
      ctx.strokeStyle = `hsla(${hue}, ${sat}%, ${lit}%, ${alpha})`;
      ctx.lineWidth = Math.max(0.75, width);
      ctx.beginPath();
      ctx.moveTo(s.pa[0], s.pa[1]);
      ctx.lineTo(s.pb[0], s.pb[1]);
      ctx.stroke();
    });

    dots.sort((a, b) => b.z - a.z);
    dots.forEach(({ p, j, hue }) => {
      const d = depth01(p[2]);
      const isHl = j === this.opts.highlight;
      const r = (isHl ? 6 : 2.6) * (1.25 - d * 0.5);
      ctx.beginPath();
      if (isHl) {
        ctx.fillStyle = "#fbbf24";
        ctx.shadowColor = "rgba(251, 191, 36, 0.9)";
        ctx.shadowBlur = 14;
      } else {
        ctx.fillStyle = `hsla(${hue}, 30%, 92%, ${1 - d * 0.5})`;
        ctx.shadowBlur = 0;
      }
      ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      if (isHl) {
        ctx.strokeStyle = "rgba(251, 191, 36, 0.55)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p[0], p[1], r + 4.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    if (this.opts.labels || this.opts.highlight >= 0) {
      ctx.font = "10.5px 'IBM Plex Mono', monospace";
      ctx.textBaseline = "middle";
      dots.forEach(({ p, j }) => {
        const isHl = j === this.opts.highlight;
        if (!this.opts.labels && !isHl) return;
        ctx.fillStyle = isHl ? "rgba(251, 191, 36, 0.95)" : "rgba(147, 161, 181, 0.75)";
        ctx.fillText(this.model.names[j], p[0] + 8, p[1] - 2);
      });
    }

    this._drawGizmo(ctx, M, W, H);
  }

  _drawGrid(ctx, M, W, H, f) {
    const s = this.gridStep;
    const N = 8;
    const y = this.floorY;
    const cx = Math.round(this.cam.tx / s) * s;
    const cz = Math.round(this.cam.tz / s) * s;
    ctx.lineWidth = 1;
    for (let i = -N; i <= N; i += 1) {
      const fade = 1 - Math.abs(i) / (N + 1);
      const a = this._project(M, cx + i * s, y, cz - N * s, W, H, f);
      const b = this._project(M, cx + i * s, y, cz + N * s, W, H, f);
      if (a && b) {
        ctx.strokeStyle = `rgba(120, 140, 170, ${0.05 + fade * 0.1})`;
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
      }
      const c = this._project(M, cx - N * s, y, cz + i * s, W, H, f);
      const d = this._project(M, cx + N * s, y, cz + i * s, W, H, f);
      if (c && d) {
        ctx.strokeStyle = `rgba(120, 140, 170, ${0.05 + fade * 0.1})`;
        ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(d[0], d[1]); ctx.stroke();
      }
    }
  }

  _drawTrails(ctx, M, W, H, f) {
    const j = this.opts.highlight;
    const len = this.opts.trailLen;
    const start = Math.max(0, this.frame - len);
    const prevById = new Map();
    for (let i = start; i <= this.frame; i += 1) {
      const fr = this.model.frames[i];
      if (!fr) continue;
      const t = (i - start) / Math.max(1, this.frame - start);
      for (const { person, out } of this.framePoints(fr)) {
        if (!this.visible.has(person.id)) continue;
        if (!person.ok[j]) { prevById.set(person.id, null); continue; }
        const pr = this._project(M, out[j * 3], out[j * 3 + 1], out[j * 3 + 2], W, H, f);
        const prev = prevById.get(person.id);
        if (prev && pr) {
          const hue = PERSON_HUES[person.id % PERSON_HUES.length];
          ctx.strokeStyle = `hsla(${hue}, 90%, 65%, ${0.05 + t * 0.55})`;
          ctx.lineWidth = 1 + t * 1.6;
          ctx.beginPath();
          ctx.moveTo(prev[0], prev[1]);
          ctx.lineTo(pr[0], pr[1]);
          ctx.stroke();
        }
        prevById.set(person.id, pr);
      }
    }
  }

  _drawGizmo(ctx, M, W) {
    const ox = W - 54;
    const oy = 54;
    const L = 22;
    const axes = [
      { v: [1, 0, 0], label: "X", color: "#f87171" },
      { v: [0, 1, 0], label: "Y", color: "#4ade80" },
      { v: [0, 0, 1], label: "Z", color: "#60a5fa" },
    ];
    ctx.font = "10px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    axes.forEach(({ v, label, color }) => {
      const [x, y, z] = v;
      const x1 = x * M.cy + z * M.sy;
      const z1 = -x * M.sy + z * M.cy;
      const y2 = y * M.cp - z1 * M.sp;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + x1 * L, oy - y2 * L);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.fillText(label, ox + x1 * (L + 9), oy - y2 * (L + 9));
    });
    ctx.globalAlpha = 1;
    ctx.textAlign = "left";
  }
}

/* ================================================================
   アプリ
   ================================================================ */

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
