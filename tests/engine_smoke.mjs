/* poselab 3D エンジン (poselab/webviewer/static/engine.js) のスモークテスト。
   ブラウザなしで Node の vm 上にエンジンを読み込み、
   1. デモデータ生成 (demoModel)
   2. 全 4 形式エクスポート → parseAny で再パースのラウンドトリップ
   3. MMPose 形式 JSON (Pose3DStudio 互換) の最小例パース
   を検証する。pytest からは tests/test_engine_js.py 経由で実行される。
   単体実行: node tests/engine_smoke.mjs
*/
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = path.dirname(fileURLToPath(import.meta.url));
const enginePath = path.join(
  here, "..", "poselab", "webviewer", "static", "engine.js",
);
const source = readFileSync(enginePath, "utf8");

// エンジンは DOM 非依存で評価できる (PoseStage は定義のみで未使用)。
// 末尾の式でトップレベル宣言 (const/class/function) を取り出す。
const context = vm.createContext({});
const api = vm.runInContext(
  source +
    "\n;({ parseAny, demoModel, collectExport, EXPORT_FORMATS, MP_NAMES," +
    " availableMetrics, metricSeries, seriesStats, smoothModel });",
  context,
  { filename: "engine.js" },
);

let failures = 0;
function check(cond, message) {
  if (!cond) {
    failures += 1;
    console.error(`NG: ${message}`);
  }
}
function close(a, b, message) {
  check(Math.abs(a - b) < 1e-4, `${message} (${a} vs ${b})`);
}

/* 1. デモデータ生成 */
const demo = api.demoModel();
check(demo.frames.length > 0, "demoModel: フレームがある");
check(demo.names.length === api.MP_NAMES.length,
  "demoModel: MediaPipe 33 関節");
check(demo.personIds.length === 2, "demoModel: 2 人");
check(demo.edges.length > 0, "demoModel: 骨格リンクがある");

/* 2. エクスポート → 再パースのラウンドトリップ (全形式) */
const FRAMES = 10;
const col = api.collectExport(demo, { start: 0, end: FRAMES - 1 });
check(col.records.length === FRAMES, "collectExport: フレーム範囲");

const refFrame = demo.frames[3];
const refPerson = refFrame.persons[1];

for (const [key, fmt] of Object.entries(api.EXPORT_FORMATS)) {
  let model;
  try {
    const text = fmt.build(col, demo, "raw");
    model = api.parseAny(text, `roundtrip_${key}.${fmt.ext}`);
  } catch (err) {
    failures += 1;
    console.error(`NG: ${key}: エクスポート/再パースで例外: ${err.message}`);
    continue;
  }
  check(model.frames.length === FRAMES, `${key}: フレーム数が保存される`);
  check(model.names.length === demo.names.length, `${key}: 関節数が保存される`);
  check(model.personIds.length === 2, `${key}: 人数が保存される`);
  // 座標値の保存 (フレーム 3・人物 1 の全関節、許容誤差 1e-4)
  const got = model.frames[3] && model.frames[3].persons.find(
    (p) => p.id === refPerson.id,
  );
  check(!!got, `${key}: 人物 ID が保存される`);
  if (got) {
    for (let j = 0; j < demo.names.length; j += 1) {
      for (let c = 0; c < 3; c += 1) {
        close(got.pts[j * 3 + c], refPerson.pts[j * 3 + c],
          `${key}: 座標 ${demo.names[j]}[${"xyz"[c]}]`);
      }
    }
  }
}

/* 3. MMPose 形式 JSON (meta_info / instance_info) の最小例 */
const mmposeJson = JSON.stringify({
  meta_info: {
    num_keypoints: 3,
    keypoint_id2name: { 0: "root", 1: "left_hip", 2: "right_hip" },
    skeleton_links: [[0, 1], [0, 2]],
  },
  instance_info: [
    {
      frame_id: 0,
      instances: [{
        keypoints: [[0, 0, 0], [0.1, 0.0, 0.9], [-0.1, 0.0, 0.9]],
        keypoint_scores: [1, 1, 1],
      }],
    },
  ],
});
const mm = api.parseAny(mmposeJson, "results_test.json");
check(mm.frames.length === 1, "mmpose: 1 フレーム");
check(mm.names.join(",") === "root,left_hip,right_hip", "mmpose: 関節名");
check(mm.edges.length === 2, "mmpose: 骨格リンク");
check(mm.defaultAxis === "zup", "mmpose: H36M 系は z-up が既定");

/* 4. 解析ヘルパ (関節角度 / 速度 / 対称性 / 平滑化) */
const metrics = api.availableMetrics(demo);
check(metrics.angles.some((a) => a.key === "left_knee"),
  "availableMetrics: MediaPipe は left_knee 角度を計算可能");
check(metrics.symmetry.some((s) => s.key === "knee"),
  "availableMetrics: 膝の左右対称性が計算可能");
check(metrics.joints.length === demo.names.length,
  "availableMetrics: 速度対象は全関節");

const angleSeries = api.metricSeries(demo, {
  person: demo.personIds[0],
  metric: { type: "angle", key: "left_knee" },
});
check(angleSeries.values.length === demo.frames.length,
  "metricSeries: 角度系列長 = フレーム数");
check(angleSeries.unit === "°", "metricSeries: 角度の単位");
const angStats = api.seriesStats(angleSeries.values);
check(angStats.n > 0, "seriesStats: 有効サンプルがある");
check(angStats.min >= 0 && angStats.max <= 180,
  "metricSeries: 関節角度は 0-180 度の範囲");

const speedSeries = api.metricSeries(demo, {
  person: demo.personIds[0],
  metric: { type: "speed", joint: 0 },
});
check(speedSeries.values[0] == null, "metricSeries: 速度の先頭フレームは null");
check(api.seriesStats(speedSeries.values).n > 0, "metricSeries: 速度が計算される");

const symSeries = api.metricSeries(demo, {
  person: demo.personIds[0],
  metric: { type: "symmetry", key: "knee" },
});
check(api.seriesStats(symSeries.values).min >= 0,
  "metricSeries: 対称性指標は非負");

// 平滑化はモデル構造 (フレーム数・関節数・人数) を保つ
const sm = api.smoothModel(demo, 5);
check(sm.frames.length === demo.frames.length, "smoothModel: フレーム数不変");
check(sm.names.length === demo.names.length, "smoothModel: 関節数不変");
check(sm.personIds.length === demo.personIds.length, "smoothModel: 人数不変");
check(sm !== demo && sm.frames !== demo.frames,
  "smoothModel: 元モデルを破壊しない (新オブジェクト)");
// window<=1 は元モデルをそのまま返す
check(api.smoothModel(demo, 1) === demo, "smoothModel: 窓 1 は無変換");

if (failures > 0) {
  console.error(`engine smoke: ${failures} 件失敗`);
  process.exit(1);
}
console.log("engine smoke: OK");
