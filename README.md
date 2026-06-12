# poselab — 研究用ヒト骨格推定ツールキット

画像・動画・カメラ入力からヒトの骨格 (33 キーポイント) を推定し、
座標データを CSV / JSON / NumPy 形式でエクスポートできるツールです。
GUI と CLI の両方から使用できます。

- 推定エンジン: [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Apache-2.0) を依存ライブラリとして利用
- 本リポジトリのコードはすべて独自実装 (MIT ライセンス)
- 2D ピクセル座標・正規化座標・3D ワールド座標 (メートル単位) ・信頼度を出力
- 複数人検出に対応 (`--num-poses`)

## インストール

Python 3.9 以上が必要です。

```bash
git clone <this-repo>
cd 3d
pip install -e .
```

GUI を使う場合は tkinter も必要です (多くの環境では同梱。Ubuntu では
`sudo apt install python3-tk`)。

初回実行時に推定モデル (約 5–30 MB、Apache-2.0) が
`~/.cache/poselab/` に自動ダウンロードされます。
保存先は環境変数 `POSELAB_CACHE_DIR` で変更できます。

## CLI の使い方

```bash
# 動画を処理して座標 CSV と骨格描画済み動画を出力
poselab --input walk.mp4 --csv walk.csv --save-video walk_annotated.mp4

# 静止画 (複数指定可)
poselab --input a.jpg b.jpg --json poses.json

# 1 枚の画像に骨格を描画して保存
poselab --input photo.jpg --save-image annotated.jpg --csv photo.csv

# カメラ 0 番をライブプレビューしながら座標を記録 (q キーで終了)
poselab --input camera:0 --show --csv live.csv

# 高精度モデル・3 人まで検出・NumPy 出力
poselab --input dance.mp4 --model heavy --num-poses 3 --npz dance.npz

# 関節角度 (肘・肩・股・膝・足首) の時系列 CSV + 5 フレーム移動平均で平滑化
poselab --input squat.mp4 --angles-csv angles.csv --csv coords.csv --smooth 5

# キーポイント名一覧
poselab --list-keypoints
```

主なオプション:

| オプション | 説明 |
| --- | --- |
| `--input` | 画像/動画パス、または `camera:0` 形式のカメラ指定 |
| `--model {lite,full,heavy}` | モデルサイズ (lite=高速 / heavy=高精度) |
| `--num-poses N` | 最大検出人数 |
| `--csv` / `--json` / `--npz` | 座標データの出力先 |
| `--angles-csv` | 関節角度 (10 関節) の時系列 CSV |
| `--smooth N` | N フレーム移動平均による座標の平滑化 |
| `--save-video` / `--save-image` | 骨格描画済みメディアの出力先 |
| `--show` | プレビューウィンドウ表示 |
| `--draw-labels` | キーポイント名も描画 |
| `--camera-mirror` | カメラ映像を左右反転 (鏡像) で処理 |
| `--max-frames N` | 処理フレーム数の上限 |

## GUI の使い方

```bash
poselab-gui
```

- **入力**: 画像・動画ファイルを開く、またはカメラ番号を指定して開始
  (ミラー表示の切り替え可)
- **ライブプレビュー**: 骨格オーバーレイ・FPS 表示付き。一時停止 / 再開、
  表示中フレームの画像保存も可能
- **記録**: 「座標を記録する」を有効にすると推定結果が蓄積され、
  CSV / JSON / NPZ / 関節角度 CSV にエクスポートできます
- **一括処理**: 動画ファイルを選ぶと、座標 (CSV + JSON) と
  骨格描画済み動画 (MP4) を進捗バー付きで一括生成します

## 出力フォーマット

### CSV (ロング形式、1 行 = 1 キーポイント)

| 列 | 意味 |
| --- | --- |
| `frame`, `timestamp_ms` | フレーム番号、タイムスタンプ |
| `person` | 人物インデックス (複数人検出時) |
| `keypoint_id`, `keypoint_name` | キーポイント番号・名前 |
| `x_px`, `y_px` | ピクセル座標 |
| `x_norm`, `y_norm` | 画像サイズで正規化した座標 (0–1) |
| `z` | 腰中心を原点とする相対深度 (近いほど負) |
| `visibility`, `presence` | 可視性・存在の推定確率 (0–1) |
| `world_x`, `world_y`, `world_z` | 3D ワールド座標 (メートル、腰中心が原点) |

pandas でそのまま解析できます:

```python
import pandas as pd
df = pd.read_csv("walk.csv")
wrist = df[df.keypoint_name == "right_wrist"]
```

### 関節角度 CSV (`--angles-csv`)

肘・肩・股関節・膝・足首 (左右計 10 関節) について、3 点のなす角を
度単位で出力します。ワールド座標 (3D) がある場合はそれを優先し
(`coordinates=world`)、なければピクセル座標 (2D) で計算します。
列: `frame, timestamp_ms, person, angle_name, angle_deg, min_visibility, coordinates`

### JSON

`metadata` (ツール情報・キーポイント名一覧) と `frames` (フレームごとの
全人物・全キーポイント) を持つ構造化データです。

### NPZ (NumPy)

- `keypoints`: `(フレーム数, 人数, 33, 5)` — `[x_px, y_px, z, visibility, presence]`
- `world`: `(フレーム数, 人数, 33, 4)` — `[x, y, z, visibility]`
- `timestamps_ms`, `frame_indices`, `keypoint_names`
- 未検出は NaN

```python
import numpy as np
data = np.load("dance.npz")
right_wrist_xy = data["keypoints"][:, 0, 16, :2]  # 16 = right_wrist
```

## Python API

```python
from poselab.backends import create_backend
from poselab.pipeline import run_pipeline
from poselab.sources import open_source

source = open_source("walk.mp4")
backend = create_backend("mediapipe", model="full", num_poses=2)
results = run_pipeline(source, backend)
for frame in results:
    for person in frame.persons:
        nose = person.keypoints[0]
        print(frame.frame_index, nose.x_px, nose.y_px, nose.visibility)
backend.close()
```

バックエンドは `poselab.backends.base.PoseBackend` を継承することで
他の推定エンジンにも差し替えられる設計です。

## テスト

```bash
pip install pytest
pytest tests/
```

## ライセンスについて

- 本リポジトリのコード: **MIT License** (LICENSE 参照)
- 依存ライブラリ: MediaPipe (Apache-2.0)、OpenCV (Apache-2.0)、
  NumPy (BSD)、Pillow (MIT-CMU) — いずれも公開 API 経由で利用しており、
  コードのコピーは含みません
- 自動ダウンロードされる Pose Landmarker モデル: Apache-2.0 (Google 提供)
- 研究利用・改変・再配布は各ライセンスの条件の範囲で自由に行えます

## 研究利用上の注意

- `z` (画像座標系の深度) は相対値であり、ワールド座標 (`world_*`) と
  スケールが異なります。3D 解析にはワールド座標の使用を推奨します
- 推定値にはノイズが含まれるため、解析前に `visibility` でのフィルタや
  平滑化を検討してください (`--smooth N` で NaN 対応の移動平均を適用
  できます。より高度な平滑化が必要なら Savitzky–Golay フィルタ等を)
- 複数人検出時の `person` インデックスはフレーム間で同一人物を保証
  しません (トラッキング ID ではありません)。`--smooth` も同一
  インデックス = 同一人物の前提で動作します
- カメラのミラー表示 (`--camera-mirror`) 使用時は、キーポイントの
  left/right が被写体の実際の左右と逆になります
