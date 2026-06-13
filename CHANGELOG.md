# Changelog

## 0.6.0 (2026-06-13)

### 追加
- **Pose3DStudio 後継のスタンドアロン GUI** (`poselab-studio` で起動):
  旧 exe と同じ Web GUI (ジョブキュー / 進捗 / 出力プレビュー / 3D
  ビューア / モデルプロファイル) を、poselab 自身のパイプライン
  (mmpose 2D + 3D リフティング = `--pose3d` 相当) へ接続するローカル
  サーバーを独自実装 (`poselab/studio/server.py`)。旧 exe は不要
  - SSE でログ / 進捗 / 出力をライブ配信、複数動画のキュー処理と並べ替え、
    キャンセル、実行前チェック (preflight)、ネイティブのファイル選択
    ダイアログ、出力フォルダ / 動画を開く、results JSON のサマリ表示
  - 旧 GUI が送るモデル / 検出器プロファイル (exe 内のコンフィグパス) は
    MMPose / MMDetection model zoo のモデル名へ自動で読み替え
  - center_root / normalize_scale は CSV の world 座標に適用して書き出す
    (関節 0 = root を原点へ移動 / 原点からの最大距離を 1 にスケール)
  - GPU は自動検出 (torch → nvidia-smi の順)。無ければ CPU で実行
- **Windows 配布版 exe の CI ビルド** (`.github/workflows/build-exe.yml` +
  `packaging/`): CUDA 11.8 版 PyTorch・OpenMMLab 一式・poselab を
  PyInstaller で同梱した `PoseLabStudio-win64-cuda118.zip` を生成
  (手動実行または vX.Y.Z タグ push で起動、Actions Artifacts に 30 日保存)
  - 解凍して `PoseLabStudio.exe` を起動するだけ。Python や依存関係の
    インストールは不要 (モデルの重みのみ初回使用時に自動ダウンロード)
  - GPU ドライバ 452 以降で CUDA 実行、無ければ自動で CPU 実行
  - `PoseLabStudio.exe --selftest` で同梱物を自己診断 (CI でも実行)
  - `PoseLabStudio.exe --cli ...` は poselab CLI として動作
    (サーバーがジョブ実行に内部使用)

## 0.5.1 (2026-06-12)

### 変更 (内部構成の再編。機能・出力は変わらない)
- 3D エンジンを `poselab/webviewer/static/engine.js` に物理分割し、
  コメントマーカーによる切り出し (`ENGINE_END_MARKER`) を廃止
  - ビューアは engine.js + app.js (UI 配線) の 2 ファイル構成に。
    ローカル配信・GitHub Pages・`--export-html` (従来どおり 1 ファイルに
    埋め込み) すべて対応済み
  - `poselab-studio build` は engine.js をそのまま IIFE で包んで連結
    (exe へ配備する app.js は従来どおり 1 ファイル)
- 3D エンジンの Node スモークテストを追加 (`tests/engine_smoke.mjs` /
  `tests/test_engine_js.py`): デモ生成 → 全 4 形式エクスポート →
  再パースのラウンドトリップと MMPose 形式の最小例パースを検証
  (node が無い環境ではスキップ)
- パッケージ版数を `poselab/__init__.py` の `__version__` に単一ソース化
  (pyproject.toml は dynamic version で参照)

## 0.5.0 (2026-06-13)

### 追加
- ビューアに「エクスポート」パネル: 読み込んだデータから**人物・関節
  (複数選択)・フレーム範囲を選んで個別にダウンロード**できる
  - 形式: ワイド CSV / ロング CSV / poselab JSON / MMPose 互換 JSON
    (いずれもビューアに再読み込み可能。関節サブセットでは骨格リンクを
    再構成)
  - 「表示変換を適用」でセンタリング・正規化・座標系変換後の座標を
    書き出せる (既定は読み込んだままの生座標)
  - Pose3DStudio の埋め込みビューアにも同機能を反映
- `--auto-output --outputs LIST`: 一括出力の形式をカンマ区切りで選択
  (`long, wide, json, angles, summary, video, image`。既定: すべて)。
  `--pose3d` との併用にも対応
- `poselab-studio`: デスクトップ版 Pose3DStudio の Web GUI ソースを
  リポジトリに収容 (`poselab/studio/gui/`)。`build` で 3D エンジン
  (ビューアと共通) と連結して生成、`deploy` で exe の `_internal/gui`
  へ配備 (既存ファイルは `.backup/` へ退避)
- 開発引き継ぎドキュメントを追加 (`CLAUDE.md`、`docs/DEVELOPMENT.md`)

## 0.4.0 (2026-06-12)

### 追加
- MMPose バックエンド (`--backend mmpose`、オプション依存):
  RTMDet 人物検出 + RTMPose 2D 推定 (COCO 17 点)。既存の CSV / JSON /
  NPZ 出力・トラッキング・描画・GUI 外の全 CLI 機能がそのまま使える。
  モデルの重みは MMPose 公式 model zoo から初回使用時に自動ダウンロード
- 3D リフティング (`--pose3d`、動画入力): 2D 推定 + VideoPose3D 系
  時系列リフティングで Human3.6M 17 点の 3D 骨格を推定
  - MMPose 互換 results JSON (`meta_info` / `instance_info`) を出力。
    Pose3DStudio の出力と互換で、`poselab-viewer` でそのまま再生可能
  - ロング / ワイド CSV (world_x/y/z に 3D 座標)、2D+3D 可視化動画
    (`--save-video`、`--h264` 対応)、`--auto-output` 併用に対応
  - `--pose2d-model` / `--lift-model` / `--det-model` 等でモデル差し替え、
    `--device` でデバイス指定
- COCO 17 点 / Human3.6M 17 点の骨格カタログを追加 (`poselab.skeleton`)
- `poselab --info` に mmpose の導入状況と実行デバイスを表示
- ビューアに座標系「Z 上向き (MMPose 3D)」を追加。MMPose 形式 JSON の
  読み込み時は自動で選択される

## 0.3.0 (2026-06-12)

### 追加
- ブラウザ 3D ビューア `poselab-viewer`: 推定結果の JSON / CSV を
  ドラッグ & ドロップで読み込み、3D 骨格をマウスで回転・ズーム・
  パンしながら再生できる (依存ライブラリゼロ、オフライン動作)
  - poselab JSON / ロング CSV / ワイド CSV に加え、MMPose 系
    3D パイプライン (Pose3DStudio など) の
    `meta_info / instance_info` 形式 JSON・汎用ワイド CSV も読める
  - 複数人の同時表示 (人物ごとに配色)、注目関節のハイライトと
    軌跡 (トレイル)、床グリッド、関節ラベル、正面/側面/上面プリセット、
    キーボードショートカット (Space / ←→ / F / R / G / L / 1-3)
  - 腰位置センタリング・体格スケール正規化の表示オプション
  - 再生速度 (×0.25–×2)・FPS 指定・ループ・PNG スナップショット
  - 合成歩行データのデモ内蔵 (`?demo=1` または「デモ」ボタン)
  - `--export-html` で CSS/JS を埋め込んだ自己完結 HTML を出力
    (GitHub Pages 等にそのまま置ける)。GitHub Pages への自動デプロイ
    ワークフローを追加
- ワイド形式 CSV (`--wide-csv`、1 行 = 1 フレーム × 1 人物。GUI の
  エクスポートにも追加)
- `--auto-output`: `<入力名>_poselab/` フォルダへ全形式
  (long/wide CSV・JSON・角度・サマリ・H.264 注釈動画) を一括出力。
  複数動画の連続バッチ処理に対応
- `--h264`: 注釈動画を ffmpeg で H.264 (yuv420p, faststart) に
  再エンコード (ブラウザや一般プレーヤーで再生可能に)
- `poselab-plot --kind pose3d`: ワールド座標の 3D 骨格ビュー
  (`--frame` でフレーム指定、`--show` でマウス回転できるウィンドウ表示)
- グラフ生成コマンド `poselab-plot`: 座標の時系列 / 軌跡プロット /
  滞在ヒートマップ、関節角度・速度・距離の時系列グラフを CSV から生成
  (CSV 種類はヘッダから自動判別、matplotlib は optional 依存 `[plot]`)
- 2 点間距離の特徴量計算 (`--distance right_wrist:nose --distance-csv`、
  複数ペア対応、ピクセルとワールド座標 (m) の両方を出力)
- GUI「シーンタグ」タブ: 行動コーディング用のラベル付き時間区間の記録
  (T キーで開始/終了) と CSV 書き出し

### 改善
- カメラが開けない場合のエラーメッセージを具体的な確認手順付きに変更
  (GUI ではダイアログ表示)
- Windows で MSMF バックエンドが失敗した場合に DirectShow で自動再試行
- 利用可能なカメラの検索機能 (`poselab --list-cameras` /
  GUI の「使えるカメラを検索」ボタン)

## 0.2.0 (2026-06-12)

### 追加
- 複数人検出の本格対応: 人物 ID トラッキング
  (`--num-poses` 2 以上で自動有効、`--no-track` で無効化)、
  P0/P1... の ID バッジ描画、NPZ / 平滑化のトラッキング ID 対応
  - マッチングは等速モデルの位置予測 + 胴体色ヒストグラムを併用し、
    交差・すれ違い・短いオクルージョンに頑健
  - 接近・交差した区間と長いオクルージョン後の再出現を自動検出し、
    「ID が入れ替わっている可能性」として処理後に警告
    (CLI 表示 / GUI ダイアログ / summary JSON の `id_warnings`)
- キーポイント軌跡 (モーショントレイル) の動画上へのプロット
  (`--trail` / `--trail-keypoints`、GUI のオーバーレイ設定、一括処理にも反映)
- GUI のダークテーマ化とタブ構成への再設計 (ツールバー、
  アクセントボタン、記録中インジケータ)
- CLI プログレスバー (%, fps, 残り時間。総数不明時はフレーム数と経過時間)
- キーポイント速度の CSV 出力 (`--velocity-csv`、px/s と m/s)
- 処理サマリの JSON 出力 (`--summary-json`、検出率・平均人数など)
- 環境診断コマンド (`poselab --info`)
- GUI: 再生位置バー (% / 時刻表示、カメラは LIVE 経過時間)
- GUI: メニューバーとキーボードショートカット
  (Ctrl+I/O 開く、Ctrl+S 全形式保存、Space 一時停止、Esc 停止)
- GUI: 関節角度のライブ表示パネル
- GUI: 全 5 形式の一括エクスポートとエクスポート時平滑化
- GUI: 終了時の検出率サマリ表示、設定の自動保存・復元
- GUI: 一括処理に関節角度 CSV を追加
- Windows の日本語 (非 ASCII) 画像パス対応

### 開発
- GitHub Actions CI (Ubuntu / Windows でのテスト、ruff、パッケージビルド)

## 0.1.x (2026-06-12)

- 初回リリース: 画像 / 動画 / カメラ入力の 33 キーポイント推定
  (MediaPipe Pose Landmarker、複数人対応)
- CSV / JSON / NPZ 座標エクスポート (2D・正規化・3D ワールド座標)
- 骨格オーバーレイ描画と注釈付き動画・画像の書き出し
- CLI (`poselab`) と Tkinter GUI (`poselab-gui`)
- 関節角度 CSV、移動平均平滑化、カメラミラー、一時停止
