# Changelog

## 0.2.0 (2026-06-12)

### 追加
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
