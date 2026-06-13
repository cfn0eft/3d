# CLAUDE.md — poselab (cfn0eft/3d) 開発ガイド

このファイルは Claude Code などの AI エージェントと開発者向けの要点集。
詳細は [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) を参照。

## このリポジトリは何か

研究用ヒト骨格推定ツールキット **poselab**。3 つの顔を持つ:

1. **poselab 本体** — MediaPipe (標準) / MMPose (オプション) バックエンドの
   推定 CLI・Tkinter GUI (`poselab/`)
2. **ブラウザ 3D ビューア** — `poselab/webviewer/static/` (依存ゼロ、
   https://cfn0eft.github.io/3d/ に Pages 自動デプロイ)
3. **PoseLab Studio** — Pose3DStudio 後継のジョブ管理 Web GUI。
   `poselab-studio` でローカル起動 (GUI: `poselab/studio/gui/`、
   サーバー: `poselab/studio/server.py`)。Windows 配布版 exe は
   `build-exe.yml` ワークフローが CUDA 同梱で CI ビルドする
   (`packaging/`)。旧 exe への `poselab-studio deploy` はレガシー

このリポジトリは旧 Pose3DStudio.exe (mmpose ベースの GPU パイプライン、
ソース非公開のビルド済み exe) の機能を「公開 API を使った独自実装」として
移植してきた経緯があり、v0.6.0 で GUI + exe ビルドまで完全移設した。
**コードはすべて独自実装、コピー禁止**の方針。

## よく使うコマンド

```bash
pip install -e ".[plot]" pytest ruff   # セットアップ (mediapipe 等が入る)
pytest tests/ -q                        # テスト (mmpose 不要、フェイク注入式)
ruff check poselab/ tests/              # lint (CI と同じ)
python -m build                         # パッケージビルド

poselab-viewer                          # ビューアをローカル起動
poselab-viewer --export-html out.html   # 自己完結 HTML

poselab-studio                          # Studio GUI をローカル起動 (要 mmpose)
poselab-studio build --out dist/studio-gui          # GUI 一式をビルド
poselab-studio deploy <exe>/_internal/gui           # 旧 exe へ配備 (レガシー)
# 配備先は環境変数 POSE3DSTUDIO_GUI でも指定可
```

配布版 exe は GitHub Actions「Build Windows exe (PoseLab Studio)」を
手動実行 (または vX.Y.Z タグ push) → Artifacts の zip。

## 絶対に守ること

- **PR のベースは必ず `main`**。スタック PR (ベース=フィーチャーブランチ) は
  マージしても main に入らない事故が実際に起きた (#9 → 回収 #10)
- **3D エンジンのソースは `poselab/webviewer/static/engine.js` だけ**。
  ビューア (index.html が engine.js → app.js の順に読み込む) と
  Pose3DStudio GUI (`poselab-studio build` が IIFE で包んで連結) の共通
  ソース。エンジンを編集したら Node スモーク
  (`pytest tests/test_engine_js.py`) を回し、`poselab-studio deploy` で
  exe にも反映する
- **CHANGELOG.md とバージョン** (`poselab/__init__.py` の `__version__` が
  唯一のソース。pyproject.toml は dynamic 参照) を機能追加のたびに更新。
  コミットメッセージは英語タイトル + 詳細本文
- ビューア/GUI の文言・ドキュメントは日本語

## 座標規約 (3D 表示で迷ったら)

- MediaPipe world / mmpose カメラ系 → **y 下向き** (ビューア軸 `ydown`、既定)
- mmpose 3D リフタ (`--pose3d`) の出力 → **z 上向き・x 反転・床基準**
  (ビューア軸 `zup`。MMPose 形式 JSON 読込時に自動選択)

## テスト方針

- mmpose / GPU は CI に無い → `tests/test_mmpose_backend.py` /
  `tests/test_pose3d.py` はフェイク inferencer 注入、
  `tests/test_studio_server.py` はフェイクのジョブコマンド注入で検証
- 3D エンジン (engine.js) のパース / エクスポートは Node スモークテスト
  (`tests/engine_smoke.mjs`。pytest から自動実行、node 無しはスキップ) で
  ラウンドトリップを検証。PoseStage の描画と UI 配線 (app.js /
  app_main.js) は実ブラウザでデモ (`?demo=1`) を確認
- mmpose の実 API 仕様が必要なら Pose3DStudio.exe の
  `_internal/mmpose/` に同梱ソースがある (ユーザーのマシン上)

## Pose3DStudio.exe との関係 (ユーザー環境)

- exe は `_internal/gui/` の 3 ファイルをディスクから毎回配信 →
  `poselab-studio deploy` だけで GUI 更新可能 (exe 再ビルド不要)
- exe の HTTP サーバーは固定ルートのみ (`/`, `/app.css`, `/app.js` + API)。
  gui フォルダに新ファイルを足しても 404 → すべて app.js に連結する設計
- exe の出力 JSON は mmpose 形式 (`meta_info`/`instance_info`) で、
  ビューアがそのまま読める
