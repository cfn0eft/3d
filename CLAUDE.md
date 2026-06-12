# CLAUDE.md — poselab (cfn0eft/3d) 開発ガイド

このファイルは Claude Code などの AI エージェントと開発者向けの要点集。
詳細は [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) を参照。

## このリポジトリは何か

研究用ヒト骨格推定ツールキット **poselab**。3 つの顔を持つ:

1. **poselab 本体** — MediaPipe (標準) / MMPose (オプション) バックエンドの
   推定 CLI・Tkinter GUI (`poselab/`)
2. **ブラウザ 3D ビューア** — `poselab/webviewer/static/` (依存ゼロ、
   https://cfn0eft.github.io/3d/ に Pages 自動デプロイ)
3. **Pose3DStudio GUI** — ユーザーのデスクトップアプリ Pose3DStudio.exe の
   Web GUI ソース (`poselab/studio/gui/`)。`poselab-studio deploy` で exe へ配備

このリポジトリは Pose3DStudio.exe (mmpose ベースの GPU パイプライン、
ソース非公開のビルド済み exe) の機能を「公開 API を使った独自実装」として
移植してきた経緯がある。**コードはすべて独自実装、コピー禁止**の方針。

## よく使うコマンド

```bash
pip install -e ".[plot]" pytest ruff   # セットアップ (mediapipe 等が入る)
pytest tests/ -q                        # テスト (mmpose 不要、フェイク注入式)
ruff check poselab/ tests/              # lint (CI と同じ)
python -m build                         # パッケージビルド

poselab-viewer                          # ビューアをローカル起動
poselab-viewer --export-html out.html   # 自己完結 HTML

poselab-studio build --out dist/studio-gui          # exe GUI をビルド
poselab-studio deploy <exe>/_internal/gui           # exe へ配備 (要 F5)
# 配備先は環境変数 POSE3DSTUDIO_GUI でも指定可
```

## 絶対に守ること

- **PR のベースは必ず `main`**。スタック PR (ベース=フィーチャーブランチ) は
  マージしても main に入らない事故が実際に起きた (#9 → 回収 #10)
- **3D エンジンのソースは `poselab/webviewer/static/app.js` だけ**。
  「アプリ」セクション見出しより前がエンジン部で、`poselab-studio build` が
  そこを切り出して exe 用 app.js を生成する。エンジンを編集したら
  `poselab-studio deploy` で exe にも反映する。見出しマーカーを変えるときは
  `poselab/studio/__init__.py` の `ENGINE_END_MARKER` と
  `tests/test_studio.py` も更新
- **CHANGELOG.md とバージョン** (pyproject.toml + `poselab/__init__.py`) を
  機能追加のたびに更新。コミットメッセージは英語タイトル + 詳細本文
- ビューア/GUI の文言・ドキュメントは日本語

## 座標規約 (3D 表示で迷ったら)

- MediaPipe world / mmpose カメラ系 → **y 下向き** (ビューア軸 `ydown`、既定)
- mmpose 3D リフタ (`--pose3d`) の出力 → **z 上向き・x 反転・床基準**
  (ビューア軸 `zup`。MMPose 形式 JSON 読込時に自動選択)

## テスト方針

- mmpose / GPU は CI に無い → `tests/test_mmpose_backend.py` /
  `tests/test_pose3d.py` はフェイク inferencer 注入でロジックを検証
- ブラウザ側 (app.js) は Python テスト対象外。変更したら実ブラウザで
  デモ (`?demo=1`) とエクスポートのラウンドトリップを確認
- mmpose の実 API 仕様が必要なら Pose3DStudio.exe の
  `_internal/mmpose/` に同梱ソースがある (ユーザーのマシン上)

## Pose3DStudio.exe との関係 (ユーザー環境)

- exe は `_internal/gui/` の 3 ファイルをディスクから毎回配信 →
  `poselab-studio deploy` だけで GUI 更新可能 (exe 再ビルド不要)
- exe の HTTP サーバーは固定ルートのみ (`/`, `/app.css`, `/app.js` + API)。
  gui フォルダに新ファイルを足しても 404 → すべて app.js に連結する設計
- exe の出力 JSON は mmpose 形式 (`meta_info`/`instance_info`) で、
  ビューアがそのまま読める
