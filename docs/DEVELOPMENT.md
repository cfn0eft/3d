# 開発メモ (DEVELOPMENT)

別の環境・別のセッションから開発を引き継ぐためのドキュメント。
要点だけ知りたい場合は [CLAUDE.md](../CLAUDE.md) を先に読むこと。

## 全体図

```
┌─ poselab (このリポジトリ、pip パッケージ) ─────────────────────┐
│                                                                  │
│  poselab/            推定エンジンと CLI                          │
│   ├ backends/        mediapipe (標準) / mmpose (オプション)      │
│   ├ pose3d.py        --pose3d: 動画→3D リフティング (mmpose)     │
│   ├ cli.py           poselab コマンド (--auto-output --outputs)  │
│   ├ gui.py           Tkinter GUI (poselab-gui)                   │
│   ├ webviewer/       ブラウザ 3D ビューア (poselab-viewer)       │
│   │   ├ static/engine.js ★3D エンジンの唯一のソース★            │
│   │   └ static/app.js    ビューアの UI 配線                      │
│   └ studio/          Pose3DStudio.exe の GUI ソースとビルダー    │
│       ├ gui/         index.html / app.css / app_main.js          │
│       └ __init__.py  poselab-studio build/deploy                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
          │ Pages 自動デプロイ                 │ poselab-studio deploy
          ▼                                    ▼
  https://cfn0eft.github.io/3d/        Pose3DStudio.exe\_internal\gui\
  (公開ビューア)                        (ユーザーのデスクトップアプリ)
```

## 経緯 (なぜこうなっているか)

- ユーザーは **Pose3DStudio.exe** という mmpose ベースの 3D 姿勢推定
  デスクトップアプリ (PyInstaller 製、ソースなし) を持っている
- その機能を本リポジトリへ「**ワークフローを参考にした独自実装**」として
  段階的に移植してきた:
  - v0.3.0: 出力ワークフロー (wide CSV / auto-output / H.264) + ブラウザ
    3D ビューア + GitHub Pages 公開
  - v0.4.0: 推定エンジン (`--backend mmpose`、`--pose3d` 3D リフティング)
  - v0.5.0: エクスポートの個別選択 (ビューアのパネル + `--outputs`)、
    exe GUI ソースの本リポジトリへの収容 (`poselab/studio/`)
  - v0.5.1: エンジンを engine.js に物理分割 (マーカー切り出しを廃止)、
    Node スモークテスト追加、バージョンの単一ソース化
- exe 自体は残っているが、GUI は本リポジトリからビルド・配備する。
  将来的には PyInstaller ビルドを CI 化して exe 自体をリポジトリから
  生成するのがロードマップの最終形

## 3D エンジンの単一ソース運用

3D エンジン (骨格カタログ / パーサ parseAny / デモ生成 / PoseStage
レンダラ / エクスポート関数群) は
**`poselab/webviewer/static/engine.js` が唯一のソース**:

- **ビューア**: index.html が `engine.js` → `app.js` (UI 配線、DOM 依存)
  の順に読み込む。`--export-html` は両方を 1 つの HTML に埋め込む
- **Pose3DStudio GUI**: `poselab-studio build` が engine.js を IIFE
  (`window.PoseLab3D`) で包み、`studio/gui/app_main.js` と連結して
  exe 用の単一 app.js を生成する (exe 側の固定ルート制約のため 1 ファイル)
- つまり**エンジンのバグ修正・機能追加は 1 か所**。変更後は
  `poselab-studio deploy` で exe にも配る (`tests/test_studio.py` が
  連結構造を守っている)
- パーサ / エクスポートは DOM 非依存の純関数なので、Node スモークテスト
  (`tests/engine_smoke.mjs`、pytest の `tests/test_engine_js.py` から自動
  実行) が「デモ生成 → 全形式エクスポート → 再パース」のラウンドトリップを
  CI で検証する。PoseStage の描画だけは実ブラウザで確認する

## Pose3DStudio.exe の技術メモ (ユーザー環境)

- 起動するとローカル HTTP サーバー (既定 7860、塞がっていれば 7861…) で
  GUI を配信し、ブラウザ/ウィンドウを開く
- **gui ファイルはリクエストごとにディスクから読む** → 配備後 F5 で反映
- サーバーは固定ルートのみ: `/`, `/app.css`, `/app.js`,
  API (`/run /enqueue /status /events /preflight /file /summary /open
  /pick-video /pick-videos /pick-folder /queue-move /clear-queue /cancel
  /gpu`)。**新しい静的ファイルは配信されない** → app.js に全部入れる
- GUI が叩く API ペイロードは `studio/gui/app_main.js` の `buildPayload()`
  参照 (モデル/検出器プロファイルのチェックポイントパス込み)
- exe の `_internal/` には mmpose/mmdet/mmengine の **ソースがそのまま同梱**
  されている。mmpose API の正確な挙動を確認したいときに読める
  (例: `_internal/mmpose/apis/inferencers/pose3d_inferencer.py`)
- 出力 results JSON は mmpose 形式。3D リフタ出力は z-up (x 反転・床基準)

## 開発環境セットアップ

```bash
git clone https://github.com/cfn0eft/3d && cd 3d
pip install -e ".[plot]" pytest ruff build
pytest tests/ -q          # 80+ 件、mediapipe は必要 / mmpose は不要
ruff check poselab/ tests/
```

mmpose バックエンドを実際に動かす場合 (GPU 推奨):

```bash
pip install -U openmim
mim install "mmcv>=2.0.1,<2.2" "mmdet>=3.1,<3.3" "mmpose>=1.2,<1.4"
poselab --info    # mmpose の検出状況を確認
```

既知の制約: Windows + Python 3.12 には mmcv<2.2 のホイールが無い
(exe は Python 3.11 同梱)。実推論の検証は 3.9–3.11 か Linux で。

## 検証レシピ

| 対象 | 方法 |
| --- | --- |
| Python 全般 | `pytest tests/ -q` + `ruff check poselab/ tests/` |
| 3D エンジン | `pytest tests/test_engine_js.py` (Node ラウンドトリップ) |
| ビューア | `poselab-viewer` → `?demo=1` でデモ、各形式をドロップ |
| エクスポート | ビューアでエクスポート → 出力ファイルを再度ドロップ (ラウンドトリップ) |
| exe GUI | `poselab-studio deploy <exe>/_internal/gui` → exe 画面で F5 → デモ再生 |
| パッケージ | `python -m build` で wheel に `webviewer/static` と `studio/gui` が入ること |

## リリース手順

1. CHANGELOG.md に追記、`poselab/__init__.py` の `__version__` を上げる
   (pyproject.toml は dynamic 参照なので触らない)
2. ブランチを push → **base=main の PR** を作る (スタック PR 禁止 —
   #9 が main に入らない事故の教訓)
3. CI (test ×3 / ruff / build) が緑になってからマージ
4. `poselab/webviewer/**` に変更があれば Pages が自動再デプロイ
   (https://cfn0eft.github.io/3d/)

## ロードマップ

- [ ] Web GUI 統合 (`poselab-web`): Pose3DStudio 風のジョブ管理 UI を
      poselab のパイプラインに接続 (studio/gui がベースになる)
- [ ] PyInstaller ビルドの CI 化 — exe をリポジトリから生成して完全移設
- [ ] mmpose バックエンドの GPU 実機スモークの定期実行
