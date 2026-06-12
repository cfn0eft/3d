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
│   │   └ static/app.js  ★3D エンジンの唯一のソース★              │
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
- exe 自体は残っているが、GUI は本リポジトリからビルド・配備する。
  将来的には PyInstaller ビルドを CI 化して exe 自体をリポジトリから
  生成するのがロードマップの最終形

## 3D エンジンの単一ソース運用

`poselab/webviewer/static/app.js` は次の構成:

```
スケルトンカタログ → パーサ (parseAny) → デモ生成 → PoseStage レンダラ
→ エクスポート関数群 →  ←★ここまでエンジン★
/* ====…
   アプリ            ←★この見出しがマーカー (ENGINE_END_MARKER)★
   ====… */
ビューアの UI 配線 (DOM 依存)
```

- ビューアはこのファイルをそのまま使う
- Pose3DStudio GUI は `poselab-studio build` がマーカーより前を切り出し、
  IIFE (`window.PoseLab3D`) で包んで `studio/gui/app_main.js` と連結する
- つまり**エンジンのバグ修正・機能追加は 1 か所**。変更後は
  `poselab-studio deploy` で exe にも配る (`tests/test_studio.py` が
  マーカー存在と連結構造を守っている)

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
| ビューア | `poselab-viewer` → `?demo=1` でデモ、各形式をドロップ |
| エクスポート | ビューアでエクスポート → 出力ファイルを再度ドロップ (ラウンドトリップ) |
| exe GUI | `poselab-studio deploy <exe>/_internal/gui` → exe 画面で F5 → デモ再生 |
| パッケージ | `python -m build` で wheel に `webviewer/static` と `studio/gui` が入ること |

## リリース手順

1. CHANGELOG.md に追記、`pyproject.toml` と `poselab/__init__.py` の
   バージョンを同時に上げる
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
