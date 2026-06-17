# Changelog

## 0.9.8 (2026-06-17)

### 追加
- **平滑化フィルタを厳密化** (`poselab/filters.py`)。研究での前処理として
  選べる手法と頑健性を追加した。`--smooth-method` で選択する。
  - **`median`** — NaN 対応の移動メディアン。単発スパイク・外れ値に頑健
    (単純移動平均は外れ値に引きずられる問題への対処)
  - **`butter`** — ゼロ位相 Butterworth ローパス (前後 2 回適用の filtfilt、
    生体力学の標準)。`--smooth-cutoff [Hz]` と動画の fps から係数を計算。
    scipy 非依存の純 numpy 実装。欠損は内部補間してフィルタの連続性を保ち、
    書き戻すのは元から有効な位置だけ
  - **`--smooth-weighted`** — moving 法で visibility を重みにした加重平均。
    低信頼度フレームの寄与を下げる
  - 既定 (`moving`) は従来どおりの中央移動平均で挙動不変。平滑化設定は
    run-manifest / サマリの `smoothing` に記録する

## 0.9.7 (2026-06-17)

### 追加
- **高次の運動学的特徴量を追加** (`poselab/kinematics.py`)。研究でよく使う
  指標を計算・出力できるようにした。
  - **加速度・ジャーク** — `--velocity-csv` の出力に `accel_px_per_s2` /
    `accel_m_per_s2` / `jerk_m_per_s3` 列を追加 (速度の 2 階・3 階差分)。
    高次微分はノイズに敏感なため `--smooth` 併用を推奨
  - **角速度** — `--angles-csv` の出力に `angular_velocity_deg_per_s` 列を追加
  - **左右対称性** — `--symmetry-csv` を新設。対応する左右の関節角度
    (肘・肩・股・膝・足首) から Symmetry Index を計算 (0=左右対称、
    大きいほど非対称。リハビリ・左右差評価向け)
  - **歩行リズムの推定** — 足首の上下動の自己相関から cadence (毎分サイクル
    数) と 1 周期を推定し、`--summary-json` の `gait` に記録 (推定できない
    場合は誤値を出さず省略)
  - いずれも既存 CSV は列追加のみで後方互換。`kinematics.py` は numpy のみ依存

## 0.9.6 (2026-06-17)

### 追加
- **低信頼度キーポイントのマスキング出力 `--mask-visibility T` を追加**。
  visibility が T 未満のキーポイント座標を欠損 (CSV は空欄、NPZ は NaN)
  として書き出す。信頼度の低い (0,0) などの推定値を「原点で検出された」と
  誤読するのを防ぐ (研究データの誠実性)。0=無効 (既定) で従来挙動は不変。
  CSV ロング/ワイド・NPZ に適用し、visibility / presence 列は残すので欠損
  理由を追跡できる
- **スコアの意味 (score_semantics) と欠損値規約 (missing_value) を run-manifest
  に記録**。MediaPipe の visibility/presence と mmpose (RTMPose) の検出スコアは
  意味が異なるため、バックエンド横断比較の誤りを防ぐ目的で出力メタに明記する。
  `--mask-visibility` の閾値も `export.mask_visibility` として記録

## 0.9.5 (2026-06-17)

### 追加
- **実行来歴 (run-manifest) の記録を追加** — 研究での再現性・引用可能性を
  高めるため、推定実行のメタ情報を 1 つの JSON にまとめて自動保存する
  (`poselab/provenance.py`)。記録内容は実行日時 (UTC)・コマンドライン・
  全 CLI 引数・バックエンド/モデル名・**入力ファイルの SHA-256 とサイズ**・
  実行環境 (Python / OpenCV / MediaPipe / mmpose / PyTorch のバージョン、
  CUDA 有無)・**出力データの単位 (px / m / 確率)** と**座標系の定義**・
  フレームレート / タイムスタンプの由来・トラッキングの有無。
  - 出力があれば既定で `<出力名>.run.json` に自動保存。`--run-manifest PATH`
    で保存先を指定、`--no-manifest` で無効化できる。`--auto-output` 時は
    各 `<名前>_poselab/` フォルダに保存される
  - JSON / サマリ / NPZ のメタデータにも単位・座標系・来歴の要約を埋め込む
    ようにし、データファイル単体でも自己記述的になった
    (`NpzExporter` に `metadata` 引数を追加し `metadata_json` として同梱)
  - これまで MediaPipe モデルが `.../latest/...` 参照で静かに入れ替わっても
    記録が残らなかった問題に対し、解決済みバージョンと入力ハッシュを残す

## 0.9.4 (2026-06-17)

### 修正
- **PoseLab Studio GUI の表示文言を日本語へ統一**。システムチェック
  (preflight) の警告 (`Input video is not set.` /
  `ffmpeg not found: H.264 re-encode will be skipped.` など) が英語のままで、
  日本語 GUI と不統一だった。サーバー側 (`server.py`) の preflight 警告と
  ジョブ投入・キュー・ダウンロードのエラーメッセージ、フロント側
  (`gui/app_main.js` / `gui/index.html`) の状態表示・アラート・ログ
  (`Run Pipeline` → `実行`、`Idle` → `待機中`、`Disconnected` → `切断`、
  キュー/履歴/ダウンロードの各ステータス等) をすべて日本語化した。
  固有名詞 (MMPose 等)・`value` 属性・座標系メタ値などの技術値は据え置き。
  併せて preflight 警告の日本語化に追従するよう
  `tests/test_studio_server.py` のアサーションを更新
- **ビューア操作部のカメラ回転ラベルを分かりやすく変更**。専門用語の
  カタカナ表記だった `ヨー` / `ピッチ` を `水平回転` / `垂直回転` にし、
  ビューアのヒント文 (「ドラッグ 回転」) と表現を揃えた
  (`gui/index.html`)

## 0.9.3 (2026-06-13)

### 修正
- **MMPose 3D の可視化動画 (`--save-video`) が matplotlib 3.10+ で
  クラッシュする不具合を修正**。mmpose の 3D 可視化
  (`local_visualizer_3d`)は matplotlib 3.8 で非推奨化・3.10 で削除された
  `FigureCanvasAgg.tostring_rgb()` を呼ぶため、新しい matplotlib では
  `AttributeError: 'FigureCanvasTkAgg' object has no attribute 'tostring_rgb'`
  で 3D 推定 (`--pose3d --save-video`)が中断していた。削除された
  `tostring_rgb` を `buffer_rgba` から再現する互換シムを追加し、併せて
  オフスクリーン描画用の非対話 Agg バックエンドへ切り替えるようにした
  (`poselab/_mpl_compat.py`)。これにより 2D+3D 可視化動画の出力が
  matplotlib のバージョンに依存せず動作する

## 0.9.2 (2026-06-13)

### 追加
- **アプリアイコンを同梱** (`packaging/installer/poselab.ico`、GUI と同じ
  ブランドマークを Pillow で生成: `packaging/make_icon.py`)。ローカル
  インストーラ (`install_local.ps1`) と Inno Setup インストーラ
  (`poselab_studio.iss`) が、スタートメニュー / デスクトップのショートカット
  および Setup.exe / アンインストール項目にこのアイコンを設定するようにした
- **ローカルインストールにアンインストーラを追加** —
  `packaging\installer\Uninstall-PoseLabStudio.cmd`
  (内部は `uninstall_local.ps1`)。インストール先一式とショートカットを削除し、
  `-Cache` 指定でダウンロード済みモデル重み (`~/.cache/poselab`,
  `~/.cache/torch`) も削除する

## 0.9.1 (2026-06-13)

### 修正
- **MMPose 3D パイプラインの既定 2D モデル名を修正**。既定値が旧命名規則の
  `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192` のままで、mmpose>=1.2 の
  metafile に該当名が無く `ValueError: Cannot find model: ... in mmpose` で
  推定・モデル事前ダウンロード(`--pose3d --prepare-models`)が失敗していた。
  現行 metafile に登録されている `rtmpose-m_8xb256-420e_aic-coco-256x192`
  (RTMPose-M / AIC+COCO 事前学習 / 256x192)へ変更(高精度プロファイルの
  `rtmpose-l_8xb256-420e_aic-coco-384x288` と命名規則を統一)

## 0.9.0 (2026-06-13)

### 追加
- **PoseLab Studio で推定バックエンドを選択可能に**。従来の
  **MMPose 3D**(RTMDet + RTMPose 2D → VideoPose3D 3D リフティング、GPU 推奨)
  に加え、**MediaPipe**(33 点・GPU 不要の軽量 2D/3D)を GUI の
  「バックエンド」から選べるようにした
  - MediaPipe 選択時はモデルサイズ(lite / full / heavy)と最大人数を
    指定でき、MMPose 専用のモデル/検出器プロファイルは自動で隠れる
  - 出力 JSON は poselab 形式(`world_keypoints` 入り)で、そのまま 3D
    ビューアで再生できる。サマリ表示(フレーム数 / 平均人数 / 平均スコア)も
    両形式に対応
  - 「モデルダウンロード」「システムチェック」も選択バックエンドに追従
    (MediaPipe では mmpose 未導入の警告を出さず、`--prepare-models` で
    Pose Landmarker の `.task` を事前取得)
  - CLI の `poselab --backend mediapipe --prepare-models` で MediaPipe
    モデルだけを事前ダウンロードできるようにした

## 0.8.0 (2026-06-13)

### 追加
- **PoseLab Studio の「モデルダウンロード」パネルを機能化**。推定モデル
  (人物検出 RTMDet / 2D RTMPose / 3D VideoPose3D)を**一覧表示**し、
  「ダウンロード」ボタンで**事前取得**できる(進捗は SSE でライブ表示、
  取得済みは再起動後も ready 表示)。未取得でも従来どおり Run 時に自動取得
  - CLI に `poselab --pose3d --prepare-models` を追加(動画不要でモデルの
    重みを事前ダウンロードして終了)。サーバーはこれをサブプロセス実行する
- 推論サブプロセスで `PYTHONWARNINGS=ignore` を設定し、mmengine の
  `pkg_resources is deprecated` 警告などがログに出ないように整理

## 0.7.3 (2026-06-13)

### 追加
- `Install-PoseLabStudio.cmd` が起動時に `git pull` で**自動更新**してから
  インストールするように変更。毎回手動で pull する必要がなくなった
  (更新できない場合は現在のバージョンで続行)

### 修正
- uv 駆動インストールで、uv のシード版 setuptools に `pkg_resources` が
  含まれず mmengine の import が `ModuleNotFoundError: No module named
  'pkg_resources'` で失敗する不具合を修正。venv 作成後に PyPI の
  setuptools (<81、pkg_resources 同梱) と wheel を入れ直す

## 0.7.2 (2026-06-13)

### 変更
- ローカルインストーラー (`install_local.ps1`) を **uv 駆動**に刷新。
  インストール状況が uv のきれいな進捗表示 (スピナー / 進捗バー /
  "Downloaded ..." / "Installed N packages in Xms") で可視化される
  - uv が専用 Python 3.11 と venv も用意するため **winget 不要**に
  - uv のターゲットは `VIRTUAL_ENV` 環境変数で渡し、numpy 制約はインライン
    指定にして、**スペースを含むパスでも安全** (引数にスペース入りパスを
    渡さない)。mmcv は `--find-links` で torch に合う wheel を取得
  - uv 本体は Astral 署名済みバイナリを取得して使用 (SAC でブロックされない)

## 0.7.1 (2026-06-13)

### 修正
- インストール先パスにスペースが含まれる (既定の
  `%LOCALAPPDATA%\PoseLab Studio` 等) と pip が
  `Could not open requirements file: ...\PoseLab` で失敗する不具合を修正。
  原因は **`PIP_CONSTRAINT` 環境変数が空白で分割される**仕様で、制約ファイルの
  パスが途中で切れていた
  - `install_local.ps1` / `setup_env.ps1` とも、制約は環境変数ではなく
    `--constraint <file>` 引数で渡すように変更 (引数なら空白入りパスでも
    1 つの値として正しく渡る)
  - CI の検証をスペース入りパスで実行するようにして、この種の回帰を検出

## 0.7.0 (2026-06-13)

### 変更 (デザイン刷新)
- **UI を「洗練ミニマル」(Linear / Vercel 系) に全面リデザイン**。
  PoseLab Studio GUI (`poselab/studio/gui/`) とブラウザ 3D ビューア
  (`poselab/webviewer/static/`) を共通のデザイントークンで統一
  - 配色をシアン→バイオレットのネオングラデーションから、単色の
    落ち着いたインディゴ (アクセント) + 近黒の背景に変更
  - ガラス(過剰な blur)・大きなグロー・虹色グラデーションを排し、
    1px のヘアライン罫・余白・控えめな影でフラットに整理
  - フォントを Inter に変更 (データ列は IBM Plex Mono のまま)
  - 機能・DOM 構造・要素 ID は不変 (CSS とフォント指定のみの変更)
- インストーラーのコンソール出力も同じトーンに整理 (ASCII のまま、
  すっきりしたヘッダ + ステップ表示)

## 0.6.5 (2026-06-13)

### 修正
- インストールスクリプトが Windows PowerShell 5.1 で、ネイティブコマンドの
  stderr 出力(例: `py` の "No suitable Python runtime"、pip の警告)を
  きっかけに即異常終了する不具合を修正。`$ErrorActionPreference='Stop'` の
  下では PS5.1 が stderr 書き込みを終了エラー化するため、Python 3.11 が
  無い環境で winget による自動導入に進む前に落ちていた
  - `$ErrorActionPreference` を `Continue` にし、失敗判定は明示的な終了
    コードチェックに統一(`install_local.ps1` / `setup_env.ps1`)。ネイティブ
    出力は `2>&1 | Out-Host` で取り込み、重要なファイル操作のみ
    `-ErrorAction Stop`
  - これにより Python 3.11 未導入の環境でも winget で自動導入して継続する

## 0.6.4 (2026-06-13)

### 修正
- インストールスクリプト (`install_local.ps1` / `setup_env.ps1` /
  `Install-PoseLabStudio.cmd`) が**日本語 Windows で文字化けして構文エラーに
  なる / cmd が行を誤読する**不具合を修正。原因は UTF-8 (BOM なし) + LF 改行で、
  Windows PowerShell 5.1 がロケールの ANSI コードページとして誤デコードし、
  cmd.exe が LF を誤読していた (例: `powershell` が `ershell` に化ける)
  - スクリプトを **ASCII (英語メッセージ)** に統一し、`.gitattributes` で
    `*.cmd` / `*.ps1` を **CRLF 改行**に固定。文字コード依存を排除
  - venv に `wheel` を先に入れて chumpy の `--no-build-isolation` ビルドを通す
    (0.6.3 から継続)

## 0.6.3 (2026-06-13)

### 追加
- **ローカルインストールスクリプト** (`packaging/installer/install_local.ps1`
  + ダブルクリック用 `Install-PoseLabStudio.cmd`): 未署名の .exe を実行
  しないため **Smart App Control にブロックされない**配布物。署名済みの
  cmd / powershell / Python だけを使い、専用 venv に GPU(または CPU)版
  PyTorch・mmpose・poselab を導入してスタートメニュー / デスクトップに
  ショートカットを作る
  - Python 3.11 が無ければ winget で導入を試みる。NVIDIA GPU を自動判定
    (CUDA 11.8 / CPU)。`-Cpu` / `-Gpu` で上書き可
  - CI が CPU モードで実走 (venv + torch + mmpose + poselab + import) して
    検証する
- インストーラー .exe が SAC でブロックされる件を README に記載

## 0.6.2 (2026-06-13)

### 修正
- オンラインインストーラーの `setup_env.ps1` が mmpose 一式
  (chumpy / openmim / mmengine / mmcv / mmdet / mmpose) と最終動作確認を
  **実際には実行していなかった**不具合を修正。ヘルパー関数の引数名
  `$Args` が PowerShell の自動変数 `$args` と衝突し、`python` が引数なしで
  起動 (即終了・終了コード 0) していたため、未導入のまま「成功」して
  いた。0.6.1 のインストーラーでは 3D 推定が動かない。CI の検証ジョブが
  mmpose のインストールと `Pose3DInferencer` の import を実通しするように
  なり、回帰を検出できる

## 0.6.1 (2026-06-13)

### 追加
- **オンラインインストーラー** (`PoseLabStudioSetup.exe`、数十 MB):
  巨大な同梱 exe (約2.85GB) のダウンロードを避け、小さなインストーラーが
  実行時に必要なものを構築する方式 (`packaging/installer/`)
  - `uv` で専用 Python 3.11 と仮想環境を用意 (システムの Python に触れない)
  - **NVIDIA GPU を自動判定**: 搭載機は CUDA 11.8 版 PyTorch、非搭載機は
    CPU 版 (約200MB) を選択。mmcv も torch に合わせて mim が自動選択
  - mmpose 一式と poselab (同梱 wheel) を導入し、import 動作確認まで実施
  - スタートメニュー / デスクトップにショートカット、アンインストーラ付き
  - 取得は PyPI / PyTorch の高速 CDN 経由でレジューム可。`uv` がキャッシュ
  - `.github/workflows/build-installer.yml` が CI でビルド。実機検証として
    **CPU レシピを最後まで実行**してから Inno Setup でインストーラーを生成。
    `vX.Y.Z` タグ時は GitHub Release に直リンクで添付
- `setup_env.ps1` は `-Cpu` / `-Gpu` で PyTorch 種別を手動指定可能

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
