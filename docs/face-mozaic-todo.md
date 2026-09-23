# face-mosaic 開発TODO

仕様書（`face-mozaic-specs.md`）に対応したタスクリスト。Phase 1（Mac実装）→ Phase 2（Windows / NPU対応）→ Phase 3（GUI化）まで全完了。

## 事前確認（実装着手前）

- [x] AVCLabs Video Blur AIの現行設定値を確認（デフォルト値を反映）
- [x] アプリ名を最終決定（`face-mosaic`）
- [x] Gitリポジトリを初期化・準備

## Phase 1: Mac向け実装

### 環境構築・リポジトリ準備
- [x] Python環境構築（venv、依存パッケージ管理方針の決定）
- [x] 主要ライブラリ導入：`onnxruntime`, `opencv-python`, `numpy`, ffmpeg（brew経由）
- [x] SCRFDモデル（ONNX、2.5G/10G/34G等）の入手・動作確認
- [x] `.gitignore`作成（`models/`, `venv/`, `__pycache__/`, テスト用動画, ローカル設定ファイル等を除外）
- [x] `scripts/download_models.py`（モデル取得スクリプト）作成（Windows側でも同じスクリプトで取得できるようにする）
- [x] `requirements-common.txt` / `requirements-mac.txt` に分けて依存関係を記録
- [x] `.github/workflows/ci.yml`作成（push/PR時のLint・テスト実行）
- [x] `.github/workflows/release.yml`作成（タグpush時に`macos-latest`でPyInstallerビルド→GitHub Releasesに添付）

### コア機能：顔検出・ぼかし
- [x] 動画入出力パイプライン実装（**ffmpeg経由**での読み込み・書き出し。音声・回転メタデータ・色空間タグを保持）
- [x] SCRFDによる顔検出モジュール実装（ONNX Runtime推論ラッパー）
- [x] フレーム間トラッキング／検出結果の平滑化実装（IoUベース等）
- [x] 前後フレームへのぼかし延長（パディング、前後個別設定・デフォルト3、フレーム数ベース）実装
- [x] ぼかし処理実装（ガウス／モザイク切り替え、強度・ブロックサイズパラメータ、マージン設定）
- [x] 出力エンコード実装（HEVC、CRFベース、初期値CRF 18〜20で実測調整）

### 誤検知対策
- [x] 検出信頼度閾値・最小顔サイズ・ランドマーク品質によるフィルタ実装
- [x] 人間 vs 非人間（動物除外）二値分類器の選定・組み込み
- [x] 肌色判定（YCbCr、彫像・銅像除外）実装
- [x] 写真・絵画除外機能（複合条件、デフォルトOFFオプション）実装
- [x] 固定除外エリア（マスク）指定の仕組み実装
- [x] **上記の誤検知対策をトラック単位（フレーム単位ではない）で判定する仕組みとして実装**（ちらつき防止）

### 色再現性
- [x] 入力動画の色空間メタデータ（色域・伝達特性・レンジ）取得・出力への継承実装（ffmpeg呼び出し部分）
- [x] **HDR（Dolby Vision）の静的タグ継承対応**（ダイナミックメタデータ/RPUの再現は対象外と明確化）
- [x] 色差検証スクリプト作成（入出力フレーム比較、平均絶対誤差／ΔE2000等）
- [x] ハードウェアデコード（VideoToolbox等）使用時の色space挙動検証
- [x] AVCLabs（Ryzen実行時）で見られたような色ズレが発生していないことを確認

### バッチ処理・パラメータ
- [x] 単一ファイル／複数ファイル選択／フォルダ指定の3パターン対応
- [x] 処理進捗表示・ログ出力実装
- [x] **エラー発生時はスキップして続行する挙動の実装**（成功/失敗一覧のサマリー出力を含む）
- [x] パラメータのCLI引数／設定ファイル（YAML/JSON）対応

### 検証・引き渡し準備
- [x] 実際の散歩動画・合成動画で一通りのE2Eテスト実施
- [x] 誤検知パターン（彫像・広告・動物等）の実例収集、パラメータ調整
- [x] 各SCRFDモデルサイズでの速度・精度の実測比較（2.5G: 66.2 FPS / 10G: 30.2 FPS / 34G: 14.3 FPS）
- [x] README作成（Mac向けセットアップ手順・使い方・Gatekeeper警告の回避方法）
- [x] Git初期コミット作成

## Phase 2: Windowsでの追加実装

- [x] `requirements-windows.txt` での Windows 環境設定整備
- [x] `scripts/download_models.py` のマルチOS完全互換（PowerShell / cmd / sh）
- [x] OpenVINO EP / DirectML EP / CPU の優先度フォールバックチェーン実装（`detector.py`）
- [x] Intel Core Ultra NPU / DirectML GPU の自動プロバイダ識別ログ出力
- [x] Windows向け ffmpeg 導入手順（winget / choco）の README 記載
- [x] `.github/workflows/release.yml` に `windows-latest` マトリックスを追加し、Windows用自動ビルド (.zip) を設定
- [x] 仕様書（`face-mozaic-specs.md`）へ Phase 2 実装完了内容を追記

## Phase 3: GUI化

- [x] GUIフレームワーク（PySide6）でのモダンデスクトップ画面構築（`face_mosaic/gui/app.py`）
- [x] パラメータ設定UI実装（モデル選択、ぼかし方式、強度スライダー、前後パディング、CRF）
- [x] 単一動画・複数ファイル・フォルダ選択UIとファイル件数表示
- [x] 画質調整UI（HEVC / H.264、CRF品質）
- [x] 非同期ワーカースレッド（`QThread`）による重い処理の分離（UIフリーズ防止・キャンセル機能）
- [x] リアルタイムプログレスバー・実行ログビューア
- [x] CLIからの `--gui` フラグおよび引数なし起動でのGUI自動ディスパッチ
- [x] GUI単体テスト（`tests/test_gui.py`）の作成・パス

## Phase 3.1: GUI UX強化・堅牢化（キュー管理・プレビュー・完全サイレント起動）

- [x] ルート起動スクリプトの集約（`run.bat` のみ、`scripts/run_gui.vbs` への委譲による完全サイレント起動）
- [x] 変換中・コーデックプローブ時の `cmd.exe` ポップアップ完全抑制（`CREATE_NO_WINDOW` & `SW_HIDE`）
- [x] マシンスペック自動診断とモデル自動推薦（RAM 32GB / Core Ultra / GPU 検出）
- [x] 2フェーズ・リアルタイム映像プレビュー（`VideoPreviewWidget`、Pass 1 検出バッジ / Pass 2 モザイク完成形描画）
- [x] 動画処理キュー（Queue Table）の動的追加・削除・状態管理
- [x] キューアイテムの一意ID（UUID）管理とインプレースセル更新（行インデックスずれ防止）
- [x] 選択項目対応の変換実行（`▶ Start Processing (N selected)`、キャンセル済み/エラー状態の自動復帰）
- [x] Start/Stop ボタンの安全なステートマシン（待機中 / 実行中 / 停止処理中（非活性） / 完了）
- [x] 個別ファイルキャンセル（`⏭ Skip / Cancel Current File`）およびコンテキストメニュー強化
- [x] キュー内ファイルの処理順序変更機能（マウスドラッグ＆ドロップ並び替え、▲ Up / ▼ Down ボタン、右クリック先頭・末尾移動、実行中ワーカースレッドとの動的同期）
- [x] 前回開いたディレクトリの完全永続記憶（ファイル/フォルダダイアログ・ドラッグ＆ドロップ両対応、`QSettings.sync` 確実化、テスト環境分離）
- [x] ファイルログ記録（`logs/face_mosaic.log`）および「📋 Copy Log」「📁 Open Log File」ボタン
- [x] キュー削除・選択・停止状態遷移・ディレクトリ記憶・処理順序並び替えの自動テスト追加（49テスト全件通過）

## Phase 3.2: 4K60fps長尺動画対応・OOM防止・チェックポイント＆レジューム・CPU/GPU/NPU最適化

- [x] 長尺4K60fps動画での OOM 解消（NumPy スライスクロップによる生バッファ参照リークの遮断、最大 128px リサイズ＋`.copy()`、トラックレット保持上限 35 枚）
- [x] OpenCV `blobFromImage` アロケータ制約の解消（純粋 NumPy 演算による前処理移行）
- [x] プレビュー画像転送時のメモリ抑制（640px 幅への事前縮小と解放）
- [x] 中間チェックポイント機構（`face_mosaic/checkpoint.py`）の実装
  - 出力先と同階層の隠しフォルダ `.{output_name}.checkpoint/` 管理
  - `state.json` および JSONL 形式の `pass1_detections.jsonl`（1000フレーム約20KB、不要キャッシュ防止）
  - 中断時の安全な再開（Pass 1 途中復帰、Pass 2 スキップ復帰）
  - 処理完了時の一時フォルダ完全自動削除とアトミックリネーム（`.part.mp4` -> `.mp4`）
- [x] OpenVINO ネイティブ推論の統合（Intel Core Ultra 5 225H / Intel Arc 130T GPU / Intel AI Boost NPU）
  - `MULTI:GPU,NPU` による 135〜155+ FPS の高速推論
  - NPU 向け静的モデルリシェイプ（`[1, 3, 640, 640]`）対応
- [x] Pass 1 動画読み込みデコード最適化（FFmpeg `-vf scale=640:360` ＋ `-threads 0` により 2.4 FPS → 約 60 FPS へ 25倍高速化）
- [x] Pass 2 レンダリング 3ステージ並行非同期パイプライン化（`cv2.VideoCapture` 高速デコード ＋ メインスレッド描画 ＋ `VideoWriter` 非同期 QSV エンコードにより 2.5 FPS → 20〜30+ FPS へ 最大10倍高速化）
- [x] 10-bit HDR / BT.2020 色再現性の完全保証（10-bit `p010le` 自動適用、`-bsf:v hevc_metadata` による VUI ビットストリーム注入、色味変化の完全防止）
- [x] 単体・結合テストの追加・更新（チェックポイント、ライフサイクル、レジューム、メモリ分離、GUIログコピー、VUI BSFタグ等 全63件テスト通過）

## Phase 3.3: マルチプラットフォーム（Windows/macOS）リリース自動化・配布ZIPパッケージ整備

- [x] Windows 向け起動バッチのリネームと整理（`start-app.bat` 新設、初回進捗表示 / 2回目以降サイレント起動、`run.bat` 互換転送）
- [x] macOS 向けワンクリックランナーの実装（`start-app.command` および `scripts/setup_and_run_mac.sh`、Python3/venv/ffmpeg自動検証、モデルダウンロードからGUI起動まで自動化）
- [x] GitHub Actions リリースワークフロー（`.github/workflows/release.yml`）の刷新
  - Windows用ジョブ（`package-windows`）：Windows専用ファイルのみをステージングして `face-mosaic-windows.zip` 生成
  - macOS用ジョブ（`package-macos`）：macOS専用ファイルのみをステージング、`chmod +x` で実行権限を保持したまま `face-mosaic-macos.zip` 生成
  - タグpush時の GitHub Releases 自動アタッチおよびワークフロー成果物アップロード
- [x] macOS Gatekeeper 対策とドキュメント整備（Finder「右クリック→開く」および `xattr` 解除手順を明記）
- [x] 仕様書・TODO・README の完全同期更新

## Phase 3.4: ハードウェア自動検知とマルチベンダー最適化（Apple Silicon / AMD Ryzen / Intel Arc）

- [x] ハードウェア種別判定の実装（`face_mosaic/hardware.py` に `get_hardware_vendor()` 追加）
  - Apple Silicon（M1/M2/M3/M4・Unified Memory）、AMD Ryzen（Zen AVX2/AVX-512）、Intel Core/Arc を自動判別
- [x] Apple Silicon 最適化
  - Pass 1 推論：ONNX Runtime で `CoreMLExecutionProvider`（Apple Neural Engine & Metal GPU）を最優先
  - Pass 2 エンコード：Apple VideoToolbox（`hevc_videotoolbox` / `h264_videotoolbox`）ハードウェアエンコード対応（10-bit HDR / `main10` 対応）
- [x] AMD Ryzen / Radeon 最適化
  - Pass 1 推論：Radeon GPU での `DmlExecutionProvider` (DirectML)、CPU 時は Zen マルチスレッド（`intra_op_num_threads`）最適化
  - Pass 2 エンコード：AMD AMF（`hevc_amf`）対応および CPU ソフトウェアエンコード時の `-threads 0` 全コア活用
- [x] Intel Arc / Core Ultra 最適化の完全維持
  - OpenVINO (`MULTI:GPU,NPU`) および Intel QSV (`hevc_qsv`) の既存最適化パスを 100% 保持
- [x] エンコーダーおよびハードウェア検知の単体テスト（`tests/test_hardware.py`, `tests/test_io_encoders.py`）追加（全70件通過）

## バックログ（優先度未定・実運用次第で検討）

- [ ] 実際に発生した誤検知フレームを蓄積し、検出モデルのハードネガティブ追加学習
- [ ] フォルダ監視による新規動画の自動検知・自動処理



