# face-mosaic 開発TODO

仕様書（`face-mozaic-specs.md`）に対応したタスクリスト。Phase 1（Mac実装→GitHubへpush）→ Phase 2（Windowsでgit clone→追加実装）→ Phase 3（GUI化）の順で進める想定。

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
- [x] `.github/workflows/release.yml`作成（タグpush時に`macos-latest`でPyInstallerビルド→GitHub Releasesに添付。Windowsジョブの追加はPhase 2で行う）

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
- [ ] リモートリポジトリ（GitHub）へのpush及び `v0.1.0` タグ付け（ユーザー側でリモート作成後に実行）

## Phase 2: Windowsでの追加実装

- [ ] Windows機で`git clone`し、`requirements-common.txt` + `requirements-windows.txt`で環境構築
- [ ] `scripts/download_models.py`でモデルを取得し、Mac版と同じ動作をまず確認（CPU実行）
- [ ] Windows向けの追加実装は`windows-npu`ブランチ等で進める
- [ ] `onnxruntime-openvino`導入、OpenVINO EP／NPUへの切り替え
- [ ] Core Ultra 5 255HのNPU実行時の制約確認（対応opset、モデル変換要否、精度への影響）
- [ ] ONNX Runtime実行プロバイダー比較（OpenVINO EP／NPU vs DirectML EP／GPU）
- [ ] Mac版との処理速度・精度差の検証
- [ ] Windows版での色再現性の再検証（AVCLabsで見られた色ズレの再発有無を確認）
- [ ] Windows向けffmpeg導入方法をREADMEに追記
- [ ] `.github/workflows/release.yml`に`windows-latest`ジョブを追加し、Windows用アーティファクトもGitHub Releasesに添付されるようにする（NPU動作確認はCIではなく手元のWindows実機で行う）
- [ ] 動作確認後、`main`ブランチへ統合
- [ ] 仕様書「Phase 2」セクションへ検証結果を追記

## Phase 3: GUI化

- [ ] GUIフレームワーク（PySide6）での基本画面構築
- [ ] パラメータ設定UI実装（スライダー・プリセット保存）
- [ ] モザイク対象プレビュー・手動選択機能実装（サムネイル一覧、除外エリア描画）
- [ ] 画質調整UI実装（解像度・ビットレート・コーデック）
- [ ] 重い処理のスレッド/プロセス分離（UIフリーズ防止）
- [ ] PyInstallerでのMac版パッケージング検証
- [ ] PyInstallerでのWindows版パッケージング検証（onnxruntime/ffmpeg同梱まわり）

## バックログ（優先度未定・実運用次第で検討）

- [ ] 実際に発生した誤検知フレームを蓄積し、検出モデルのハードネガティブ追加学習
- [ ] フォルダ監視による新規動画の自動検知・自動処理
