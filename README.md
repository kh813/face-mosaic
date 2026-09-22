# face-mosaic

街中散歩動画（iPhone 4K60fps等）の顔に自動でモザイク／ぼかしを適用する、高速・完全ローカル動作のAI CLIツールです。

- **完全ローカル完結**：クラウド送信なし（プライバシー・著作権配慮）
- **高精度顔検出**：SCRFD (ONNX Runtime / CoreML & CPU高速推論)
- **ちらつき防止・前後フレームパディング**：IoU追跡 ＋ 前後Nフレームへのぼかし延長
- **誤検知防止（トラック単位判定）**：YCbCr肌色判定（彫像・銅像除外）、動物除外判定、写真・ポスター除外
- **色再現性（カラーマネジメント）**：BT.2020 / BT.709 静的メタデータタグ・レンジの完全継承
- **バッチ処理**：単一動画／複数指定／フォルダ一括処理、エラー時スキップ＆サマリー出力

---

## 1. 必要要件

- **macOS** (Apple Silicon / Intel) または **Windows**
- **Python 3.10以上**
- **ffmpeg**
  - macOS: `brew install ffmpeg`
  - Windows: `choco install ffmpeg` または `winget install Gyan.FFmpeg`

---

## 2. セットアップ手順

### 2.1 Windows の場合（ワンクリック自動セットアップ推奨）

Windows環境では、`run.bat` を実行するだけで、**Python環境（未導入時はポータブル版Python）、仮想環境（venv）、FFmpeg、SCRFD顔検出モデルのダウンロードからGUI起動まで全自動**で行われます。

```cmd
# リポジトリ直下のバッチファイルをダブルクリックまたはコマンドラインから実行
run.bat
```

手動で構築する場合：
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements/requirements-windows.txt
python scripts/download_models.py
```

### 2.2 macOS の場合

#### リポジトリの準備・仮想環境構築

```bash
git clone https://github.com/your-username/face-mosaic.git
cd face-mosaic

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージ導入
pip install --upgrade pip
pip install -r requirements/requirements-mac.txt
```

#### SCRFDモデルのダウンロード

自動ダウンロードスクリプトを実行してONNXモデルを取得します。

```bash
python scripts/download_models.py
```

`models/` ディレクトリに以下のモデルが配置されます：
- `scrfd_2.5g_bnkps.onnx` (高速・60+ FPS)
- `scrfd_10g_bnkps.onnx` (標準・推奨デフォルト)
- `scrfd_34g_gnkps.onnx` (高精度)

---

## 3. 使い方

### 3.1 基本的な実行

```bash
# 単一ファイルの処理
python -m face_mosaic.cli path/to/walk_video.mov

# 複数ファイルの一括指定
python -m face_mosaic.cli video1.mov video2.mp4 video3.mov

# フォルダ内の動画を一括バッチ処理（再帰的探索）
python -m face_mosaic.cli /path/to/videos/ --output-dir /path/to/output/
```

### 3.2 オプション指定

```bash
# モザイク方式・ブロックサイズ指定
python -m face_mosaic.cli video.mp4 --blur-type mosaic --strength 20

# ガウスぼかし強度・検出閾値・前後パディングフレーム数指定
python -m face_mosaic.cli video.mp4 --blur-type gaussian --strength 45 --conf 0.5 --pad-back 3 --pad-fwd 3

# 高速モデル (2.5G)・H.264エンコーダ指定
python -m face_mosaic.cli video.mp4 --model scrfd_2.5g_bnkps.onnx --codec h264 --crf 18

# フォルダ監視モード（新規動画が追加されたら自動処理）
python -m face_mosaic.cli /path/to/watch_folder --watch

# 設定ファイル (YAML) を指定して実行

python -m face_mosaic.cli video.mp4 --config configs/config.default.yaml
```

### 3.3 コマンドライン引数一覧

| 引数 | 説明 | デフォルト値 |
|---|---|---|
| `inputs` | 処理対象の動画ファイルまたはフォルダ（複数指定可） | 必須 |
| `--config, -c` | 設定ファイル（YAML）のパス | `configs/config.default.yaml` |
| `--model, -m` | 使用するSCRFDモデル名 | `scrfd_10g_bnkps.onnx` |
| `--blur-type, -b` | ぼかし方式 (`gaussian` または `mosaic`) | `gaussian` |
| `--strength, -s` | ぼかし強度（奇数整数）またはモザイクブロックサイズ | `31` / `16` |
| `--conf` | 顔検出の信頼度閾値 (0.0〜1.0) | `0.5` |
| `--pad-back` | 顔検出前のぼかし延長フレーム数 (N) | `3` |
| `--pad-fwd` | 顔検出後のぼかし延長フレーム数 (M) | `3` |
| `--codec` | 出力動画コーデック (`hevc` または `h264`) | `hevc` |
| `--crf` | CRF品質パラメータ（数値が小さいほど高画質） | `18` |
| `--output-dir, -o` | 出力フォルダ（省略時は入力と同じフォルダ） | `""` |
| `--no-recursive` | フォルダ探索時に再帰探索を無効化 | `False` |

### 3.4 GUIモード（デスクトップUI）

Windows環境では `run.bat` をダブルクリックすることで、コマンドプロンプト（`cmd.exe`）の黒い画面を表示させずにクリーンにGUIが起動します。

- **cmd.exe の非表示（サイレント起動＆変換中のポップアップ完全抑制）**:
  - 起動時だけでなく、変換中の FFmpeg / FFprobe / QSV コーデック判定などの子プロセス呼び出し時にも一切コンソールウィンドウを出現させない完全サイレント設計（`CREATE_NO_WINDOW` & `SW_HIDE`）。
- **高度な処理キュー管理（Queue Table）**:
  - 投入された動画ファイルの一覧、状態（待機中 / 処理中 / 完了 / キャンセル / エラー）、進捗率がテーブル形式で視認可能。
  - **マウスドラッグ＆ドロップおよびボタンによる処理順序の変更**:
    - **マウスドラッグ**: テーブル内の動画行をマウスで直接ドラッグ＆ドロップして、希望の位置へ直感的に並び替え可能。
    - **▲ Up / ▼ Down ボタン**: 選択した動画行をボタン1クリックで上下へ移動（ショートカットキー `Alt+Up` / `Alt+Down` にも対応）。
    - **右クリックメニュー**: `▲ Move Up`、`▼ Move Down`、`⤒ Move to Top`（先頭へ移動）、`⤓ Move to Bottom`（末尾へ移動）を即座に実行可能。
    - **実行中の動的順序同期**: バックグラウンドで変換が実行中であっても、待機中の動画の処理順序を並び替えると、次の処理順序へリアルタイムに即座反映されます。
  - **選択ファイルに応じた柔軟な実行**: キュー内の行をクリックして選択すると、ボタンが `▶ Start Processing (1 selected)` に変化し、選択した動画のみを即座に変換可能（キャンセル済みやエラーの動画も自動初期化して再実行）。未選択時はキュー内の未完了動画を順次一括処理します。
  - **安全なキュー除外**: 処理中または停止後にファイルを削除してもワーカースレッドのインデックスが狂わない一意ID（UUID）管理。
  - **右クリックコンテキストメニュー**:
    - `▶ Process This Video`: 選択した動画のみを即座に実行
    - `▶ Process Selected Videos (N)`: 複数選択した動画のみを実行
    - `▶ Process From This Video to End`: 選択行からキューの最後まで順次実行
    - `▲ Move Up` / `▼ Move Down` / `⤒ Move to Top` / `⤓ Move to Bottom`: 処理順序の移動
    - `✕ Remove from Queue`: キューから安全に除外
- **安全な Start / Stop ステートマシン**:
  - **待機中 (Idle)**: `▶ Start Processing`（青色・有効）
  - **実行中 (Running)**: `⏹ Stop Processing`（赤色・有効）
  - **停止処理中 (Stopping)**: `⏳ Stopping...`（グレー・非活性無効化）
    - 停止ボタン押下後、バックグラウンドの FFmpeg プロセスがクリーンに終了するまでの間ボタンが無効化され、重複起動や二重停止の競合を確実に防止します。
  - **きめ細やかなキャンセル機能**:
    - `⏭ Skip / Cancel Current File`: 現在処理中の動画のみを即座にスキップし、キューに残っている次の動画へ安全に進みます。
- **2フェーズ・リアルタイムプレビュー**:
  - **Pass 1（顔検出フェーズ）**: 緑色のバウンディングボックスと信頼度スコアがリアルタイム描画され、検出状況を即座に確認可能。
  - **Pass 2（モザイク描画フェーズ）**: 実際にモザイク・ぼかしが施された完成形映像がリアルタイム表示されます。
- **スペック自動判定＆最適モデルの自動選択**: 起動時にCPU、RAM（32GB等）、GPU（DirectML/DirectX 12）を自動診断し、最適なモデル（Core Ultra 32GB機等では高精度 `scrfd_34g_gnkps.onnx`）を自動選択します。
- **最後に開いたフォルダの記憶 & ドラッグ＆ドロップ対応**:
  - ファイル追加（`+ Add File(s)...`）やフォルダ追加（`+ Add Folder...`）で選択したディレクトリを永続的に自動記憶（`QSettings.sync`）。次回ダイアログ表示時やアプリ再起動後も同一フォルダを素早く開けます。
  - エクスプローラー / Finder から動画ファイルやフォルダをGUI上に直接ドラッグ＆ドロップしてキューに追加可能（親ディレクトリも自動記憶）。
  - テーブルの空き領域を右クリックしてコンテキストメニューから素早くファイル・フォルダを追加可能。
- **実行ログ＆トラブルシューティング支援**:
  - GUI上のログ表示に加え、`logs/face_mosaic.log` にタイムスタンプ付きで詳細な処理ログおよびスタックトレースを自動保存。
  - 「📋 Copy Log」ボタン（クリップボードへ一括コピー）および「📁 Open Log File」ボタン（ログファイルを即座に開く）を標準装備。
- **ワンクリック再生＆出力フォルダ表示**: 処理完了後、「▶ Open Processed Video」および「📁 Open Output Folder」ボタンから即座に再生確認・フォルダオープンが可能です。

---

## 4. Gatekeeper警告の回避方法 (Mac版配布バイナリ利用時)

GitHub Releases等からダウンロードした未署名のビルドバイナリをmacOSで初回実行する際、「開発元が未確認のため開けません」というGatekeeper警告が表示される場合があります。

1. Finderで実行ファイルを「Controlキーを押しながらクリック」（右クリック）し、「開く」を選択します。
2. 表示されるダイアログで「開く」をクリックすると、次回以降は通常通り起動できるようになります。
3. または、ターミナルで以下のコマンドを実行して隔離属性を解除します：
   ```bash
   xattr -d com.apple.quarantine ./face-mosaic
   ```

---

## 5. テスト・ベンチマークの実行

```bash
# 単体・結合テストの実行
pytest -v

# SCRFD推論速度のベンチマーク計測
python scripts/benchmark.py
```

---

## 6. ハードウェアアクセラレーション (GPU / NPU) と高速化仕様

本ツールは、Windows・macOS双方のハードウェア性能を最大限に引き出す最適化パイプラインを備えています。

### 6.1 GPU アクセラレーション (DirectML / DirectX 12)

Windows環境（Intel Arc, NVIDIA GeForce, AMD Radeon等）では、`onnxruntime-directml` により **DirectX 12 経由で GPU 推論が自動適用** されます。

- **実機での推論ベンチマーク性能（Intel Arc 130T GPU 測定値）**:
  - **標準 10G モデル (`scrfd_10g_bnkps.onnx`)**: **約 43.5 FPS**（CPU比 約4.0倍高速化）
  - **高速 2.5G モデル (`scrfd_2.5g_bnkps.onnx`)**: **約 78.3 FPS**（実時間 60fps を上回る超高速処理）
- **非同期フレームプリフェッチ**:
  バックグラウンドスレッドで先行して動画フレームを読み込み・バッファリングすることで、GPU の I/O 待機時間をゼロにし、推論を途切れず連続稼働させます。

### 6.2 ハードウェア動画エンコード (Intel Quick Sync Video / QSV)

Pass 2（モザイク描画後の動画書き出しフェーズ）では、CPU ソフトウェアエンコード（`libx265`）に代わり、Intel GPU 内蔵の専用メディアエンジン **`hevc_qsv` / `h264_qsv` を自動活用** します。
これにより、エンコード時間の大幅な短縮と CPU 負荷の低減を同時に実現しています（環境に応じてソフトウェアエンコードへの安全な自動フォールバック機能付き）。

### 6.3 タスクマネージャー上の GPU 使用率について

「CPU使用率が下がり、GPU使用率が10〜20%前後に見える」現象は、**GPU演算能力が非常に高く、推論がわずか約20msという一瞬で完了している（GPUに十分な余力がある）** ことを示しています。

### 6.4 GPU パワーを余すところなく活用するための推奨設定

Intel Arc 等の大容量 VRAM（16GB 等）や高性能 GPU をお使いの場合、以下の設定を行うことで GPU の演算能力をフルに引き出し、最高精度の検出結果を得られます：

1. **最高精度モデル (`scrfd_34g_gnkps.onnx`) の選択**:
   GUI のモデル選択で **`scrfd_34g_gnkps.onnx (High Acc)`**（CLIでは `--model scrfd_34g_gnkps.onnx`）を選択してください。
   GPU の並列演算ユニットをより高密度に活用しながら、遠くの極小の顔や横顔・うつむき顔の検出率を大幅に向上させることができます。
2. **複数動画の一括バッチ処理**:
   フォルダ指定（`run.bat C:\path\to\videos`）や複数ファイル選択でまとめて処理を流すことで、非同期パイプラインと QSV エンコーダーが常に高効率で GPU を駆動し続けます。

