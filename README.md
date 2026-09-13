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

## 2. セットアップ手順 (macOS)

### 2.1 リポジトリの準備・仮想環境構築

```bash
git clone https://github.com/your-username/face-mosaic.git
cd face-mosaic

# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージ導入
pip install --upgrade pip
pip install -r requirements-mac.txt
```

### 2.2 SCRFDモデルのダウンロード

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
