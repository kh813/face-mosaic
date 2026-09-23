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
| `--strength, -s` | ぼかし強度（奇数整数）またはモザイクブロックサイズ | `51` / `28` |
| `--conf` | 顔検出の信頼度閾値 (0.0〜1.0) | `0.5` |
| `--pad-back` | 顔検出前のぼかし延長フレーム数 (N)（事前モザイク） | `12` |
| `--pad-fwd` | 顔検出後のぼかし延長フレーム数 (M) | `8` |
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

### 3.5 中断・再開機能（チェックポイント＆レジューム）とキャッシュ管理

4K 60fps の長尺動画や長時間のバッチ処理中に予期せぬ中断（強制終了・電源断・キャンセル）が発生しても、**最初からやり直すことなく安全に途中再開できるチェックポイント機構**を備えています。

- **保存場所とファイル構成**:
  - 出力先フォルダ内の出力ファイルと同階層に一時作業フォルダ `.{出力ファイル名}.checkpoint/` を自動生成します。
  - 出力動画自体は書き出し完了まで `{出力ファイル名}.part.mp4` として一時保存されます。
  ```text
  出力先フォルダ/
  ├── .IMG_2365_mosaic.checkpoint/     # 中間チェックポイント（完了時に自動削除）
  │   ├── state.json                  # 全体進捗メタデータ（~300バイト）
  │   ├── pass1_detections.jsonl      # Pass 1 フレーム毎の顔検出座標ログ（~20KB/1000フレーム）
  │   └── pass1_frame_bboxes.json     # Pass 2 用の確定バウンディングボックス
  └── IMG_2365_mosaic.part.mp4         # エンコード中の一時動画ファイル
  ```
- **数十GBの不要キャッシュが残らない超軽量設計**:
  - **非圧縮フレームや画像キャッシュは一切ディスクに書き出しません**。
  - 保存するのは **顔検出の座標情報（バウンディングボックスの数値）のみ** です。4K 60fps の1時間動画（約21万フレーム）であっても、チェックポイントの総容量はわずか数MB〜十数MBに収まります。
- **自動クリーンアップ**:
  - 処理が 100% 完了した瞬間に `.{出力ファイル名}.checkpoint/` は **自動的に完全消去** され、`.part.mp4` が最終的な `.mp4` にアトミックにリネームされます。
- **再開（レジューム）の動作**:
  - **Pass 1（顔検出）の途中で中断**: 最後に検出が記録されたフレームから即座に再開します。
  - **Pass 2（モザイク描画）の途中で中断**: 時間のかかる Pass 1 を丸ごとスキップし、Pass 2 から即座に再開します。

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
# 全単体・結合テストの実行（全63テスト）
pytest -v

# SCRFD推論速度のベンチマーク計測
python scripts/benchmark.py
```

---

## 6. ハードウェアアクセラレーション (CPU / GPU / NPU) と高速化仕様

本ツールは、Windows・macOS双方のハードウェア性能（特に最新の Intel Core Ultra CPU / Intel Arc GPU / Intel AI Boost NPU）を限界まで引き出すハイブリッド高速化パイプラインを備えています。

### 6.1 OpenVINO による GPU + NPU ハイブリッド推論 (Intel Core Ultra)

Windows（Intel Core Ultra 環境）では、ネイティブ OpenVINO ランタイムを通じて **Intel Arc GPU と Intel AI Boost NPU の協調処理 (`MULTI:GPU,NPU`)** を自動活用します。

- **実機での推論ベンチマーク性能（Intel Core Ultra 5 225H / Arc 130T GPU）**:
  - **OpenVINO GPU / NPU 推論**: **約 135〜155 FPS**（4K 60fps 動画の推論速度を 2.5倍以上上回るリアルタイム超高速処理）
  - **DirectML (DirectX 12) / ONNX Runtime**: 約 16〜43 FPS（DirectML 環境への自動安全フォールバック対応）
- **長尺動画での OOM 防止（メモリリーク完全遮断）**:
  - 4K 生フレーム（1フレーム約 25MB）への NumPy スライス参照を遮断し、顔トラッキング用クロップ画像を最大 128px にリサイズした上で独立メモリバッファとして保持。
  - 数万フレーム（数十GB規模）の 4K 60fps 動画であっても、メモリ使用量が数GB以内に一定制御され、クラッシュなく安定して完走します。

### 6.2 Pass 1（顔検出）の動画デコード超高速化

4K 動画（3840×2160）をそのままデコードすると、プロセス間パイプに毎秒約 1.5GB の非圧縮データが流れ込み、デコード速度が約 2.4 FPS でボトルネックになっていました。
本ツールでは、顔検出モデルの入力サイズ（640×640）に合わせて **FFmpeg 内部でスケールダウン（`-vf scale=640:360`）した上でパイプ転送し、マルチスレッド（`-threads 0`）を適用** しています。
- パイプ転送量を 1/36 に削減し、Pass 1 の動画読み込み速度が **約 2.4 FPS → 約 60 FPS（約25倍）** に飛躍的に高速化されます。

### 6.3 Pass 2（モザイク描画・レンダリング）の 3ステージ非同期パイプライン高速化

Pass 2（レンダリング）において CPU 使用率が 15% 前後で停滞していたボトルネックは、Windows の匿名パイプ（Anonymous Pipe）を経由して 4K 非圧縮フレーム（1フレーム約 25MB、60fps で毎秒 1.5GB）を同期転送する際に発生する OS レベルのコンテキストスイッチ待ちでした。

本ツールでは、**3ステージ並行非同期パイプライン** を導入することで、Pass 2 の書き出し速度を **約 2.5 FPS → 約 20〜30+ FPS（最大10倍以上）** に高速化しました：
1. **ステージ 1（バックグラウンド・デコーダー）**: `cv2.VideoCapture` によるプロセス内ハードウェア高速デコード（Direct3D 11 / Media Foundation）で、OS パイプを介さず直接フレームをプリフェッチキューへ先読み。
2. **ステージ 2（メインスレッド・モザイク演算）**: メモリ上でゼロコピー演算を行い、モザイク・ぼかしを高速描画（80+ FPS）。
3. **ステージ 3（バックグラウンド・エンコーダー）**: 非同期キュー経由で Intel Arc GPU のメディアエンジン（`hevc_qsv` / `h264_qsv`）へストリーミング転送し、ハードウェアエンコード。

### 6.4 色再現性（カラーマネジメント）の完全保証：色味変更の完全防止

iPhone 4K 10-bit HDR（BT.2020 / HLG）などの広色域・高ダイナミックレンジ映像において、**再エンコードによる色褪せ・変色・白飛びを 100% 防止** します：
- **10-bit 階調の維持 (`p010le`)**: 入力動画が 10-bit（`yuv420p10le`）の場合、Intel QSV でも自動的に 10-bit ピクセルフォーマット（`p010le`）を適用し、8-bit 圧縮によるバンディングや色階調の破綻を防ぎます。
- **HEVC VUI ビットストリーム注入 (`hevc_metadata`)**:
  コンテナタグだけでなく、HEVC ビットストリームの NAL ユニット内部（VUI: Video Usability Information）に直接 **ITU-T H.273 規格コード** を埋め込みます：
  - `colour_primaries=9` (BT.2020)
  - `transfer_characteristics=18` (HLG / ARIB STD-B67)
  - `matrix_coefficients=9` (BT.2020 non-constant luminance)
  - `color_range=tv` (Limited Range)
  これにより、YouTube や QuickTime Player、VLC 等のあらゆる再生環境で、元動画と完全に同一の発色・色域・コントラストが正確に再現されます。

---

## 7. 利用規約・免責事項 (Disclaimer & Terms of Use)

本ソフトウェアをご利用になる前に、以下の免責事項を必ずお読みください。本ソフトウェアを使用した時点で、本免責事項に同意したものとみなされます。

> [!IMPORTANT]
> **目視による最終確認の義務**
> 本ソフトウェアは、AI（コンピュータビジョン・機械学習モデル）を用いて顔検出およびモザイク／ぼかし処理を**自動化・補助するツール**です。
> カメラアングル、被写体の移動速度、遮蔽物、極端な照明変化、解像度等により、**顔の検出漏れやモザイクの途切れが発生する可能性を技術的に 100% 排除することはできません**。
>
> 処理後の動画をインターネット（YouTube、SNS等）へ公開・共有する前に、**利用者の責任において必ず全編の目視確認を行い、必要に応じて手動での追加修正を行ってください**。

1. **保証の否認 (Disclaimer of Warranty)**:
   本ソフトウェアは Apache License 2.0 に基づき「現状有姿 (AS IS)」で提供され、商品性、特定目的への適合性、権利非侵害、および検出の完全性を含め、明示的にも黙示的にもいかなる保証も行いません。
2. **責任の制限 (Limitation of Liability)**:
   本ソフトウェアの利用、誤動作、検出漏れ、または利用不能に起因して生じたいかなる損害（肖像権侵害、プライバシー侵害、名誉毀損、第三者との紛争、営業損害、データの消失、その他直接的・間接的・派生的損害を含む）について、開発者およびコントリビューターは理由の如何を問わず一切の法的責任を負いません。
3. **第三者権利侵害の責任**:
   動画を撮影・加工・公開・配信することに伴うすべての法的責任（肖像権、プライバシー権、著作権等の遵守）は、本ソフトウェアの利用者が単独で負うものとします。

---

## 8. ライセンス (License)

本プロジェクトは **[Apache License, Version 2.0](LICENSE)** の下で公開されています。
詳細はリポジトリ直下の `LICENSE` ファイルをご参照ください。

