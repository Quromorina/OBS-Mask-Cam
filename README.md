# 🎭 OBS Mask Cam

> AIがリアルタイムで顔を検出し、お好みの画像を顔の上に追従させる軽量な仮想カメラアプリケーション。

## 💡 なぜこれを作ったのか？

配信やオンラインミーティング、Discordのボイスチャットなどで「顔出しはしたくないけれど、カメラで様子は見せたい」という場合に、簡単に顔を隠せるツールが欲しかったので作りました。

## ⚠️ 前提条件 (必ずご確認ください)

- **OBS Studio**: 本アプリは仮想カメラを使用するため、事前に [OBS Studio](https://obsproject.com/ja/download) がインストールされていることが**前提**となります。
- **GPUは不要**: 顔検出には軽量なモデル（YuNet）を使い、CPUで動作します。グラフィックボード（GPU）の有無は問いません。

## 🚀 クイックスタート (使い方)

最短で使い始めるための手順です。本アプリ自体の複雑な環境構築は不要です。

1. [最新版のリリース (v1.0.0)](https://github.com/Quromorina/OBS-Mask-Cam/releases/latest) から `OBSMaskCam_Setup_v1.0.0.exe` をダウンロードします。
2. ダウンロードしたインストーラーを実行し、画面の案内に従ってインストールを行います。
3. デスクトップまたはスタートメニューから **OBS Mask Cam** を起動します。
4. **OBS Studio** などの配信ソフトを開き、カメラ入力のソースとして `OBS Virtual Camera` を選択します。
   *🎉 これで、あなたの顔の上に画像が追従して表示されます！*

---

## 🎨 コントロールパネルの主な機能

直感的な専用UIから、配信中でもリアルタイムに各種設定を変更できます。

### 1. オリジナルマスク画像の追加と削除
デフォルトのアイコン画像の代わりに、好きな画像（PNG/JPG/WebPなど）を顔に被せることができます。
- **追加**: アプリ画面の「➕ 新しいマスクを追加」をクリックして画像を選びます。背景が透過されたPNG画像を使うと、より自然に合成されます。
- **自動最適化**: 4Kなどの高解像度画像を追加した場合、動作を軽くするために自動で最適サイズ（長辺最大1000px）に縮小して保存されます。
- **削除**: 不要になった画像は「🗑️ 選択中のマスクを削除」ボタンからいつでも整理できます。

### 2. その他の調整機能
| 設定項目 | 説明 |
|---|---|
| **映像ソース** | 使用するWebカメラをプルダウンから即座に切り替えられます。 |
| **マスクの大きさ** | スライダーを動かし、顔のサイズに対する画像の比率を直感的に調整できます。 |
| **顔を見失ったときのマスク保持** | 顔の検出が一瞬途切れても、指定した秒数（0.5〜5.0 秒、初期値 1.5 秒）はマスクをその場に残します。長くするほど素顔が映りにくくなります。 |
| **一時無効化** | ボタン一つで、マスク処理を一瞬でOFFにし素のカメラ映像に戻せます。 |

### 3. 「いま顔が隠れているか」の表示
画面上部に、現在の送出状態が常に表示されます。配信前・配信中はここを確認してください。

| 表示 | 意味 |
|---|---|
| ✅ マスク適用中（緑） | 選んだマスク画像で顔を隠しています。 |
| ⚠ ぼかしで顔を隠しています（橙） | マスク画像が読み込めない・選ばれていないなどの理由で、代わりにモザイクで顔を隠しています。 |
| ⚠ 画面全体をぼかしています（橙） | 顔検出が動いていないため、顔の位置が分からず画面全体をモザイクにしています。 |
| 🚫 マスクOFF（赤） | **素顔がそのまま映ります。** |
| ❌ 映像の送出が止まっています（赤） | エラーで停止しています。理由は画面とログに出ます。 |

> **Note**: マスクがONのあいだ、マスク画像が使えなくても素のカメラ映像にはせず、モザイクで代用します。ただし顔検出そのものが顔を見つけられなかった場合（横顔、手で隠れている、暗いなど）は隠せません。大事な場面では、保持時間を長めにする・マスクを大きめにするなどで余裕を持たせてください。

---

## 🛠 開発者向けセットアップ

ご自身でソースコードから実行・改変したい場合の手順です。

**前提条件**: Python 3.10+、OpenCV 4.x 系（`requirements.txt` の `opencv-python` 4.13 を想定。顔検出に `cv2.FaceDetectorYN` を使います）

```bash
# 1. リポジトリのクローン
git clone https://github.com/Quromorina/OBS-Mask-Cam.git
cd OBS-Mask-Cam

# 2. 依存関係のインストール
pip install -r requirements.txt

# 3. アプリの起動
python main.py
```
> **Tip**: 顔検出用AIモデル（`face_detection_yunet_2023mar.onnx`）はリポジトリに同梱しています。ダウンロードは行いません（アプリはネットワーク通信をしません）。

Windows 専用です（OBS Virtual Camera・DirectShow を使うため）。

画像処理と顔検出の単体テスト（GUI・カメラ不要）は次で実行できます。

```bash
python -m unittest discover -s tests -v
```

---

## 🆘 トラブルシューティング

**Q. 「カメラが見つかりません」または映像が真っ暗になる**  
A. Windowsの「設定」>「プライバシーとセキュリティ」>「カメラ」にて、**「アプリにカメラへのアクセスを許可する」がオン**になっているか確認してください。また、Zoomなど他の通話アプリが既にカメラを使用中の場合は、一度終了させてください。

**Q. マスクの追従が遅い、映像がカクカクする**  
A. 顔検出はCPUで行っています。他の重いアプリ（ゲーム、動画のエンコードなど）がCPUを使い切っていないか確認してください。カメラ側の設定で解像度やフレームレートが極端に高くなっている場合も、下げると改善することがあります。

**Q. 「顔検出モデルを読み込めませんでした」「映像処理がエラーで停止しました」と表示される**  
A. 理由は画面のエラー表示と、ログファイル `%APPDATA%\OBSMaskCam\app.log` に記録されます。不具合報告の際はこのログを添えてください（ログにはユーザー名を含むフォルダのパスが入ります。公開の場に貼るときは伏せてください）。

---

## 🔐 配布物の確認方法（SHA-256）

配布している EXE・インストーラには**コード署名がありません**。そのため初回実行時に Windows SmartScreen の警告が出ます。ダウンロードしたファイルが本物かどうかは、リリースページに記載した SHA-256 と照合して確認してください。

```powershell
Get-FileHash .\OBSMaskCam_Setup_v1.0.0.exe -Algorithm SHA256
```

表示された `Hash` が、リリースページの値と一致していれば改ざんされていません。

### リリースする側の手順（メンテナ向け）

1. クリーンな作業ツリー（`git status` が空）で `pip install -r requirements-build.txt` → `pyinstaller OBSMaskCam.spec` → Inno Setup で `OBSMaskCam_Installer.iss` をビルドする。
2. できたインストーラの SHA-256 を計算する：`Get-FileHash .\InnoSetup_Output\OBSMaskCam_Setup_vX.Y.Z.exe -Algorithm SHA256`
3. GitHub のリリースノートに、ファイル名・SHA-256・ビルド元のコミット ID を記載する。
4. インストーラに入る初期マスクは `default_masks/` の中の、git にコミットした PNG だけです。`masks/` の手元の画像は入りません。

## 🧠 同梱モデルについて

| 項目 | 内容 |
|---|---|
| モデル | YuNet（顔検出。顔の枠・5点のランドマーク・信頼度を出力） |
| ファイル | `face_detection_yunet_2023mar.onnx`（232,589 バイト） |
| SHA-256 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| 配布元 | [OpenCV Zoo `models/face_detection_yunet`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)（学習元は [ShiqiYu/libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)） |
| ライセンス | MIT License（Copyright (c) 2020 Shiqi Yu）。ライセンス文は [`licenses/YuNet-LICENSE.txt`](licenses/YuNet-LICENSE.txt) に同梱しています。 |

SHA-256 は OpenCV Zoo が同ファイルについて記録している値（Git LFS の oid）と同じです。手元のファイルは次で確認できます。

```powershell
Get-FileHash .\face_detection_yunet_2023mar.onnx -Algorithm SHA256
```

---

## 🏗 技術スタック

高速で軽量な動作を実現するため、以下の技術スタックを中心に構成しています。

- **AI顔検出モデル**: [YuNet (OpenCV Zoo)](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **AI推論エンジン**: OpenCV DNN（`cv2.FaceDetectorYN`、CPU推論）
- **GUIフレームワーク**: CustomTkinter
- **仮想カメラ連携**: pyvirtualcam
- **ビルド・パッケージング**: PyInstaller / Inno Setup

## 📄 ライセンス

ソースコードは [MIT License](LICENSE) © [Quromorina](https://github.com/Quromorina) です。

同梱している顔検出モデル（YuNet）も MIT License です（著作権者は別です。上の「同梱モデルについて」を参照してください）。
