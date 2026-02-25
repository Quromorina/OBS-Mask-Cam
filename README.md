# 🎭 OBS Mask Cam

AIがリアルタイムで顔を検出し、マスク画像を重ねてOBS Virtual Cameraに出力するアプリケーションです。  
配信やビデオ通話で顔を隠したい時に使えます。

---

## ✨ 特徴

- **リアルタイム顔検出** — YOLOv8ベースのAIモデル（ONNX）を使用
- **GPU推論対応** — DirectMLにより NVIDIA / AMD / Intel どのGPUでも高速推論
- **Python不要** — ビルド済みEXEをダウンロードするだけで使える
- **マスク追加可能** — 好きな画像をマスクとして追加できる
- **OBS Virtual Camera** — OBSやZoom、Discordなどに仮想カメラとして出力

---

## 📦 必要なもの

| 必要なソフト | 説明 |
|---|---|
| **OBS Studio** | Virtual Cameraのドライバとして必要です。[ダウンロード](https://obsproject.com/ja/download) |
| **Webカメラ** | PCに内蔵 or USB接続のカメラ |

> [!NOTE]
> GPUがなくてもCPUモードで動作しますが、推論速度が遅くなります。

---

## 🚀 使い方

### EXE版（推奨）

1. [Releases](https://github.com/Quromorina/OBS-Mask-Cam/releases) から最新の `OBSMaskCam.zip` をダウンロード
2. ZIPを解凍
3. `OBSMaskCam.exe` をダブルクリック
4. **OBS Studio** を起動し、ソースに「映像キャプチャデバイス」→「OBS Virtual Camera」を選択

### Python版（開発者向け）

```bash
git clone https://github.com/Quromorina/OBS-Mask-Cam.git
cd OBS-Mask-Cam
pip install -r requirements.txt
python main.py
```

---

## 🎭 マスクの追加方法

1. アプリのコントロールパネルで「➕ 新しいマスクを追加」をクリック
2. 好きな画像ファイル（PNG/JPG/BMP/WebP）を選択
3. 自動的に透過対応PNGに変換されて `masks/` フォルダに保存されます

> [!TIP]
> 透過PNGを使うと、顔の上に自然に画像が重なります。

---

## 🎮 コントロールパネル

| 機能 | 説明 |
|---|---|
| **映像ソース** | 使用するカメラを選択 |
| **マスク選択** | 使うマスク画像を切り替え |
| **マスクON/OFF** | マスクの有効/無効を切り替え |
| **マスクの大きさ** | 顔に対するマスクのサイズを調整 |
| **動きの滑らかさ** | マスクの追尾の滑らかさを調整 |

---

## ⚠ トラブルシューティング

### 「OBS Virtual Cameraが見つかりません」

- **OBS Studio** がインストールされていることを確認してください
- OBSを一度起動してから再度アプリを起動してください

### 「カメラが見つかりません」

- Webカメラが正しく接続されているか確認してください
- 他のアプリがカメラを占有していないか確認してください

### マスクが遅い・カクカクする

- GPU推論で動作しているか確認してください（コントロールパネル上部に表示）
- 「⚠ CPU推論（低速）」と表示されている場合、GPUドライバの更新を試してください

---

## 🛠 技術スタック

- **顔検出**: YOLOv8n-face (ONNX)
- **推論エンジン**: ONNX Runtime + DirectML
- **GUI**: CustomTkinter
- **仮想カメラ**: pyvirtualcam
- **ビルド**: PyInstaller

---

## 📄 ライセンス

MIT License
