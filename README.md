# OBS-Mask-Cam

🎭 YOLOv8 を使用したリアルタイム顔認識マスクツール

このプロジェクトは、Webカメラからの映像を解析し、検出された顔に自動でマスク画像を合成して仮想カメラへ出力します。OBS 等の配信ソフトでそのまま使用することが可能です。

## 機能

- **リアルタイム顔認識**: YOLOv8n-face を使用した高速・高精度な顔検出
- **仮想カメラ出力**: `pyvirtualcam` を使用し、OBS 等で認識可能な仮想デバイスとして出力
- **動的切り替え**: 実行中に `M` キーでマスクの有効/無効を切り替え可能
- **GPU サポート**: CUDA が利用可能な環境では自動的に GPU を使用して推論

## セットアップ

### 必要なもの
- Python 3.8以上
- Webカメラ
- [OBS Virtual Camera](https://obsproject.com/) (Windows の場合、OBSをインストールすると自動で入ります)

### インストール

```bash
# リポジトリをクローン（またはダウンロード）
git clone https://github.com/Quromorina/OBS-Mask-Cam.git
cd OBS-Mask-Cam

# 依存ライブラリのインストール
pip install -r requirements.txt
```

> [!NOTE]
> `pyvirtualcam` の動作には、対応する仮想カメラドライバ（OBS Virtual Camera など）が必要です。

## 使い方

1. `main.py` を実行します。
   ```bash
   python main.py
   ```
2. OBS 等の配信ソフトを開き、映像キャプチャデバイスから **"OBS Virtual Camera"** を選択します。
3. 実行中のコンソール（またはダミーウィンドウ）で以下の操作が可能です：
   - `M` キー: マスクの ON/OFF 切り替え
   - `Q` キー: プログラムの終了

## カスタマイズ

- **マスク画像の変更**: `mask.png` を好きな画像（背景透過の PNG 推奨）に差し替えることで、マスクを変更できます。
- **設定の変更**: `main.py` 内の `scale` 変数を調整することで、マスクの大きさを変更できます。

## ライセンス

[MIT License](LICENSE) (または任意のライセンス)
