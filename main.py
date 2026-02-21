import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from ultralytics import YOLO
import time
import torch

# 設定
width, height, fps = 1280, 720, 30
scale = 1.8
mask_enabled = True
infer_interval = 0.02  # 秒（推論頻度：0.5なら2回/秒、1.0なら1回/秒）

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

overlay_img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
model = YOLO("yolov8n-face.pt")
# デバイス自動選択（CUDA が使えなければ CPU にフォールバック）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# アルファ合成
def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    bh, bw = background.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)
    overlay_x1, overlay_y1 = max(0, -x), max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    if x1 >= x2 or y1 >= y2:
        return background
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    b, g, r, a = cv2.split(overlay_crop)
    mask = cv2.merge((a, a, a))
    overlay_rgb = cv2.merge((b, g, r))
    roi = background[y1:y2, x1:x2]
    img1_bg = cv2.bitwise_and(roi, 255 - mask)
    img2_fg = cv2.bitwise_and(overlay_rgb, mask)
    background[y1:y2, x1:x2] = cv2.add(img1_bg, img2_fg)
    return background

# ダミーウィンドウ（キー入力用）
cv2.namedWindow("HiddenWindow", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HiddenWindow", 1, 1)

# 推論保持
last_boxes = []
last_infer_time = 0

# スムージング用のバッファ
# {id: {"history": [(cx, cy, face_size), ...], "last_update": time}}
face_history = {}
SMOOTH_FRAMES = 5
DISTANCE_THRESHOLD = 50  # 前フレームとの距離のしきい値

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR) as cam:
    print(f'✅ Virtual camera started: {cam.device}')
    print("🎭 Mキーでマスク切替、Qキーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("HiddenWindow", np.zeros((1, 1), dtype=np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):
            mask_enabled = not mask_enabled
            print("🎭 マスクON" if mask_enabled else "🙈 マスクOFF")
        elif key == ord('q'):
            print("👋 終了します")
            break

        now = time.time()
        # 一定間隔ごとに推論
        if now - last_infer_time > infer_interval:
            results = model.predict(frame, verbose=False)[0]
            current_faces = []
            for box in results.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                face_size = int(max(x2 - x1, y2 - y1) * scale)
                current_faces.append((cx, cy, face_size))
            
            # --- スムージング処理 ---
            new_face_history = {}
            for (cx, cy, fs) in current_faces:
                # 既存の顔と紐付け（距離が近いものを探す）
                matched_id = None
                min_dist = DISTANCE_THRESHOLD
                for fid, data in face_history.items():
                    prev_cx, prev_cy, _ = data["history"][-1]
                    dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        matched_id = fid
                
                if matched_id is not None:
                    # 履歴を更新
                    history = face_history[matched_id]["history"]
                    history.append((cx, cy, fs))
                    if len(history) > SMOOTH_FRAMES:
                        history.pop(0)
                    new_face_history[matched_id] = {"history": history, "last_update": now}
                else:
                    # 新しい顔として登録
                    new_id = time.time() + np.random.rand()
                    new_face_history[new_id] = {"history": [(cx, cy, fs)], "last_update": now}
            
            # 古いデータを削除（一定時間更新がないもの）
            for fid, data in face_history.items():
                if now - data["last_update"] < 0.5 and fid not in new_face_history:
                     # 推論しなかっただけで、まだ存在している可能性が高い場合は前回の履歴を維持
                     new_face_history[fid] = data

            face_history = new_face_history
            last_infer_time = now

        # マスク合成（スムージングされたデータを使用）
        if mask_enabled:
            for fid, data in face_history.items():
                # 平均値を計算
                history = data["history"]
                avg_cx = int(np.mean([h[0] for h in history]))
                avg_cy = int(np.mean([h[1] for h in history]))
                avg_fs = int(np.mean([h[2] for h in history]))

                try:
                    resized = cv2.resize(overlay_img, (avg_fs, avg_fs))
                    x_offset = avg_cx - avg_fs // 2
                    y_offset = avg_cy - avg_fs // 2
                    frame = overlay_transparent(frame, resized, x_offset, y_offset)
                except cv2.error:
                    continue

        cam.send(frame)
        cam.sleep_until_next_frame()

cv2.destroyAllWindows()
