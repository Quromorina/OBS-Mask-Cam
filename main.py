import logging
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
from ultralytics import YOLO
import time
import torch
import threading
import customtkinter as ctk

# --- 設定（初期値） ---
class AppConfig:
    width, height, fps = 1280, 720, 30
    scale = 1.8
    mask_enabled = True
    infer_interval = 0.02
    smooth_frames = 5
    distance_threshold = 50
    running = True

config = AppConfig()

# --- アルファ合成 ---
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

# --- カメラ・AI処理スレッド ---
def camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    overlay_img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
    model = YOLO("yolov8n-face.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    face_history = {}
    last_infer_time = 0

    with pyvirtualcam.Camera(width=config.width, height=config.height, fps=config.fps, fmt=PixelFormat.BGR) as cam:
        print(f'✅ Virtual camera started: {cam.device}')

        while config.running:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            if now - last_infer_time > config.infer_interval:
                results = model.predict(frame, verbose=False)[0]
                current_faces = []
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    face_size = int(max(x2 - x1, y2 - y1) * config.scale)
                    current_faces.append((cx, cy, face_size))
                
                new_face_history = {}
                for (cx, cy, fs) in current_faces:
                    matched_id = None
                    min_dist = config.distance_threshold
                    for fid, data in face_history.items():
                        prev_cx, prev_cy, _ = data["history"][-1]
                        dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = fid
                    
                    if matched_id is not None:
                        history = face_history[matched_id]["history"]
                        history.append((cx, cy, fs))
                        if len(history) > 20: # 最大履歴長
                            history.pop(0)
                        new_face_history[matched_id] = {"history": history, "last_update": now}
                    else:
                        new_id = time.time() + np.random.rand()
                        new_face_history[new_id] = {"history": [(cx, cy, fs)], "last_update": now}
                
                for fid, data in face_history.items():
                    if now - data["last_update"] < 0.5 and fid not in new_face_history:
                        new_face_history[fid] = data

                face_history = new_face_history
                last_infer_time = now

            if config.mask_enabled:
                for fid, data in face_history.items():
                    history = data["history"]
                    current_history = history[-config.smooth_frames:]
                    avg_cx = int(np.mean([h[0] for h in current_history]))
                    avg_cy = int(np.mean([h[1] for h in current_history]))
                    avg_fs = int(np.mean([h[2] for h in current_history]))

                    try:
                        resized = cv2.resize(overlay_img, (avg_fs, avg_fs))
                        x_offset = avg_cx - avg_fs // 2
                        y_offset = avg_cy - avg_fs // 2
                        frame = overlay_transparent(frame, resized, x_offset, y_offset)
                    except cv2.error:
                        continue

            cam.send(frame)
            cam.sleep_until_next_frame()

    cap.release()

# --- GUIクラス ---
class ControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OBS Mask Cam - コントロールパネル")
        self.geometry("450x400")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # タイトル
        self.label_title = ctk.CTkLabel(self, text="🎭 OBS Mask Cam 設定", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.pack(pady=20)

        # マスク有効/無効トグル
        self.switch_mask = ctk.CTkSwitch(self, text="マスクを有効にする", command=self.toggle_mask)
        self.switch_mask.select()
        self.switch_mask.pack(pady=10)

        # マスクサイズ調整
        self.label_scale = ctk.CTkLabel(self, text=f"マスクの大きさ: {config.scale:.1f}")
        self.label_scale.pack(pady=(10, 0))
        self.slider_scale = ctk.CTkSlider(self, from_=1.0, to=4.0, command=self.update_scale)
        self.slider_scale.set(config.scale)
        self.slider_scale.pack(padx=20, fill="x")

        # スムージング調整
        self.label_smooth = ctk.CTkLabel(self, text=f"動きの滑らかさ (フレーム): {config.smooth_frames}")
        self.label_smooth.pack(pady=(10, 0))
        self.slider_smooth = ctk.CTkSlider(self, from_=1, to=20, number_of_steps=19, command=self.update_smooth)
        self.slider_smooth.set(config.smooth_frames)
        self.slider_smooth.pack(padx=20, fill="x")

        # 推論間隔調整
        self.label_infer = ctk.CTkLabel(self, text=f"認識の頻度 (ミリ秒): {int(config.infer_interval * 1000)}")
        self.label_infer.pack(pady=(10, 0))
        self.slider_infer = ctk.CTkSlider(self, from_=0, to=500, command=self.update_infer)
        self.slider_infer.set(config.infer_interval * 1000)
        self.slider_infer.pack(padx=20, fill="x")

        # 終了ボタン
        self.btn_quit = ctk.CTkButton(self, text="アプリを終了", fg_color="red", hover_color="darkred", command=self.on_closing)
        self.btn_quit.pack(pady=30)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_mask(self):
        config.mask_enabled = self.switch_mask.get() == 1
        print("🎭 マスクON" if config.mask_enabled else "🙈 マスクOFF")

    def update_scale(self, val):
        config.scale = val
        self.label_scale.configure(text=f"マスクの大きさ: {config.scale:.1f}")

    def update_smooth(self, val):
        config.smooth_frames = int(val)
        self.label_smooth.configure(text=f"動きの滑らかさ (フレーム): {config.smooth_frames}")

    def update_infer(self, val):
        config.infer_interval = val / 1000.0
        self.label_infer.configure(text=f"認識の頻度 (ミリ秒): {int(val)}")

    def on_closing(self):
        config.running = False
        self.destroy()

if __name__ == "__main__":
    # カメラ処理を別スレッドで開始
    thread = threading.Thread(target=camera_thread, daemon=True)
    thread.start()

    # GUIを開始
    app = ControlApp()
    app.mainloop()
