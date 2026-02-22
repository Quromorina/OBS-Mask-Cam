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
import os
import shutil
from tkinter import filedialog
from PIL import Image
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

# --- 設定（初期値） ---
class AppConfig:
    width, height, fps = 1280, 720, 30
    scale = 1.8
    mask_enabled = True
    infer_interval = 0.02
    smooth_frames = 5
    distance_threshold = 50
    running = True
    current_mask_name = "mask.png"
    mask_files = []
    need_reload_list = False
    camera_index = 0
    camera_list = []

config = AppConfig()

# --- カメラ検出 ---
def get_camera_list():
    if HAS_PYGRABBER:
        try:
            graph = FilterGraph()
            return graph.get_input_devices()
        except Exception:
            return ["0"]
    else:
        # フォールバック: OpenCVでの簡易スキャン
        available = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                available.append(str(i))
                cap.release()
        return available if available else ["0"]

config.camera_list = get_camera_list()

# --- フォルダ準備 ---
MASK_DIR = "masks"
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)

def get_mask_list():
    files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    return files if files else ["mask.png"]

config.mask_files = get_mask_list()
if config.mask_files:
    config.current_mask_name = config.mask_files[0]

# --- アルファ合成 ---
def overlay_transparent(background, overlay, x, y):
    if overlay.shape[2] != 4:
        # アルファチャンネルがない場合は単に上書き（安全策）
        h, w = overlay.shape[:2]
        bh, bw = background.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, bw), min(y + h, bh)
        if x1 >= x2 or y1 >= y2: return background
        overlay_rgb = overlay[:, :, :3]
        background[y1:y2, x1:x2] = overlay_rgb[0:y2-y1, 0:x2-x1]
        return background

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
    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    last_camera_index = config.camera_index

    def load_mask(name):
        path = os.path.join(MASK_DIR, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        
        if len(img.shape) == 2: # グレースケール
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    overlay_img = load_mask(config.current_mask_name)
    loaded_mask_name = config.current_mask_name

    model = YOLO("yolov8n-face.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    face_history = {}
    last_infer_time = 0

    with pyvirtualcam.Camera(width=config.width, height=config.height, fps=config.fps, fmt=PixelFormat.BGR) as cam:
        print(f'✅ Virtual camera started: {cam.device}')

        while config.running:
            # カメラソースの切り替えチェック
            if last_camera_index != config.camera_index:
                print(f"🔄 カメラ切り替え中: Index {config.camera_index}")
                cap.release()
                cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                last_camera_index = config.camera_index

            # マスクリストの更新チェック
            if config.need_reload_list:
                config.mask_files = get_mask_list()
                config.need_reload_list = False

            # マスクの切り替え判定
            if loaded_mask_name != config.current_mask_name:
                new_img = load_mask(config.current_mask_name)
                if new_img is not None:
                    overlay_img = new_img
                    loaded_mask_name = config.current_mask_name

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
                        if len(history) > 20: 
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

            if config.mask_enabled and overlay_img is not None:
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
# --- GUIクラス ---
class ControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OBS Mask Cam - コントロールパネル")
        self.geometry("450x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # フォント設定
        self.font_main = ("MS Gothic", 12)
        self.font_bold = ("MS Gothic", 14, "bold")
        self.font_title = ("MS Gothic", 24, "bold")
        self.font_button = ("MS Gothic", 18, "bold")

        # タイトル
        self.label_title = ctk.CTkLabel(self, text="🎭 OBS Mask Cam", font=self.font_title)
        self.label_title.pack(pady=15)

        # --- カメラ選択エリア ---
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(pady=10, padx=20, fill="x")
        
        self.label_camera = ctk.CTkLabel(self.camera_frame, text="映像ソース (カメラ):", font=self.font_bold)
        self.label_camera.pack(side="left", padx=10, pady=10)
        
        # 名前で表示
        self.option_camera = ctk.CTkOptionMenu(self.camera_frame, values=config.camera_list, 
                                               font=self.font_main, command=self.update_camera_choice)
        if config.camera_index < len(config.camera_list):
            self.option_camera.set(config.camera_list[config.camera_index])
        else:
            self.option_camera.set(config.camera_list[0])
        self.option_camera.pack(side="right", padx=10, pady=10)

        # --- プレビューエリア ---
        self.preview_frame = ctk.CTkFrame(self, width=200, height=200)
        self.preview_frame.pack(pady=10)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="画像なし", font=self.font_main)
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # --- マスク選択・追加 ---
        self.label_mask_choice = ctk.CTkLabel(self, text="使用するマスク選択:", font=self.font_bold)
        self.label_mask_choice.pack(pady=(10, 0))
        
        self.option_mask = ctk.CTkOptionMenu(self, values=config.mask_files, font=self.font_main, dropdown_font=self.font_main, command=self.update_mask_choice)
        self.option_mask.set(config.current_mask_name)
        self.option_mask.pack(pady=5)

        self.btn_add_mask = ctk.CTkButton(self, text="➕ 新しいマスクを追加", font=self.font_main, fg_color="#2b719e", command=self.add_mask_file)
        self.btn_add_mask.pack(pady=5)

        # --- マスクON/OFF 大ボタン ---
        self.btn_toggle = ctk.CTkButton(self, text="マスクを無効にする", height=60, font=self.font_button,
                                        fg_color="#2d8659", hover_color="#236b47", command=self.toggle_mask)
        self.btn_toggle.pack(pady=20, padx=40, fill="x")
        self.update_toggle_button_ui()

        # --- 設定エリア ---
        self.settings_frame = ctk.CTkFrame(self)
        self.settings_frame.pack(pady=10, padx=20, fill="x")

        # マスクサイズ調整
        self.label_scale = ctk.CTkLabel(self.settings_frame, text=f"マスクの大きさ: {config.scale:.1f}", font=self.font_main)
        self.label_scale.pack(pady=(10, 0))
        self.slider_scale = ctk.CTkSlider(self.settings_frame, from_=1.0, to=4.0, command=self.update_scale)
        self.slider_scale.set(config.scale)
        self.slider_scale.pack(padx=20, pady=(0, 10), fill="x")

        # スムージング調整
        self.label_smooth = ctk.CTkLabel(self.settings_frame, text=f"動きの滑らかさ: {config.smooth_frames}", font=self.font_main)
        self.label_smooth.pack(pady=(10, 0))
        self.slider_smooth = ctk.CTkSlider(self.settings_frame, from_=1, to=20, number_of_steps=19, command=self.update_smooth)
        self.slider_smooth.set(config.smooth_frames)
        self.slider_smooth.pack(padx=20, pady=(0, 10), fill="x")

        # 推論間隔調整
        self.label_infer = ctk.CTkLabel(self.settings_frame, text=f"認識の頻度: {int(config.infer_interval * 1000)}ms", font=self.font_main)
        self.label_infer.pack(pady=(10, 0))
        self.slider_infer = ctk.CTkSlider(self.settings_frame, from_=1, to=500, command=self.update_infer)
        self.slider_infer.set(config.infer_interval * 1000)
        self.slider_infer.pack(padx=20, pady=(0, 20), fill="x")

        # 終了ボタン
        self.btn_quit = ctk.CTkButton(self, text="アプリを終了", height=40, font=self.font_bold, fg_color="#9e2b2b", hover_color="#7a2222", command=self.on_closing)
        self.btn_quit.pack(pady=(20, 30))

        self.update_preview()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_preview(self):
        try:
            path = os.path.join(MASK_DIR, config.current_mask_name)
            if os.path.exists(path):
                img = Image.open(path)
                aspect = img.width / img.height
                if aspect > 1:
                    w, h = 180, int(180 / aspect)
                else:
                    w, h = int(180 * aspect), 180
                
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                self.preview_label.configure(image=ctk_img, text="")
        except Exception as e:
            self.preview_label.configure(image=None, text="エラー", font=self.font_main)
            print(f"Preview error: {e}")

    def add_mask_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if file_path:
            try:
                # 画像を読み込んでチャンネル数を確認
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is None: return

                if len(img.shape) == 2: # グレースケール
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3: # BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                
                # ファイル名を強制的に .png に変更して保存
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                new_file_name = f"{base_name}.png"
                dest_path = os.path.join(MASK_DIR, new_file_name)
                
                cv2.imwrite(dest_path, img)
                print(f"✅ 画像を透過対応PNGとして保存しました: {new_file_name}")

                config.need_reload_list = True
                time.sleep(0.1)
                new_list = get_mask_list()
                self.option_mask.configure(values=new_list)
                self.option_mask.set(new_file_name)
                self.update_mask_choice(new_file_name)
            except Exception as e:
                print(f"Error adding file: {e}")

    def update_mask_choice(self, choice):
        config.current_mask_name = choice
        self.update_preview()
        print(f"🎭 マスク切り替え: {choice}")

    def update_camera_choice(self, choice):
        try:
            index = config.camera_list.index(choice)
            config.camera_index = index
            print(f"🎬 カメラ選択: {choice} (Index: {index})")
        except ValueError:
            pass

    def toggle_mask(self):
        config.mask_enabled = not config.mask_enabled
        self.update_toggle_button_ui()
        print("🎭 マスクON" if config.mask_enabled else "🙈 マスクOFF")

    def update_toggle_button_ui(self):
        if config.mask_enabled:
            self.btn_toggle.configure(text="マスクを無効にする", fg_color="#2d8659", hover_color="#236b47")
        else:
            self.btn_toggle.configure(text="マスクを有効にする", fg_color="#5a5a5a", hover_color="#4a4a4a")

    def update_scale(self, val):
        config.scale = val
        self.label_scale.configure(text=f"マスクの大きさ: {config.scale:.1f}")

    def update_smooth(self, val):
        config.smooth_frames = int(val)
        self.label_smooth.configure(text=f"動きの滑らかさ: {config.smooth_frames}")

    def update_infer(self, val):
        config.infer_interval = max(0.001, val / 1000.0)
        self.label_infer.configure(text=f"認識の頻度: {int(val)}ms")

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
