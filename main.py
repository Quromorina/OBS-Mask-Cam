import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import os
import sys

# EXE環境ではonnxruntimeのDLLパスを事前にPATHに追加
if getattr(sys, 'frozen', False):
    _ort_capi = os.path.join(sys._MEIPASS, 'onnxruntime', 'capi')
    if os.path.isdir(_ort_capi):
        os.environ['PATH'] = _ort_capi + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(_ort_capi)

import onnxruntime as ort
import time
import threading
import customtkinter as ctk
import shutil
from tkinter import filedialog
from PIL import Image
try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False

# --- 日本語パス対応のOpenCVヘルパー ---
def imread_safe(path, flags=cv2.IMREAD_UNCHANGED):
    """日本語パスでも読み込めるcv2.imread代替"""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, flags)
    except Exception:
        return None

def imwrite_safe(path, img):
    """日本語パスでも保存できるcv2.imwrite代替"""
    ext = os.path.splitext(path)[1]
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(path)
        return True
    return False

# --- パス解決（EXE対応） ---
IS_FROZEN = getattr(sys, 'frozen', False)
if IS_FROZEN:
    # PyInstallerでビルドされたEXEの場合
    APP_DIR = os.path.dirname(sys.executable)
    BUNDLE_DIR = getattr(sys, '_MEIPASS', APP_DIR)
else:
    # 通常のPython実行
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    BUNDLE_DIR = APP_DIR

MODEL_PATH = os.path.join(BUNDLE_DIR, "yolov8n-face.onnx")
MASK_DIR = os.path.join(APP_DIR, "masks")

print(f"📁 APP_DIR: {APP_DIR}")
print(f"📁 BUNDLE_DIR: {BUNDLE_DIR}")
print(f"📁 MODEL_PATH: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
print(f"📁 MASK_DIR: {MASK_DIR}")

# --- 設定（初期値） ---
class AppConfig:
    width, height, fps = 1280, 720, 30
    scale = 1.8
    mask_enabled = True
    infer_interval = 0.02  # 固定値: 20ms (変更不可)
    smooth_frames = 5
    distance_threshold = 50
    running = True
    current_mask_name = ""
    mask_files = []
    need_reload_list = False
    camera_index = 0
    camera_list = []
    provider_name = ""           # GPU/CPU表示用
    startup_error = ""           # 起動時エラーメッセージ

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

# --- フォルダ準備＆デフォルトマスク初期化 ---
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)
    print(f"📁 masksフォルダ作成: {MASK_DIR}")

# masksフォルダ内に画像が1つもない場合、モザイクマスクを自動生成
def _generate_default_mask():
    """デフォルトのモザイクパターンマスクを生成"""
    size = 256
    img = np.zeros((size, size, 4), dtype=np.uint8)
    block = 16
    rng = np.random.RandomState(42)  # 再現性のため固定シード
    for y in range(0, size, block):
        for x in range(0, size, block):
            c = rng.randint(60, 200)
            img[y:y+block, x:x+block] = (c, c, c, 220)
    path = os.path.join(MASK_DIR, "mosaic.png")
    imwrite_safe(path, img)
    print(f"🎭 デフォルトモザイクマスク生成: {path}")
    return "mosaic.png"

def get_mask_list():
    files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    return files

config.mask_files = get_mask_list()
if not config.mask_files:
    # マスクが1つもない → デフォルト生成
    default_name = _generate_default_mask()
    config.mask_files = get_mask_list()

if config.mask_files:
    config.current_mask_name = config.mask_files[0]
    print(f"🎭 初期マスク: {config.current_mask_name}")
else:
    config.current_mask_name = ""
    print("⚠ マスク画像が見つかりません")

# --- ONNX推論ヘルパー ---
def create_onnx_session(model_path):
    """作成ONNX Runtimeセッション（DirectML → CPU fallback）"""
    providers = []
    available = ort.get_available_providers()
    print(f"🔍 利用可能なProvider: {available}")
    if 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    config.provider_name = active_provider
    print(f"✅ Using provider: {active_provider}")
    return session

def preprocess(frame, input_size=640):
    """フレームをONNX入力形式にリサイズ＆正規化"""
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # パディングして正方形に
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # BGR→RGB, HWC→CHW, 正規化
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, 0, 0  # pad_x, pad_y は0（左上寄せ）

def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression"""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep

def postprocess(output, scale, orig_w, orig_h, conf_threshold=0.5, iou_threshold=0.45):
    """YOLOv8出力を処理して顔バウンディングボックスを返す"""
    # output shape: (1, 5, 8400) → transpose → (8400, 5) : [cx, cy, w, h, conf]
    predictions = output[0].transpose()
    
    # 信頼度フィルタ
    scores = predictions[:, 4]
    mask = scores > conf_threshold
    predictions = predictions[mask]
    scores = scores[mask]
    
    if len(predictions) == 0:
        return []
    
    # cx, cy, w, h → x1, y1, x2, y2
    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # NMS
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep]
    
    # スケールを元画像サイズに戻す
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale
    
    # クリッピング
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    return boxes.astype(int)

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
    # カメラ確認
    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        config.startup_error = "camera"
        print("❌ カメラが見つかりません")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    last_camera_index = config.camera_index

    def load_mask(name):
        if not name:
            print("⚠ マスク名が空です")
            return None
        path = os.path.join(MASK_DIR, name)
        if not os.path.exists(path):
            print(f"⚠ マスクファイルが見つかりません: {path}")
            return None
        img = imread_safe(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠ マスク画像の読み込み失敗: {path}")
            return None
        
        if len(img.shape) == 2: # グレースケール
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3: # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        print(f"✅ マスク読み込み成功: {name} ({img.shape})")
        return img

    overlay_img = load_mask(config.current_mask_name)
    loaded_mask_name = config.current_mask_name

    # ONNX Runtimeセッション作成
    session = create_onnx_session(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    face_history = {}
    last_infer_time = 0

    # Virtual Camera接続
    try:
        cam = pyvirtualcam.Camera(width=config.width, height=config.height, fps=config.fps, fmt=PixelFormat.BGR)
    except Exception as e:
        config.startup_error = "virtualcam"
        print(f"❌ OBS Virtual Cameraが見つかりません: {e}")
        cap.release()
        return

    print(f'✅ Virtual camera started: {cam.device}')

    try:
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
                # カメラからフレーム取得失敗 → 黒画面を送る
                frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (config.width//4, config.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cam.send(frame)
                cam.sleep_until_next_frame()
                continue

            now = time.time()
            if now - last_infer_time > config.infer_interval:
                # ONNX推論
                blob, scale, _, _ = preprocess(frame, 640)
                output = session.run(None, {input_name: blob})[0]
                boxes = postprocess(output, scale, frame.shape[1], frame.shape[0])
                
                current_faces = []
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    face_size = int(max(x2 - x1, y2 - y1) * config.scale)
                    current_faces.append((cx, cy, face_size))
                
                new_face_history = {}
                matched_old_ids = set()  # マッチ済みIDを追跡して二重マッチを防止
                for (cx, cy, fs) in current_faces:
                    matched_id = None
                    min_dist = config.distance_threshold
                    for fid, data in face_history.items():
                        if fid in matched_old_ids:  # 既にマッチ済みならスキップ
                            continue
                        prev_cx, prev_cy, _ = data["history"][-1]
                        dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            matched_id = fid
                    
                    if matched_id is not None:
                        matched_old_ids.add(matched_id)  # マッチ済みとして記録
                        history = face_history[matched_id]["history"]
                        history.append((cx, cy, fs))
                        if len(history) > 20: 
                            history.pop(0)
                        new_face_history[matched_id] = {"history": history, "last_update": now}
                    else:
                        new_id = time.time() + np.random.rand()
                        new_face_history[new_id] = {"history": [(cx, cy, fs)], "last_update": now}
                
                # ゴースト保持: 現在の検出に近いものは移動した顔なので保持しない
                for fid, data in face_history.items():
                    if fid in matched_old_ids or fid in new_face_history:
                        continue
                    if now - data["last_update"] < 0.5:
                        ghost_cx, ghost_cy, _ = data["history"][-1]
                        near_current = any(
                            np.sqrt((cx - ghost_cx)**2 + (cy - ghost_cy)**2) < config.distance_threshold * 3
                            for (cx, cy, fs) in current_faces
                        )
                        if not near_current:
                            new_face_history[fid] = data

                face_history = new_face_history
                last_infer_time = now

            if config.mask_enabled and overlay_img is not None:
                for fid, data in face_history.items():
                    history = data["history"]
                    current_history = history[-config.smooth_frames:]
                    avg_cx = int(np.mean([h[0] for h in current_history]))
                    avg_cy = int(np.mean([h[1] for h in current_history]))
                    
                    # 適応的スムージング: サイズ変化が大きい時は即追従
                    sizes = [h[2] for h in current_history]
                    latest_fs = sizes[-1]
                    if len(sizes) >= 2:
                        size_change_ratio = abs(sizes[-1] - sizes[-2]) / max(sizes[-2], 1)
                        if size_change_ratio > 0.15:  # 15%以上のサイズ変化
                            # 最新2フレームだけで平均（即追従）
                            avg_fs = int(np.mean(sizes[-2:]))
                        else:
                            avg_fs = int(np.mean(sizes))
                    else:
                        avg_fs = latest_fs

                    try:
                        resized = cv2.resize(overlay_img, (avg_fs, avg_fs))
                        x_offset = avg_cx - avg_fs // 2
                        y_offset = avg_cy - avg_fs // 2
                        frame = overlay_transparent(frame, resized, x_offset, y_offset)
                    except cv2.error:
                        continue

            cam.send(frame)
            cam.sleep_until_next_frame()
    finally:
        cam.close()
        cap.release()

# --- GUIクラス ---
class ControlApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("OBS Mask Cam - コントロールパネル")
        self.geometry("450x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # フォント設定
        self.font_main = ("MS Gothic", 12)
        self.font_bold = ("MS Gothic", 14, "bold")
        self.font_title = ("MS Gothic", 24, "bold")
        self.font_button = ("MS Gothic", 18, "bold")
        self.font_small = ("MS Gothic", 10)

        # タイトル
        self.label_title = ctk.CTkLabel(self, text="🎭 OBS Mask Cam", font=self.font_title)
        self.label_title.pack(pady=15)

        # --- ステータス表示 ---
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(pady=5, padx=20, fill="x")
        
        self.label_provider = ctk.CTkLabel(self.status_frame, text="⏳ 起動中...", 
                                           font=self.font_small, text_color="#888888")
        self.label_provider.pack(pady=5)
        
        self.label_error = None
        
        # provider情報をポーリングで更新
        self._poll_provider_status()


        # --- カメラ選択エリア ---
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(pady=10, padx=20, fill="x")
        
        self.label_camera = ctk.CTkLabel(self.camera_frame, text="映像ソース (カメラ):", font=self.font_bold)
        self.label_camera.pack(side="left", padx=10, pady=10)
        
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

        self.label_scale = ctk.CTkLabel(self.settings_frame, text=f"マスクの大きさ: {config.scale:.1f}", font=self.font_main)
        self.label_scale.pack(pady=(10, 0))
        self.slider_scale = ctk.CTkSlider(self.settings_frame, from_=1.0, to=4.0, command=self.update_scale)
        self.slider_scale.set(config.scale)
        self.slider_scale.pack(padx=20, pady=(0, 10), fill="x")

        self.label_smooth = ctk.CTkLabel(self.settings_frame, text=f"動きの滑らかさ: {config.smooth_frames}", font=self.font_main)
        self.label_smooth.pack(pady=(10, 0))
        self.slider_smooth = ctk.CTkSlider(self.settings_frame, from_=1, to=20, number_of_steps=19, command=self.update_smooth)
        self.slider_smooth.set(config.smooth_frames)
        self.slider_smooth.pack(padx=20, pady=(0, 20), fill="x")

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
                img = imread_safe(file_path, cv2.IMREAD_UNCHANGED)
                if img is None: return

                if len(img.shape) == 2: # グレースケール
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3: # BGR
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                
                # ファイル名を強制的に .png に変更して保存
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                new_file_name = f"{base_name}.png"
                dest_path = os.path.join(MASK_DIR, new_file_name)
                
                imwrite_safe(dest_path, img)
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

    def _poll_provider_status(self):
        """camera_threadのprovider情報が確定するまでポーリング"""
        if config.provider_name:
            # provider確定 → 表示更新
            if "Dml" in config.provider_name:
                self.label_provider.configure(text="✅ GPU推論 (DirectML)", text_color="#2d8659")
            else:
                self.label_provider.configure(text="⚠ CPU推論（低速）", text_color="#9e6b2b")
            return
        
        if config.startup_error:
            # エラー発生
            self.label_provider.configure(text="⚠ CPU推論（低速）", text_color="#9e6b2b")
            if config.startup_error == "virtualcam":
                self.label_error = ctk.CTkLabel(self.status_frame, 
                    text="❌ OBS Virtual Cameraが見つかりません\nOBS Studioをインストールしてください", 
                    font=self.font_small, text_color="#e74c3c", wraplength=380)
                self.label_error.pack(pady=5)
            elif config.startup_error == "camera":
                self.label_error = ctk.CTkLabel(self.status_frame, 
                    text="❌ カメラが見つかりません\nWebカメラを接続してください", 
                    font=self.font_small, text_color="#e74c3c", wraplength=380)
                self.label_error.pack(pady=5)
            return
        
        # まだ確定してない → 500ms後に再チェック
        self.after(500, self._poll_provider_status)

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
