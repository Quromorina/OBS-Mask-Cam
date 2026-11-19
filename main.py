import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import os
import sys
import time
import threading
import logging
import logging.handlers
import customtkinter as ctk
import shutil
from tkinter import filedialog, messagebox
from PIL import Image
from mask_safety import (normalize_mask_image, fit_frame, obscure_frame, cover_face,
                         decide_mask_status, save_mask_png)
from face_detector import FaceDetector, MODEL_FILENAME
from camera_recovery import configure_capture, ReopenTimer
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
MODEL_PATH = os.path.join(BUNDLE_DIR, MODEL_FILENAME)

# 共有データ（ユーザーごとのAppData\Roamingに保存）
USER_DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "OBSMaskCam")
MASK_DIR = os.path.join(USER_DATA_DIR, "masks")
APP_MASK_DIR = os.path.join(APP_DIR, "masks")
LOG_PATH = os.path.join(USER_DATA_DIR, "app.log")

# --- ログ（EXEはコンソールが無いので、原因を追えるようファイルにも残す） ---
def setup_logging():
    logger = logging.getLogger("obsmaskcam")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    try:
        os.makedirs(USER_DATA_DIR, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"ログファイルを開けません: {e}")
    return logger

log = setup_logging()
log.info("起動: APP_DIR=%s MODEL_PATH=%s (exists=%s)", APP_DIR, MODEL_PATH, os.path.exists(MODEL_PATH))

print(f"📁 APP_DIR: {APP_DIR}")
print(f"📁 BUNDLE_DIR: {BUNDLE_DIR}")
print(f"📁 MODEL_PATH: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
print(f"📁 MASK_DIR: {MASK_DIR}")

# --- 設定（初期値） ---
class AppConfig:
    width, height, fps = 1280, 720, 30
    scale = 1.8
    mask_enabled = True
    infer_interval = 0.01  # 固定値: 10ms
    smooth_frames = 5
    distance_threshold = 50
    running = True
    current_mask_name = ""
    mask_files = []
    need_reload_list = False
    camera_index = 0
    camera_list = []
    camera_mapping = {}          # カメラ名 -> システムIndexの対応
    provider_name = ""           # 顔検出の方式の表示用（空のあいだは「起動中」）
    startup_error = ""           # エラー種別: camera / virtualcam / model / internal
    error_detail = ""            # 画面に出すエラーの理由
    face_hold_seconds = 1.5      # 顔を見失ってからマスクをその場に残す秒数
    # 現在の送出状態: starting / ok / fallback / detector_error / off / no_camera / stopped
    mask_status = "starting"

config = AppConfig()

# --- カメラ検出 ---
def get_camera_list():
    input_names = []
    mapping = {}
    if HAS_PYGRABBER:
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            for i, name in enumerate(devices):
                if "OBS Virtual Camera" not in name:
                    input_names.append(name)
                    mapping[name] = i
            return input_names if input_names else ["0"], mapping if input_names else {"0": 0}
        except Exception:
            return ["0"], {"0": 0}
    else:
        # フォールバック: OpenCVでの簡易スキャン
        available = []
        mapping = {}
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                name = str(i)
                available.append(name)
                mapping[name] = i
                cap.release()
        return available if available else ["0"], mapping if mapping else {"0": 0}

config.camera_list, config.camera_mapping = get_camera_list()
if len(config.camera_list) > 0:
    config.camera_index = config.camera_mapping.get(config.camera_list[0], 0)
else:
    config.camera_index = 0

# --- フォルダ準備＆デフォルトマスク初期化 ---
if not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR, exist_ok=True)
    print(f"📁 masksフォルダ作成: {MASK_DIR}")
    if os.path.exists(APP_MASK_DIR):
        for f in os.listdir(APP_MASK_DIR):
            src_file = os.path.join(APP_MASK_DIR, f)
            if os.path.isfile(src_file):
                try:
                    shutil.copy2(src_file, MASK_DIR)
                except Exception as e:
                    print(f"初期マスクのコピー失敗: {e}")

def get_mask_list():
    files = [f for f in os.listdir(MASK_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    return files

# 古いモザイク画像の削除（非推奨化）
old_mosaic_path = os.path.join(MASK_DIR, "mosaic.png")
if os.path.exists(old_mosaic_path):
    try:
        os.remove(old_mosaic_path)
    except Exception:
        pass

# icon.png が MASK_DIR になければコピーして生成
dest_icon_path = os.path.join(MASK_DIR, "icon.png")
if not os.path.exists(dest_icon_path):
    app_icon_path = os.path.join(APP_DIR, "icon.png")
    if not os.path.exists(app_icon_path):
        app_icon_path = os.path.join(APP_DIR, "icon.ico")
    if not os.path.exists(app_icon_path):
        app_icon_path = os.path.join(BUNDLE_DIR, "icon.ico")
    try:
        if os.path.exists(app_icon_path):
            img = Image.open(app_icon_path)
            img.save(dest_icon_path, "PNG")
            print(f"🎭 デフォルトマスク（アイコン）生成: {dest_icon_path}")
    except Exception as e:
        print(f"デフォルトマスク生成エラー: {e}")

config.mask_files = get_mask_list()

if config.mask_files:
    config.current_mask_name = config.mask_files[0]
    print(f"🎭 初期マスク: {config.current_mask_name}")
else:
    config.current_mask_name = ""
    print("⚠ マスク画像が見つかりません")

# --- マスク画像の読み込み ---
def load_mask(name):
    """マスク画像を 8bit BGRA で返す。使えないときは None（呼び出し側はぼかしで代用する）。"""
    if not name:
        log.warning("マスクが選択されていません。ぼかしで代用します")
        return None
    path = os.path.join(MASK_DIR, name)
    if not os.path.exists(path):
        log.warning("マスクファイルが見つかりません: %s", path)
        return None
    img = normalize_mask_image(imread_safe(path, cv2.IMREAD_UNCHANGED))
    if img is None:
        log.warning("マスク画像を読み込めません: %s", path)
        return None
    if img[:, :, 3].max() == 0:
        log.warning("マスク画像が完全に透明です（顔を隠せません）: %s", path)
        return None
    log.info("マスク読み込み成功: %s %s", name, img.shape)
    return img

# --- カメラを開く ---
def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    if cap.isOpened():
        # MJPG を要求して USB の帯域を節約する（無圧縮だと帯域不足で止まるカメラがある）
        actual_w, actual_h, fourcc = configure_capture(cap, config.width, config.height, config.fps)
        log.info("カメラ(Index %s)を開きました: %sx%s %s", index, actual_w, actual_h, fourcc)
        if (actual_w, actual_h) != (config.width, config.height):
            # 要求どおりの解像度にならないカメラ。毎フレーム fit_frame で出力サイズへ合わせる
            log.warning("カメラ(Index %s)の実解像度は %sx%s です。%sx%s に合わせて送出します",
                        index, actual_w, actual_h, config.width, config.height)
    return cap

# 顔検出が失敗したときに再試行する間隔（秒）
DETECTOR_RETRY_SECONDS = 1.0

# --- カメラ・AI処理スレッド ---
def camera_thread():
    """スレッドの入口。想定外の例外でも黙って終わらせず、理由を画面とログに残す。"""
    try:
        _camera_loop()
    except Exception as e:
        log.exception("映像処理スレッドが例外で停止しました")
        if not config.startup_error:
            config.error_detail = f"{type(e).__name__}: {e}"
            config.startup_error = "internal"
    finally:
        config.mask_status = "stopped"

def _camera_loop():
    # カメラ確認
    cap = open_camera(config.camera_index)
    if not cap.isOpened():
        config.startup_error = "camera"
        log.error("カメラが見つかりません (Index %s)", config.camera_index)
        cap.release()
        return
    last_camera_index = config.camera_index
    cam = None
    reopen_timer = ReopenTimer()

    try:
        loaded_mask_name = config.current_mask_name
        overlay_img = load_mask(loaded_mask_name)

        # 顔検出（YuNet / OpenCV DNN・CPU）の準備。起動時に一度ダミー画像で動作確認もする
        try:
            detector = FaceDetector(MODEL_PATH)
            config.provider_name = "YuNet (OpenCV DNN / CPU)"
            log.info("顔検出の準備完了: %s (OpenCV %s)", config.provider_name, cv2.__version__)
        except Exception as e:
            config.error_detail = f"{type(e).__name__}: {e}"
            config.startup_error = "model"
            log.exception("顔検出モデルを読み込めません: %s", MODEL_PATH)
            return

        face_history = {}
        last_infer_time = 0
        detector_ok = True

        # Virtual Camera接続
        try:
            cam = pyvirtualcam.Camera(width=config.width, height=config.height, fps=config.fps, fmt=PixelFormat.BGR)
        except Exception as e:
            config.startup_error = "virtualcam"
            config.error_detail = f"{type(e).__name__}: {e}"
            log.exception("OBS Virtual Camera に接続できません")
            return

        log.info("Virtual camera started: %s", cam.device)

        while config.running:
            # カメラソースの切り替えチェック
            if last_camera_index != config.camera_index:
                print(f"🔄 カメラ切り替え中: Index {config.camera_index}")
                
                # スムーズな切り替えのため、一旦ダミー画面を出力
                frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Switching Camera...", (config.width//4, config.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cam.send(frame)
                cam.sleep_until_next_frame()
                
                cap.release()
                time.sleep(0.5) # リソース解放を待機
                
                last_camera_index = config.camera_index
                cap = open_camera(last_camera_index)
                reopen_timer.reset()
                continue

            # マスクリストの更新チェック
            if config.need_reload_list:
                config.mask_files = get_mask_list()
                config.need_reload_list = False

            # マスクの切り替え判定
            # 読み込めなかった場合は古いマスクを出し続けず、ぼかしで代用する（overlay_img = None）
            if loaded_mask_name != config.current_mask_name:
                loaded_mask_name = config.current_mask_name
                overlay_img = load_mask(loaded_mask_name)

            ret, frame = cap.read()
            # カメラの実解像度が出力サイズと違っても、ここで必ず width x height に揃える
            frame = fit_frame(frame, config.width, config.height) if ret else None
            if frame is None:
                # カメラからフレーム取得失敗 → 黒画面を送る
                frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not available", (config.width//4, config.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                config.mask_status = "no_camera"
                cam.send(frame)
                cam.sleep_until_next_frame()
                # 途切れたままなら、カメラを開き直す（アプリの再起動を不要にする）
                if reopen_timer.on_failure(time.time()) and config.running:
                    log.warning("カメラ(Index %s)の映像が途切れたため、開き直します", last_camera_index)
                    cap.release()
                    time.sleep(0.5)  # リソース解放を待機
                    cap = open_camera(last_camera_index)
                    if not cap.isOpened():
                        log.warning("カメラ(Index %s)を開けませんでした。少し待って再試行します", last_camera_index)
                continue
            if reopen_timer.on_success():
                log.info("カメラの映像が戻りました")

            now = time.time()
            infer_wait = config.infer_interval if detector_ok else DETECTOR_RETRY_SECONDS
            if now - last_infer_time > infer_wait:
                # 顔検出。失敗したら顔の位置が分からないので、画面全体をぼかす側に倒す
                # （ランドマークと信頼度も返るが、いまのマスク合成は bbox だけを使う）
                try:
                    boxes, _landmarks, _scores = detector.detect(frame)
                    if not detector_ok:
                        log.info("顔検出が復帰しました")
                    detector_ok = True
                except Exception:
                    if detector_ok:
                        log.exception("顔検出に失敗しました。復帰するまで画面全体をぼかします")
                    detector_ok = False
                    boxes = []


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
                    if now - data["last_update"] < config.face_hold_seconds:
                        ghost_cx, ghost_cy, _ = data["history"][-1]
                        near_current = any(
                            np.sqrt((cx - ghost_cx)**2 + (cy - ghost_cy)**2) < config.distance_threshold * 3
                            for (cx, cy, fs) in current_faces
                        )
                        if not near_current:
                            new_face_history[fid] = data

                face_history = new_face_history
                last_infer_time = now

            all_masks_applied = True
            covered_faces = 0
            if not config.mask_enabled:
                pass
            elif not detector_ok:
                frame = obscure_frame(frame)
            else:
                for fid, data in face_history.items():
                    history = data["history"]
                    current_history = history[-config.smooth_frames:]
                    
                    # 適応的スムージング: 速い動きの時は即追従 (XY軸)
                    if len(current_history) >= 2:
                        dx = abs(current_history[-1][0] - current_history[-2][0])
                        dy = abs(current_history[-1][1] - current_history[-2][1])
                        move_dist = np.sqrt(dx**2 + dy**2)
                        
                        # 顔サイズに対する移動量の比率で判定
                        face_ref = max(current_history[-1][2], 1)
                        move_ratio = move_dist / face_ref
                        ds = abs(current_history[-1][2] - current_history[-2][2])
                        scale_ratio = ds / face_ref

                        if move_ratio > 0.25 or scale_ratio > 0.08:  # しきい値を下げてより機敏に
                            # 最新2フレームだけで平均（即追従）
                            use_hist = current_history[-2:]
                        else:
                            use_hist = current_history
                    else:
                        use_hist = current_history
                    
                    avg_cx = int(np.mean([h[0] for h in use_hist]))
                    avg_cy = int(np.mean([h[1] for h in use_hist]))
                    
                    # サイズ(Z軸)も適応的スムージング：変化に敏感にする
                    sizes = [h[2] for h in current_history]
                    latest_fs = sizes[-1]
                    if len(sizes) >= 2:
                        size_change_ratio = abs(sizes[-1] - sizes[-2]) / max(sizes[-2], 1)
                        if size_change_ratio > 0.05: # 5%以上の変化でもう追従
                            avg_fs = int((sizes[-1] * 2 + sizes[-2]) / 3) # 最新の重みを高く
                        else:
                            # 変化が少ない場合でも最新寄りに
                            weights = np.linspace(0.5, 1.5, len(sizes))
                            avg_fs = int(np.average(sizes, weights=weights))
                    else:
                        avg_fs = latest_fs

                    # マスク画像を重ねられないときは、その顔をモザイクで潰す（素通しにしない）
                    frame, used_mask = cover_face(frame, overlay_img, avg_cx, avg_cy, avg_fs)
                    covered_faces += 1
                    if not used_mask:
                        all_masks_applied = False

            # 顔が 0 件のときは「マスク適用中」ではなく「顔未検出」と出す
            status = decide_mask_status(config.mask_enabled, detector_ok, overlay_img is not None,
                                        covered_faces, all_masks_applied)

            if status != config.mask_status:
                log.info("送出状態: %s -> %s", config.mask_status, status)
                config.mask_status = status

            cam.send(frame)
            cam.sleep_until_next_frame()
    finally:
        if cam is not None:
            cam.close()
        cap.release()

# --- GUIクラス ---
class ControlApp(ctk.CTk):
    # config.mask_status -> (表示文言, 背景色)
    MASK_STATUS_UI = {
        "starting":       ("⏳ 起動中...", "#5a5a5a"),
        "ok":             ("✅ マスク適用中", "#2d8659"),
        "no_face":        ("👤 顔未検出（隠す対象の顔が見つかっていません）", "#3a6ea5"),
        "no_face_fallback": ("⚠ 顔未検出・マスク画像なし（顔が映ったら、ぼかしで隠します）", "#9e6b2b"),
        "fallback":       ("⚠ マスク画像が使えないため、ぼかしで顔を隠しています", "#9e6b2b"),
        "detector_error": ("⚠ 顔検出が止まっているため、画面全体をぼかしています", "#9e6b2b"),
        "off":            ("🚫 マスクOFF：素顔がそのまま映ります", "#c0392b"),
        "no_camera":      ("⚠ カメラ映像なし（黒画面を送出中）", "#5a5a5a"),
        "stopped":        ("❌ 映像の送出が止まっています", "#c0392b"),
    }
    ERROR_TEXT = {
        "virtualcam": "❌ OBS Virtual Cameraが見つかりません\nOBS Studioをインストールしてください",
        "camera": "❌ カメラが見つかりません\nWebカメラを接続してください",
        "model": "❌ 顔検出モデルを読み込めませんでした",
        "internal": "❌ 映像処理がエラーで停止しました",
    }

    def __init__(self):
        super().__init__()

        self.title("OBS Mask Cam - コントロールパネル")
        self.geometry("450x940")
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

        # いま顔が隠れているかどうかを、常に目立つ形で出す
        self.label_mask_status = ctk.CTkLabel(self.status_frame, text=self.MASK_STATUS_UI["starting"][0],
                                              font=self.font_bold, text_color="#ffffff",
                                              fg_color=self.MASK_STATUS_UI["starting"][1],
                                              corner_radius=6, height=40, wraplength=380)
        self.label_mask_status.pack(pady=(0, 8), padx=10, fill="x")
        self._shown_mask_status = "starting"
        self._provider_shown = False

        self.label_error = None

        # 推論方法・エラー・マスクの適用状態をポーリングで更新
        self._poll_status()


        # --- カメラ選択エリア ---
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(pady=10, padx=20, fill="x")
        
        self.label_camera = ctk.CTkLabel(self.camera_frame, text="映像ソース (カメラ):", font=self.font_bold)
        self.label_camera.pack(side="left", padx=10, pady=10)
        
        self.option_camera = ctk.CTkOptionMenu(self.camera_frame, values=config.camera_list, 
                                               font=self.font_main, command=self.update_camera_choice)
        current_name = config.camera_list[0] if config.camera_list else "0"
        for name, idx in config.camera_mapping.items():
            if idx == config.camera_index:
                current_name = name
                break
        self.option_camera.set(current_name)
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
        self.btn_add_mask.pack(pady=(5, 5))

        self.btn_del_mask = ctk.CTkButton(self, text="🗑️ 選択中のマスクを削除", font=self.font_main, fg_color="#7a2b2b", hover_color="#5a1818", command=self.delete_mask_file)
        self.btn_del_mask.pack(pady=(0, 5))

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

        # 検出が途切れたとき、マスクをその場に残しておく時間（長いほど素顔が出にくい）
        self.label_hold = ctk.CTkLabel(self.settings_frame, font=self.font_main,
                                       text=f"顔を見失ったときのマスク保持: {config.face_hold_seconds:.1f} 秒")
        self.label_hold.pack(pady=(5, 0))
        self.slider_hold = ctk.CTkSlider(self.settings_frame, from_=0.5, to=5.0, number_of_steps=9, command=self.update_hold)
        self.slider_hold.set(config.face_hold_seconds)
        self.slider_hold.pack(padx=20, pady=(0, 10), fill="x")


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
                # 8bit BGRA に正規化（16bit PNG などをそのまま保存しない）
                img = normalize_mask_image(imread_safe(file_path, cv2.IMREAD_UNCHANGED))
                if img is None:
                    messagebox.showerror("追加できません", "この画像は読み込めませんでした。PNG / JPG などの別の画像を選んでください。")
                    return

                # --- 画像が巨大すぎる場合は縮小する ---
                MAX_MASK_SIZE = 1000
                h, w = img.shape[:2]
                if max(h, w) > MAX_MASK_SIZE:
                    scale_ratio = MAX_MASK_SIZE / float(max(h, w))
                    new_w = int(w * scale_ratio)
                    new_h = int(h * scale_ratio)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    print(f"ℹ️ 画像が大きすぎるため、{new_w}x{new_h} に最適化して保存します。")
                
                # ファイル名を強制的に .png に変更して保存
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                new_file_name = f"{base_name}.png"
                dest_path = os.path.join(MASK_DIR, new_file_name)
                
                # 保存できたことを確かめてから、一覧の更新と選択の切り替えに進む
                if not save_mask_png(dest_path, img):
                    log.error("マスク画像を保存できませんでした: %s", dest_path)
                    messagebox.showerror(
                        "保存できませんでした",
                        f"マスク画像「{new_file_name}」を保存できませんでした。\n"
                        "保存先のフォルダに書き込めるか、空き容量があるかを確認してください。\n"
                        f"保存先: {MASK_DIR}\n\n"
                        "いま選択中のマスクは変更していません。")
                    return
                print(f"✅ 画像を透過対応PNGとして保存しました: {new_file_name}")

                config.need_reload_list = True
                time.sleep(0.1)
                new_list = get_mask_list()
                self.option_mask.configure(values=new_list)
                self.option_mask.set(new_file_name)
                self.update_mask_choice(new_file_name)
            except Exception as e:
                log.exception("マスク画像の追加に失敗しました")
                messagebox.showerror("追加できませんでした", f"マスク画像の追加中にエラーが起きました。\n{e}")

    def delete_mask_file(self):
        current = config.current_mask_name
        if not current:
            return
        if current == "icon.png":
            messagebox.showwarning("削除不可", "デフォルトのアイコン画像は削除できません。")
            return
            
        if messagebox.askyesno("削除の確認", f"マスク画像「{current}」を削除してもよろしいですか？\n削除された画像は元に戻せません。"):
            try:
                path = os.path.join(MASK_DIR, current)
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️ マスクを削除しました: {current}")
                    
                    config.need_reload_list = True
                    time.sleep(0.1)
                    new_list = get_mask_list()
                    self.option_mask.configure(values=new_list)
                    if new_list:
                        new_choice = new_list[0]
                        self.option_mask.set(new_choice)
                        self.update_mask_choice(new_choice)
                    else:
                        self.option_mask.set("")
                        self.update_mask_choice("")
            except Exception as e:
                messagebox.showerror("エラー", f"削除に失敗しました: {e}")

    def update_mask_choice(self, choice):
        config.current_mask_name = choice
        self.update_preview()
        print(f"🎭 マスク切り替え: {choice}")

    def update_camera_choice(self, choice):
        if choice in config.camera_mapping:
            config.camera_index = config.camera_mapping[choice]
            print(f"🎬 カメラ選択: {choice} (Index: {config.camera_index})")

    def toggle_mask(self):
        config.mask_enabled = not config.mask_enabled
        self.update_toggle_button_ui()
        print("🎭 マスクON" if config.mask_enabled else "🙈 マスクOFF")

    def update_toggle_button_ui(self):
        if config.mask_enabled:
            self.btn_toggle.configure(text="マスクを無効にする", fg_color="#2d8659", hover_color="#4a4a4a")
        else:
            self.btn_toggle.configure(text="マスクを有効にする", fg_color="#5a5a5a", hover_color="#236b47")

    def update_scale(self, val):
        config.scale = val
        self.label_scale.configure(text=f"マスクの大きさ: {config.scale:.1f}")

    def update_hold(self, val):
        config.face_hold_seconds = round(float(val), 1)
        self.label_hold.configure(text=f"顔を見失ったときのマスク保持: {config.face_hold_seconds:.1f} 秒")

    def _poll_status(self):
        """camera_threadの状態（推論方法・エラー・マスクの適用状態）を画面へ反映し続ける"""
        # 顔検出の方式（準備できたら一度だけ）
        if config.provider_name and not self._provider_shown:
            self._provider_shown = True
            self.label_provider.configure(text=f"✅ 顔検出: {config.provider_name}", text_color="#2d8659")

        # エラー（起動時だけでなく、動作中に止まった場合も出す）
        if config.startup_error and self.label_error is None:
            if not config.provider_name:
                self.label_provider.configure(text="❌ 起動できませんでした", text_color="#e74c3c")
            text = self.ERROR_TEXT.get(config.startup_error, self.ERROR_TEXT["internal"])
            if config.startup_error in ("model", "internal"):
                if config.error_detail:
                    text += f"\n{config.error_detail[:200]}"
                text += f"\nログ: {LOG_PATH}"
            self.label_error = ctk.CTkLabel(self.status_frame, text=text,
                font=self.font_small, text_color="#e74c3c", wraplength=380)
            self.label_error.pack(pady=5)

        # マスクの適用状態
        status = config.mask_status
        if status != self._shown_mask_status:
            self._shown_mask_status = status
            text, color = self.MASK_STATUS_UI.get(status, self.MASK_STATUS_UI["stopped"])
            self.label_mask_status.configure(text=text, fg_color=color)

        self.after(300, self._poll_status)

    # 終了時に、カメラ処理スレッドがカメラを閉じ終わるまで待つ最長秒数
    CLOSE_WAIT_SECONDS = 5.0

    def on_closing(self):
        """カメラを閉じてから終了する。閉じずに終わると、次に使うときカメラが反応しないことがある。"""
        if not config.running:
            return  # 終了処理中に、もう一度押された
        config.running = False
        self.withdraw()  # 待っているあいだ、ウィンドウは先に消す
        self._close_deadline = time.time() + self.CLOSE_WAIT_SECONDS
        self._wait_camera_closed()

    def _wait_camera_closed(self):
        worker = camera_worker
        if worker is not None and worker.is_alive() and time.time() < self._close_deadline:
            self.after(100, self._wait_camera_closed)
            return
        if worker is not None and worker.is_alive():
            log.warning("カメラの終了処理が %s 秒で終わらなかったため、そのまま終了します", self.CLOSE_WAIT_SECONDS)
        self.destroy()

# カメラ処理スレッド（終了時に、カメラを閉じ終わるのを待つために参照する）
camera_worker = None

if __name__ == "__main__":
    # カメラ処理を別スレッドで開始
    camera_worker = threading.Thread(target=camera_thread, daemon=True)
    camera_worker.start()

    # GUIを開始
    app = ControlApp()
    app.mainloop()
