import cv2
import numpy as np
try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat
    HAS_PYVIRTUALCAM = True
except ImportError:
    pyvirtualcam = None
    PixelFormat = None
    HAS_PYVIRTUALCAM = False
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
try:
    import customtkinter as ctk
    HAS_CUSTOMTKINTER = True
except ImportError:
    ctk = None
    HAS_CUSTOMTKINTER = False
import shutil
from tkinter import filedialog, messagebox
from PIL import Image
try:
    import supervision as sv
    from trackers import ByteTrackTracker
    HAS_BYTETRACK = True
except ImportError:
    HAS_BYTETRACK = False
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

def ascii_model_path(path):
    """OpenCV DNNが日本語パスを読めない環境向けに、AppData側へモデルをキャッシュする"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass

    cache_dir = os.path.join(USER_DATA_DIR, "models")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, os.path.basename(path))
    if not os.path.exists(cached) or os.path.getsize(cached) != os.path.getsize(path):
        shutil.copy2(path, cached)
    return cached

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
MODEL_DIR = os.path.join(BUNDLE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(BUNDLE_DIR, "yolov8n-face.onnx")
YUNET_MODEL_PATH = os.path.join(MODEL_DIR, "face_detection_yunet_2026may.onnx")
SCRFD_MODEL_PATH = os.path.join(MODEL_DIR, "scrfd_2.5g_kps.onnx")
MODEL_PATH = YOLO_MODEL_PATH

# 共有データ（ユーザーごとのAppData\Roamingに保存）
USER_DATA_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "OBSMaskCam")
MASK_DIR = os.path.join(USER_DATA_DIR, "masks")
APP_MASK_DIR = os.path.join(APP_DIR, "masks")

print(f"📁 APP_DIR: {APP_DIR}")
print(f"📁 BUNDLE_DIR: {BUNDLE_DIR}")
print(f"📁 MODEL_PATH: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
print(f"📁 YUNET_MODEL_PATH: {YUNET_MODEL_PATH} (exists: {os.path.exists(YUNET_MODEL_PATH)})")
print(f"📁 SCRFD_MODEL_PATH: {SCRFD_MODEL_PATH} (exists: {os.path.exists(SCRFD_MODEL_PATH)})")
print(f"📁 MASK_DIR: {MASK_DIR}")

# --- 設定（初期値） ---
class AppConfig:
    width, height, fps = 1280, 720, 30
    scale = 1.8
    mask_enabled = True
    infer_interval = 0.01  # 固定値: 10ms
    detector_backend = os.environ.get("OBS_MASK_CAM_DETECTOR", "scrfd").lower()
    model_input_size = 640
    conf_threshold = 0.35
    iou_threshold = 0.45
    max_candidates = 300
    max_faces = 20
    tile_detection = True
    yunet_max_side = 960
    use_external_tracker = True
    smooth_frames = 3
    distance_threshold = 50
    track_hold_seconds = 0.12
    next_face_id = 1
    running = True
    current_mask_name = ""
    mask_files = []
    need_reload_list = False
    camera_index = 0
    camera_list = []
    camera_mapping = {}          # カメラ名 -> システムIndexの対応
    provider_name = ""           # GPU/CPU表示用
    startup_error = ""           # 起動時エラーメッセージ

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
    """YOLOv8のletterbox前処理。中央パディングで遠景・端の顔の座標ズレを抑える"""
    h, w = frame.shape[:2]
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    # パディングして正方形に
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # BGR→RGB, HWC→CHW, 正規化
    blob = padded[:, :, ::-1].transpose(2, 0, 1)
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, scale, pad_x, pad_y

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

def postprocess(output, scale, pad_x, pad_y, orig_w, orig_h,
                conf_threshold=0.35, iou_threshold=0.45,
                max_candidates=300, max_faces=20, return_scores=False):
    """YOLOv8出力を処理して顔バウンディングボックスを返す"""
    # output shape: (1, 5, 8400) → transpose → (8400, 5) : [cx, cy, w, h, conf]
    predictions = output[0].transpose()
    
    # 信頼度フィルタ
    scores = predictions[:, 4]
    mask = scores > conf_threshold
    predictions = predictions[mask]
    scores = scores[mask]
    
    if len(predictions) == 0:
        if return_scores:
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=np.float32)
        return []

    if len(predictions) > max_candidates:
        top_idx = np.argpartition(scores, -max_candidates)[-max_candidates:]
        predictions = predictions[top_idx]
        scores = scores[top_idx]
    
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
    scores = scores[keep]

    if len(boxes) > max_faces:
        top_faces = scores.argsort()[::-1][:max_faces]
        boxes = boxes[top_faces]
        scores = scores[top_faces]
    
    # スケールを元画像サイズに戻す
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    
    # クリッピング
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
    
    boxes = boxes.astype(int)
    if return_scores:
        return boxes, scores.astype(np.float32)
    return boxes

def detect_faces(session, input_name, frame, use_tiling=False, return_scores=False):
    """全体検出に2x2タイル検出を追加し、遠くの小さい顔も拾いやすくする"""
    regions = [(0, 0, frame.shape[1], frame.shape[0])]
    if use_tiling:
        h, w = frame.shape[:2]
        overlap_x = w // 12
        overlap_y = h // 12
        mid_x = w // 2
        mid_y = h // 2
        regions.extend([
            (0, 0, min(w, mid_x + overlap_x), min(h, mid_y + overlap_y)),
            (max(0, mid_x - overlap_x), 0, w, min(h, mid_y + overlap_y)),
            (0, max(0, mid_y - overlap_y), min(w, mid_x + overlap_x), h),
            (max(0, mid_x - overlap_x), max(0, mid_y - overlap_y), w, h),
        ])

    all_boxes = []
    all_scores = []
    for x1, y1, x2, y2 in regions:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        blob, scale, pad_x, pad_y = preprocess(crop, config.model_input_size)
        output = session.run(None, {input_name: blob})[0]
        boxes, scores = postprocess(
            output, scale, pad_x, pad_y, crop.shape[1], crop.shape[0],
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            max_candidates=config.max_candidates,
            max_faces=config.max_faces,
            return_scores=True,
        )
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] += x1
        boxes[:, [1, 3]] += y1
        all_boxes.append(boxes)
        all_scores.append(scores)

    if not all_boxes:
        if return_scores:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return []

    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    keep = nms(boxes.astype(np.float32), scores, config.iou_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    if len(boxes) > config.max_faces:
        top_faces = scores.argsort()[::-1][:config.max_faces]
        boxes = boxes[top_faces]
        scores = scores[top_faces]
    if return_scores:
        return boxes.astype(np.float32), scores.astype(np.float32)
    return boxes.astype(int)

class YoloFaceDetector:
    name = "YOLOv8n-face"

    def __init__(self, model_path):
        self.session = create_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.use_tiling = config.tile_detection and "Dml" in config.provider_name
        if self.use_tiling:
            print("✅ YOLOタイル検出ON: 遠くの小さい顔も検出しやすくします")

    def detect(self, frame):
        return detect_faces(self.session, self.input_name, frame, self.use_tiling, return_scores=True)

class YuNetFaceDetector:
    name = "YuNet"

    def __init__(self, model_path):
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("このOpenCVにはFaceDetectorYNがありません")
        model_path = ascii_model_path(model_path)
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (config.width, config.height),
            score_threshold=float(config.conf_threshold),
            nms_threshold=float(config.iou_threshold),
            top_k=max(5000, config.max_candidates),
        )
        self.input_size = None
        config.provider_name = "OpenCV YuNet"

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, float(config.yunet_max_side) / max(w, h))
        if scale < 1.0:
            det_w, det_h = int(w * scale), int(h * scale)
            det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            det_w, det_h = w, h
            det_frame = frame

        if self.input_size != (det_w, det_h):
            self.detector.setInputSize((det_w, det_h))
            self.input_size = (det_w, det_h)
        _, faces = self.detector.detect(det_frame)
        if faces is None or len(faces) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        faces = faces[:config.max_faces]
        boxes = np.empty((len(faces), 4), dtype=np.float32)
        boxes[:, 0] = faces[:, 0] / scale
        boxes[:, 1] = faces[:, 1] / scale
        boxes[:, 2] = (faces[:, 0] + faces[:, 2]) / scale
        boxes[:, 3] = (faces[:, 1] + faces[:, 3]) / scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        scores = faces[:, -1].astype(np.float32)
        return boxes, scores

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

class SCRFDFaceDetector:
    name = "SCRFD_2.5G_KPS"

    def __init__(self, model_path):
        self.session = create_onnx_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.fmc = 3
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.input_size = (config.model_input_size, config.model_input_size)
        self.input_mean = 127.5
        self.input_std = 128.0
        self.center_cache = {}

    def _anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key not in self.center_cache:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
            if len(self.center_cache) < 100:
                self.center_cache[key] = centers
        return self.center_cache[key]

    def detect(self, frame):
        input_w, input_h = self.input_size
        im_h, im_w = frame.shape[:2]
        im_ratio = float(im_h) / im_w
        model_ratio = float(input_h) / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = float(new_h) / im_h
        resized = cv2.resize(frame, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            (input_w, input_h),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        scores_all = []
        boxes_all = []

        for idx, stride in enumerate(self.strides):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            if scores.ndim == 3:
                scores = scores[0]
                bbox_preds = bbox_preds[0]
            feat_h = input_h // stride
            feat_w = input_w // stride
            anchor_centers = self._anchor_centers(feat_h, feat_w, stride)
            pos_inds = np.where(scores.ravel() >= config.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue
            boxes = distance2bbox(anchor_centers, bbox_preds)
            scores_all.append(scores.ravel()[pos_inds])
            boxes_all.append(boxes[pos_inds])

        if not boxes_all:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = np.vstack(boxes_all) / det_scale
        scores = np.concatenate(scores_all).astype(np.float32)
        order = scores.argsort()[::-1]
        if len(order) > config.max_candidates:
            order = order[:config.max_candidates]
        boxes = boxes[order]
        scores = scores[order]

        keep = nms(boxes.astype(np.float32), scores, config.iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        if len(boxes) > config.max_faces:
            top_faces = scores.argsort()[::-1][:config.max_faces]
            boxes = boxes[top_faces]
            scores = scores[top_faces]

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, im_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, im_h)
        return boxes.astype(np.float32), scores.astype(np.float32)

def create_face_detector():
    requested = config.detector_backend
    candidates = {
        "scrfd": [(SCRFDFaceDetector, SCRFD_MODEL_PATH), (YuNetFaceDetector, YUNET_MODEL_PATH), (YoloFaceDetector, YOLO_MODEL_PATH)],
        "yunet": [(YuNetFaceDetector, YUNET_MODEL_PATH), (SCRFDFaceDetector, SCRFD_MODEL_PATH), (YoloFaceDetector, YOLO_MODEL_PATH)],
        "yolo": [(YoloFaceDetector, YOLO_MODEL_PATH), (SCRFDFaceDetector, SCRFD_MODEL_PATH), (YuNetFaceDetector, YUNET_MODEL_PATH)],
    }.get(requested, [])
    if not candidates:
        candidates = [(SCRFDFaceDetector, SCRFD_MODEL_PATH), (YuNetFaceDetector, YUNET_MODEL_PATH), (YoloFaceDetector, YOLO_MODEL_PATH)]

    errors = []
    for detector_cls, model_path in candidates:
        if not os.path.exists(model_path):
            errors.append(f"{detector_cls.name}: model not found: {model_path}")
            continue
        try:
            detector = detector_cls(model_path)
            config.detector_backend = detector.name
            print(f"✅ Face detector: {detector.name}")
            return detector
        except Exception as e:
            errors.append(f"{detector_cls.name}: {e}")

    raise RuntimeError("顔検出モデルを初期化できません: " + " / ".join(errors))

def update_face_tracks(current_faces, face_history, now):
    """最大20人程度を想定した軽量トラッキング。全候補ペアを評価してID入れ替わりを抑える"""
    candidates = []
    for face_idx, (cx, cy, fs) in enumerate(current_faces):
        for fid, data in face_history.items():
            prev_cx, prev_cy, prev_fs = data["history"][-1]
            dist = float(np.hypot(cx - prev_cx, cy - prev_cy))
            size_diff = abs(fs - prev_fs) / max(fs, prev_fs, 1)
            gate = max(config.distance_threshold, min(220, max(fs, prev_fs) * 0.85))
            if dist <= gate and size_diff <= 0.65:
                score = (dist / gate) + size_diff * 0.35
                candidates.append((score, face_idx, fid))

    candidates.sort(key=lambda item: item[0])
    matched_faces = set()
    matched_tracks = set()
    new_face_history = {}

    for _, face_idx, fid in candidates:
        if face_idx in matched_faces or fid in matched_tracks:
            continue
        cx, cy, fs = current_faces[face_idx]
        history = face_history[fid]["history"]
        history.append((cx, cy, fs))
        if len(history) > 20:
            history.pop(0)
        new_face_history[fid] = {"history": history, "last_update": now}
        matched_faces.add(face_idx)
        matched_tracks.add(fid)

    for face_idx, (cx, cy, fs) in enumerate(current_faces):
        if face_idx in matched_faces:
            continue
        fid = config.next_face_id
        config.next_face_id += 1
        new_face_history[fid] = {"history": [(cx, cy, fs)], "last_update": now}

    for fid, data in face_history.items():
        if fid in matched_tracks:
            continue
        if now - data["last_update"] > config.track_hold_seconds:
            continue
        ghost_cx, ghost_cy, ghost_fs = data["history"][-1]
        near_current = any(
            np.hypot(cx - ghost_cx, cy - ghost_cy) < max(config.distance_threshold, ghost_fs * 0.9)
            for (cx, cy, fs) in current_faces
        )
        if not near_current:
            new_face_history[fid] = data

    return new_face_history

def create_external_tracker():
    if not (config.use_external_tracker and HAS_BYTETRACK):
        return None
    try:
        return ByteTrackTracker(
            lost_track_buffer=20,
            frame_rate=float(config.fps),
            track_activation_threshold=max(0.25, config.conf_threshold),
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.1,
            high_conf_det_threshold=max(0.45, config.conf_threshold + 0.1),
        )
    except Exception as e:
        print(f"⚠ ByteTrack初期化失敗。内蔵trackerへフォールバックします: {e}")
        return None

def update_face_tracks_with_bytetrack(boxes, scores, tracker, face_history, now):
    if tracker is None:
        current_faces = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            face_size = int(max(x2 - x1, y2 - y1) * config.scale)
            current_faces.append((cx, cy, face_size))
        return update_face_tracks(current_faces, face_history, now)

    if len(boxes) == 0:
        detections = sv.Detections.empty()
    else:
        detections = sv.Detections(
            xyxy=boxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )

    tracked = tracker.update(detections)
    track_ids = tracked.tracker_id
    if track_ids is None or len(track_ids) == 0:
        return {
            fid: data for fid, data in face_history.items()
            if now - data["last_update"] <= config.track_hold_seconds
        }

    new_face_history = {}
    for box, fid in zip(tracked.xyxy, track_ids):
        fid = int(fid)
        if fid < 0:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        face_size = int(max(x2 - x1, y2 - y1) * config.scale)
        history = face_history.get(fid, {"history": []})["history"]
        history.append((cx, cy, face_size))
        if len(history) > 20:
            history.pop(0)
        new_face_history[fid] = {"history": history, "last_update": now}

    for fid, data in face_history.items():
        if fid in new_face_history or now - data["last_update"] > config.track_hold_seconds:
            continue
        ghost_cx, ghost_cy, ghost_fs = data["history"][-1]
        near_fresh_track = any(
            np.hypot(fresh["history"][-1][0] - ghost_cx, fresh["history"][-1][1] - ghost_cy)
            < max(config.distance_threshold, ghost_fs * 0.9)
            for fresh in new_face_history.values()
        )
        if not near_fresh_track:
            new_face_history[fid] = data

    return new_face_history

def smoothed_mask_pose(history):
    if not history:
        return None

    current_history = history[-config.smooth_frames:]
    latest_cx, latest_cy, latest_fs = current_history[-1]

    if len(current_history) >= 2:
        prev_cx, prev_cy, prev_fs = current_history[-2]
        first_cx, first_cy, _ = current_history[0]
        face_ref = max(latest_fs, 1)
        frame_move = float(np.hypot(latest_cx - prev_cx, latest_cy - prev_cy)) / face_ref
        span_move = float(np.hypot(latest_cx - first_cx, latest_cy - first_cy)) / face_ref

        if frame_move > 0.10 or span_move > 0.18:
            xy_hist = current_history[-1:]
        elif frame_move > 0.05:
            xy_hist = current_history[-2:]
        else:
            xy_hist = current_history
    else:
        xy_hist = current_history

    if len(xy_hist) == 1:
        avg_cx, avg_cy = int(xy_hist[-1][0]), int(xy_hist[-1][1])
    else:
        weights = np.linspace(0.35, 1.0, len(xy_hist))
        avg_cx = int(np.average([h[0] for h in xy_hist], weights=weights))
        avg_cy = int(np.average([h[1] for h in xy_hist], weights=weights))

    sizes = [h[2] for h in current_history]
    latest_fs = sizes[-1]
    if len(sizes) >= 2:
        size_change_ratio = abs(sizes[-1] - sizes[-2]) / max(sizes[-2], 1)
        if size_change_ratio > 0.04:
            avg_fs = int(sizes[-1] * 0.75 + sizes[-2] * 0.25)
        else:
            weights = np.linspace(0.5, 1.5, len(sizes))
            avg_fs = int(np.average(sizes, weights=weights))
    else:
        avg_fs = latest_fs

    return avg_cx, avg_cy, max(1, avg_fs)

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
    resized_mask_cache = {}

    # 顔検出器作成
    detector = create_face_detector()
    external_tracker = create_external_tracker()
    if external_tracker is not None:
        print("✅ ByteTrack tracker ON: 複数人のID維持を強化します")
    else:
        print("ℹ️ 内蔵trackerを使用します")

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
                
                # スムーズな切り替えのため、一旦ダミー画面を出力
                frame = np.zeros((config.height, config.width, 3), dtype=np.uint8)
                cv2.putText(frame, "Switching Camera...", (config.width//4, config.height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                cam.send(frame)
                cam.sleep_until_next_frame()
                
                cap.release()
                time.sleep(0.5) # リソース解放を待機
                
                cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                last_camera_index = config.camera_index
                continue

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
                    resized_mask_cache.clear()

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
                boxes, scores = detector.detect(frame)
                face_history = update_face_tracks_with_bytetrack(boxes, scores, external_tracker, face_history, now)
                last_infer_time = now

            if config.mask_enabled and overlay_img is not None:
                for fid, data in face_history.items():
                    pose = smoothed_mask_pose(data["history"])
                    if pose is None:
                        continue
                    avg_cx, avg_cy, avg_fs = pose

                    try:
                        mask_size = max(1, int(round(avg_fs / 4) * 4))
                        if mask_size not in resized_mask_cache:
                            if len(resized_mask_cache) > 64:
                                resized_mask_cache.clear()
                            resized_mask_cache[mask_size] = cv2.resize(overlay_img, (mask_size, mask_size))
                        resized = resized_mask_cache[mask_size]
                        x_offset = avg_cx - mask_size // 2
                        y_offset = avg_cy - mask_size // 2
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



    def _poll_provider_status(self):
        """camera_threadのprovider情報が確定するまでポーリング"""
        if config.provider_name:
            # provider確定 → 表示更新
            if "Dml" in config.provider_name:
                self.label_provider.configure(text="✅ GPU推論 (DirectML)", text_color="#2d8659")
            elif "YuNet" in config.provider_name:
                self.label_provider.configure(text="✅ 高速検出 (YuNet/OpenCV)", text_color="#2d8659")
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
