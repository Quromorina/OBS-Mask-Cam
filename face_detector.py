"""YuNet（OpenCV 同梱の cv2.FaceDetectorYN）による顔検出。

モデルは OpenCV Zoo の face_detection_yunet_2023mar.onnx（MIT License）。
推論は OpenCV の DNN モジュール（CPU）で行うため、ONNX Runtime / DirectML は使わない。

検出に失敗したとき（モデルが壊れている、出力の形が想定と違うなど）は、
空の結果を返さずに例外を送出する。呼び出し側（main.py）はそれを受けて
画面全体をぼかす。「検出できなかった」と「顔がいない」を取り違えて素顔を流さないため。
"""
import cv2
import numpy as np

MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

# 検出にかける画像の長辺（px）。1280x720 のフレームは 640x360 に縮小して検出する。
# YuNet が検出できる顔はおよそ 10〜300px 四方なので、縮小したほうが
# カメラに近い大きな顔を拾いやすく、CPU でも十分に速い。
DETECT_MAX_SIDE = 640
# これ未満の信頼度の検出は捨てる。顔を隠す用途なので、見逃しより誤検出を許容して低めにしている
# （OpenCV Zoo のラッパー yunet.py の既定値と同じ 0.6）。
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000

# FaceDetectorYN の出力 1 行の列数: x, y, w, h, ランドマーク 5 点 (x, y) x 5, score
_FACE_COLUMNS = 15


def detection_size(frame_w, frame_h, max_side=DETECT_MAX_SIDE):
    """検出にかける画像サイズ (幅, 高さ) を返す。縦横比は保ち、拡大はしない。"""
    frame_w, frame_h = int(frame_w), int(frame_h)
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"フレームサイズが不正です: {frame_w}x{frame_h}")
    scale = min(1.0, max_side / max(frame_w, frame_h))
    return max(1, int(round(frame_w * scale))), max(1, int(round(frame_h * scale)))


def parse_faces(raw, scale_x, scale_y, frame_w, frame_h):
    """FaceDetectorYN.detect() の出力を、元フレームの座標に直して返す。

    戻り値は (boxes, landmarks, scores)。
      boxes     : (N, 4) int   [x1, y1, x2, y2]（フレーム内にクリップ済み）
      landmarks : (N, 5, 2) float32  右目・左目・鼻・口の右端・口の左端の (x, y)
      scores    : (N,) float32  信頼度
    顔が無いとき（raw が None または 0 行）は N = 0 の配列を返す。
    出力の形や値が想定外のときは ValueError（＝検出失敗として扱わせる）。
    """
    empty = (np.zeros((0, 4), dtype=int),
             np.zeros((0, 5, 2), dtype=np.float32),
             np.zeros((0,), dtype=np.float32))
    if raw is None:
        return empty

    faces = np.asarray(raw, dtype=np.float32)
    if faces.size == 0:
        return empty
    if faces.ndim != 2 or faces.shape[1] != _FACE_COLUMNS:
        raise ValueError(f"顔検出の出力の形が想定外です: {faces.shape}")
    if not np.isfinite(faces).all():
        raise ValueError("顔検出の出力に NaN / inf が含まれています")

    x1 = faces[:, 0] * scale_x
    y1 = faces[:, 1] * scale_y
    x2 = (faces[:, 0] + faces[:, 2]) * scale_x
    y2 = (faces[:, 1] + faces[:, 3]) * scale_y
    boxes = np.stack([
        np.clip(x1, 0, frame_w),
        np.clip(y1, 0, frame_h),
        np.clip(x2, 0, frame_w),
        np.clip(y2, 0, frame_h),
    ], axis=1).astype(int)

    landmarks = faces[:, 4:14].reshape(-1, 5, 2) * np.array([scale_x, scale_y], dtype=np.float32)
    scores = faces[:, 14]

    # 画面の外に完全に出ている検出（クリップ後に面積が無い）は隠す対象が無いので捨てる
    visible = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[visible], landmarks[visible].astype(np.float32), scores[visible]


def _create_detector(model_path, input_size):
    """FaceDetectorYN を作る。

    まず通常どおりファイルパスで読み込む。Windows の OpenCV は日本語などを含むパスを
    開けないことがあるので、失敗したらファイルの中身をメモリに読んで渡す方法で再試行する。
    """
    try:
        return cv2.FaceDetectorYN.create(
            model_path, "", input_size, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
    except Exception as path_error:
        try:
            model_bytes = np.fromfile(model_path, dtype=np.uint8)
            if model_bytes.size == 0:
                raise ValueError("モデルファイルが空です")
            return cv2.FaceDetectorYN.create(
                "onnx", model_bytes, np.empty(0, dtype=np.uint8),
                input_size, SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
        except Exception as buffer_error:
            raise RuntimeError(
                f"YuNet モデルを読み込めません: {model_path} "
                f"(パス指定: {path_error} / メモリ読み込み: {buffer_error})") from buffer_error


class FaceDetector:
    """YuNet で顔を検出する。カメラ処理スレッドからのみ呼ぶこと（スレッドセーフではない）。"""

    def __init__(self, model_path):
        self._input_size = (DETECT_MAX_SIDE, DETECT_MAX_SIDE)
        self._detector = _create_detector(model_path, self._input_size)
        # モデルや OpenCV の版が合わない場合に、配信が始まる前（起動時）に気づけるよう一度流しておく
        self.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    def detect(self, frame):
        """BGR フレームから顔を検出し、(boxes, landmarks, scores) を返す（parse_faces を参照）。"""
        frame_h, frame_w = frame.shape[:2]
        size = detection_size(frame_w, frame_h)
        if size != self._input_size:
            self._detector.setInputSize(size)
            self._input_size = size

        if size == (frame_w, frame_h):
            small = frame
        else:
            small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

        _, raw = self._detector.detect(small)
        return parse_faces(raw, frame_w / size[0], frame_h / size[1], frame_w, frame_h)
