"""マスクを出せないときでも素顔を送出しないための画像処理ヘルパー。

GUI・カメラ・ONNX Runtime に依存しない純粋な関数だけを置き、
tests/test_mask_safety.py から単体で確認できるようにしている。
"""
import os

import cv2
import numpy as np

# ぼかし（モザイク）の粗さ。マスク領域を一辺あたり何ブロックに潰すか。
# マスク領域は顔の約1.8倍なので、8ブロックなら顔は4〜5ブロック程度になり判別できない。
MOSAIC_BLOCKS = 8
# 画面全体を潰すときのブロック数（横, 縦）
FULL_FRAME_BLOCKS = (16, 9)
# モザイク処理自体が失敗したときの塗りつぶし色（BGR）
FILL_COLOR = (64, 64, 64)
# 顔を隠す領域の一辺の下限（px）。検出結果のサイズが 0〜数 px でも、
# 「何も隠さない」「1px だけ隠す」にならないよう、ここまで広げる。
MIN_COVER_SIZE = 16
# モザイク 1 ブロックの一辺の下限（px）。領域が小さいときはブロック数を減らす。
# （8px の領域を 8 ブロックに潰しても元の画素のままで、隠したことにならないため）
MIN_BLOCK_PX = 4


def normalize_mask_image(img):
    """読み込んだマスク画像を 8bit BGRA に揃える。使えない画像は None を返す。

    16bit PNG などをそのまま合成に渡すと cv2.error になり、
    以前はその例外が握りつぶされて素顔が送出されていた。
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None

    # --- ビット深度を 8bit に ---
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype == np.bool_:
        img = img.astype(np.uint8) * 255
    elif np.issubdtype(img.dtype, np.floating):
        # 浮動小数の画像は 0.0〜1.0 とみなす
        img = (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # --- チャンネルを BGRA に ---
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGRA)
        elif channels == 2:  # グレー＋アルファ
            gray, alpha = img[:, :, 0], img[:, :, 1]
            img = np.dstack([gray, gray, gray, alpha])
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif channels != 4:
            return None
    else:
        return None

    return np.ascontiguousarray(img)


def fit_frame(frame, width, height):
    """カメラのフレームを出力サイズ（width x height, 8bit BGR）に合わせる。

    カメラが要求どおりの解像度を返さない場合は、縦横比を保って縮小・拡大し、
    余白を黒で埋める。扱えないフレームは None を返す。
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
        return None

    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim != 3 or frame.shape[2] != 3:
        return None

    h, w = frame.shape[:2]
    if (w, h) == (width, height):
        return frame

    scale = min(width / w, height / h)
    new_w = max(1, min(width, int(round(w * scale))))
    new_h = max(1, min(height, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def obscure_region(frame, cx, cy, size):
    """(cx, cy) を中心とする一辺 size の正方形を、判別できない粗さのモザイクで潰す。

    size が MIN_COVER_SIZE より小さい（0 や負の値を含む）ときは MIN_COVER_SIZE に広げる。
    何もせずに返すのは、領域が完全に画面の外にあるときだけ。
    """
    cx, cy = int(cx), int(cy)
    size = max(int(size), MIN_COVER_SIZE)
    half = size // 2

    h, w = frame.shape[:2]
    x1, y1 = max(cx - half, 0), max(cy - half, 0)
    x2, y2 = min(cx - half + size, w), min(cy - half + size, h)
    if x1 >= x2 or y1 >= y2:
        return frame

    try:
        # 領域が小さい（または画面端で切れて細い）ときはブロック数を減らし、
        # 1 ブロックが MIN_BLOCK_PX 未満にならないようにする。最小は 1 ブロック（平均色で塗る）。
        blocks_x = max(1, min(MOSAIC_BLOCKS, (x2 - x1) // MIN_BLOCK_PX))
        blocks_y = max(1, min(MOSAIC_BLOCKS, (y2 - y1) // MIN_BLOCK_PX))
        small = cv2.resize(frame[y1:y2, x1:x2], (blocks_x, blocks_y),
                           interpolation=cv2.INTER_AREA)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                         interpolation=cv2.INTER_NEAREST)
    except Exception:
        # モザイクにも失敗したら単色で塗りつぶす（素顔は出さない）
        frame[y1:y2, x1:x2] = FILL_COLOR
    return frame


def obscure_frame(frame):
    """顔の位置が分からないとき用。画面全体を粗いモザイクにする。"""
    h, w = frame.shape[:2]
    try:
        small = cv2.resize(frame, FULL_FRAME_BLOCKS, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    except Exception:
        frame[:] = FILL_COLOR
        return frame


def overlay_transparent(background, overlay, x, y):
    """BGRA の overlay を background の (x, y) にアルファ合成する。"""
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


def cover_face(frame, overlay, cx, cy, size):
    """顔の位置にマスク画像を重ねる。重ねられなければモザイクで潰す。

    戻り値は (frame, マスク画像を使えたか)。False のときはモザイクで代用している。
    どの経路でも「何もせずに返す」ことはしない。
    """
    cx, cy = int(cx), int(cy)
    # 小さすぎるサイズは下限まで広げる（1px のマスクを重ねて「適用した」ことにしない）
    size = max(int(size), MIN_COVER_SIZE)
    # 8bit BGRA 以外（16bit・アルファなし等）は合成せずモザイクにする。
    # 環境によっては cv2 が例外を出さずに崩れた合成をするため、例外頼みにしない。
    usable = (
        overlay is not None
        and getattr(overlay, "dtype", None) == np.uint8
        and getattr(overlay, "ndim", 0) == 3
        and overlay.shape[2] == 4
    )
    if usable:
        try:
            resized = cv2.resize(overlay, (size, size))
            frame = overlay_transparent(frame, resized, cx - size // 2, cy - size // 2)
            return frame, True
        except Exception:
            pass
    return obscure_region(frame, cx, cy, size), False


def decide_mask_status(mask_enabled, detector_ok, mask_usable, face_count, all_masks_applied=True):
    """送出状態（画面に出す状態名）を決める。

    - "off":             マスクOFF（素顔が映る）
    - "detector_error":  顔検出が動いておらず、画面全体をぼかしている
    - "no_face":         顔が見つかっていない（隠す対象がない）。マスク画像は使える
    - "no_face_fallback": 顔が見つかっていない。マスク画像も使えない（見つかればぼかしで隠す）
    - "ok":              見つかった顔すべてにマスク画像を重ねた
    - "fallback":        見つかった顔のうち 1 つ以上を、マスク画像の代わりにぼかしで隠した

    face_count は「いま隠している顔の数」（検出が途切れて保持中のものを含む）。
    """
    if not mask_enabled:
        return "off"
    if not detector_ok:
        return "detector_error"
    if face_count <= 0:
        return "no_face" if mask_usable else "no_face_fallback"
    return "ok" if (mask_usable and all_masks_applied) else "fallback"


def save_mask_png(path, img):
    """マスク画像を PNG として path に保存する。保存できたときだけ True を返す。

    - 日本語パスでも書けるよう、cv2.imwrite ではなく imencode + tofile を使う。
    - いったん一時ファイルに書き、読み戻して確認してから置き換える。
      失敗したとき、同名の既存マスクは壊さずに残る。
    - 例外は出さない（失敗はすべて False）。
    """
    tmp_path = f"{path}.tmp"
    try:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return False
        buf.tofile(tmp_path)
        # 書けたつもりで壊れている（容量不足など）場合を、読み戻して検出する
        written = cv2.imdecode(np.fromfile(tmp_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if written is None or written.shape != img.shape:
            return False
        os.replace(tmp_path, path)
        return True
    except Exception:
        return False
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
