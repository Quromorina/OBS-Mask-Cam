import argparse
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np


def ascii_path(path, fallback_name):
    try:
        path.encode("ascii")
        return path, False
    except UnicodeEncodeError:
        cached = os.path.join(tempfile.gettempdir(), fallback_name)
        if not os.path.exists(cached) or os.path.getsize(cached) != os.path.getsize(path):
            shutil.copy2(path, cached)
        return cached, True


def writer_path(path):
    try:
        path.encode("ascii")
        return path, None
    except UnicodeEncodeError:
        temp_path = os.path.join(tempfile.gettempdir(), "obs_mask_cam_render.mp4")
        return temp_path, path


def load_mask(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"mask image not readable: {path}")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--detector", default="scrfd", choices=["scrfd", "yunet", "yolo"])
    parser.add_argument("--mask", default="")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.8)
    args = parser.parse_args()

    os.environ["OBS_MASK_CAM_DETECTOR"] = args.detector
    os.environ.setdefault("APPDATA", tempfile.gettempdir())

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    import main as app

    app.config.scale = args.scale
    detector = app.create_face_detector()
    tracker = app.create_external_tracker()
    face_history = {}

    input_path, _ = ascii_path(os.path.abspath(args.input), "obs_mask_cam_input.mp4")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"video not readable: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames > 0:
        total = min(total, args.max_frames)

    out_abs = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    write_to, final_copy = writer_path(out_abs)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(write_to, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"video writer not opened: {write_to}")

    mask_path = args.mask or os.path.join(app.MASK_DIR, app.config.current_mask_name)
    mask_img = load_mask(mask_path)
    resized_cache = {}
    start = time.perf_counter()
    frame_idx = 0

    while True:
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        boxes, scores = detector.detect(frame)
        face_history = app.update_face_tracks_with_bytetrack(boxes, scores, tracker, face_history, now)

        for data in face_history.values():
            history = data["history"][-app.config.smooth_frames:]
            avg_cx = int(np.mean([h[0] for h in history]))
            avg_cy = int(np.mean([h[1] for h in history]))
            avg_fs = int(np.mean([h[2] for h in history]))
            mask_size = max(1, int(round(avg_fs / 4) * 4))
            if mask_size not in resized_cache:
                if len(resized_cache) > 64:
                    resized_cache.clear()
                resized_cache[mask_size] = cv2.resize(mask_img, (mask_size, mask_size))
            resized = resized_cache[mask_size]
            frame = app.overlay_transparent(frame, resized, avg_cx - mask_size // 2, avg_cy - mask_size // 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 120 == 0:
            elapsed = time.perf_counter() - start
            print(f"rendered {frame_idx}/{total} frames ({frame_idx / max(elapsed, 0.001):.1f} fps)")

    cap.release()
    writer.release()
    if final_copy:
        shutil.copy2(write_to, final_copy)
    elapsed = time.perf_counter() - start
    print(f"done: {out_abs}")
    print(f"frames={frame_idx} elapsed={elapsed:.1f}s avg_fps={frame_idx / max(elapsed, 0.001):.1f}")


if __name__ == "__main__":
    main()
