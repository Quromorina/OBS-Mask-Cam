"""face_detector の単体テスト（GUI・カメラ不要）。

実行: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from face_detector import (  # noqa: E402
    MODEL_FILENAME,
    FaceDetector,
    detection_size,
    parse_faces,
)

MODEL_PATH = os.path.join(ROOT, MODEL_FILENAME)


def yunet_row(x, y, w, h, score=0.9):
    """FaceDetectorYN の出力 1 行（15 列）を作る。ランドマークは箱の中心に置く。"""
    cx, cy = x + w / 2, y + h / 2
    return [x, y, w, h] + [cx, cy] * 5 + [score]


class DetectionSizeTest(unittest.TestCase):
    def test_hd_frame_is_halved(self):
        self.assertEqual(detection_size(1280, 720), (640, 360))

    def test_small_frame_is_not_enlarged(self):
        self.assertEqual(detection_size(320, 240), (320, 240))

    def test_portrait_frame_keeps_aspect(self):
        self.assertEqual(detection_size(720, 1280), (360, 640))

    def test_invalid_size_raises(self):
        with self.assertRaises(ValueError):
            detection_size(0, 720)


class ParseFacesTest(unittest.TestCase):
    def test_no_faces(self):
        for raw in (None, np.zeros((0, 15), np.float32)):
            boxes, landmarks, scores = parse_faces(raw, 2.0, 2.0, 1280, 720)
            self.assertEqual(boxes.shape, (0, 4))
            self.assertEqual(landmarks.shape, (0, 5, 2))
            self.assertEqual(scores.shape, (0,))

    def test_box_is_scaled_back_to_frame_coordinates(self):
        raw = np.array([yunet_row(100, 50, 80, 100, score=0.95)], np.float32)
        boxes, landmarks, scores = parse_faces(raw, 2.0, 2.0, 1280, 720)
        self.assertEqual(boxes.tolist(), [[200, 100, 360, 300]])
        self.assertEqual(landmarks.shape, (1, 5, 2))
        self.assertTrue(np.allclose(landmarks[0], [280, 200]))
        self.assertAlmostEqual(float(scores[0]), 0.95, places=5)

    def test_box_is_clipped_to_frame(self):
        raw = np.array([yunet_row(-20, -10, 100, 100), yunet_row(600, 300, 100, 100)], np.float32)
        boxes, _, _ = parse_faces(raw, 2.0, 2.0, 1280, 720)
        self.assertEqual(boxes.tolist(), [[0, 0, 160, 180], [1200, 600, 1280, 720]])

    def test_box_fully_outside_frame_is_dropped(self):
        raw = np.array([yunet_row(700, 100, 50, 50), yunet_row(100, 100, 50, 50)], np.float32)
        boxes, landmarks, scores = parse_faces(raw, 2.0, 2.0, 1280, 720)
        self.assertEqual(boxes.tolist(), [[200, 200, 300, 300]])
        self.assertEqual(landmarks.shape, (1, 5, 2))
        self.assertEqual(scores.shape, (1,))

    def test_unexpected_output_raises_instead_of_reporting_no_faces(self):
        # 「検出できなかった」を「顔がいない」と取り違えると素顔が流れるので、必ず例外にする
        with self.assertRaises(ValueError):
            parse_faces(np.zeros((1, 5), np.float32), 1.0, 1.0, 1280, 720)
        with self.assertRaises(ValueError):
            parse_faces(np.zeros((15,), np.float32), 1.0, 1.0, 1280, 720)
        bad = np.array([yunet_row(100, 100, 50, 50)], np.float32)
        bad[0, 2] = np.nan
        with self.assertRaises(ValueError):
            parse_faces(bad, 1.0, 1.0, 1280, 720)


@unittest.skipUnless(os.path.exists(MODEL_PATH) and hasattr(cv2, "FaceDetectorYN"),
                     "YuNet モデルまたは cv2.FaceDetectorYN が無い環境")
class FaceDetectorModelTest(unittest.TestCase):
    """同梱モデルを実際に読み込んで推論が通ること（顔画像は使わない）。"""

    def test_blank_frames_of_various_sizes_have_no_faces(self):
        detector = FaceDetector(MODEL_PATH)
        for h, w in [(720, 1280), (480, 640), (1080, 1920), (240, 320)]:
            boxes, landmarks, scores = detector.detect(np.zeros((h, w, 3), np.uint8))
            self.assertEqual(boxes.shape, (0, 4), msg=f"{w}x{h}")
            self.assertEqual(landmarks.shape, (0, 5, 2))
            self.assertEqual(scores.shape, (0,))

    def test_missing_model_raises(self):
        with self.assertRaises(Exception):
            FaceDetector(os.path.join(ROOT, "no_such_model.onnx"))


if __name__ == "__main__":
    unittest.main()
