"""mask_safety の単体テスト（GUI・カメラ・ONNX Runtime 不要）。

実行: python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mask_safety  # noqa: E402
from mask_safety import (  # noqa: E402
    MIN_COVER_SIZE,
    cover_face,
    decide_mask_status,
    fit_frame,
    normalize_mask_image,
    obscure_frame,
    obscure_region,
    save_mask_png,
)


def noisy_frame(height=720, width=1280):
    """モザイクで必ず変化が出るよう、乱数で埋めたフレームを作る。"""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class NormalizeMaskImageTest(unittest.TestCase):
    def assert_bgra8(self, img, height, width):
        self.assertIsNotNone(img)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (height, width, 4))

    def test_16bit_bgra_is_reduced_to_8bit(self):
        src = np.full((10, 12, 4), 65535, dtype=np.uint16)
        out = normalize_mask_image(src)
        self.assert_bgra8(out, 10, 12)
        self.assertEqual(int(out.max()), 255)

    def test_16bit_png_roundtrip(self):
        src = np.full((8, 8, 4), 40000, dtype=np.uint16)
        ok, buf = cv2.imencode(".png", src)
        self.assertTrue(ok)
        decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded.dtype, np.uint16)
        self.assert_bgra8(normalize_mask_image(decoded), 8, 8)

    def test_grayscale_and_bgr_get_opaque_alpha(self):
        gray = normalize_mask_image(np.full((5, 6), 100, dtype=np.uint8))
        self.assert_bgra8(gray, 5, 6)
        self.assertTrue((gray[:, :, 3] == 255).all())

        bgr = normalize_mask_image(np.full((5, 6, 3), 100, dtype=np.uint8))
        self.assert_bgra8(bgr, 5, 6)
        self.assertTrue((bgr[:, :, 3] == 255).all())

    def test_gray_alpha_and_16bit_gray(self):
        gray_alpha = np.dstack([np.full((4, 4), 10, np.uint8), np.full((4, 4), 200, np.uint8)])
        out = normalize_mask_image(gray_alpha)
        self.assert_bgra8(out, 4, 4)
        self.assertTrue((out[:, :, 3] == 200).all())

        self.assert_bgra8(normalize_mask_image(np.full((4, 4), 65535, np.uint16)), 4, 4)

    def test_float_image(self):
        out = normalize_mask_image(np.ones((4, 4, 3), dtype=np.float32))
        self.assert_bgra8(out, 4, 4)
        self.assertEqual(int(out[:, :, :3].max()), 255)

    def test_unusable_inputs_return_none(self):
        self.assertIsNone(normalize_mask_image(None))
        self.assertIsNone(normalize_mask_image(np.zeros((0, 0, 4), np.uint8)))
        self.assertIsNone(normalize_mask_image(np.zeros((4, 4, 5), np.uint8)))


class FitFrameTest(unittest.TestCase):
    def test_same_size_is_returned_as_is(self):
        frame = noisy_frame()
        self.assertIs(fit_frame(frame, 1280, 720), frame)

    def test_other_resolutions_are_fitted(self):
        for h, w in [(480, 640), (1080, 1920), (720, 960), (1280, 720)]:
            out = fit_frame(noisy_frame(h, w), 1280, 720)
            self.assertEqual(out.shape, (720, 1280, 3), msg=f"{w}x{h}")
            self.assertEqual(out.dtype, np.uint8)

    def test_4_3_frame_is_pillarboxed(self):
        out = fit_frame(np.full((480, 640, 3), 255, np.uint8), 1280, 720)
        # 640x480 → 960x720 になり、左右に 160px ずつ黒帯
        self.assertTrue((out[:, :160] == 0).all())
        self.assertTrue((out[:, 1120:] == 0).all())
        self.assertTrue((out[:, 160:1120] == 255).all())

    def test_gray_and_bgra_frames(self):
        self.assertEqual(fit_frame(np.zeros((720, 1280), np.uint8), 1280, 720).shape, (720, 1280, 3))
        self.assertEqual(fit_frame(np.zeros((720, 1280, 4), np.uint8), 1280, 720).shape, (720, 1280, 3))

    def test_unusable_frames_return_none(self):
        self.assertIsNone(fit_frame(None, 1280, 720))
        self.assertIsNone(fit_frame(np.zeros((0, 0, 3), np.uint8), 1280, 720))
        self.assertIsNone(fit_frame(np.zeros((720, 1280, 3), np.float32), 1280, 720))


class ObscureTest(unittest.TestCase):
    def test_region_is_changed_and_outside_untouched(self):
        original = noisy_frame()
        frame = obscure_region(original.copy(), 640, 360, 200)
        self.assertFalse(np.array_equal(frame[260:460, 540:740], original[260:460, 540:740]))
        self.assertTrue(np.array_equal(frame[:260], original[:260]))
        self.assertTrue(np.array_equal(frame[:, :540], original[:, :540]))

    def test_region_partly_outside_frame(self):
        original = noisy_frame()
        frame = obscure_region(original.copy(), 0, 0, 200)
        self.assertFalse(np.array_equal(frame[:100, :100], original[:100, :100]))
        obscure_region(original.copy(), 5000, 5000, 200)  # 完全に画面外でも例外にならない
        obscure_region(original.copy(), 640, 360, 0)

    def test_tiny_sizes_still_obscure_pixels(self):
        # size が 0〜1（や負の値）でも、何もせず返さない。下限の広さまで潰す。
        half = MIN_COVER_SIZE // 2
        for size in (-5, 0, 1, 2, 3, 7, MIN_COVER_SIZE):
            original = noisy_frame()
            frame = obscure_region(original.copy(), 640, 360, size)
            region = (slice(360 - half, 360 + half), slice(640 - half, 640 + half))
            self.assertFalse(np.array_equal(frame[region], original[region]), msg=f"size={size}")
            # 元の画素がそのまま残っていない（全画素が元と違う色になっている）
            unchanged = (frame[region] == original[region]).all(axis=2)
            self.assertLess(unchanged.mean(), 0.05, msg=f"size={size}")
            # 領域の外は触らない
            self.assertTrue(np.array_equal(frame[:360 - half], original[:360 - half]))
            self.assertTrue(np.array_equal(frame[:, :640 - half], original[:, :640 - half]))
            self.assertTrue(np.array_equal(frame[360 + half:], original[360 + half:]))
            self.assertTrue(np.array_equal(frame[:, 640 + half:], original[:, 640 + half:]))

    def test_small_region_uses_coarse_blocks(self):
        # 小さい領域を 8 ブロックに潰すと元の画素のままになるので、ブロックを粗くする
        original = noisy_frame()
        frame = obscure_region(original.copy(), 640, 360, 8)  # 下限の 16px に広がる
        region = frame[352:368, 632:648]
        colors = np.unique(region.reshape(-1, 3), axis=0)
        self.assertLessEqual(len(colors), 16)  # 16px なら 4x4 ブロック以下

    def test_thin_region_at_frame_edge_is_obscured(self):
        # 画面端で 1px 幅に切れた領域も、元の画素のままにしない（平均色で塗る）
        original = noisy_frame()
        frame = obscure_region(original.copy(), -7, 360, 16)  # x は 0 の 1 列だけ残る
        self.assertFalse(np.array_equal(frame[352:368, 0:1], original[352:368, 0:1]))
        self.assertEqual(len(np.unique(frame[352:368, 0].reshape(-1, 3), axis=0)), 4)

    def test_odd_size_covers_full_width(self):
        original = noisy_frame()
        frame = obscure_region(original.copy(), 640, 360, 201)
        # 一辺 201 なら x は 540〜740（741 の手前）まで。右端の列も潰れている
        self.assertFalse(np.array_equal(frame[260:461, 740], original[260:461, 740]))
        self.assertTrue(np.array_equal(frame[:, 741:], original[:, 741:]))

    def test_mosaic_failure_falls_back_to_fill(self):
        original = noisy_frame()
        with mock.patch.object(mask_safety.cv2, "resize", side_effect=cv2.error("boom")):
            frame = obscure_region(original.copy(), 640, 360, 1)
        half = MIN_COVER_SIZE // 2
        self.assertTrue((frame[360 - half:360 + half, 640 - half:640 + half]
                         == mask_safety.FILL_COLOR).all())

    def test_obscure_frame_changes_everything(self):
        original = noisy_frame()
        out = obscure_frame(original.copy())
        self.assertEqual(out.shape, original.shape)
        self.assertFalse(np.array_equal(out, original))
        # 16x9 ブロックのモザイクなので、1ブロック（80x80）の中は単色になる
        block = out[:80, :80].reshape(-1, 3)
        self.assertTrue((block == block[0]).all())


class CoverFaceTest(unittest.TestCase):
    """マスクがONのとき、どの経路でも顔の領域が素通しにならないこと。"""

    def assert_face_hidden(self, overlay):
        original = noisy_frame()
        frame, used_mask = cover_face(original.copy(), overlay, 640, 360, 200)
        self.assertFalse(np.array_equal(frame[260:460, 540:740], original[260:460, 540:740]))
        return frame, used_mask

    def test_valid_mask_is_overlaid(self):
        overlay = np.zeros((50, 50, 4), np.uint8)
        overlay[:, :] = (0, 0, 255, 255)  # 不透明な赤
        frame, used_mask = self.assert_face_hidden(overlay)
        self.assertTrue(used_mask)
        self.assertTrue((frame[260:460, 540:740] == (0, 0, 255)).all())

    def test_missing_mask_falls_back_to_mosaic(self):
        _, used_mask = self.assert_face_hidden(None)
        self.assertFalse(used_mask)

    def test_16bit_mask_falls_back_to_mosaic(self):
        # 正規化を通さずに 16bit のまま来ても、素通しにせずモザイクにする
        _, used_mask = self.assert_face_hidden(np.full((50, 50, 4), 65535, np.uint16))
        self.assertFalse(used_mask)

    def test_mask_without_alpha_falls_back_to_mosaic(self):
        _, used_mask = self.assert_face_hidden(np.full((50, 50, 3), 128, np.uint8))
        self.assertFalse(used_mask)

    def test_normalized_16bit_mask_is_overlaid(self):
        overlay = normalize_mask_image(np.full((50, 50, 4), 65535, np.uint16))
        _, used_mask = self.assert_face_hidden(overlay)
        self.assertTrue(used_mask)

    def test_tiny_face_size_is_still_covered(self):
        # 顔サイズが 0〜1 でも、マスクあり・なしのどちらの経路でも下限の広さを隠す
        half = MIN_COVER_SIZE // 2
        region = (slice(360 - half, 360 + half), slice(640 - half, 640 + half))
        red = np.zeros((50, 50, 4), np.uint8)
        red[:, :] = (0, 0, 255, 255)
        for size in (-3, 0, 1):
            original = noisy_frame()
            frame, used_mask = cover_face(original.copy(), red, 640, 360, size)
            self.assertTrue(used_mask, msg=f"size={size}")
            self.assertTrue((frame[region] == (0, 0, 255)).all(), msg=f"size={size}")

            frame, used_mask = cover_face(original.copy(), None, 640, 360, size)
            self.assertFalse(used_mask, msg=f"size={size}")
            unchanged = (frame[region] == original[region]).all(axis=2)
            self.assertLess(unchanged.mean(), 0.05, msg=f"size={size}")

    def test_overlay_failure_falls_back_to_mosaic(self):
        original = noisy_frame()
        overlay = np.full((50, 50, 4), 255, np.uint8)
        with mock.patch.object(mask_safety, "overlay_transparent", side_effect=cv2.error("boom")):
            frame, used_mask = cover_face(original.copy(), overlay, 640, 360, 200)
        self.assertFalse(used_mask)
        self.assertFalse(np.array_equal(frame[260:460, 540:740], original[260:460, 540:740]))

    def test_mask_partly_outside_frame(self):
        overlay = np.full((50, 50, 4), 255, np.uint8)
        frame, used_mask = cover_face(noisy_frame(), overlay, 10, 10, 200)
        self.assertTrue(used_mask)
        self.assertTrue((frame[:110, :110] == 255).all())


class DecideMaskStatusTest(unittest.TestCase):
    def test_off_and_detector_error_take_priority(self):
        self.assertEqual(decide_mask_status(False, True, True, 1), "off")
        self.assertEqual(decide_mask_status(False, False, False, 0), "off")
        self.assertEqual(decide_mask_status(True, False, True, 1), "detector_error")
        self.assertEqual(decide_mask_status(True, False, True, 0), "detector_error")

    def test_no_face_is_not_reported_as_applied(self):
        self.assertEqual(decide_mask_status(True, True, True, 0), "no_face")
        self.assertEqual(decide_mask_status(True, True, False, 0), "no_face_fallback")

    def test_faces_covered(self):
        self.assertEqual(decide_mask_status(True, True, True, 1), "ok")
        self.assertEqual(decide_mask_status(True, True, True, 3, all_masks_applied=True), "ok")
        self.assertEqual(decide_mask_status(True, True, True, 2, all_masks_applied=False), "fallback")
        self.assertEqual(decide_mask_status(True, True, False, 1), "fallback")
        self.assertEqual(decide_mask_status(True, True, False, 1, all_masks_applied=False), "fallback")


class SaveMaskPngTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.img = np.zeros((20, 30, 4), np.uint8)
        self.img[:, :] = (10, 20, 30, 200)

    def read(self, path):
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    def test_saves_and_can_be_read_back(self):
        path = os.path.join(self.dir, "マスク.png")  # 日本語のファイル名
        self.assertTrue(save_mask_png(path, self.img))
        self.assertTrue(np.array_equal(self.read(path), self.img))
        self.assertEqual(os.listdir(self.dir), ["マスク.png"])  # 一時ファイルを残さない

    def test_missing_folder_returns_false(self):
        path = os.path.join(self.dir, "no_such_dir", "a.png")
        self.assertFalse(save_mask_png(path, self.img))
        self.assertFalse(os.path.exists(path))

    def test_encode_failure_returns_false(self):
        path = os.path.join(self.dir, "a.png")
        with mock.patch.object(mask_safety.cv2, "imencode", return_value=(False, None)):
            self.assertFalse(save_mask_png(path, self.img))
        with mock.patch.object(mask_safety.cv2, "imencode", side_effect=cv2.error("boom")):
            self.assertFalse(save_mask_png(path, self.img))
        self.assertFalse(save_mask_png(path, None))
        self.assertEqual(os.listdir(self.dir), [])

    def test_failed_save_keeps_existing_mask(self):
        path = os.path.join(self.dir, "a.png")
        self.assertTrue(save_mask_png(path, self.img))
        other = np.full((5, 5, 4), 255, np.uint8)
        # 書き込んだ内容が壊れていた（読み戻せない）場合
        with mock.patch.object(mask_safety.cv2, "imdecode", return_value=None):
            self.assertFalse(save_mask_png(path, other))
        self.assertTrue(np.array_equal(self.read(path), self.img))
        self.assertEqual(os.listdir(self.dir), ["a.png"])

    def test_replace_failure_returns_false(self):
        path = os.path.join(self.dir, "a.png")
        with mock.patch.object(mask_safety.os, "replace", side_effect=PermissionError("locked")):
            self.assertFalse(save_mask_png(path, self.img))
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
