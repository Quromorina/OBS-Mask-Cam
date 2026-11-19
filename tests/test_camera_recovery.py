"""camera_recovery の単体テスト（カメラ不要）。

実行: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_recovery import ReopenTimer, configure_capture, fourcc_to_str  # noqa: E402

MJPG = cv2.VideoWriter_fourcc(*"MJPG")
YUY2 = cv2.VideoWriter_fourcc(*"YUY2")


class ReopenTimerTest(unittest.TestCase):
    def test_does_not_reopen_on_short_dropout(self):
        timer = ReopenTimer(after=2.0, interval=3.0)
        self.assertFalse(timer.on_failure(100.0))
        self.assertFalse(timer.on_failure(101.9))

    def test_reopens_once_dropout_lasts(self):
        timer = ReopenTimer(after=2.0, interval=3.0)
        timer.on_failure(100.0)
        self.assertTrue(timer.on_failure(102.0))

    def test_retries_at_interval_while_still_failing(self):
        timer = ReopenTimer(after=2.0, interval=3.0)
        timer.on_failure(100.0)
        self.assertTrue(timer.on_failure(102.0))
        self.assertFalse(timer.on_failure(104.9))   # 開き直した直後は連打しない
        self.assertTrue(timer.on_failure(105.0))
        self.assertFalse(timer.on_failure(106.0))
        self.assertTrue(timer.on_failure(108.0))

    def test_success_resets_and_reports_recovery(self):
        timer = ReopenTimer(after=2.0, interval=3.0)
        self.assertFalse(timer.on_success())        # 途切れていなければ「復帰」ではない
        timer.on_failure(100.0)
        self.assertTrue(timer.on_success())
        self.assertFalse(timer.on_success())
        # 復帰後にまた途切れたら、短い途切れでは開き直さない
        self.assertFalse(timer.on_failure(200.0))
        self.assertFalse(timer.on_failure(201.0))
        self.assertTrue(timer.on_failure(202.0))


class FakeCapture:
    """cv2.VideoCapture の set/get だけを真似る。accept_mjpg=False は MJPG 非対応のカメラ。"""

    def __init__(self, accept_mjpg=True, size=(1280, 720)):
        self.accept_mjpg = accept_mjpg
        self.calls = []
        self.props = {cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480,
                      cv2.CAP_PROP_FOURCC: YUY2}
        self.size = size

    def set(self, prop, value):
        self.calls.append(prop)
        if prop == cv2.CAP_PROP_FOURCC:
            if value == MJPG and not self.accept_mjpg:
                return False
        elif prop == cv2.CAP_PROP_FRAME_WIDTH:
            value = self.size[0]
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            value = self.size[1]
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)


class ConfigureCaptureTest(unittest.TestCase):
    def test_requests_mjpg_after_resolution(self):
        # DirectShow では解像度より先に FOURCC を設定すると YUY2 に戻る（C920 で確認）
        cap = FakeCapture()
        result = configure_capture(cap, 1280, 720, 30)
        self.assertEqual(result, (1280, 720, "MJPG"))
        self.assertEqual(cap.calls[-1], cv2.CAP_PROP_FOURCC)
        self.assertLess(cap.calls.index(cv2.CAP_PROP_FRAME_HEIGHT), cap.calls.index(cv2.CAP_PROP_FOURCC))

    def test_reports_actual_format_when_mjpg_unsupported(self):
        cap = FakeCapture(accept_mjpg=False, size=(640, 480))
        self.assertEqual(configure_capture(cap, 1280, 720, 30), (640, 480, "YUY2"))

    def test_fourcc_to_str_handles_garbage(self):
        self.assertEqual(fourcc_to_str(MJPG), "MJPG")
        self.assertEqual(fourcc_to_str(0), "????")
        self.assertEqual(fourcc_to_str(-1.0), "????")


if __name__ == "__main__":
    unittest.main()
