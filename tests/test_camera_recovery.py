"""camera_recovery の単体テスト（カメラ不要）。

実行: python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_recovery import (ReopenTimer, close_wait_action, configure_capture,  # noqa: E402
                             fourcc_to_str, is_blank_frame, is_stalled_read, reopen_capture)

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


class OpenGraceTest(unittest.TestCase):
    """開いた直後の遅い読み取りで、開き直しを繰り返さないこと。"""

    def test_no_reopen_during_grace_after_open(self):
        timer = ReopenTimer(after=2.0, interval=3.0, open_grace=5.0)
        timer.on_opened(100.0)
        # 開いた直後から 1 秒ごとに失敗が続いても、猶予のあいだは開き直さない
        for now in (100.0, 101.0, 102.0, 103.0, 104.9):
            self.assertFalse(timer.on_failure(now))
        self.assertTrue(timer.on_failure(105.0))    # 猶予を過ぎても戻らなければ開き直す

    def test_slow_camera_does_not_loop(self):
        # 開いてから最初のフレームまで 4 秒かかるカメラ。開き直しは一度も起きない
        timer = ReopenTimer(after=2.0, interval=3.0, open_grace=5.0)
        timer.on_opened(0.0)
        reopens = sum(timer.on_failure(float(now)) for now in range(0, 4))
        self.assertEqual(reopens, 0)
        self.assertTrue(timer.on_success())
        # そのあとの途切れは、いつもどおり after 秒で開き直す（猶予は「開いた直後」だけ）
        self.assertFalse(timer.on_failure(10.0))
        self.assertTrue(timer.on_failure(12.0))

    def test_each_reopen_gets_its_own_grace(self):
        timer = ReopenTimer(after=2.0, interval=3.0, open_grace=5.0)
        timer.on_opened(0.0)
        self.assertFalse(timer.on_failure(0.1))
        self.assertTrue(timer.on_failure(5.0))
        timer.on_opened(6.5)                        # 開き直しに 1.5 秒かかった
        self.assertFalse(timer.on_failure(6.6))
        self.assertFalse(timer.on_failure(11.4))    # 開き直しを求めた時刻ではなく、開き終わった時刻から数える
        self.assertTrue(timer.on_failure(11.5))

    def test_without_on_opened_behaves_as_before(self):
        timer = ReopenTimer(after=2.0, interval=3.0, open_grace=5.0)
        timer.on_failure(100.0)
        self.assertTrue(timer.on_failure(102.0))


class StalledReadTest(unittest.TestCase):
    def setUp(self):
        self.black = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.picture = np.full((720, 1280, 3), 90, dtype=np.uint8)

    def test_slow_and_black_is_stalled(self):
        # USB の帯域不足の C920: 約 1 秒待たされ、真っ黒な画像が「成功」で返る
        self.assertTrue(is_stalled_read(1.0, self.black))
        self.assertTrue(is_stalled_read(0.9, self.black))

    def test_slow_but_real_picture_is_not_stalled(self):
        # 開いた直後の最初のフレームなど、遅いだけで中身のある画像は映像として使う
        self.assertFalse(is_stalled_read(1.5, self.picture))
        self.assertFalse(is_stalled_read(3.0, self.picture))

    def test_fast_black_is_not_stalled(self):
        # レンズを覆っているだけ（読み取りは速い）なら、途切れ扱いにしない
        self.assertFalse(is_stalled_read(0.03, self.black))
        self.assertFalse(is_stalled_read(0.89, self.black))

    def test_blank_frame(self):
        self.assertTrue(is_blank_frame(None))
        self.assertTrue(is_blank_frame(np.zeros((0, 0, 3), dtype=np.uint8)))
        self.assertTrue(is_blank_frame(self.black))
        nearly = self.black.copy()
        nearly[10, 10] = 4                          # 圧縮のにじみ程度は真っ黒のうち
        self.assertTrue(is_blank_frame(nearly))
        dark = self.black.copy()
        dark[300:320, 600:640] = 30                 # 暗い部屋でも、何か映っていれば真っ黒ではない
        self.assertFalse(is_blank_frame(dark))


class Cap:
    """開く・閉じるだけを数える偽のカメラ。"""

    def __init__(self, name):
        self.name = name
        self.released = 0

    def release(self):
        self.released += 1


class ReopenCaptureTest(unittest.TestCase):
    def run_reopen(self, running_answers):
        """is_running が順に running_answers を返すときの結果と、起きたことを返す。"""
        answers = iter(running_answers)
        old, events, opened = Cap("old"), [], []

        def open_fn():
            events.append("open")
            opened.append(Cap("new"))
            return opened[-1]

        result = reopen_capture(old, open_fn, lambda: next(answers),
                                settle_seconds=0.5, sleep=lambda seconds: events.append(("sleep", seconds)))
        return result, old, opened, events

    def test_reopens_when_still_running(self):
        result, old, opened, events = self.run_reopen([True, True, True])
        self.assertIs(result, opened[0])
        self.assertEqual(old.released, 1)
        self.assertEqual(opened[0].released, 0)
        self.assertEqual(events, [("sleep", 0.5), "open"])

    def test_does_not_open_when_shutdown_started_before_release_finished(self):
        result, old, opened, events = self.run_reopen([False])
        self.assertIsNone(result)
        self.assertEqual((old.released, opened, events), (1, [], []))

    def test_does_not_open_when_shutdown_started_while_settling(self):
        result, old, opened, events = self.run_reopen([True, False])
        self.assertIsNone(result)
        self.assertEqual((old.released, opened, events), (1, [], [("sleep", 0.5)]))

    def test_releases_new_capture_when_shutdown_started_while_opening(self):
        # open_camera が数秒かかり、そのあいだにウィンドウが閉じられた場合
        result, old, opened, events = self.run_reopen([True, True, False])
        self.assertIsNone(result)
        self.assertEqual(old.released, 1)
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0].released, 1)     # 開いたカメラを閉じてから抜ける


class CloseWaitTest(unittest.TestCase):
    def test_waits_while_worker_alive_before_deadline(self):
        self.assertEqual(close_wait_action(True, 104.9, 105.0), "wait")

    def test_gives_up_at_deadline(self):
        self.assertEqual(close_wait_action(True, 105.0, 105.0), "timed_out")
        self.assertEqual(close_wait_action(True, 110.0, 105.0), "timed_out")

    def test_closes_as_soon_as_worker_finished(self):
        self.assertEqual(close_wait_action(False, 100.0, 105.0), "closed")
        self.assertEqual(close_wait_action(False, 110.0, 105.0), "closed")   # 期限後でも警告は不要


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
