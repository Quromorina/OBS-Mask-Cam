"""カメラの開き方と、映像が途切れたときに開き直す判断。

USB 2.0 のハブにオーディオインターフェースや高速ポーリングのマウスと一緒につながっていると、
カメラに回せる帯域が足りず、無圧縮（YUY2）の 1280x720 は送れないことがある。
そのため MJPG（圧縮）を要求して、必要な帯域を減らす。
"""
import cv2

# 映像が途切れてから、カメラを開き直すまでの秒数
REOPEN_AFTER_SECONDS = 2.0
# 開き直しても戻らないとき、次に試すまでの秒数
REOPEN_INTERVAL_SECONDS = 3.0


def fourcc_to_str(value):
    """CAP_PROP_FOURCC の数値を 'MJPG' のような文字列にする（ログ用）。"""
    v = int(value)
    codes = [(v >> (8 * i)) & 0xFF for i in range(4)]
    return "".join(chr(c) if 0x20 <= c < 0x7F else "?" for c in codes)


def configure_capture(cap, width, height, fps):
    """解像度・fps・MJPG を要求し、カメラが実際に受け付けた (幅, 高さ, 形式) を返す。

    DirectShow では、解像度を設定したあとに FOURCC を設定しないと MJPG にならない
    （C920 で確認）。カメラが MJPG に対応していなければ、元の形式のまま動く。
    """
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)))


class ReopenTimer:
    """フレームの読み取りに失敗し続けたら、カメラを開き直す時機を知らせる。

    - 途切れてから after 秒たったら、最初の開き直しを求める
    - それでも戻らなければ、interval 秒ごとに開き直しを求める
    - 1 フレームでも読めたら、最初の状態に戻る
    """

    def __init__(self, after=REOPEN_AFTER_SECONDS, interval=REOPEN_INTERVAL_SECONDS):
        self.after = after
        self.interval = interval
        self.reset()

    def reset(self):
        self._fail_since = None
        self._last_reopen = None

    def on_success(self):
        """フレームを読めたときに呼ぶ。途切れていた状態から戻ったなら True を返す。"""
        was_failing = self._fail_since is not None
        self.reset()
        return was_failing

    def on_failure(self, now):
        """読み取りに失敗したときに呼ぶ。いま開き直すべきなら True を返す。"""
        if self._fail_since is None:
            self._fail_since = now
        if now - self._fail_since < self.after:
            return False
        if self._last_reopen is not None and now - self._last_reopen < self.interval:
            return False
        self._last_reopen = now
        return True
