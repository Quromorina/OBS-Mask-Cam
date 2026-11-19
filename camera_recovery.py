"""カメラの開き方と、映像が途切れたときに開き直す判断。

USB 2.0 のハブにオーディオインターフェースや高速ポーリングのマウスと一緒につながっていると、
カメラに回せる帯域が足りず、無圧縮（YUY2）の 1280x720 は送れないことがある。
そのため MJPG（圧縮）を要求して、必要な帯域を減らす。
"""
import time

import cv2

# 映像が途切れてから、カメラを開き直すまでの秒数
REOPEN_AFTER_SECONDS = 2.0
# 開き直しても戻らないとき、次に試すまでの秒数
REOPEN_INTERVAL_SECONDS = 3.0
# 1 回の読み取りにこれ以上かかったら、映像が届いていないとみなす。
# DirectShow は新しいフレームが来ないと約 1 秒待ってから、空（真っ黒）の画像を「成功」として返す
# （USB の帯域不足の C920 で確認）。暗所で fps が落ちても 1 フレーム 0.2 秒程度なので区別できる。
STALLED_READ_SECONDS = 0.9
# 待たされた末の画像は、全画素がこの値以下なら「真っ黒（映像は届いていない）」とみなす
BLANK_FRAME_MAX_VALUE = 4
# カメラを開いてから、この秒数は開き直しを求めない。
# 開いた直後は最初のフレームが遅いカメラがあり、そこで開き直すと「開く → 遅い → 開き直す」を繰り返す
OPEN_GRACE_SECONDS = 5.0
# カメラを閉じてから開き直すまでに待つ秒数（リソースの解放待ち）
REOPEN_SETTLE_SECONDS = 0.5


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


def is_blank_frame(frame):
    """画像が無い、または真っ黒なら True。"""
    if frame is None or frame.size == 0:
        return True
    return int(frame.max()) <= BLANK_FRAME_MAX_VALUE


def is_stalled_read(elapsed, frame, threshold=STALLED_READ_SECONDS):
    """「成功」と返った読み取りが、実は映像の届いていない空読みなら True。

    待たされた（threshold 秒以上）うえに真っ黒、の両方がそろったときだけ空読みとみなす。
    遅いだけで中身のある画像（開いた直後の最初のフレームなど）は、映像として使う。
    """
    return elapsed >= threshold and is_blank_frame(frame)


def reopen_capture(cap, open_fn, is_running, settle_seconds=REOPEN_SETTLE_SECONDS, sleep=time.sleep):
    """cap を閉じ、open_fn() で開き直して返す。終了が始まっていたら None を返す。

    閉じる・開くは数秒かかることがあり、そのあいだにアプリの終了が始まることがある。
    その場合は開かずにやめ、開いてしまったあとならすぐ閉じる（カメラを開いたまま終了しない）。
    None を返したとき、カメラは閉じた状態になっている。
    """
    cap.release()
    if not is_running():
        return None
    sleep(settle_seconds)
    if not is_running():
        return None
    new_cap = open_fn()
    if not is_running():
        new_cap.release()
        return None
    return new_cap


def close_wait_action(worker_alive, now, deadline):
    """終了時の待ち方を決める。

    - "wait": カメラ処理がまだ動いていて、期限前。もう少し待つ
    - "timed_out": 期限を過ぎてもカメラ処理が終わらない。待つのをやめて終了する
    - "closed": カメラ処理は終わっている。終了してよい
    """
    if not worker_alive:
        return "closed"
    return "wait" if now < deadline else "timed_out"


class ReopenTimer:
    """フレームの読み取りに失敗し続けたら、カメラを開き直す時機を知らせる。

    - 途切れてから after 秒たったら、最初の開き直しを求める
    - それでも戻らなければ、interval 秒ごとに開き直しを求める
    - 1 フレームでも読めたら、最初の状態に戻る
    - カメラを開いた直後（on_opened から open_grace 秒）は、開き直しを求めない
    """

    def __init__(self, after=REOPEN_AFTER_SECONDS, interval=REOPEN_INTERVAL_SECONDS,
                 open_grace=OPEN_GRACE_SECONDS):
        self.after = after
        self.interval = interval
        self.open_grace = open_grace
        self._opened_at = None
        self.reset()

    def reset(self):
        self._fail_since = None
        self._last_reopen = None

    def on_opened(self, now):
        """カメラを開いた（開き直した）直後に呼ぶ。ここから open_grace 秒は開き直しを求めない。"""
        self.reset()
        self._opened_at = now

    def on_success(self):
        """フレームを読めたときに呼ぶ。途切れていた状態から戻ったなら True を返す。"""
        was_failing = self._fail_since is not None
        self.reset()
        return was_failing

    def on_failure(self, now):
        """読み取りに失敗したときに呼ぶ。いま開き直すべきなら True を返す。"""
        if self._fail_since is None:
            self._fail_since = now
        if self._opened_at is not None and now - self._opened_at < self.open_grace:
            return False
        if now - self._fail_since < self.after:
            return False
        if self._last_reopen is not None and now - self._last_reopen < self.interval:
            return False
        self._last_reopen = now
        return True
