"""Fast screen capture via Android's built-in `screenrecord` + local ffmpeg.

`adb exec-out screenrecord --output-format=h264` streams a raw H.264 Annex B
bitstream over ADB. We pipe it into ffmpeg to decode to raw BGR frames; a
background thread keeps re-encoding the latest frame as JPEG so the agent
main loop can read it instantly.

screenrecord has a hard 3-minute per-session limit on the device. We respawn
the pipeline every ~170s; this causes a brief (<1s) gap in frames per restart
during which `grab_jpeg()` keeps returning the last captured frame.

End-to-end latency: ~100-200ms.
Throughput: 25-30 fps (Android caps screenrecord at the display refresh rate).

Prerequisites on Mac:
    brew install ffmpeg
    # scrcpy NOT required (we use the device's built-in recorder)
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from mac_agent.capture import AdbCapture

# Android's hard limit is 180s; stay safely below.
SCREENRECORD_TIME_LIMIT_S = 170


class ScreenrecordCapture:
    """Drop-in replacement for AdbCapture with much faster grab_jpeg()."""

    def __init__(self, serial: Optional[str] = None,
                 max_size: int = 1280, bit_rate: int = 4_000_000,
                 jpeg_quality: int = 70) -> None:
        self.serial = serial
        self._jpeg_quality = jpeg_quality
        self._bit_rate = bit_rate

        # Reuse AdbCapture for device + orientation-aware size detection.
        adb = AdbCapture(serial)
        self._serial = adb.check_device()
        dw, dh = adb.screen_size()
        self._device_size = (dw, dh)

        # screenrecord scales to whatever --size we request; preserve aspect.
        scale = max_size / max(dw, dh)
        ow = max(2, int(round(dw * scale)) & ~1)
        oh = max(2, int(round(dh * scale)) & ~1)
        self._out_w, self._out_h = ow, oh
        self._frame_size = ow * oh * 3
        self._capture_size = f"{ow}x{oh}"
        print(f"[capture] device={dw}x{dh}  capture={ow}x{oh}  "
              f"bit_rate={bit_rate // 1000}kbps")

        self._latest_jpg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._frames_read = 0
        self._sessions = 0
        self._stop = False

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        self._wait_for_first_frame(timeout=15.0)
        print(f"[capture] first frame OK")

    # ---- public API (matches AdbCapture) ----

    def grab_jpeg(self, quality: int = 70) -> bytes:
        with self._lock:
            if self._latest_jpg is None:
                raise RuntimeError("no frame available yet")
            return self._latest_jpg

    def screen_size(self) -> Tuple[int, int]:
        return self._device_size

    def check_device(self) -> str:
        return self._serial

    def close(self) -> None:
        self._stop = True

    @property
    def frames_read(self) -> int:
        with self._lock:
            return self._frames_read

    # ---- internals ----

    def _spawn_pipeline(self) -> subprocess.Popen:
        """Spawn a shell-wrapped 'adb screenrecord | ffmpeg' pipeline.

        Returns a single shell process whose stdout streams raw BGR frames.
        We wrap in shell because connecting two `Popen` directly via PIPE +
        PIPE has a deadlock with this particular adb+ffmpeg combo on macOS;
        letting the shell wire the inner pipe works reliably.
        """
        serial_part = f"-s {self.serial} " if self.serial else ""
        cmd = (
            f"adb {serial_part}exec-out screenrecord --output-format=h264 "
            f"--size {self._capture_size} --bit-rate {self._bit_rate} "
            f"--time-limit {SCREENRECORD_TIME_LIMIT_S} - | "
            f"ffmpeg -loglevel error "
            f"-f h264 -i - -f rawvideo -pix_fmt bgr24 -"
        )
        return subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _reader_loop(self) -> None:
        while not self._stop:
            proc = self._spawn_pipeline()
            self._sessions += 1
            print(f"[capture] session#{self._sessions} started (pid={proc.pid})")
            try:
                self._read_session(proc)
            except Exception as e:
                print(f"[capture] read error: {e}")
            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            if self._stop:
                return
            time.sleep(0.3)

    def _read_session(self, proc: subprocess.Popen) -> None:
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("pipeline has no stdout")
        while not self._stop:
            buf = self._read_exact(stdout, self._frame_size)
            if buf is None:
                stderr_tail = b""
                try:
                    if proc.stderr:
                        stderr_tail = proc.stderr.read() or b""
                except Exception:
                    pass
                msg = stderr_tail.decode(errors="ignore")[-300:]
                if msg.strip():
                    print(f"[capture] pipeline stderr: {msg}")
                return
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                (self._out_h, self._out_w, 3))
            ok, jpg = cv2.imencode(
                ".jpg", arr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality])
            if ok:
                with self._lock:
                    self._latest_jpg = jpg.tobytes()
                    self._frames_read += 1

    @staticmethod
    def _read_exact(stdout, n: int) -> Optional[bytes]:
        buf = bytearray()
        while len(buf) < n:
            chunk = stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _wait_for_first_frame(self, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._latest_jpg is not None:
                    return
            time.sleep(0.1)
        raise RuntimeError(
            f"no frame within {timeout}s. Try the manual command first:\n"
            f"  adb exec-out screenrecord --output-format=h264 "
            f"--size {self._capture_size} --time-limit 5 - > /tmp/test.h264"
        )
