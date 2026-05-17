"""Android emulator screen capture via ADB.

Demo-grade: ~200ms per frame. Good enough for first demo (target ~5 fps).
For 30 fps low-latency, swap to scrcpy-server (see README roadmap).
"""
from __future__ import annotations

import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np


class AdbCapture:
    def __init__(self, serial: Optional[str] = None):
        self.serial = serial
        self._base = ["adb"] + (["-s", serial] if serial else [])

    def grab_jpeg(self, quality: int = 70) -> bytes:
        """Grab one frame as JPEG bytes.

        Pipeline: ADB screencap (PNG) -> cv2 decode -> cv2 JPEG encode.
        We re-encode because JPEG is ~10x smaller than PNG over the wire.
        """
        png_bytes = subprocess.check_output(
            self._base + ["exec-out", "screencap", "-p"]
        )
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("screencap returned non-decodable bytes")
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return jpg.tobytes()

    def screen_size(self) -> Tuple[int, int]:
        """Return (width, height) of the emulator screen in the CURRENT orientation.

        `wm size` reports the natural (portrait) physical size. If the device is
        currently rotated to landscape (rotation 1 or 3), we swap so callers get
        the same coordinate system that `input tap` and the game UI use.
        """
        out = subprocess.check_output(self._base + ["shell", "wm", "size"]).decode()
        # Example: "Physical size: 1440x2560"
        size_str = out.strip().split(":")[-1].strip()
        w_nat, h_nat = (int(s) for s in size_str.split("x"))

        rotation = self._current_rotation()
        if rotation in (1, 3):
            return h_nat, w_nat  # landscape
        return w_nat, h_nat       # portrait (or unknown -> assume natural)

    def _current_rotation(self) -> int:
        """Best-effort current display rotation in {0,1,2,3}. Returns 0 on failure."""
        try:
            out = subprocess.check_output(
                self._base + ["shell", "dumpsys", "input"],
                stderr=subprocess.DEVNULL,
            ).decode(errors="ignore")
        except Exception:
            return 0
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("SurfaceOrientation:"):
                try:
                    return int(line.split(":", 1)[1].strip())
                except ValueError:
                    return 0
        return 0

    def check_device(self) -> str:
        """Return first connected device serial, raise if none."""
        out = subprocess.check_output(["adb", "devices"]).decode()
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        devices = [ln.split()[0] for ln in lines[1:] if ln.endswith("device")]
        if not devices:
            raise RuntimeError(
                "No ADB device. Start emulator and run `adb connect 127.0.0.1:7555` "
                "(MuMu) or `adb connect 127.0.0.1:5555` (BlueStacks)."
            )
        return devices[0]
