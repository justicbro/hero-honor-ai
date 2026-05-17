"""Generate a fake template that matches the mock client's synthetic frame.

We crop the "MOCK FRAME" text region from the same image the mock client sends.
This lets us exercise the full server pipeline (decode -> match -> respond Action)
without a real emulator.
"""
from pathlib import Path

import cv2

from tests.mock_client import make_fake_frame
import numpy as np


def main() -> None:
    jpg = make_fake_frame()
    arr = np.frombuffer(jpg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    crop = img[h // 2 - 60: h // 2 + 20, w // 4: w // 4 + 600]
    out = Path(__file__).parent.parent / "server" / "templates" / "tap_mock.png"
    cv2.imwrite(str(out), crop)
    print(f"wrote {out} shape={crop.shape}")


if __name__ == "__main__":
    main()
