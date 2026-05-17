"""Mock Mac agent: sends a synthetic frame and prints the action.

Useful for verifying server is reachable BEFORE setting up the Mac side.

Usage (from anywhere with network access to your server):
    PYTHONPATH=. python -m tests.mock_client --server ws://YOUR_SERVER_IP:8765
"""
from __future__ import annotations

import argparse
import asyncio
import time

import cv2
import numpy as np
import websockets

from shared.protocol import Action, Hello


def make_fake_frame(w: int = 1280, h: int = 720) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "MOCK FRAME", (w // 4, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    assert ok
    return jpg.tobytes()


async def run(server_url: str, n: int) -> None:
    frame = make_fake_frame()
    async with websockets.connect(server_url, max_size=4 * 1024 * 1024) as ws:
        await ws.send(Hello(width=1280, height=720, fps=5.0).to_json())
        for i in range(n):
            t0 = time.time()
            await ws.send(frame)
            reply = await ws.recv()
            rtt = (time.time() - t0) * 1000
            action = Action.from_json(reply)
            print(f"[mock] #{i} rtt={rtt:.1f}ms action={action}")
            await asyncio.sleep(0.2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:8765")
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    asyncio.run(run(args.server, args.n))
