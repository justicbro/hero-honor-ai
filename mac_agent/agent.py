"""Mac agent main loop.

Run on your Mac (must use -m, not direct script path):
    # Fast (Android screenrecord + ffmpeg, ~25fps; needs `brew install ffmpeg`):
    PYTHONPATH=. python -m mac_agent.agent --server ws://YOUR_SERVER_IP:8765 --fps 10

    # After connect: wait until you press Enter before any tap/swipe executes
    # (heroes stop once you Ctrl+C previous run; reconnect with --pause-actions):
    PYTHONPATH=. python -m mac_agent.agent --pause-actions ...

    # Compatibility (ADB screencap, ~0.6fps; works without extra deps):
    PYTHONPATH=. python -m mac_agent.agent --server ws://YOUR_SERVER_IP:8765 \
        --capture adb --fps 5

Debug: dump the first captured frame to disk so you can sanity-check what the
server sees and pick button coordinates from it:
    ... --save-frame /tmp/first.jpg
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import websockets

from mac_agent.capture import AdbCapture
from mac_agent.control import AdbControl
from shared.protocol import Action, Hello


def make_capture(backend: str, max_size: int, jpeg_quality: int):
    """Return a capture object with grab_jpeg / screen_size / check_device."""
    if backend == "screenrecord":
        from mac_agent.screenrecord_capture import ScreenrecordCapture
        return ScreenrecordCapture(max_size=max_size, jpeg_quality=jpeg_quality)
    if backend == "adb":
        return AdbCapture()
    raise ValueError(f"unknown --capture backend: {backend}")


def make_control(backend: str, screen_w: int, screen_h: int,
                 release_joy_before_skill_tap: bool):
    """Return a control object with execute(Action)."""
    if backend == "sendevent":
        from mac_agent.sendevent_control import SendeventControl
        return SendeventControl(
            screen_w=screen_w, screen_h=screen_h,
            release_joy_before_skill_tap=release_joy_before_skill_tap,
        )
    if backend == "adb":
        return AdbControl()
    raise ValueError(f"unknown --control backend: {backend}")


async def run(server_url: str, fps: float, jpeg_quality: int,
              save_frame: str | None, dry_run: bool,
              capture_backend: str, capture_max_size: int,
              control_backend: str,
              pause_actions: bool,
              release_joy_before_skill_tap: bool,
              recv_timeout: float | None) -> None:
    cap = make_capture(capture_backend, capture_max_size, jpeg_quality)
    serial = cap.check_device()
    w, h = cap.screen_size()
    ctl = make_control(control_backend, w, h, release_joy_before_skill_tap)
    interval = 1.0 / fps
    recv_budget = (
        recv_timeout if recv_timeout is not None
        else max(0.65, interval * 3.0))
    recv_skips = 0
    print(f"[agent] capture={capture_backend} control={control_backend} "
          f"device={serial} screen={w}x{h} -> {server_url} @ {fps}fps "
          f"recv_timeout≤{recv_budget:.2f}s"
          + (" [DRY-RUN: actions not executed]" if dry_run else ""))

    # One RTT must cover JPEG uplink + infer + JSON downlink. Using exactly
    # `interval` drops many replies on Wi‑Fi / remote servers → hero never walks.

    loop = asyncio.get_running_loop()

    async def wait_enter() -> None:
        prompt = ("[agent] Connected. Arrange the game screen, bring up the pause "
                  "keyboard if needed, then press ENTER to START sending clicks.\n")
        sys.stderr.write(prompt)
        sys.stderr.flush()
        await loop.run_in_executor(None, sys.stdin.readline)

    try:
        async with websockets.connect(server_url, max_size=4 * 1024 * 1024) as ws:
            await ws.send(Hello(width=w, height=h, fps=fps).to_json())
            if pause_actions:
                await wait_enter()
                print("[agent] resume: executing server actions.")

            frame_id = 0
            while True:
                t0 = time.time()
                try:
                    frame = cap.grab_jpeg(quality=jpeg_quality)
                except Exception as e:
                    print(f"[agent] capture error: {e}")
                    await asyncio.sleep(0.5)
                    continue

                if save_frame and frame_id == 0:
                    Path(save_frame).write_bytes(frame)
                    print(f"[agent] saved first frame -> {save_frame} "
                          f"({len(frame)} bytes)")

                await ws.send(frame)

                try:
                    reply = await asyncio.wait_for(
                        ws.recv(), timeout=recv_budget)
                    action = Action.from_json(reply)
                    if action.type != "noop":
                        if not dry_run:
                            ctl.execute(action)
                        print(f"[agent] frame#{frame_id} -> {action}"
                              + (" [skipped]" if dry_run else ""))
                except asyncio.TimeoutError:
                    recv_skips += 1
                    if recv_skips == 1 or recv_skips % 45 == 0:
                        print(f"[agent] WARN: recv timed out "
                              f"({recv_budget:.2f}s) x{recv_skips} — "
                              f"moves/skills skipped; raise "
                              f"--recv-timeout or lower --fps")

                frame_id += 1
                dt = time.time() - t0
                if dt < interval:
                    await asyncio.sleep(interval - dt)
    finally:
        ctl.close()
        print("[agent] control released (joystick fingertip UP if "
              "sendevent mode).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="ws://127.0.0.1:8765",
                    help="ws://host:port of inference server")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="How often to send frames to server (default 10).")
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--save-frame", default=None,
                    help="Save the first captured JPEG to this path and continue.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions but don't actually inject them via ADB.")
    ap.add_argument("--capture", choices=["screenrecord", "adb"],
                    default="screenrecord",
                    help="Frame source. 'screenrecord' uses Android's built-in "
                         "H.264 recorder + local ffmpeg (~25fps, needs ffmpeg); "
                         "'adb' is screencap-based, slow but no extra deps.")
    ap.add_argument("--capture-max-size", type=int, default=1280,
                    help="Max output dimension for screenrecord (default 1280).")
    ap.add_argument("--control", choices=["sendevent", "adb"],
                    default="sendevent",
                    help="Touch backend. 'sendevent' is low-level multi-touch "
                         "(persistent finger -> smooth joystick); 'adb' uses "
                         "`adb shell input` (simpler, but jittery walking).")
    ap.add_argument("--pause-actions", action="store_true",
                    help="After WebSocket HELLO succeeds, wait for Enter before "
                         "executing taps/swipes. Use when reconnecting: stop the "
                         "old agent first (Ctrl+C releases touch), start with "
                         "this flag, fix your camera, then resume.")
    ap.add_argument(
        "--release-joystick-before-skill-tap", action="store_true",
        help="Sendevent-only: briefly lift the virtual joystick before each tap. "
             "Older King-of-Glory-style workaround when skills ignore taps with "
             "the stick held. Default OFF so vision chase persists while combos "
             "fire; enable if taps fail to register on your build.")
    ap.add_argument(
        "--recv-timeout", type=float, default=None, metavar="SEC",
        help="Max seconds to wait for inference reply after each frame (default "
             "max(0.65, 3/fps)). Increase when remote server/Wi‑Fi is slow — "
             "otherwise swipes/taps are silently dropped.")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.server, args.fps, args.jpeg_quality,
                        args.save_frame, args.dry_run,
                        args.capture, args.capture_max_size,
                        args.control, args.pause_actions,
                        args.release_joystick_before_skill_tap,
                        args.recv_timeout))
    except KeyboardInterrupt:
        print("\n[agent] stopped by user")


if __name__ == "__main__":
    main()
