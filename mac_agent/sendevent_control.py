"""Android touch injection via streamed ``adb shell`` + ``sendevent``.

``adb shell input swipe`` ends each gesture with ACTION_UP, so repeated joystick
drags feel jerky. Protocol-B multitouch keeps slot 1 as a persistent joystick
finger and slot 0 for transient skill taps.

Joystick swipes mirror ``Action(swipe)``: DOWN at thumbstick center ``(x,y)``,
then MOVE to rim ``(x2,y2)`` — required by many MOBA engines.

Optional legacy workaround: lift joystick briefly before skill taps when
``release_joy_before_skill_tap=True`` (hurts walking during combos).

Landscape capture coordinates map to portrait touch axes via ``_screen_to_touch``.
"""
from __future__ import annotations

import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from shared.protocol import Action

EV_SYN = 0
EV_KEY = 1
EV_ABS = 3

SYN_REPORT = 0
BTN_TOUCH = 330

ABS_MT_SLOT = 47
ABS_MT_POSITION_X = 53
ABS_MT_POSITION_Y = 54
ABS_MT_TRACKING_ID = 57

SLOT_TAP = 0
SLOT_JOY = 1


@dataclass
class TouchDevice:
    path: str
    max_x: int
    max_y: int


class SendeventControl:
    def __init__(self, serial: Optional[str] = None,
                 screen_w: int = 2560, screen_h: int = 1440,
                 device_override: Optional[TouchDevice] = None,
                 release_joy_before_skill_tap: bool = False):
        self._serial = serial
        self._base = ["adb"] + (["-s", serial] if serial else [])
        self._sw = screen_w
        self._sh = screen_h
        self._lock = threading.Lock()
        self._release_joy_before_skill_tap = release_joy_before_skill_tap

        self._device = device_override or self._probe_touch_device()
        flag = (" releaseJoyBeforeTap=on" if release_joy_before_skill_tap else "")
        print(f"[sendevent] touch device {self._device.path} "
              f"max=({self._device.max_x},{self._device.max_y}) "
              f"screen={screen_w}x{screen_h}{flag}")

        self._shell = subprocess.Popen(
            self._base + ["shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        self._next_tracking_id = 1000
        self._joy_down = False
        self._joy_hold_until: float = 0.0
        self._active_touches: set[int] = set()

        self._stop_watchdog = threading.Event()
        self._watchdog = threading.Thread(target=self._watchdog_loop,
                                          daemon=True)
        self._watchdog.start()

    def _probe_touch_device(self) -> TouchDevice:
        try:
            out = subprocess.check_output(
                self._base + ["shell", "getevent", "-pl"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode(errors="ignore")
        except Exception as e:
            print(f"[sendevent] probe failed ({e}); using MuMu defaults")
            return TouchDevice("/dev/input/event1", 1440, 2560)

        devs: List[TouchDevice] = []
        cur_path: Optional[str] = None
        cur_max_x: Optional[int] = None
        cur_max_y: Optional[int] = None
        for raw in out.splitlines():
            line = raw.rstrip()
            m = re.match(r"add device \d+: (\S+)", line)
            if m:
                if (cur_path is not None
                        and cur_max_x is not None and cur_max_y is not None):
                    devs.append(TouchDevice(cur_path, cur_max_x, cur_max_y))
                cur_path = m.group(1)
                cur_max_x = None
                cur_max_y = None
                continue
            if "ABS_MT_POSITION_X" in line:
                mm = re.search(r"max\s+(\d+)", line)
                if mm:
                    cur_max_x = int(mm.group(1))
            elif "ABS_MT_POSITION_Y" in line:
                mm = re.search(r"max\s+(\d+)", line)
                if mm:
                    cur_max_y = int(mm.group(1))
        if (cur_path is not None
                and cur_max_x is not None and cur_max_y is not None):
            devs.append(TouchDevice(cur_path, cur_max_x, cur_max_y))

        if not devs:
            print("[sendevent] no MT device found; using MuMu defaults")
            return TouchDevice("/dev/input/event1", 1440, 2560)
        return devs[0]

    def _screen_to_touch(self, sx: int, sy: int) -> Tuple[int, int]:
        sx = max(0, min(self._sw - 1, int(sx)))
        sy = max(0, min(self._sh - 1, int(sy)))
        tx = self._device.max_x - int(sy * self._device.max_x / self._sh)
        ty = int(sx * self._device.max_y / self._sw)
        return tx, ty

    def _send(self, events: List[Tuple[int, int, int]]) -> None:
        path = self._device.path
        lines = [f"sendevent {path} {t} {c} {v}".encode()
                 for (t, c, v) in events]
        payload = b"\n".join(lines) + b"\n"
        try:
            assert self._shell.stdin is not None
            self._shell.stdin.write(payload)
            self._shell.stdin.flush()
        except (BrokenPipeError, AssertionError):
            print("[sendevent] shell pipe broken; respawning")
            self._respawn_shell()
            assert self._shell.stdin is not None
            self._shell.stdin.write(payload)
            self._shell.stdin.flush()

    def _respawn_shell(self) -> None:
        try:
            self._shell.terminate()
        except Exception:
            pass
        self._shell = subprocess.Popen(
            self._base + ["shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._joy_down = False
        self._active_touches.clear()

    def _next_tid(self) -> int:
        self._next_tracking_id += 1
        return self._next_tracking_id

    def _slot_down(self, slot: int, tx: int, ty: int) -> None:
        first_touch = not self._active_touches
        self._active_touches.add(slot)
        events: List[Tuple[int, int, int]] = [
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_TRACKING_ID, self._next_tid()),
            (EV_ABS, ABS_MT_POSITION_X, tx),
            (EV_ABS, ABS_MT_POSITION_Y, ty),
        ]
        if first_touch:
            events.append((EV_KEY, BTN_TOUCH, 1))
        events.append((EV_SYN, SYN_REPORT, 0))
        self._send(events)

    def _slot_move(self, slot: int, tx: int, ty: int) -> None:
        self._send([
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_POSITION_X, tx),
            (EV_ABS, ABS_MT_POSITION_Y, ty),
            (EV_SYN, SYN_REPORT, 0),
        ])

    def _slot_up(self, slot: int) -> None:
        if slot not in self._active_touches:
            return
        self._active_touches.discard(slot)
        events: List[Tuple[int, int, int]] = [
            (EV_ABS, ABS_MT_SLOT, slot),
            (EV_ABS, ABS_MT_TRACKING_ID, -1),
        ]
        if not self._active_touches:
            events.append((EV_KEY, BTN_TOUCH, 0))
        events.append((EV_SYN, SYN_REPORT, 0))
        self._send(events)

    def _tap(self, sx: int, sy: int, hold_ms: int = 60) -> None:
        tx, ty = self._screen_to_touch(sx, sy)
        with self._lock:
            self._slot_down(SLOT_TAP, tx, ty)
        time.sleep(max(0.02, hold_ms / 1000.0))
        with self._lock:
            self._slot_up(SLOT_TAP)

    def _joystick_from_to(self,
                          sx1: int, sy1: int,
                          sx2: int, sy2: int,
                          hold_ms: int) -> None:
        tx1, ty1 = self._screen_to_touch(sx1, sy1)
        tx2, ty2 = self._screen_to_touch(sx2, sy2)
        with self._lock:
            if not self._joy_down:
                self._slot_down(SLOT_JOY, tx1, ty1)
                self._joy_down = True
                if tx2 != tx1 or ty2 != ty1:
                    self._slot_move(SLOT_JOY, tx2, ty2)
            else:
                self._slot_move(SLOT_JOY, tx2, ty2)
            # Hold long enough that sparse frames / recv jitter do not lift the
            # finger between server swipes — otherwise the hero stands still.
            self._joy_hold_until = time.time() + max(2.2, hold_ms / 1000.0)

    def _joystick_release(self) -> None:
        with self._lock:
            if self._joy_down:
                self._slot_up(SLOT_JOY)
                self._joy_down = False

    def _watchdog_loop(self) -> None:
        while not self._stop_watchdog.is_set():
            time.sleep(0.05)
            if self._joy_down and time.time() > self._joy_hold_until:
                self._joystick_release()

    def execute(self, action: Action) -> None:
        if action.type == "tap":
            if self._release_joy_before_skill_tap:
                self._joystick_release()
                time.sleep(0.04)
            t = threading.Thread(
                target=self._tap,
                args=(action.x, action.y, action.duration_ms),
                daemon=True,
            )
            t.start()
        elif action.type == "swipe":
            hold_ms = max(action.duration_ms, 600)
            self._joystick_from_to(action.x, action.y, action.x2, action.y2,
                                   hold_ms=hold_ms)
        # noop: watchdog may release joystick when stale

    def close(self) -> None:
        self._stop_watchdog.set()
        try:
            self._joystick_release()
        except Exception:
            pass
        try:
            self._shell.terminate()
        except Exception:
            pass
