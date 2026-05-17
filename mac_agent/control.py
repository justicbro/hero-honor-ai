"""ADB-based input injection for tap / swipe actions."""
from __future__ import annotations

import subprocess
from typing import Optional

from shared.protocol import Action


class AdbControl:
    def __init__(self, serial: Optional[str] = None):
        self._base = ["adb"] + (["-s", serial] if serial else [])

    def execute(self, action: Action) -> None:
        if action.type == "tap":
            self._run(["shell", "input", "tap", str(action.x), str(action.y)])
        elif action.type == "swipe":
            self._run([
                "shell", "input", "swipe",
                str(action.x), str(action.y),
                str(action.x2), str(action.y2),
                str(action.duration_ms),
            ])
        # noop: do nothing

    def _run(self, args: list[str]) -> None:
        subprocess.run(self._base + args, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
