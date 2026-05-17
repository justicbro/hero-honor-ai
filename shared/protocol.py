"""WebSocket message protocol shared by Mac agent and inference server.

Uplink   (Mac -> Server): binary frame  = JPEG bytes
Downlink (Server -> Mac): text frame    = JSON-encoded Action

Action JSON examples:
    {"type": "tap",   "x": 100, "y": 200, "x2": 0, "y2": 0, "duration_ms": 100}
    {"type": "swipe", "x": 100, "y": 200, "x2": 300, "y2": 400, "duration_ms": 300}
    {"type": "noop",  "x": 0,   "y": 0,   "x2": 0, "y2": 0,   "duration_ms": 0}

Walk + cast (MOBA): same frame requests joystick nudge then skill tap:
    {"type": "tap", "x": 1900, "y": 1200, "duration_ms": 60,
     "pre_joystick_pad": true, "jx": 460, "jy": 1080, "jx2": 520, "jy2": 950}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Literal

ActionType = Literal["tap", "swipe", "noop"]


@dataclass
class Action:
    type: ActionType = "noop"
    x: int = 0
    y: int = 0
    x2: int = 0
    y2: int = 0
    duration_ms: int = 100
    # If type=="tap" and True: client runs joystick (jx,jy)->(jx2,jy2) first,
    # then the skill tap ("walk while casting" on multitouch stacks).
    pre_joystick_pad: bool = False
    jx: int = 0
    jy: int = 0
    jx2: int = 0
    jy2: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Action":
        d = json.loads(s)
        names = {f.name for f in fields(cls)}
        base = asdict(cls())
        for k, v in d.items():
            if k in names:
                base[k] = v
        return cls(**base)


@dataclass
class Hello:
    """First text message Mac agent sends after WebSocket upgrade."""
    width: int
    height: int
    fps: float

    def to_json(self) -> str:
        return json.dumps({"hello": asdict(self)})

    @classmethod
    def from_json(cls, s: str) -> "Hello":
        data = json.loads(s)["hello"]
        return cls(**data)
