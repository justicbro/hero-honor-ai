"""WebSocket message protocol shared by Mac agent and inference server.

Uplink   (Mac -> Server): binary frame  = JPEG bytes
Downlink (Server -> Mac): text frame    = JSON-encoded Action

Action JSON examples:
    {"type": "tap",   "x": 100, "y": 200, "x2": 0, "y2": 0, "duration_ms": 100}
    {"type": "swipe", "x": 100, "y": 200, "x2": 300, "y2": 400, "duration_ms": 300}
    {"type": "noop",  "x": 0,   "y": 0,   "x2": 0, "y2": 0,   "duration_ms": 0}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Action":
        data = json.loads(s)
        return cls(**data)


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
