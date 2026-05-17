"""Low-cost computer vision primitives for the inference server.

Currently provides HSV color-based HP-bar detection. In MOBAs, friendly /
enemy / neutral targets are color-coded above their head via a thin
horizontal HP bar; finding those bars is dramatically cheaper than running
an object detector and works without any training data.

Pipeline:
    BGR frame  -->  HSV mask of target color  -->  morphology close to merge
    HP segments  -->  contour boundingRect filter (thin + wide = HP bar).

Color presets are empirical guesses for Honor of Kings; if they miss your
HUD theme, run with `--vision-debug-dir` and tweak `HSV_PRESETS`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class HsvRange:
    """An HSV color band. h_low > h_high means wraparound (e.g. red)."""
    h_low: int
    h_high: int
    s_low: int = 100
    v_low: int = 80

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        if self.h_low <= self.h_high:
            return cv2.inRange(
                hsv,
                (self.h_low, self.s_low, self.v_low),
                (self.h_high, 255, 255),
            )
        m1 = cv2.inRange(
            hsv, (0, self.s_low, self.v_low), (self.h_high, 255, 255))
        m2 = cv2.inRange(
            hsv, (self.h_low, self.s_low, self.v_low), (180, 255, 255))
        return cv2.bitwise_or(m1, m2)


# Empirical HSV ranges for Honor of Kings HP strips.
# Enemy champions **and enemy lane minions** share the same red bar palette (tier 0).
# Friendly minions read green (tier 2) — excluded when max_combat_tier<=1 so we
# don't chase allied waves as combat targets.
HSV_PRESETS: dict[str, HsvRange] = {
    # Red wrap — widen hue / relax S,V slightly so JPEG + distant tiny minions hit.
    "red":    HsvRange(165, 22, s_low=78, v_low=55),
    # Green / teal nameplates — training dummies sometimes read as green-ish.
    "green":  HsvRange(38, 92, s_low=85, v_low=65),
    # Yellow / gold bars.
    "yellow": HsvRange(18, 36, s_low=95, v_low=115),
    # Orange trim on some HP bars.
    "orange": HsvRange(5, 22, s_low=120, v_low=110),
    # Purple / blue-violet (enemy when you are red team).
    "purple": HsvRange(120, 158, s_low=65, v_low=65),
    # Shield / tower / ward bars.
    "cyan":   HsvRange(78, 108, s_low=85, v_low=85),
}


@dataclass
class Detection:
    """A detected horizontal bar in image (capture) coordinates."""
    x: int        # bbox center x
    y: int        # bbox center y
    w: int
    h: int
    label: str


def find_hp_bars(
    frame_bgr: np.ndarray,
    hsv_ranges,
    labels=None,
    min_width: int = 12,
    max_width: int = 400,
    # King of Glory segmented enemy bars (+ level orb) easily exceed ~14 px tall at
    # 720-class captures; Zhuang Zhou's bar hit 17 px in customer screenshots.
    max_height: int = 22,
    min_aspect: float = 2.2,
) -> List[Detection]:
    """Locate thin horizontal HP bars in the frame.

    `hsv_ranges` may be a single `HsvRange` or a list/dict of them.
    `labels` (optional) is a list of human-readable labels aligned with
    `hsv_ranges` (only used when `hsv_ranges` is a list).

    Filter rules (a typical HP bar):
        - min_width <= width <= max_width  (in capture-frame coordinates)
        - height <= max_height
        - width >= min_aspect * height     (horizontal aspect)
    """
    if isinstance(hsv_ranges, HsvRange):
        ranges = [(labels or "target", hsv_ranges)] if isinstance(
            labels, str) else [("target", hsv_ranges)]
    elif isinstance(hsv_ranges, dict):
        ranges = list(hsv_ranges.items())
    else:
        if labels is None:
            labels = [f"c{i}" for i in range(len(hsv_ranges))]
        ranges = list(zip(labels, hsv_ranges))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    rh, rw = frame_bgr.shape[:2]
    # Stretchier horizontal morph on wide ROIs heals JPEG-fractured reds.
    kw = max(9, min(rw // 45, 25))
    kh = max(2, rh // 100)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    # Slight vertical dilation reconnects anti-aliased / split HP bar pixels.
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    out: List[Detection] = []
    for label, hsv_range in ranges:
        mask = hsv_range.mask(hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)
        mask = cv2.dilate(mask, kernel_v, iterations=1)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (min_width <= w <= max_width
                    and h <= max_height
                    and w >= min_aspect * h):
                out.append(Detection(x + w // 2, y + h // 2, w, h, label))
    return out


_COLOR_BGR = {
    "red":    (0, 0, 255),
    "green":  (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "purple": (255, 0, 255),
    "cyan":   (255, 255, 0),
}


def annotate(frame_bgr: np.ndarray,
             detections: List[Detection]) -> np.ndarray:
    """Return a copy of `frame_bgr` with boxes drawn around each detection."""
    viz = frame_bgr.copy()
    for d in detections:
        x1 = d.x - d.w // 2
        y1 = d.y - d.h // 2
        color = _COLOR_BGR.get(d.label, (0, 255, 0))
        cv2.rectangle(viz, (x1, y1), (x1 + d.w, y1 + d.h), color, 2)
        cv2.putText(viz, d.label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return viz
