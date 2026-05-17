"""Pluggable decision module.

Starts with OpenCV template matching: drop PNG button images into
`server/templates/*.png`, and we'll tap the center of the highest-scoring match.

Also provides `FixedTapDecider` for the "training-mode mash button" demo: it
ignores the frame contents and just emits a tap at a fixed coordinate every N
frames. Useful for proving the end-to-end loop (capture -> WS -> tap) before
any vision model exists.

To upgrade later, implement `Decider.decide(jpg_bytes) -> Action` with a
PyTorch / YOLO / RL policy. The WebSocket protocol stays unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from shared.protocol import Action

# Templates with names starting with "tap_" are tapped; others are ignored for now.
TAP_PREFIX = "tap_"
MATCH_THRESHOLD = 0.85

# When multiple-colored masks fire on the same frame, cyan often latches thin UI /
# ground trims that sit closer to screen center than the enemy health bar — we
# prefer champ/dummy palettes first when picking chase / attack centroid.
LABEL_COMBAT_PRIORITY: dict[str, int] = {
    # Enemy champions & enemy lane minions both resolve as preset "red".
    "red": 0,
    "purple": 0,
    "yellow": 1,
    "orange": 1,
    "green": 2,
    "cyan": 3,
}


class Decider(Protocol):
    def decide(self, jpg_bytes: bytes) -> Action: ...


class TemplateMatcher:
    def __init__(self, template_dir: Path):
        self.template_dir = template_dir
        self.templates: dict[str, np.ndarray] = {}
        for p in sorted(template_dir.glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[inference] WARN: cannot read {p}")
                continue
            self.templates[p.stem] = img
        if not self.templates:
            print(f"[inference] no templates in {template_dir}. Add tap_*.png files "
                  f"to enable actions; until then the bot will only watch.")
        else:
            print(f"[inference] loaded {len(self.templates)} templates: "
                  f"{list(self.templates)}")

    def decide(self, jpg_bytes: bytes) -> Action:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return Action(type="noop")

        best_name: str | None = None
        best_val: float = -1.0
        best_loc: tuple[int, int] = (0, 0)
        best_tpl_shape: tuple[int, int] = (0, 0)

        for name, tpl in self.templates.items():
            if not name.startswith(TAP_PREFIX):
                continue
            if tpl.shape[0] > frame.shape[0] or tpl.shape[1] > frame.shape[1]:
                continue
            res = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_name = name
                best_loc = max_loc
                best_tpl_shape = tpl.shape[:2]

        if best_name is None or best_val < MATCH_THRESHOLD:
            return Action(type="noop")

        cx = best_loc[0] + best_tpl_shape[1] // 2
        cy = best_loc[1] + best_tpl_shape[0] // 2
        print(f"[inference] matched {best_name} score={best_val:.3f} at ({cx},{cy})")
        return Action(type="tap", x=int(cx), y=int(cy))


class FixedTapDecider:
    """Emit `tap(x, y)` every `every_n` frames, regardless of frame content.

    Coordinates can be absolute pixels (e.g. "1180,600") or relative ratios in
    [0, 1] (e.g. "0.92,0.85"). Relative needs `set_screen_size()` to be called
    after the Hello message arrives.
    """

    def __init__(self, x: float, y: float, every_n: int = 5, duration_ms: int = 60):
        self._rel = (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0)
        self._x_raw = x
        self._y_raw = y
        self._x_px = int(x) if not self._rel else 0
        self._y_px = int(y) if not self._rel else 0
        self._every_n = max(1, every_n)
        self._duration_ms = duration_ms
        self._counter = 0
        mode = "relative" if self._rel else "absolute"
        print(f"[inference] FixedTapDecider {mode} ({x},{y}) every {every_n} frames")

    def set_screen_size(self, w: int, h: int) -> None:
        if self._rel:
            self._x_px = int(self._x_raw * w)
            self._y_px = int(self._y_raw * h)
            print(f"[inference] resolved tap target -> ({self._x_px},{self._y_px}) "
                  f"on {w}x{h}")

    def decide(self, jpg_bytes: bytes) -> Action:
        self._counter += 1
        if self._counter % self._every_n != 0:
            return Action(type="noop")
        return Action(type="tap", x=self._x_px, y=self._y_px,
                      duration_ms=self._duration_ms)


def parse_demo_tap(spec: str) -> tuple[float, float, int]:
    """Parse "x,y" or "x,y@N" into (x, y, every_n)."""
    every_n = 5
    if "@" in spec:
        spec, n_str = spec.rsplit("@", 1)
        every_n = int(n_str)
    x_str, y_str = spec.split(",")
    x = float(x_str)
    y = float(y_str)
    return x, y, every_n


# Default Honor of Kings button positions in landscape view, as ratios.
# Verified for 2560x1440: A confirmed via input tap (2355,1224). Others are
# best-guess starting points — fine-tune if your skill isn't landing.
DEFAULT_BUTTONS: dict[str, tuple[float, float]] = {
    "A": (0.920, 0.850),   # basic attack          (verified 2560x1440)
    "1": (0.780, 0.850),   # skill 1               (unverified guess)
    "2": (0.830, 0.700),   # skill 2               (unverified guess)
    "3": (0.920, 0.550),   # skill 3 (ult)         (unverified guess)
    "F": (0.665, 0.900),   # summoner spell        (verified)
    "H": (0.571, 0.892),   # heal / restore        (verified)
    "B": (0.515, 0.903),   # recall to base        (verified)
}

# Movement joystick center (bottom-left), as ratio of landscape screen.
DEFAULT_JOYSTICK: tuple[float, float] = (0.18, 0.75)

# How far to drag the joystick in screen-relative units. ~0.08 of screen width
# is well past the joystick dead zone, gives near-max walk speed.
DEFAULT_MOVE_DISTANCE_REL: float = 0.08

# Direction unit vectors in screen coords (y grows downward).
DIRECTIONS: dict[str, tuple[float, float]] = {
    "N": (0.0, -1.0),
    "S": (0.0, 1.0),
    "E": (1.0, 0.0),
    "W": (-1.0, 0.0),
    "NE": (0.7071, -0.7071),
    "NW": (-0.7071, -0.7071),
    "SE": (0.7071, 0.7071),
    "SW": (-0.7071, 0.7071),
}


def parse_button_overrides(specs: list[str]) -> dict[str, tuple[float, float]]:
    """Parse --button overrides like ["A=2355,1224", "1=0.78,0.85"]."""
    out: dict[str, tuple[float, float]] = {}
    for spec in specs:
        name, coord = spec.split("=", 1)
        x_str, y_str = coord.split(",")
        out[name.upper()] = (float(x_str), float(y_str))
    return out


def _vision_filtered_hp_dets(
    frame_bgr: np.ndarray,
    *,
    roi: tuple[float, float, float, float],
    hsv_ranges,
    color_names: list[str],
    self_exclude_x_rel: float,
    self_exclude_y_rel: float,
    self_exclude_cy_shift: float,
    max_combat_tier: int,
    find_bars,
    det_exclude_top_frac: float = 0.0,
    det_exclude_bottom_frac: float = 0.0,
) -> tuple[list, tuple[int, int, int, int], int, int]:
    """Run HP-bar detection + green self-ellipse peel + tier cap. Shared by
    vision deciders.

    Horizontal HP bars shouldn't appear in extreme top (score/KDA HUD) or
    bottom band (skill cluster). centroid ``d.y`` is checked in *full-frame*
    coordinates after ROI shift."""

    dropped_vertical = 0
    h_img, w_img = frame_bgr.shape[:2]
    top, left, right, bottom = roi
    y0 = int(h_img * top)
    x0 = int(w_img * left)
    y1 = h_img - int(h_img * bottom)
    x1 = w_img - int(w_img * right)
    roi_frame = frame_bgr[y0:y1, x0:x1]

    roi_dets = find_bars(roi_frame, hsv_ranges, labels=color_names)
    dets = []
    for d in roi_dets:
        shifted = type(d)(x=d.x + x0, y=d.y + y0,
                          w=d.w, h=d.h, label=d.label)
        dets.append(shifted)

    cx_self = w_img // 2
    cy_geom = h_img // 2
    cy_exclude = int(cy_geom + self_exclude_cy_shift * h_img)
    ex_rx = self_exclude_x_rel * w_img
    ex_ry = self_exclude_y_rel * h_img
    dropped_self = 0
    if ex_rx > 0 and ex_ry > 0:
        kept: list = []
        for d in dets:
            if (d.label or "").lower() != "green":
                kept.append(d)
                continue
            inside = ((((d.x - cx_self) / ex_rx) ** 2)
                      + (((d.y - cy_exclude) / ex_ry) ** 2) < 1.0)
            if inside:
                dropped_self += 1
                continue
            kept.append(d)
        dets = kept

    dets = [
        d for d in dets
        if LABEL_COMBAT_PRIORITY.get((d.label or "").lower(), 99)
        <= max_combat_tier
    ]

    y_top_line = (
        float(det_exclude_top_frac) * h_img if det_exclude_top_frac > 0 else None)
    y_bot_line = (
        (1.0 - float(det_exclude_bottom_frac)) * h_img
        if det_exclude_bottom_frac > 0 else None)
    if y_top_line is not None or y_bot_line is not None:
        kept_v: list = []
        for d in dets:
            if y_top_line is not None and d.y < y_top_line:
                dropped_vertical += 1
                continue
            if y_bot_line is not None and d.y > y_bot_line:
                dropped_vertical += 1
                continue
            kept_v.append(d)
        dets = kept_v

    return dets, (x0, y0, x1, y1), dropped_self, dropped_vertical


def _skill_roi_mean_v(frame_bgr: np.ndarray, w_img: int, h_img: int,
                      rx: float, ry: float, pad_rel: float = 0.048) -> float:
    """Mean V channel over a square ROI centered at skill button (ratios)."""
    cx = int(rx * w_img) if 0.0 <= rx <= 1.0 else int(rx)
    cy = int(ry * h_img) if 0.0 <= ry <= 1.0 else int(ry)
    pw = max(8, int(pad_rel * w_img))
    ph = max(8, int(pad_rel * h_img))
    x0 = max(0, cx - pw)
    x1 = min(w_img, cx + pw)
    y0 = max(0, cy - ph)
    y1 = min(h_img, cy + ph)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 128.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def pick_best_hp_det(dlist: list, cx_self: int, cy_geom: int) -> object:
    """Prefer large tier‑0/red bars, then tier‑1 yellow/orange, else closest."""
    def _tier_hp(d: object) -> int:
        return LABEL_COMBAT_PRIORITY.get((getattr(d, "label") or "").lower(),
                                         99)

    def _dist_sq(d: object) -> float:
        return float((d.x - cx_self) ** 2 + (d.y - cy_geom) ** 2)

    def _area(d: object) -> float:
        return float(max(1, d.w) * max(1, d.h))

    tier0 = [x for x in dlist if _tier_hp(x) == 0]
    if tier0:
        return max(tier0, key=lambda d: (_area(d), -_dist_sq(d)))
    tier1 = [x for x in dlist if _tier_hp(x) == 1]
    if tier1:
        return max(tier1, key=lambda d: (_area(d), -_dist_sq(d)))
    rest = sorted(dlist, key=lambda x: (_tier_hp(x), _dist_sq(x)))
    return rest[0]


def pick_steer_hp_det(
        dlist: list,
        cx_self: int,
        cy_geom: int,
        h_img: int,
        foot_bias_rel: float,
        max_combat_tier: int,
        min_dist_px: float,
) -> object | None:
    """Pick whose direction to steer toward — not always the same as ``pick_best_hp_det``.

    ``pick_best_hp_det`` biases large tier‑0 bars, which sometimes sit ambiguously
    near the HUD / screen pivot → ``dx≈dy≈0`` and joystick swipes degenerate to
    "finger on center", so the hero never walks. Prefer the nearest plausible
    target with offset ≥ ``min_dist_px``; if none, fall back to the farthest
    candidate so the swipe still pushes toward some off‑screen enemy.
    """
    cand: list[tuple[float, int, object]] = []
    for d in dlist:
        tier = LABEL_COMBAT_PRIORITY.get(
            (getattr(d, "label", None) or "").lower(), 99)
        if tier > max_combat_tier:
            continue
        _, _, dist = enemy_foot_dx_dy_dist(
            d, cx_self, cy_geom, h_img, foot_bias_rel)
        cand.append((dist, tier, d))
    if not cand:
        return None
    meaningful = [t for t in cand if t[0] >= min_dist_px]
    if meaningful:
        meaningful.sort(key=lambda t: (t[1], t[0]))
        return meaningful[0][2]
    cand.sort(key=lambda t: -t[0])
    return cand[0][2]


def enemy_foot_dx_dy_dist(best: object,
                          cx_self: int, cy_geom: int, h_img: int,
                          foot_bias_rel: float) -> tuple[float, float,
                                                         float]:
    """Screen-space vector from pivot to approximate enemy feet + distance."""
    import math

    foot_y = min(int(best.y + foot_bias_rel * h_img), h_img - 2)
    dx = float(best.x - cx_self)
    dy = float(foot_y - cy_geom)
    dist = math.hypot(dx, dy)
    return dx, dy, dist


class RotationDecider:
    """Cycle through a sequence of named buttons, one per `every_n` frames.

    Example rotation string "1AAA2AAA3AAAAAA": fire skill 1, three basic
    attacks, skill 2, three basics, ult, then six basics, then loop.
    """

    def __init__(self, rotation: str, every_n: int = 1,
                 buttons: dict[str, tuple[float, float]] | None = None,
                 duration_ms: int = 60):
        buttons = buttons or DEFAULT_BUTTONS
        self._buttons = buttons
        self._sequence = [c for c in rotation.upper() if c in buttons]
        if not self._sequence:
            raise ValueError(
                f"rotation '{rotation}' has no known button chars. "
                f"Known: {sorted(buttons)}"
            )
        self._every_n = max(1, every_n)
        self._duration_ms = duration_ms
        self._frame_counter = 0
        self._step_idx = 0
        self._w = 0
        self._h = 0
        print(f"[inference] RotationDecider sequence={self._sequence} "
              f"every {every_n} frames")

    def set_screen_size(self, w: int, h: int) -> None:
        self._w, self._h = w, h
        # Resolve each button to absolute px for logging.
        resolved = {
            name: (int(rx * w) if 0 <= rx <= 1 else int(rx),
                   int(ry * h) if 0 <= ry <= 1 else int(ry))
            for name, (rx, ry) in self._buttons.items()
        }
        print(f"[inference] resolved buttons on {w}x{h}: {resolved}")

    def decide(self, jpg_bytes: bytes) -> Action:
        self._frame_counter += 1
        if self._frame_counter % self._every_n != 0:
            return Action(type="noop")
        if self._w == 0:
            return Action(type="noop")

        name = self._sequence[self._step_idx % len(self._sequence)]
        self._step_idx += 1
        rx, ry = self._buttons[name]
        x = int(rx * self._w) if 0 <= rx <= 1 else int(rx)
        y = int(ry * self._h) if 0 <= ry <= 1 else int(ry)
        print(f"[inference] step#{self._step_idx} {name} -> ({x},{y})")
        return Action(type="tap", x=x, y=y, duration_ms=self._duration_ms)


class VisionAttackDecider:
    """See an HP bar -> chase + attack the target.

    Per frame:
        1. Decode JPEG -> BGR.
        2. Search for HP bars matching `hsv_color` (red=enemy, yellow=dummy, ...).
        3. If none found -> noop.
        4. Pick the bar closest to the screen center (hero is always centered
           in MOBA cameras).
        5. If that bar is FAR from center -> swipe joystick toward it (chase).
           If CLOSE to center -> tap basic-attack button (auto-lock kicks in).

    Annotated debug frames are written to `debug_dir/latest.jpg` every frame
    when set; periodic snapshots also go to `frame_NNNNNN.jpg`.
    """

    def __init__(self, hsv_color: str = "red",
                 attack_xy: tuple[float, float] = (0.92, 0.85),
                 chase: bool = True,
                 joystick_xy: tuple[float, float] = (0.18, 0.75),
                 joystick_distance_rel: float = 0.08,
                 # HP-bar centroids hover above enemy models → pure distance to the
                 # geometric screen midpoint under-estimates "we are in melee".
                 attack_range_rel: float = 0.32,
                 # Shift target point downward (toward champion feet/body hitbox).
                 attack_foot_bias_rel: float = 0.085,
                 swipe_ms: int = 300,
                 swipe_min_interval_ms: int = 80,
                 attack_min_interval_ms: int = 250,
                 fallback_direction: str | None = None,
                 # Own nameplate/HUD strip sits ABOVE the hero model center.
                 # Old defaults used the geometric screen midpoint with a HUGE
                 # vertical ellipse (0.45 * H) → most HP bars anywhere above /
                 # near-center were falsely dropped as \"self\". We now tighten
                 # the ellipse and shift its center upward toward the player's
                 # HP bar (~10–15% frame height higher than geometric center).
                 self_exclude_x_rel: float = 0.13,
                 self_exclude_y_rel: float = 0.10,
                 self_exclude_center_y_shift_rel: float = -0.11,
                 roi: tuple[float, float, float, float] = (0.04, 0.18, 0.15, 0.26),
                 debug_dir: "Path | None" = None,
                 debug_every_n: int = 30,
                 duration_ms: int = 60,
                 # Ignore HUD-ish palettes unless explicitly widened:
                 # LABEL_COMBAT_PRIORITY tier 0–1 = red/purple/yellow/orange,
                 # tier 2 = green (minions/allies), tier 3 = cyan (shields/UI).
                 max_combat_tier: int = 1,
                 # Only tap basic attack when best target tier <= this.
                 # Default 0 = red/purple (enemy heroes & enemy red lane bars).
                 # Yellow/orange (jungles / lots of HUD false positives) still get
                 # chased but will NOT spam AA in melee → fixes \"empty swings\".
                 tap_max_combat_tier: int = 0,
                 # Drop centroid y inside top HUD / bottom skill band (false HP
                 # bars on scoreboard vs skill trims). Pass 0.0 to disable.
                 det_exclude_top_frac: float = 0.09,
                 det_exclude_bottom_frac: float = 0.42):
        from server.vision import HSV_PRESETS, annotate, find_hp_bars

        color_names = [c.strip().lower() for c in hsv_color.split(",")
                       if c.strip()]
        unknown = [c for c in color_names if c not in HSV_PRESETS]
        if unknown:
            raise ValueError(
                f"unknown color(s) {unknown}. "
                f"Known: {list(HSV_PRESETS)}")
        if not color_names:
            color_names = ["red"]
        self._hsv_ranges = [HSV_PRESETS[c] for c in color_names]
        self._color_names = color_names
        self._color_label = ",".join(color_names)
        self._attack_xy = attack_xy
        self._duration_ms = duration_ms
        # Anything inside an axis-aligned ellipse around the hero (always at
        # screen center in MOBA cameras) is almost certainly the hero's own
        # HP bar -- filter those out so we don't try to "attack ourselves".
        self._self_exclude_x_rel = self_exclude_x_rel
        self._self_exclude_y_rel = self_exclude_y_rel
        self._self_exclude_cy_shift = self_exclude_center_y_shift_rel

        self._chase = chase
        self._joystick_xy = joystick_xy
        self._joystick_dist_rel = joystick_distance_rel
        self._attack_range_rel = attack_range_rel
        self._foot_bias_rel = attack_foot_bias_rel
        self._swipe_ms = swipe_ms
        # With the sendevent control backend, every swipe is interpreted as
        # "move the persistent joystick finger to (x2,y2)", so we *want* to
        # reissue often to track moving targets. The lower bound below just
        # avoids spamming if the source fps is very high.
        self._swipe_reissue_after_ms = swipe_min_interval_ms
        self._attack_min_interval_ms = attack_min_interval_ms
        self._next_swipe_allowed_ms: float = 0.0
        self._next_attack_allowed_ms: float = 0.0

        self._fallback_dir: tuple[float, float] | None = None
        if fallback_direction:
            key = fallback_direction.upper()
            if key not in DIRECTIONS:
                raise ValueError(
                    f"unknown fallback direction '{fallback_direction}'. "
                    f"Use one of {sorted(DIRECTIONS)}"
                )
            self._fallback_dir = DIRECTIONS[key]
            self._fallback_label = key
        else:
            self._fallback_label = "none"
        # (top, left, right, bottom) fractions to chop off before detection
        # Honor of Kings landscape HUD: top score bar, bottom skills,
        # left minimap, right skill cluster.
        self._roi = roi

        from pathlib import Path
        self._debug_dir = Path(debug_dir) if debug_dir else None
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_every = max(1, debug_every_n)

        self._frame_id = 0
        self._w = 0
        self._h = 0
        if max_combat_tier < 0 or max_combat_tier > 3:
            raise ValueError("max_combat_tier must be in [0, 3]")
        self._max_combat_tier = max_combat_tier
        if tap_max_combat_tier < 0 or tap_max_combat_tier > max_combat_tier:
            raise ValueError(
                "tap_max_combat_tier must satisfy 0 <= tap_max_combat_tier "
                "<= max_combat_tier")
        self._tap_max_tier = tap_max_combat_tier
        for name, fv in (
                ("det_exclude_top_frac", det_exclude_top_frac),
                ("det_exclude_bottom_frac", det_exclude_bottom_frac),
        ):
            if fv < 0.0 or fv > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._det_ex_top = det_exclude_top_frac
        self._det_ex_bot = det_exclude_bottom_frac
        self._find_bars = find_hp_bars
        self._annotate = annotate

        print(f"[vision] VisionAttackDecider colors={self._color_label} "
              f"max_combat_tier={max_combat_tier} tap_max_tier="
              f"{tap_max_combat_tier} "
              f"chase={chase} attack_xy={attack_xy} "
              f"attack_range_rel={attack_range_rel} foot_bias_rel="
              f"{attack_foot_bias_rel} "
              f"self_exclude=({self_exclude_x_rel}x{self_exclude_y_rel}), "
              f"cy_shift={self_exclude_center_y_shift_rel:+g} fallback_dir="
              f"{self._fallback_label} det_y_exclude_top="
              f"{det_exclude_top_frac} bot={det_exclude_bottom_frac}")

    def set_screen_size(self, w: int, h: int) -> None:
        self._w, self._h = w, h
        print(f"[vision] device size {w}x{h}")

    def decide(self, jpg_bytes: bytes) -> Action:
        import math
        import time
        self._frame_id += 1
        now_ms = time.time() * 1000.0
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return Action(type="noop")

        h_img, w_img = frame.shape[:2]
        cx_self = w_img // 2
        cy_geom = h_img // 2
        cy_exclude = int(cy_geom + self._self_exclude_cy_shift * h_img)
        ex_rx = self._self_exclude_x_rel * w_img
        ex_ry = self._self_exclude_y_rel * h_img

        dets, (x0, y0, x1, y1), dropped_self, dropped_vert = (
            _vision_filtered_hp_dets(
                frame,
                roi=self._roi,
                hsv_ranges=self._hsv_ranges,
                color_names=self._color_names,
                self_exclude_x_rel=self._self_exclude_x_rel,
                self_exclude_y_rel=self._self_exclude_y_rel,
                self_exclude_cy_shift=self._self_exclude_cy_shift,
                max_combat_tier=self._max_combat_tier,
                find_bars=self._find_bars,
                det_exclude_top_frac=self._det_ex_top,
                det_exclude_bottom_frac=self._det_ex_bot,
            ))

        if self._debug_dir:
            viz = self._annotate(frame, dets) if dets else frame.copy()
            cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if ex_rx > 0 and ex_ry > 0:
                cv2.ellipse(viz, (cx_self, cy_exclude),
                            (int(ex_rx), int(ex_ry)), 0, 0, 360,
                            (255, 0, 0), 2)
            cv2.imwrite(str(self._debug_dir / "latest.jpg"), viz)
            if self._frame_id % self._debug_every == 0:
                cv2.imwrite(
                    str(self._debug_dir / f"frame_{self._frame_id:06d}.jpg"),
                    viz,
                )

        if self._frame_id % 30 == 0:
            counts: dict[str, int] = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            msg = f"[vision] frame#{self._frame_id}: dets={counts}"
            extras: list[str] = []
            if dropped_self:
                extras.append(f"self_roi_green={dropped_self}")
            if dropped_vert:
                extras.append(f"top_bot_band={dropped_vert}")
            if extras:
                msg += f" ({', '.join(extras)})"
            print(msg)

        if not dets:
            if self._fallback_dir is None:
                if self._frame_id % 30 == 0:
                    print(f"[vision] frame#{self._frame_id}: no target")
                return Action(type="noop")
            if now_ms < self._next_swipe_allowed_ms:
                return Action(type="noop")
            ux, uy = self._fallback_dir
            jx, jy = self._joystick_xy
            jcx = int(jx * self._w)
            jcy = int(jy * self._h)
            deflection = int(self._joystick_dist_rel * self._w)
            ex = int(jcx + ux * deflection)
            ey = int(jcy + uy * deflection)
            self._next_swipe_allowed_ms = now_ms + self._swipe_reissue_after_ms
            if self._frame_id % 30 == 0:
                print(f"[vision] frame#{self._frame_id}: no target -> "
                      f"WANDER {self._fallback_label} "
                      f"swipe({jcx},{jcy})->({ex},{ey}) hold={self._swipe_ms}ms")
            return Action(type="swipe", x=jcx, y=jcy, x2=ex, y2=ey,
                          duration_ms=self._swipe_ms)

        cx, cy = cx_self, cy_geom

        best = pick_best_hp_det(dets, cx_self, cy_geom)
        # Distance from pivot to approximate enemy *body/feet*: HP bar centroid
        # sits high on screen → shift down toward where combat range actually is.
        foot_y = min(
            int(best.y + self._foot_bias_rel * h_img),
            h_img - 2,
        )
        dx = best.x - cx_self
        dy = foot_y - cy_geom
        dist = math.hypot(dx, dy)
        dist_rel = dist / max(w_img, h_img)

        tier_best = LABEL_COMBAT_PRIORITY.get((best.label or "").lower(), 99)
        in_melee = dist_rel < self._attack_range_rel
        allow_tap = tier_best <= self._tap_max_tier

        if not self._chase:
            if in_melee and allow_tap:
                if now_ms < self._next_attack_allowed_ms:
                    return Action(type="noop")
                self._next_attack_allowed_ms = (
                    now_ms + self._attack_min_interval_ms)
                print(f"[vision] frame#{self._frame_id}: target at "
                      f"({best.x},{best.y}) foot_adj_y={foot_y} "
                      f"dist_rel={dist_rel:.2f} tier={tier_best} -> ATTACK")
                ax, ay = self._attack_xy
                x = int(ax * self._w) if 0 <= ax <= 1 else int(ax)
                y = int(ay * self._h) if 0 <= ay <= 1 else int(ay)
                return Action(type="tap", x=x, y=y,
                              duration_ms=self._duration_ms)
            return Action(type="noop")

        if in_melee and allow_tap:
            if now_ms < self._next_attack_allowed_ms:
                return Action(type="noop")
            self._next_attack_allowed_ms = now_ms + self._attack_min_interval_ms
            print(f"[vision] frame#{self._frame_id}: target at "
                  f"({best.x},{best.y}) foot_adj_y={foot_y} "
                  f"dist_rel={dist_rel:.2f} tier={tier_best} -> ATTACK")
            ax, ay = self._attack_xy
            x = int(ax * self._w) if 0 <= ax <= 1 else int(ax)
            y = int(ay * self._h) if 0 <= ay <= 1 else int(ay)
            return Action(type="tap", x=x, y=y,
                          duration_ms=self._duration_ms)

        # Close enough geometrically but palette is yellow/orange/etc. → no AA.
        if in_melee and self._chase and not allow_tap:
            if self._frame_id % 60 == 0:
                print(f"[vision] frame#{self._frame_id}: melee dist_rel="
                      f"{dist_rel:.2f} but tier={tier_best} > tap_max="
                      f"{self._tap_max_tier} -> HOLD (no empty AA)")

        if dist_rel >= self._attack_range_rel:
            if now_ms < self._next_swipe_allowed_ms:
                return Action(type="noop")

            ux = dx / max(dist, 1e-6)
            uy = dy / max(dist, 1e-6)
            jx, jy = self._joystick_xy
            jcx = int(jx * self._w)
            jcy = int(jy * self._h)
            deflection = int(self._joystick_dist_rel * self._w)
            ex = int(jcx + ux * deflection)
            ey = int(jcy + uy * deflection)
            self._next_swipe_allowed_ms = now_ms + self._swipe_reissue_after_ms
            print(f"[vision] frame#{self._frame_id}: target at "
                  f"({best.x},{best.y}) foot_adj_y={foot_y} dist_rel="
                  f"{dist_rel:.2f} tier={tier_best} -> CHASE "
                  f"swipe({jcx},{jcy})->({ex},{ey}) hold={self._swipe_ms}ms")
            return Action(type="swipe", x=jcx, y=jcy, x2=ex, y2=ey,
                          duration_ms=self._swipe_ms)

        return Action(type="noop")


class VisionEnemyComboDecider:
    """Fire a skill sequence when enemy HP bars are visible.

    The *first* skill in the combo string is tapped immediately. The decider
    then watches that skill's on-screen icon ROI: when brightness drops (cooldown
    sweep / dim), it assumes the cast went out and taps the remaining skills in
    order with short gaps between them.

    This is heuristic — tune ROI via ``--button 2=...`` if detection mis-fires.
    """

    def __init__(
            self,
            combo: str,
            buttons: dict[str, tuple[float, float]],
            hsv_color: str = "red",
            self_exclude_x_rel: float = 0.13,
            self_exclude_y_rel: float = 0.10,
            self_exclude_center_y_shift_rel: float = -0.11,
            roi: tuple[float, float, float, float] = (
                0.04, 0.18, 0.15, 0.26),
            debug_dir: "Path | None" = None,
            debug_every_n: int = 30,
            max_combat_tier: int = 1,
            gate_skill_roi_pad_rel: float = 0.048,
            skip_frames_after_gate_skill: int = 5,
            cooldown_v_ratio: float = 0.90,
            cooldown_abs_drop: float = 18.0,
            cooldown_consec_frames: int = 2,
            gate_timeout_frames: int = 22,
            post_skill_gap_frames: int = 4,
            refractory_frames: int = 40,
            enemy_stable_frames: int = 2,
            duration_ms: int = 60,
            det_exclude_top_frac: float = 0.09,
            det_exclude_bottom_frac: float = 0.42,
            chase_before_combo: bool = True,
            combo_allow_max_tier: int = 0,
            attack_range_rel: float = 0.26,
            attack_foot_bias_rel: float = 0.085,
            # Below this pivot→foot distance (÷ max frame dim) we never START the
            # combo circle — steer first. Dodges phantom “on‑HUD” centroid picks that
            # yield zero joystick deflection while still looking “close”.
            min_steer_dist_rel: float = 0.07,
            joystick_xy: tuple[float, float] = (0.18, 0.75),
            joystick_distance_rel: float = 0.12,
            swipe_ms: int = 300,
            swipe_min_interval_ms: int = 80,
            fallback_direction: str | None = None,
            # When wait_gate exceeds gate_timeout_frames without seeing icon dim,
            # older builds always TAP the rest of the combo — that produces many
            # empty casts when the gate skill never left cooldown visually.
            gate_timeout_force_tail: bool = False,
    ):
        from pathlib import Path

        from server.vision import HSV_PRESETS, annotate, find_hp_bars

        color_names = [c.strip().lower() for c in hsv_color.split(",")
                       if c.strip()]
        unknown = [c for c in color_names if c not in HSV_PRESETS]
        if unknown:
            raise ValueError(
                f"unknown color(s) {unknown}. Known: {list(HSV_PRESETS)}")
        if not color_names:
            color_names = ["red"]

        seq = [c for c in combo.strip() if c in buttons]
        if len(seq) < 2:
            raise ValueError(
                f"combo {combo!r} needs at least 2 skill keys present in "
                f"buttons (got {seq}). Example: \"231\"")

        self._combo = seq
        self._buttons = buttons
        self._hsv_ranges = [HSV_PRESETS[c] for c in color_names]
        self._color_names = color_names
        self._color_label = ",".join(color_names)
        self._self_exclude_x_rel = self_exclude_x_rel
        self._self_exclude_y_rel = self_exclude_y_rel
        self._self_exclude_cy_shift = self_exclude_center_y_shift_rel
        self._roi = roi
        self._debug_dir = Path(debug_dir) if debug_dir else None
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)
        self._debug_every = max(1, debug_every_n)
        if max_combat_tier < 0 or max_combat_tier > 3:
            raise ValueError("max_combat_tier must be in [0, 3]")
        self._max_combat_tier = max_combat_tier
        self._find_bars = find_hp_bars
        self._annotate = annotate

        self._gate_pad = gate_skill_roi_pad_rel
        self._skip_after_gate = max(0, skip_frames_after_gate_skill)
        self._cd_v_ratio = cooldown_v_ratio
        self._cd_abs_drop = cooldown_abs_drop
        self._cd_consec = max(1, cooldown_consec_frames)
        self._gate_timeout = max(1, gate_timeout_frames)
        self._post_gap = max(0, post_skill_gap_frames)
        self._refrac = max(0, refractory_frames)
        self._enemy_stable = max(1, enemy_stable_frames)
        self._duration_ms = duration_ms
        for name, fv in (
                ("det_exclude_top_frac", det_exclude_top_frac),
                ("det_exclude_bottom_frac", det_exclude_bottom_frac),
        ):
            if fv < 0.0 or fv > 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        self._det_ex_top = det_exclude_top_frac
        self._det_ex_bot = det_exclude_bottom_frac
        ms = float(min_steer_dist_rel)
        if ms < 0.0:
            raise ValueError("min_steer_dist_rel cannot be negative")
        if ms >= attack_range_rel:
            ms = attack_range_rel * 0.5
            print(f"[vision-combo] WARN: min_steer_dist_rel capped to "
                  f"{ms:.4f} (must stay < attack_range_rel={attack_range_rel})")

        if combo_allow_max_tier > max_combat_tier:
            raise ValueError("combo_allow_max_tier cannot exceed "
                             "max_combat_tier")
        self._combo_allow_max_tier = combo_allow_max_tier
        self._chase_before_combo = chase_before_combo
        self._attack_range_rel = attack_range_rel
        self._min_steer_dist_rel = ms
        self._foot_bias_rel = attack_foot_bias_rel
        self._joystick_xy = joystick_xy
        self._joystick_dist_rel = joystick_distance_rel
        self._swipe_ms = swipe_ms
        self._swipe_reissue_after_ms = swipe_min_interval_ms
        self._next_swipe_allowed_ms = 0.0
        self._fallback_dir: tuple[float, float] | None = None
        if fallback_direction:
            key = fallback_direction.upper()
            if key not in DIRECTIONS:
                raise ValueError(
                    f"unknown fallback direction '{fallback_direction}'. "
                    f"Use one of {sorted(DIRECTIONS)}"
                )
            self._fallback_dir = DIRECTIONS[key]
            self._fallback_label = key
        else:
            self._fallback_label = "none"

        self._frame_id = 0
        self._w = 0
        self._h = 0
        self._phase = "idle"
        self._enemy_streak = 0
        self._wait_frames = 0
        self._cd_streak = 0
        self._gap_left = 0
        self._refrac_left = 0
        self._gate_ref_v = 0.0
        self._next_idx = 0
        self._gate_timeout_force_tail = gate_timeout_force_tail

        print(f"[vision-combo] VisionEnemyComboDecider combo={''.join(seq)} "
              f"(gate skill {seq[0]!r}) chase={chase_before_combo} "
              f"range<{attack_range_rel} steer≥{ms} "
              f"combo_allow_tier≤{combo_allow_max_tier}"
              f" colors={self._color_label} max_combat_tier={max_combat_tier} "
              f"skip_gate={self._skip_after_gate} gate_timeout="
              f"{gate_timeout_frames} force_tail_on_timeout="
              f"{gate_timeout_force_tail} fallback={self._fallback_label}")

    def set_screen_size(self, w: int, h: int) -> None:
        self._w, self._h = w, h
        print(f"[vision-combo] device size {w}x{h}")

    def _tap(self, skill: str) -> Action:
        rx, ry = self._buttons[skill]
        x = int(rx * self._w) if 0.0 <= rx <= 1.0 else int(rx)
        y = int(ry * self._h) if 0.0 <= ry <= 1.0 else int(ry)
        return Action(type="tap", x=x, y=y, duration_ms=self._duration_ms)

    def _wander_swipe_if_any(self, now_ms: float) -> Action:
        if self._fallback_dir is None:
            return Action(type="noop")
        if now_ms < self._next_swipe_allowed_ms:
            return Action(type="noop")
        ux, uy = self._fallback_dir
        jx, jy = self._joystick_xy
        jcx = int(jx * self._w)
        jcy = int(jy * self._h)
        deflection = int(self._joystick_dist_rel * self._w)
        ex = int(jcx + ux * deflection)
        ey = int(jcy + uy * deflection)
        self._next_swipe_allowed_ms = now_ms + self._swipe_reissue_after_ms
        return Action(type="swipe", x=jcx, y=jcy, x2=ex, y2=ey,
                      duration_ms=self._swipe_ms)

    def _chase_best_swipe(self, dets: list, h_img: int, w_img: int,
                          cx_self: int, cy_geom: int,
                          now_ms: float,
                          *,
                          silent: bool = False) -> Action:
        if not self._chase_before_combo or not dets:
            return Action(type="noop")
        wref = float(max(w_img, h_img))
        min_px = self._min_steer_dist_rel * wref
        best = pick_steer_hp_det(
            dets, cx_self, cy_geom, h_img, self._foot_bias_rel,
            self._max_combat_tier, min_px)
        if best is None:
            return Action(type="noop")
        dx, dy, dist = enemy_foot_dx_dy_dist(
            best, cx_self, cy_geom, h_img, self._foot_bias_rel)
        ux = dx / max(dist, 1e-6)
        uy = dy / max(dist, 1e-6)
        jx, jy = self._joystick_xy
        jcx = int(jx * self._w)
        jcy = int(jy * self._h)
        deflection = int(self._joystick_dist_rel * self._w)
        ex = int(jcx + ux * deflection)
        ey = int(jcy + uy * deflection)
        if not silent and self._frame_id % 45 == 0:
            tier_b = LABEL_COMBAT_PRIORITY.get(
                (best.label or "").lower(), 99)
            print(f"[vision-combo] frame#{self._frame_id}: CHASE toward "
                  f"({best.x},{best.y}) tier={tier_b}")
        return Action(type="swipe", x=jcx, y=jcy, x2=ex, y2=ey,
                      duration_ms=self._swipe_ms)

    def decide(self, jpg_bytes: bytes) -> Action:
        import time as _time_mod

        self._frame_id += 1
        now_ms = _time_mod.time() * 1000.0
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return Action(type="noop")
        if self._w == 0:
            return Action(type="noop")

        h_img, w_img = frame.shape[:2]
        cx_self = w_img // 2
        cy_geom = h_img // 2
        cy_exclude = int(cy_geom + self._self_exclude_cy_shift * h_img)
        ex_rx = self._self_exclude_x_rel * w_img
        ex_ry = self._self_exclude_y_rel * h_img

        dets, (x0, y0, x1, y1), dropped_self, dropped_vert = (
            _vision_filtered_hp_dets(
                frame,
                roi=self._roi,
                hsv_ranges=self._hsv_ranges,
                color_names=self._color_names,
                self_exclude_x_rel=self._self_exclude_x_rel,
                self_exclude_y_rel=self._self_exclude_y_rel,
                self_exclude_cy_shift=self._self_exclude_cy_shift,
                max_combat_tier=self._max_combat_tier,
                find_bars=self._find_bars,
                det_exclude_top_frac=self._det_ex_top,
                det_exclude_bottom_frac=self._det_ex_bot,
            ))

        if self._debug_dir:
            viz = self._annotate(frame, dets) if dets else frame.copy()
            cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 255), 2)
            if ex_rx > 0 and ex_ry > 0:
                cv2.ellipse(viz, (cx_self, cy_exclude),
                            (int(ex_rx), int(ex_ry)), 0, 0, 360,
                            (255, 0, 0), 2)
            cv2.putText(
                viz, f"combo:{self._phase}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imwrite(str(self._debug_dir / "latest.jpg"), viz)
            if self._frame_id % self._debug_every == 0:
                cv2.imwrite(
                    str(self._debug_dir / f"frame_{self._frame_id:06d}.jpg"),
                    viz,
                )

        enemy_here = len(dets) > 0

        if self._frame_id % 30 == 0:
            counts: dict[str, int] = {}
            for d in dets:
                counts[d.label] = counts.get(d.label, 0) + 1
            msg = (f"[vision-combo] frame#{self._frame_id}: phase="
                   f"{self._phase} dets={counts}")
            extras2: list[str] = []
            if dropped_self:
                extras2.append(f"self_roi_green={dropped_self}")
            if dropped_vert:
                extras2.append(f"top_bot_band={dropped_vert}")
            if extras2:
                msg += f" ({', '.join(extras2)})"
            print(msg)

        gate_skill = self._combo[0]
        grx, gry = self._buttons[gate_skill]

        def gate_roi_v() -> float:
            return _skill_roi_mean_v(
                frame, w_img, h_img, grx, gry, pad_rel=self._gate_pad)

        if self._phase == "refractory":
            if self._refrac_left > 0:
                self._refrac_left -= 1
            if self._refrac_left <= 0:
                self._phase = "idle"
                self._enemy_streak = 0
            move = Action(type="noop")
            if enemy_here:
                move = self._chase_best_swipe(
                    dets, h_img, w_img, cx_self, cy_geom, now_ms,
                    silent=True)
            elif move.type == "noop":
                move = self._wander_swipe_if_any(now_ms)
            return move

        if self._phase == "idle":
            if not enemy_here:
                self._enemy_streak = 0
                return self._wander_swipe_if_any(now_ms)

            best = pick_best_hp_det(dets, cx_self, cy_geom)
            tier_best = LABEL_COMBAT_PRIORITY.get(
                (best.label or "").lower(), 99)
            _, _, dist = enemy_foot_dx_dy_dist(
                best, cx_self, cy_geom, h_img, self._foot_bias_rel)
            dist_rel = dist / float(max(w_img, h_img))
            far = dist_rel >= self._attack_range_rel
            tier_block = tier_best > self._combo_allow_max_tier
            too_centered = dist_rel < self._min_steer_dist_rel
            if far or tier_block or too_centered:
                self._enemy_streak = 0
                return self._chase_best_swipe(
                    dets, h_img, w_img, cx_self, cy_geom, now_ms)

            self._enemy_streak += 1
            if self._enemy_streak < self._enemy_stable:
                # Keep drifting toward lock range instead of emitting noop stalls
                # (looks like «only skills» when TAP 2 soon follows).
                return self._chase_best_swipe(
                    dets, h_img, w_img, cx_self, cy_geom, now_ms,
                    silent=True)

            self._gate_ref_v = gate_roi_v()
            self._phase = "wait_gate"
            self._wait_frames = 0
            self._cd_streak = 0
            tb = tier_best
            print(f"[vision-combo] frame#{self._frame_id}: melee+combo OK "
                  f"tier={tb} -> TAP {gate_skill} "
                  f"ref_V={self._gate_ref_v:.1f}")
            return self._tap(gate_skill)

        if self._phase == "wait_gate":
            self._wait_frames += 1
            if not enemy_here:
                print(f"[vision-combo] frame#{self._frame_id}: lost enemy "
                      f"during gate -> abort")
                self._phase = "idle"
                self._enemy_streak = 0
                self._wait_frames = 0
                self._cd_streak = 0
                return self._wander_swipe_if_any(now_ms)

            chase_move = self._chase_best_swipe(
                dets, h_img, w_img, cx_self, cy_geom, now_ms, silent=True)
            best = pick_best_hp_det(dets, cx_self, cy_geom)
            tier_best = LABEL_COMBAT_PRIORITY.get(
                (best.label or "").lower(), 99)
            _, _, dist_w = enemy_foot_dx_dy_dist(
                best, cx_self, cy_geom, h_img, self._foot_bias_rel)
            dist_rel_w = dist_w / float(max(w_img, h_img))
            if tier_best > self._combo_allow_max_tier:
                print(f"[vision-combo] frame#{self._frame_id}: tier={tier_best} "
                      f"> allow {self._combo_allow_max_tier} -> abort gate")
                self._phase = "idle"
                self._enemy_streak = 0
                self._wait_frames = 0
                self._cd_streak = 0
                return chase_move
            if dist_rel_w >= self._attack_range_rel:
                self._phase = "idle"
                self._enemy_streak = 0
                self._wait_frames = 0
                self._cd_streak = 0
                return chase_move

            cur_v = gate_roi_v()
            if self._wait_frames <= self._skip_after_gate:
                return chase_move

            if (cur_v < self._gate_ref_v * self._cd_v_ratio
                    or cur_v < self._gate_ref_v - self._cd_abs_drop):
                self._cd_streak += 1
            else:
                self._cd_streak = 0

            timed_out = self._wait_frames > self._gate_timeout
            cd_ok = self._cd_streak >= self._cd_consec

            if cd_ok:
                print(f"[vision-combo] frame#{self._frame_id}: gate skill "
                      f"{gate_skill} cooldown seen (V={cur_v:.1f}) -> tail")
                self._next_idx = 1
                self._phase = (
                    "gap" if self._next_idx < len(self._combo) - 1 else
                    "fire_last")
                self._gap_left = self._post_gap
                sk = self._combo[self._next_idx]
                print(f"[vision-combo] frame#{self._frame_id}: TAP {sk}")
                return self._tap(sk)
            if timed_out:
                if self._gate_timeout_force_tail:
                    print(f"[vision-combo] frame#{self._frame_id}: gate "
                          f"timeout (V now {cur_v:.1f}) -> force tail")
                    self._next_idx = 1
                    self._phase = (
                        "gap" if self._next_idx < len(self._combo) - 1 else
                        "fire_last")
                    self._gap_left = self._post_gap
                    sk = self._combo[self._next_idx]
                    print(f"[vision-combo] frame#{self._frame_id}: TAP {sk}")
                    return self._tap(sk)
                print(f"[vision-combo] frame#{self._frame_id}: gate timeout "
                      f"(V={cur_v:.1f}) -> bail (enable "
                      f"--vision-combo-force-tail-on-timeout to restore old)")
                self._phase = "idle"
                self._enemy_streak = 0
                self._wait_frames = 0
                self._cd_streak = 0
                return chase_move

            return chase_move

        if self._phase == "gap":
            gm = Action(type="noop")
            if enemy_here:
                gm = self._chase_best_swipe(
                    dets, h_img, w_img, cx_self, cy_geom, now_ms, silent=True)
            else:
                gm = self._wander_swipe_if_any(now_ms)
            if self._gap_left > 0:
                self._gap_left -= 1
                return gm
            self._next_idx += 1
            if self._next_idx >= len(self._combo):
                self._phase = "refractory"
                self._refrac_left = self._refrac
                return gm
            sk = self._combo[self._next_idx]
            if self._next_idx == len(self._combo) - 1:
                self._phase = "fire_last"
            else:
                self._phase = "gap"
                self._gap_left = self._post_gap
            print(f"[vision-combo] frame#{self._frame_id}: TAP {sk}")
            return self._tap(sk)

        if self._phase == "fire_last":
            self._phase = "refractory"
            self._refrac_left = self._refrac
            return Action(type="noop")

        return Action(type="noop")


class MoveDecider:
    """Continuously drag the joystick toward a fixed compass direction.

    Each `decide()` returns a swipe from joystick center to a point offset in
    the direction. Re-fired every frame so the joystick stays held -> the hero
    walks. Direction is one of N/S/E/W/NE/NW/SE/SW.
    """

    def __init__(self, direction: str, swipe_ms: int = 250,
                 joystick: tuple[float, float] = DEFAULT_JOYSTICK,
                 distance_rel: float = DEFAULT_MOVE_DISTANCE_REL):
        d = direction.upper()
        if d not in DIRECTIONS:
            raise ValueError(f"unknown direction '{direction}'. "
                             f"Use one of {sorted(DIRECTIONS)}")
        self._direction = d
        self._dx_unit, self._dy_unit = DIRECTIONS[d]
        self._swipe_ms = swipe_ms
        self._joystick = joystick
        self._distance_rel = distance_rel
        self._w = 0
        self._h = 0
        print(f"[inference] MoveDecider direction={d} swipe_ms={swipe_ms}")

    def set_screen_size(self, w: int, h: int) -> None:
        self._w, self._h = w, h
        jx, jy = self._joystick
        cx = int(jx * w)
        cy = int(jy * h)
        dist = int(self._distance_rel * w)
        ex = int(cx + self._dx_unit * dist)
        ey = int(cy + self._dy_unit * dist)
        print(f"[inference] joystick center=({cx},{cy}) -> drag to ({ex},{ey})")

    def decide(self, jpg_bytes: bytes) -> Action:
        if self._w == 0:
            return Action(type="noop")
        jx, jy = self._joystick
        cx = int(jx * self._w)
        cy = int(jy * self._h)
        dist = int(self._distance_rel * self._w)
        ex = int(cx + self._dx_unit * dist)
        ey = int(cy + self._dy_unit * dist)
        return Action(type="swipe", x=cx, y=cy, x2=ex, y2=ey,
                      duration_ms=self._swipe_ms)
