"""Inference WebSocket server.

Run on your server (must use -m, not direct script path):
    PYTHONPATH=. python -u -m server.server --host 0.0.0.0 --port 8765

All print output is also appended to logs/server.log under the repo root
(use --no-log-file or --log-file PATH to change).

Demo modes:
    # Mash the basic-attack button (bottom-right) every 5 frames:
    ... --demo-tap 0.92,0.85
    # Same, every 3 frames:
    ... --demo-tap 0.92,0.85@3
    # Absolute pixel coords:
    ... --demo-tap 1180,600
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Protocol

import websockets

from server.inference import (
    DEFAULT_BUTTONS,
    FixedTapDecider,
    MoveDecider,
    RotationDecider,
    TemplateMatcher,
    VisionAttackDecider,
    VisionEnemyComboDecider,
    parse_button_overrides,
    parse_demo_tap,
)
from shared.protocol import Action, Hello


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_PATH = _REPO_ROOT / "logs" / "server.log"


class _Tee:
    """Duplicate writes to several text streams (for terminal + log file)."""

    __slots__ = ("_streams",)

    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _setup_file_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(path, "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    print(f"[server] logging to {path}")


class Decider(Protocol):
    def decide(self, jpg_bytes: bytes) -> Action: ...


async def handle(ws, decider: Decider) -> None:
    peer = getattr(ws, "remote_address", "?")
    print(f"[server] client connected: {peer}")
    try:
        hello_raw = await ws.recv()
        if isinstance(hello_raw, (bytes, bytearray)):
            print("[server] expected hello text, got binary -- closing")
            return
        hello = Hello.from_json(hello_raw)
        print(f"[server] hello from {peer}: {hello}")
        if hasattr(decider, "set_screen_size"):
            decider.set_screen_size(hello.width, hello.height)

        frame_id = 0
        t_last = time.time()
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                t0 = time.time()
                action = decider.decide(bytes(msg))
                infer_ms = (time.time() - t0) * 1000
                await ws.send(action.to_json())
                frame_id += 1
                if frame_id % 30 == 0:
                    fps = 30 / (time.time() - t_last + 1e-9)
                    t_last = time.time()
                    print(f"[server] frame#{frame_id} infer={infer_ms:.1f}ms "
                          f"fps~{fps:.1f}")
            else:
                # Ignore unsolicited text frames; respond with noop to keep RTT alive.
                await ws.send(Action(type="noop").to_json())
    except websockets.ConnectionClosed:
        print(f"[server] client disconnected: {peer}")


async def main(host: str, port: int, templates_dir: Path,
               demo_tap: str | None, rotation: str | None,
               rotation_every: int,
               button_overrides: list[str],
               move: str | None, move_swipe_ms: int,
               vision_combo: str | None,
               vision_combo_skip_after_first: int,
               vision_combo_v_ratio: float,
               vision_combo_cd_consec: int,
               vision_combo_gate_timeout: int,
               vision_combo_post_gap: int,
               vision_combo_allow_max_tier: int,
               vision_combo_force_tail_on_timeout: bool,
               vision_combo_min_bar_rel: float,
               vision_combo_enemy_stable_frames: int,
               vision_attack: bool, vision_color: str,
               vision_debug_dir: str | None,
               vision_roi: str | None,
               vision_fallback_dir: str | None,
               vision_attack_range: float | None,
               vision_attack_foot_bias: float | None,
               vision_combat_max_tier: int,
               vision_tap_max_tier: int,
               vision_det_exclude_top: float,
               vision_det_exclude_bottom: float,
               vision_move_only: bool) -> None:
    if vision_move_only and not vision_combo and not vision_attack:
        print("[server] WARN: --vision-move-only only applies with "
              "--vision-combo or --vision-attack; ignoring")
        vision_move_only = False

    decider: Decider
    buttons = dict(DEFAULT_BUTTONS)
    buttons.update(parse_button_overrides(button_overrides))
    vision_kwargs: dict = {}
    if vision_roi:
        parts = [float(x) for x in vision_roi.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "--vision-roi must be 'top,left,right,bottom' fractions")
        vision_kwargs["roi"] = tuple(parts)

    if vision_combo:
        if vision_attack:
            print("[server] note: --vision-combo active; ignoring "
                  "--vision-attack for this run")
        vcombo_kwargs = dict(
            combo=vision_combo.strip(),
            buttons=buttons,
            hsv_color=vision_color,
            debug_dir=vision_debug_dir,
            max_combat_tier=vision_combat_max_tier,
            skip_frames_after_gate_skill=vision_combo_skip_after_first,
            cooldown_v_ratio=vision_combo_v_ratio,
            cooldown_consec_frames=vision_combo_cd_consec,
            gate_timeout_frames=vision_combo_gate_timeout,
            post_skill_gap_frames=vision_combo_post_gap,
            det_exclude_top_frac=vision_det_exclude_top,
            det_exclude_bottom_frac=vision_det_exclude_bottom,
            combo_allow_max_tier=vision_combo_allow_max_tier,
            fallback_direction=vision_fallback_dir,
            gate_timeout_force_tail=vision_combo_force_tail_on_timeout,
            combo_min_bar_area_rel=vision_combo_min_bar_rel,
            enemy_stable_frames=vision_combo_enemy_stable_frames,
            attacks_enabled=not vision_move_only,
            **vision_kwargs,
        )
        if vision_attack_range is not None:
            vcombo_kwargs["attack_range_rel"] = vision_attack_range
        if vision_attack_foot_bias is not None:
            vcombo_kwargs["attack_foot_bias_rel"] = vision_attack_foot_bias
        decider = VisionEnemyComboDecider(**vcombo_kwargs)
    elif vision_attack:
        kwargs = dict(vision_kwargs)
        if vision_attack_range is not None:
            kwargs["attack_range_rel"] = vision_attack_range
        if vision_attack_foot_bias is not None:
            kwargs["attack_foot_bias_rel"] = vision_attack_foot_bias
        kwargs["max_combat_tier"] = vision_combat_max_tier
        kwargs["tap_max_combat_tier"] = vision_tap_max_tier
        kwargs["det_exclude_top_frac"] = vision_det_exclude_top
        kwargs["det_exclude_bottom_frac"] = vision_det_exclude_bottom
        kwargs["attacks_enabled"] = not vision_move_only
        decider = VisionAttackDecider(
            hsv_color=vision_color,
            attack_xy=buttons["A"],
            debug_dir=vision_debug_dir,
            fallback_direction=vision_fallback_dir,
            **kwargs,
        )
    elif move:
        decider = MoveDecider(move, swipe_ms=move_swipe_ms)
    elif rotation:
        decider = RotationDecider(rotation, every_n=rotation_every,
                                  buttons=buttons)
    elif demo_tap:
        x, y, every_n = parse_demo_tap(demo_tap)
        decider = FixedTapDecider(x, y, every_n=every_n)
    else:
        decider = TemplateMatcher(templates_dir)

    async def _handler(ws):
        await handle(ws, decider)

    async with websockets.serve(_handler, host, port, max_size=4 * 1024 * 1024):
        print(f"[server] listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-file",
        default=str(_DEFAULT_LOG_PATH),
        metavar="PATH",
        help="Append stdout/stderr to this UTF-8 log file (default: "
             "<repo>/logs/server.log).",
    )
    ap.add_argument("--no-log-file", action="store_true",
                    help="Do not write a log file; terminal only.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--templates", default=str(Path(__file__).parent / "templates"))
    ap.add_argument("--demo-tap", default=None,
                    help='Fixed-tap demo. Format: "x,y" or "x,y@N". '
                         'x,y can be absolute pixels or ratios in [0,1].')
    ap.add_argument("--rotation", default=None,
                    help='Skill rotation, e.g. "1AAA2AAA3AAAAAA". '
                         'A=basic, 1/2/3=skills, B=recall, H=heal.')
    ap.add_argument("--rotation-every", type=int, default=1,
                    help="Advance rotation every N frames (default 1).")
    ap.add_argument("--button", action="append", default=[],
                    help='Override default button coord, e.g. "1=1997,1224" '
                         'or "2=0.83,0.70". Repeat the flag for multiple.')
    ap.add_argument("--move", default=None,
                    help='Continuous joystick move. One of N/S/E/W/NE/NW/SE/SW.')
    ap.add_argument("--move-swipe-ms", type=int, default=250,
                    help="Duration of each joystick swipe (default 250ms).")
    ap.add_argument("--vision-combo", default=None,
                    metavar="SEQ",
                    help='Vision combo when enemy HP bar visible, e.g. "231". '
                         "First skill fires immediately; waits for that skill "
                         "icon to dim (cooldown) before tapping the rest. "
                         "Overrides --vision-attack when both are set.")
    ap.add_argument("--vision-combo-skip-after-first", type=int, default=5,
                    help="Frames after tapping the first combo skill before "
                         "reading cooldown brightness (default 5).")
    ap.add_argument("--vision-combo-v-ratio", type=float, default=0.90,
                    help="Cooldown if ROI mean V < ref_V * this ratio "
                         "(default 0.90; lower=stricter).")
    ap.add_argument("--vision-combo-cd-consec", type=int, default=2,
                    help="Consecutive cooldown frames required before "
                         "continuing (default 2).")
    ap.add_argument("--vision-combo-gate-timeout", type=int, default=22,
                    help="Max frames in wait_gate before aborting or forcing "
                         "tail (see --vision-combo-force-tail-on-timeout; default "
                         "22 @10fps ~2.2s).")
    ap.add_argument("--vision-combo-force-tail-on-timeout", action="store_true",
                    help="If wait_gate times out without detecting icon dim, still "
                         "tap the remainder of the combo (legacy; often empty). "
                         "Default: bail to idle/chase instead.")
    ap.add_argument("--vision-combo-min-bar-rel", type=float, default=0.001,
                    metavar="FRAC",
                    help="Minimum picked HP-bar bounding-box area (w×h) as "
                         "a fraction of frame width×height before TAP ping "
                         "gate skill — filters tiny HUD false positives "
                         "(default 0.001).")
    ap.add_argument("--vision-combo-enemy-stable", type=int, default=5,
                    metavar="N",
                    dest="vision_combo_enemy_stable_frames",
                    help="Frames the melee+latch must stay valid before firing "
                         "gate skill (default 5).")
    ap.add_argument("--vision-combo-post-gap", type=int, default=4,
                    help="No-op frames between middle and last skill taps "
                         "(default 4).")
    ap.add_argument("--vision-combo-allow-tier", type=int, default=0,
                    metavar="T",
                    dest="vision_combo_allow_max_tier",
                    help="Only start the combo after gate when best bar tier "
                         "≤ T (same scale as LABEL_COMBAT_PRIORITY; default 0 = "
                         "red/purple). Yellow/orange (1) chase only unless you "
                         "raise this.")
    ap.add_argument("--vision-attack", action="store_true",
                    help="Only tap basic-attack when an HP bar is visible.")
    ap.add_argument("--vision-move-only", action="store_true",
                    help="With --vision-combo or --vision-attack: disable all "
                         "taps (no skills, no basic attack). Only chase via "
                         "joystick + optional --vision-fallback-dir when no "
                         "target.")
    ap.add_argument("--vision-color", default="red,purple",
                    help="HP bar color(s) to detect, comma-separated. "
                         "Options: red,green,yellow,orange,purple,cyan. "
                         "Default red,purple (enemy-style bars only; skips "
                         "yellow/orange to reduce HUD false combos). Training "
                         "dummies often need explicit e.g. 'red,green,yellow'.")
    ap.add_argument("--vision-debug-dir", default=None,
                    help="If set, write annotated frames here. "
                         "Open 'latest.jpg' inside to see what server sees.")
    ap.add_argument("--vision-roi", default=None,
                    help="Override detection ROI as 'top,left,right,bottom' "
                         "fractions to chop off HUD (default trims map + skills).")
    ap.add_argument("--vision-fallback-dir", default=None,
                    help="When no HP bar visible, keep walking in this "
                         "compass direction (N/S/E/W/NE/NW/SE/SW). Omit to "
                         "stand still when no target. Applies to vision-attack "
                         "and vision-combo.")
    ap.add_argument("--vision-attack-range", type=float, default=None,
                    help="When dist(frame) drops below this (vs max frame dim), "
                         "tap basic attack instead of chase (override default "
                         "~0.32).")
    ap.add_argument("--vision-attack-foot-bias", type=float, default=None,
                    help="Fraction of frame height added below HP bar centroid "
                         "when judging melee distance (override default "
                         "~0.085).")
    ap.add_argument("--vision-combat-max-tier", type=int, default=0,
                    help="Consider bars up to this LABEL_COMBAT_PRIORITY tier "
                         "for chase/picking targets (0=red/purple, 1=yellow/"
                         "orange, 2=green creeps, 3=cyan UI). Default 0.")
    ap.add_argument("--vision-tap-max-tier", type=int, default=0,
                    help="Only tap basic attack when best target tier <= this "
                         "(default 0 = red/purple only; yellow/orange can still "
                         "be chased but won't melee-AA → fewer empty swings).")
    ap.add_argument("--vision-det-exclude-top", type=float, default=0.09,
                    help="Discard HP-bar dets whose centroid y is above this "
                         "fraction of frame height (HUD). 0=off. Default 0.09.")
    ap.add_argument("--vision-det-exclude-bottom", type=float, default=0.42,
                    help="Discard dets whose centroid is in bottom this "
                         "fraction (skill/UI). 0=off. Default 0.42.")
    args = ap.parse_args()
    if not args.no_log_file:
        _setup_file_logging(Path(args.log_file).expanduser().resolve())

    asyncio.run(main(args.host, args.port, Path(args.templates), args.demo_tap,
                     args.rotation, args.rotation_every, args.button,
                     args.move, args.move_swipe_ms,
                     args.vision_combo,
                     args.vision_combo_skip_after_first,
                     args.vision_combo_v_ratio,
                     args.vision_combo_cd_consec,
                     args.vision_combo_gate_timeout,
                     args.vision_combo_post_gap,
                     args.vision_combo_allow_max_tier,
                     args.vision_combo_force_tail_on_timeout,
                     args.vision_combo_min_bar_rel,
                     args.vision_combo_enemy_stable_frames,
                     args.vision_attack, args.vision_color,
                     args.vision_debug_dir, args.vision_roi,
                     args.vision_fallback_dir,
                     args.vision_attack_range, args.vision_attack_foot_bias,
                     args.vision_combat_max_tier, args.vision_tap_max_tier,
                     args.vision_det_exclude_top, args.vision_det_exclude_bottom,
                     args.vision_move_only))
