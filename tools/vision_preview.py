#!/usr/bin/env python3
"""Run the same HP-bar vision + filters as the server on a still image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from server.inference import _vision_filtered_hp_dets
from server.vision import HSV_PRESETS, annotate, find_hp_bars


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw HP-bar detections on an image (server-equivalent ROI).")
    ap.add_argument("image", type=Path, help="Input screenshot (BGR from file)")
    ap.add_argument(
        "-o", "--output", type=Path,
        default=Path("_vision_preview.jpg"),
        help="Where to save annotated JPG")
    ap.add_argument(
        "--vision-color", default="red,purple",
        help="Comma colors, same as server --vision-color")
    ap.add_argument(
        "--vision-roi", default="0.04,0.18,0.15,0.26",
        help="top,left,right,bottom fractions")
    ap.add_argument("--vision-combat-max-tier", type=int, default=0)
    ap.add_argument("--vision-det-exclude-top", type=float, default=0.09)
    ap.add_argument("--vision-det-exclude-bottom", type=float, default=0.42)
    args = ap.parse_args()

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"cannot read image: {args.image}")

    color_names = [c.strip().lower() for c in args.vision_color.split(",")
                   if c.strip()]
    unknown = [c for c in color_names if c not in HSV_PRESETS]
    if unknown:
        raise SystemExit(f"unknown colors: {unknown}")
    if not color_names:
        color_names = ["red"]

    roi_parts = [float(x) for x in args.vision_roi.split(",")]
    if len(roi_parts) != 4:
        raise SystemExit("--vision-roi needs 4 fractions")
    roi = tuple(roi_parts)  # type: ignore[assignment]

    h_img, w_img = frame.shape[:2]
    cx_self = w_img // 2
    cy_geom = h_img // 2
    self_exclude_cy_shift = -0.11
    cy_exclude = int(cy_geom + self_exclude_cy_shift * h_img)
    ex_rx = 0.13 * w_img
    ex_ry = 0.10 * h_img

    hsv_ranges = [HSV_PRESETS[c] for c in color_names]
    dets, (x0, y0, x1, y1), dropped_self, dropped_vert = (
        _vision_filtered_hp_dets(
            frame,
            roi=roi,
            hsv_ranges=hsv_ranges,
            color_names=color_names,
            self_exclude_x_rel=0.13,
            self_exclude_y_rel=0.10,
            self_exclude_cy_shift=self_exclude_cy_shift,
            max_combat_tier=args.vision_combat_max_tier,
            find_bars=find_hp_bars,
            det_exclude_top_frac=args.vision_det_exclude_top,
            det_exclude_bottom_frac=args.vision_det_exclude_bottom,
        ))

    viz = annotate(frame, dets) if dets else frame.copy()
    cv2.rectangle(viz, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if ex_rx > 0 and ex_ry > 0:
        cv2.ellipse(viz, (cx_self, cy_exclude),
                    (int(ex_rx), int(ex_ry)), 0, 0, 360, (255, 0, 0), 2)
    if args.vision_det_exclude_top > 0:
        yl = int(args.vision_det_exclude_top * h_img)
        cv2.line(viz, (0, yl), (w_img - 1, yl), (255, 0, 255), 2)
    if args.vision_det_exclude_bottom > 0:
        yl = int((1.0 - args.vision_det_exclude_bottom) * h_img)
        cv2.line(viz, (0, yl), (w_img - 1, yl), (255, 0, 255), 2)

    meta = {
        "input": str(args.image.resolve()),
        "counts": {},
        "dropped_inside_self_roi": dropped_self,
        "dropped_top_bottom_band": dropped_vert,
        "n_dets_after_tier_cap": len(dets),
        "combo_would_fire": len(dets) > 0,
    }
    for d in dets:
        meta["counts"][d.label] = meta["counts"].get(d.label, 0) + 1

    out_path = args.output.resolve()
    cv2.imwrite(str(out_path), viz, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[vision-preview] wrote {out_path}")
    print(f"[vision-preview] meta   {sidecar}")


if __name__ == "__main__":
    main()
