"""
CLI entry: same filter implementations as the Gradio app.

Example:
  python main.py ../data/lena.png out.png --filter "Bloom" --bloom-thresh 160
"""
from __future__ import annotations

import argparse
import sys

import cv2

from facefiltering import FILTER_NAMES, apply_filter
from facefiltering.validate import FilterInputError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a retained filter (BGR pipeline).")
    p.add_argument("input", help="Input image path")
    p.add_argument("output", help="Output image path")
    p.add_argument("--filter", required=True, choices=FILTER_NAMES, help="Filter display name")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma exponent")
    p.add_argument("--dodge-strength", type=float, default=0.55, help="Dodge strength (0..0.95)")
    p.add_argument("--bloom-thresh", type=int, default=180, help="Bloom threshold (0..255)")
    p.add_argument("--bloom-sigma", type=float, default=2.5, help="Bloom blur sigma")
    p.add_argument("--bloom-intensity", type=float, default=0.7, help="Bloom glow intensity")
    p.add_argument("--orton-sigma", type=float, default=2.0, help="Orton blur sigma")
    p.add_argument("--orton-strength", type=float, default=0.6, help="Orton blend strength")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Failed to read: {args.input}", file=sys.stderr)
        return 1

    kwargs = {
        "gamma": args.gamma,
        "dodge_strength": args.dodge_strength,
        "bloom_thresh": args.bloom_thresh,
        "bloom_sigma": args.bloom_sigma,
        "bloom_intensity": args.bloom_intensity,
        "orton_sigma": args.orton_sigma,
        "orton_strength": args.orton_strength,
    }

    try:
        out = apply_filter(args.filter, bgr, **kwargs)
    except (FilterInputError, ValueError) as e:
        print(e, file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Filter error: {e}", file=sys.stderr)
        return 3

    if not cv2.imwrite(args.output, out):
        print(f"Failed to write: {args.output}", file=sys.stderr)
        return 4
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
