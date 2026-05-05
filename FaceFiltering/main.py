"""
CLI entry: same filter implementations as the Gradio app.

Example:
  python main.py ../data/lena.png out.png --filter "Sobel (magnitude)" --ksize 3
"""
from __future__ import annotations

import argparse
import sys

import cv2

from facefiltering import FILTER_NAMES, apply_filter
from facefiltering.validate import FilterInputError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply a classical filter (BGR pipeline).")
    p.add_argument("input", help="Input image path")
    p.add_argument("output", help="Output image path")
    p.add_argument("--filter", required=True, choices=FILTER_NAMES, help="Filter display name")
    p.add_argument("--ksize", type=int, default=3, help="Sobel / Laplacian / median kernel (odd)")
    p.add_argument("--sigma", type=float, default=1.0, help="Unsharp Gaussian sigma")
    p.add_argument("--amount", type=float, default=1.5, help="Unsharp amount")
    p.add_argument("--gauss-sigma", type=float, default=2.0, help="Gaussian blur sigma")
    p.add_argument(
        "--gauss-ksize",
        type=int,
        default=0,
        help="Gaussian blur kernel (0=auto from sigma; else odd, e.g. 5)",
    )
    p.add_argument("--thresh", type=int, default=127, help="Binary threshold 0..255")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma exponent")
    p.add_argument("--cutoff", type=float, default=0.08, help="High-pass relative cutoff")
    p.add_argument("--dilate-ksize", type=int, default=5, help="Dilation kernel size")
    p.add_argument("--dilate-iter", type=int, default=1, help="Dilation iterations")
    p.add_argument("--erode-ksize", type=int, default=5, help="Erosion kernel size")
    p.add_argument("--erode-iter", type=int, default=1, help="Erosion iterations")
    p.add_argument("--canny-t1", type=int, default=80, help="Canny threshold1")
    p.add_argument("--canny-t2", type=int, default=160, help="Canny threshold2")
    p.add_argument("--canny-ap", type=int, default=3, help="Canny aperture size (3/5/7)")
    p.add_argument("--canny-l2", action="store_true", help="Use L2 gradient in Canny")
    p.add_argument("--psf", type=int, default=15, help="Wiener PSF size (odd)")
    p.add_argument("--ns", type=float, default=1e-3, help="Wiener noise-to-signal ratio")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Failed to read: {args.input}", file=sys.stderr)
        return 1

    kwargs = {
        "ksize": args.ksize,
        "sigma": args.sigma,
        "amount": args.amount,
        "gauss_sigma": args.gauss_sigma,
        "gauss_ksize": args.gauss_ksize,
        "thresh": args.thresh,
        "gamma": args.gamma,
        "cutoff": args.cutoff,
        "iter": args.dilate_iter,
        "erode_ksize": args.erode_ksize,
        "erode_iter": args.erode_iter,
        "canny_t1": args.canny_t1,
        "canny_t2": args.canny_t2,
        "canny_ap": args.canny_ap,
        "canny_l2": args.canny_l2,
        "psf": args.psf,
        "ns": args.ns,
    }
    if args.filter == "Morphological dilation":
        kwargs["ksize"] = args.dilate_ksize

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
