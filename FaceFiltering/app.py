"""
Gradio UI: compact layout, theme, and context-sensitive parameters.
Run from FaceFiltering folder:  python app.py
"""
from __future__ import annotations

import base64
import os
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from facefiltering import (
    CONVOLUTION_FILTERS,
    FILTER_FUNCTIONS,
    FILTER_METHODOLOGIES,
    FILTER_NAMES,
    FUNCTION_NAMES,
    METHODOLOGY_DESCRIPTIONS,
    filters_for_methodology_and_function,
    apply_filter,
)
from facefiltering.validate import FilterInputError

# Display names (must match registry / filter modules)
_SO = "Sobel (magnitude)"
_LA = "Laplacian (abs)"
_CA = "Canny edge detection"
_US = "Unsharp mask"
_GB = "Gaussian blur"
_BT = "Binary threshold"
_GM = "Gamma"
_HE = "Histogram equalization"
_HP = "High-pass (Fourier)"
_MD = "Median"
_DI = "Morphological dilation"
_ER = "Morphological erosion"
_WN = "Wiener deconvolution"
_DG = "Dodge"
_SW = "Swirl"
_BL = "Bloom"
_PO = "Posterize"
_CH = "Crosshatch threshold"
_ZM = "Zoom"
_LD = "Lens distortion"
_CONV_FILTER_SET = set(CONVOLUTION_FILTERS)
_LOGO_PATH = Path(__file__).resolve().parent / "facefiltering" / "logo" / "logo.png"

_HOW_IT_WORKS_CODE: dict[str, str] = {
    _SO: (
        "# 1) Convolve with x/y Sobel kernels to get directional gradients\n"
        "gx = convolve_gray(g, kx)\n"
        "gy = convolve_gray(g, ky)\n"
        "# 2) Combine both directions into one edge-strength magnitude map\n"
        "mag = np.hypot(gx, gy)\n"
        "# 3) Rescale to 0..255 for display\n"
        "out = normalize_to_u8(mag)"
    ),
    _LA: (
        "# 1) Apply Laplacian (second derivative) to highlight fast intensity changes\n"
        "lap = convolve_gray(g, lap_kernel)\n"
        "# 2) Take absolute response and normalize to display range\n"
        "out = normalize_to_u8(np.abs(lap))"
    ),
    _CA: (
        "# 1) Smooth first to reduce noise before edge detection\n"
        "blur = convolve_gray(g, gaussian_kernel(5, 1.0))\n"
        "# 2) Compute gradient magnitude and direction\n"
        "mag, angle = gradients(blur)\n"
        "# 3) Keep only local maxima along edge direction (thin edges)\n"
        "nms = non_max_suppression(mag, angle)\n"
        "# 4) Use double thresholds and edge tracking by hysteresis\n"
        "edges = hysteresis(nms, t1, t2)"
    ),
    _US: (
        "# 1) Build a blurred version (low-frequency content)\n"
        "blur = convolve_bgr(bgr, gaussian_kernel(k, sigma))\n"
        "# 2) Add scaled high-frequency residual to sharpen details\n"
        "out = bgr + amount * (bgr - blur)\n"
        "# 3) Clamp to valid uint8 range\n"
        "out = clip_u8(out)"
    ),
    _GB: (
        "# 1) Build Gaussian kernel from size/sigma\n"
        "kernel = gaussian_kernel(k, sigma)\n"
        "# 2) Convolve each channel to smooth high-frequency noise/details\n"
        "out = convolve_bgr(bgr, kernel)\n"
        "# 3) Clamp to displayable uint8\n"
        "out = clip_u8(out)"
    ),
    _BT: (
        "# Compare each grayscale pixel with threshold T\n"
        "# >=T becomes white (255), <T becomes black (0)\n"
        "bw = np.where(gray >= T, 255, 0).astype(np.uint8)"
    ),
    _GM: (
        "# 1) Build lookup-table from gamma curve (power-law mapping)\n"
        "lut = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)\n"
        "# 2) Map every pixel using the LUT\n"
        "out = lut[bgr]"
    ),
    _HE: (
        "# 1) Count intensities and accumulate to CDF\n"
        "hist = np.bincount(y.ravel(), minlength=256)\n"
        "cdf = hist.cumsum()\n"
        "# 2) Convert CDF to equalization LUT\n"
        "lut = build_equalization_lut(cdf)\n"
        "# 3) Remap luminance with LUT to improve contrast\n"
        "y_eq = lut[y]"
    ),
    _HP: (
        "# 1) Transform image to frequency domain (centered spectrum)\n"
        "F = np.fft.fftshift(np.fft.fft2(g))\n"
        "# 2) Suppress low frequencies using high-pass mask\n"
        "F_hp = F * gaussian_highpass_mask\n"
        "# 3) Inverse transform back to spatial domain\n"
        "out = np.real(np.fft.ifft2(np.fft.ifftshift(F_hp)))"
    ),
    _MD: (
        "# 1) Extract kxk neighborhoods around each pixel\n"
        "patches = sliding_window_view(channel, (k, k))\n"
        "# 2) Replace center by median value (robust to impulse noise)\n"
        "out = np.median(patches, axis=(-2, -1))"
    ),
    _DI: (
        "# 1) Build structuring element (shape template)\n"
        "kernel = elliptical_kernel(k)\n"
        "# 2) Dilation keeps local maximum under the kernel\n"
        "out = morphology_bgr(bgr, kernel, op='dilate', iterations=it)"
    ),
    _ER: (
        "# 1) Build structuring element (shape template)\n"
        "kernel = elliptical_kernel(k)\n"
        "# 2) Erosion keeps local minimum under the kernel\n"
        "out = morphology_bgr(bgr, kernel, op='erode', iterations=it)"
    ),
    _WN: (
        "# 1) Compute Wiener gain from blur transfer function and noise factor\n"
        "W = np.conj(H) / (np.abs(H)**2 + K)\n"
        "# 2) Apply gain in frequency domain to estimate sharp image spectrum\n"
        "F_hat = G * W\n"
        "# 3) Inverse FFT, then clamp back to image range\n"
        "f = np.real(np.fft.ifft2(F_hat))\n"
        "out = clip_u8(f * 255)"
    ),
    _DG: (
        "# 1) Apply stable dodge-style tone mapping\n"
        "denom = 255 - strength * bgr\n"
        "# 2) Bright areas are boosted more strongly\n"
        "out = (bgr * 255) / max(denom, 1)\n"
        "# 3) Clamp to uint8\n"
        "out = clip_u8(out)"
    ),
    _SW: (
        "# 1) Convert each output pixel to polar coordinates around center\n"
        "r, theta = to_polar(x, y, cx, cy)\n"
        "# 2) Rotate angle by radius-dependent offset (stronger near center)\n"
        "theta_src = theta - strength * (radius - r) / radius\n"
        "# 3) Sample source image with bilinear interpolation\n"
        "out = bilinear_sample(src_x, src_y)"
    ),
    _BL: (
        "# 1) Keep only bright pixels above threshold\n"
        "bright = bgr * (gray >= T)\n"
        "# 2) Blur bright layer to create glow\n"
        "glow = convolve_bgr(bright, gaussian_kernel(k, sigma))\n"
        "# 3) Add glow back to original image\n"
        "out = clip_u8(bgr + intensity * glow)"
    ),
    _PO: (
        "# 1) Map intensities to a small number of discrete bins\n"
        "q = round((I/255) * (levels-1)) / (levels-1)\n"
        "# 2) Scale quantized values back to 0..255\n"
        "out = clip_u8(q * 255)"
    ),
    _CH: (
        "# 1) Split luminance into a few darkness bands\n"
        "band = floor(gray / (256/levels))\n"
        "# 2) Draw hatch line patterns for darker bands\n"
        "out[(darkness>=1) & diag1] = 0\n"
        "out[(darkness>=2) & diag2] = 0\n"
        "# 3) Return stylized black/white hatch image\n"
        "out = to_bgr_from_gray(out)"
    ),
    _ZM: (
        "# 1) Build inverse zoom mapping around center\n"
        "src_x = cx + (x - cx)/factor\n"
        "src_y = cy + (y - cy)/factor\n"
        "# 2) Resample source coordinates using bilinear interpolation\n"
        "out = bilinear_sample(src_x, src_y)"
    ),
    _LD: (
        "# 1) Compute normalized radius from optical center\n"
        "r2 = x_n**2 + y_n**2\n"
        "# 2) Apply radial lens model (barrel/pincushion)\n"
        "src = coord / (1 + k*r2)\n"
        "# 3) Bilinear sample distorted coordinates\n"
        "out = bilinear_sample(src_x, src_y)"
    ),
}


def _format_filter_label(name: str) -> str:
    if name in _CONV_FILTER_SET:
        return f"{name} [Convolution]"
    return name


def _filter_dropdown_choices(names: list[str]):
    """Gradio choices as (label, value), keeping value as raw filter name."""
    return [(_format_filter_label(n), n) for n in names]


def _with_code(md: str, filter_name: str) -> str:
    snippet = _HOW_IT_WORKS_CODE.get(filter_name)
    if not snippet:
        return md
    return md + f"\n\n### Core computation\n```python\n{snippet}\n```"


def _logo_html(height_px: int = 68) -> str:
    if not _LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    return (
        '<div style="display:flex; justify-content:flex-end;">'
        f'<img src="data:image/png;base64,{b64}" alt="logo" style="height:{height_px}px; width:auto;" />'
        "</div>"
    )

_CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; margin: auto !important; }
footer { display: none !important; }
.compact-params .wrap { gap: 0.35rem !important; }
.compact-params label { font-size: 0.82rem !important; }
.subtle { opacity: 0.75; font-size: 0.9rem !important; }
.ff-title { text-align: center; margin: 0.1rem 0 0.25rem 0; }

/* Theme toggle (client-side). Default: dark. */
:root{
  /* default (dark) */
  --ff-bg:#0b1220; --ff-fg:#e5e7eb; --ff-card:#0f172a; --ff-border:#243043;
  --ff-input:#0b1220; --ff-input-fg:#e5e7eb; --ff-muted:#94a3b8;
}
html[data-ff-theme="light"]{
  --ff-bg:#ffffff; --ff-fg:#0f172a; --ff-card:#ffffff; --ff-border:#e5e7eb;
  --ff-input:#ffffff; --ff-input-fg:#0f172a; --ff-muted:#64748b;
}
html{ color-scheme: dark; background: var(--ff-bg) !important; }
html[data-ff-theme="light"]{ color-scheme: light; }

html, body { background: var(--ff-bg) !important; color: var(--ff-fg) !important; }
body { margin: 0 !important; }

/* Gradio outer shells (varies by version/build). These are the ones that cause the \"window background\" issue. */
body > *, #root, #app, .app, .gradio-container, .gradio-container > .main, .container, .main, .wrap, .svelte-1, .svelte-2 {
  background: var(--ff-bg) !important;
  color: var(--ff-fg) !important;
}

/* Cards / panels inside the app */
.block, .gr-panel, .gr-box, .gr-group, .gr-form, .gr-prose, .prose {
  background: var(--ff-card) !important;
  color: var(--ff-fg) !important;
  border-color: var(--ff-border) !important;
}

/* Text / headings (force readable in light theme too) */
label, .label, .markdown, .gr-markdown, .prose, p, span, div { color: var(--ff-fg) !important; }
.prose h1, .prose h2, .prose h3, .prose h4, .prose strong, .prose b { color: var(--ff-fg) !important; }
.subtle { color: var(--ff-muted) !important; }

/* Inputs */
input, textarea, select {
  background: var(--ff-input) !important;
  color: var(--ff-input-fg) !important;
  border-color: var(--ff-border) !important;
}

/* Dropdown/pills */
.gr-dropdown, .wrap.svelte-*, .token, .item, .selected, .choices, .choice {
  color: var(--ff-fg) !important;
}

/* Buttons */
button, .gr-button, .primary, .secondary {
  border-color: var(--ff-border) !important;
}
"""


def _param_row_updates(filter_name: str):
    """Show only sliders that apply to the selected filter."""
    return (
        gr.update(visible=filter_name in (_SO, _LA, _MD)),
        gr.update(visible=filter_name == _CA),
        gr.update(visible=filter_name == _CA),
        gr.update(visible=filter_name == _CA),
        gr.update(visible=filter_name == _US),
        gr.update(visible=filter_name == _US),
        gr.update(visible=filter_name == _GB),
        gr.update(visible=filter_name == _GB),
        gr.update(visible=filter_name == _BT),
        gr.update(visible=filter_name == _GM),
        gr.update(visible=filter_name == _HP),
        gr.update(visible=filter_name == _DI),
        gr.update(visible=filter_name == _DI),
        gr.update(visible=filter_name == _ER),
        gr.update(visible=filter_name == _ER),
        gr.update(visible=filter_name == _WN),
        gr.update(visible=filter_name == _WN),
        gr.update(visible=filter_name == _DG),
        gr.update(visible=filter_name == _SW),
        gr.update(visible=filter_name == _SW),
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _PO),
        gr.update(visible=filter_name == _CH),
        gr.update(visible=filter_name == _CH),
        gr.update(visible=filter_name == _ZM),
        gr.update(visible=filter_name == _LD),
    )

def _render_heatmap(mat: np.ndarray, size: int = 220) -> np.ndarray:
    """Render a small matrix as a colored heatmap (RGB)."""
    x = mat.astype(np.float64)
    if x.size == 0:
        x = np.zeros((1, 1), dtype=np.float64)
    mn, mx = float(np.min(x)), float(np.max(x))
    if mx - mn < 1e-12:
        x01 = np.zeros_like(x, dtype=np.float64)
    else:
        x01 = (x - mn) / (mx - mn)
    img = (x01 * 255.0).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rows, cols = x.shape
    cell_h = max(1, size // max(rows, 1))
    cell_w = max(1, size // max(cols, 1))
    show_values = (rows * cols <= 144) and (min(cell_h, cell_w) >= 18)
    if show_values:
        for r in range(rows):
            for c in range(cols):
                v = float(x[r, c])
                if abs(v - round(v)) < 1e-9:
                    txt = str(int(round(v)))
                elif abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
                    txt = f"{v:.1e}"
                else:
                    txt = f"{v:.2f}"

                y0 = r * cell_h
                x0 = c * cell_w
                y1 = min(size - 1, (r + 1) * cell_h - 1)
                x1 = min(size - 1, (c + 1) * cell_w - 1)
                yc = (y0 + y1) // 2
                xc = (x0 + x1) // 2

                # Choose text color by local background brightness.
                bg = img[yc, xc].astype(np.float64)
                lum = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
                color = (0, 0, 0) if lum > 150 else (255, 255, 255)

                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = float(max(0.3, min(0.65, min(cell_h, cell_w) / 42.0)))
                thick = 1
                (tw, th), _ = cv2.getTextSize(txt, font, scale, thick)
                tx = max(0, min(size - tw, xc - tw // 2))
                ty = max(th + 1, min(size - 1, yc + th // 2))
                cv2.putText(img, txt, (tx, ty), font, scale, color, thick, cv2.LINE_AA)

    return img


def _render_curve(gamma: float, w: int = 360, h: int = 220) -> np.ndarray:
    """Render gamma curve y = (x/255)^(1/gamma) in RGB."""
    g = float(max(gamma, 1e-6))
    inv = 1.0 / g
    xs = np.arange(256, dtype=np.float64)
    ys = ((xs / 255.0) ** inv) * 255.0
    ys = np.clip(ys, 0, 255)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    # axes
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    pts = []
    for x, y in zip(xs, ys):
        px = int(30 + (x / 255.0) * (w - 40))
        py = int((h - 30) - (y / 255.0) * (h - 40))
        pts.append((px, py))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (20, 70, 200), 2)
    return canvas


def _render_hist(gray_u8: np.ndarray, w: int = 360, h: int = 220) -> np.ndarray:
    """Render a simple histogram plot in RGB."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / (hist.max() + 1e-9)
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    for i in range(256):
        x = int(30 + (i / 255.0) * (w - 40))
        y = int((h - 30) - hist[i] * (h - 40))
        cv2.line(canvas, (x, h - 30), (x, y), (40, 40, 40), 1)
    return canvas


def _render_hist_with_vline(gray_u8: np.ndarray, v: int, w: int = 360, h: int = 220) -> np.ndarray:
    img = _render_hist(gray_u8, w=w, h=h)
    v = int(max(0, min(255, v)))
    x = int(30 + (v / 255.0) * (w - 40))
    cv2.line(img, (x, 10), (x, h - 30), (220, 50, 50), 2)
    cv2.putText(img, f"T={v}", (max(32, x - 20), 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 50, 50), 2, cv2.LINE_AA)
    return img


def _render_cdf(gray_u8: np.ndarray, w: int = 360, h: int = 220) -> np.ndarray:
    """CDF curve for histogram equalization mapping."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).reshape(-1).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf = cdf / (cdf[-1] + 1e-12)
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    pts = []
    for i in range(256):
        x = int(30 + (i / 255.0) * (w - 40))
        y = int((h - 30) - cdf[i] * (h - 40))
        pts.append((x, y))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (20, 120, 60), 2)
    cv2.putText(canvas, "CDF mapping", (32, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 120, 60), 2, cv2.LINE_AA)
    return canvas


def _render_sorted_values(vals: np.ndarray, w: int = 360, h: int = 220) -> np.ndarray:
    """Plot sorted neighborhood values with median highlighted."""
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    v = np.sort(vals.astype(np.float64).reshape(-1))
    if v.size == 0:
        return canvas
    mn, mx = float(v.min()), float(v.max())
    if mx - mn < 1e-12:
        v01 = np.zeros_like(v)
    else:
        v01 = (v - mn) / (mx - mn)
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    n = v.size
    for i in range(n):
        x = int(30 + (i / max(1, n - 1)) * (w - 40))
        y = int((h - 30) - v01[i] * (h - 40))
        cv2.circle(canvas, (x, y), 2, (40, 40, 40), -1)
    mid = n // 2
    x = int(30 + (mid / max(1, n - 1)) * (w - 40))
    y = int((h - 30) - v01[mid] * (h - 40))
    cv2.circle(canvas, (x, y), 5, (220, 50, 50), -1)
    cv2.putText(canvas, f"median={int(round(v[mid]))}", (32, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 50, 50), 2, cv2.LINE_AA)
    return canvas


def _pick_patch(gray_u8: np.ndarray, k: int) -> np.ndarray:
    """Take a kxk patch around the center (clamped)."""
    h, w = gray_u8.shape
    cy, cx = h // 2, w // 2
    r = k // 2
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    patch = gray_u8[y0:y1, x0:x1]
    if patch.shape != (k, k):
        patch = cv2.copyMakeBorder(patch, 0, k - patch.shape[0], 0, k - patch.shape[1], cv2.BORDER_REPLICATE)
    return patch


def _render_before_after(a: np.ndarray, b: np.ndarray, w_each: int = 180, h: int = 220) -> np.ndarray:
    """Side-by-side small grayscale images in one RGB canvas."""
    a = cv2.resize(a, (w_each, h), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (w_each, h), interpolation=cv2.INTER_NEAREST)
    a_rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    b_rgb = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    out = np.concatenate([a_rgb, b_rgb], axis=1)
    cv2.putText(out, "before", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "after", (w_each + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def build_theory(
    image: np.ndarray | None,
    filter_name: str,
    ksize: int,
    canny_t1: int,
    canny_t2: int,
    canny_ap: int,
    sigma: float,
    amount: float,
    gauss_sigma: float,
    gauss_ksize: int,
    thresh: int,
    gamma: float,
    cutoff: float,
    dilate_ksize: int,
    dilate_iter: int,
    erode_ksize: int,
    erode_iter: int,
    psf: int,
    wiener_ns: float,
    dodge_strength: float,
    swirl_strength: float,
    swirl_radius: float,
    bloom_thresh: int,
    bloom_sigma: float,
    bloom_intensity: float,
    poster_levels: int,
    hatch_levels: int,
    hatch_step: int,
    zoom_factor: float,
    lens_strength: float,
):
    """
    Returns:
      - markdown with LaTeX
      - viz1 RGB image (kernel/mask/curve/hist)
      - viz2 RGB image (optional; otherwise None)
    """
    viz1 = None
    viz2 = None

    if filter_name == _SO:
        md = (
            "### Sobel (edge strength)\n"
            "Compute gradients and magnitude:\n\n"
            "$$G_x = I * S_x,\\; G_y = I * S_y,\\; |G| = \\sqrt{G_x^2 + G_y^2}$$\n\n"
            "For `ksize=3`, the classic kernels are:\n\n"
            "$$S_x=\\begin{bmatrix}-1&0&1\\\\-2&0&2\\\\-1&0&1\\end{bmatrix},\\quad"
            "S_y=\\begin{bmatrix}-1&-2&-1\\\\0&0&0\\\\1&2&1\\end{bmatrix}$$"
        )
        sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        viz1 = _render_heatmap(sx)
        viz2 = _render_heatmap(sy)
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _LA:
        md = (
            "### Laplacian (2nd derivative)\n"
            "$$\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$$\n\n"
            "For `ksize=3`, a common discrete kernel is:\n\n"
            "$$L=\\begin{bmatrix}0&1&0\\\\1&-4&1\\\\0&1&0\\end{bmatrix}$$"
        )
        L = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        viz1 = _render_heatmap(L)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _CA:
        t1 = int(canny_t1)
        t2 = int(canny_t2)
        ap = int(canny_ap)
        md = (
            "### Canny edge detection\n"
            "Multi-stage edge detector (gradient → non-maximum suppression → hysteresis thresholding).\n\n"
            "$$\\text{edges} = \\mathrm{Canny}(I; T_1, T_2)$$\n\n"
            f"Current **T1={t1}**, **T2={t2}**, **aperture={ap}**."
        )
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            viz1 = _render_hist_with_vline(gray, t1)
            # second line for T2
            h2 = _render_hist_with_vline(gray, t2)
            viz1 = np.concatenate([viz1[:110], h2[110:]], axis=0) if viz1.shape == h2.shape else viz1
            edges = cv2.Canny(gray, t1, t2, apertureSize=ap)
            viz2 = _render_before_after(_pick_patch(gray, 31), _pick_patch(edges, 31))
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _GB:
        md = (
            "### Gaussian blur (low-pass)\n"
            "$$G(x,y)=\\frac{1}{2\\pi\\sigma^2}\\,e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$\n\n"
            "The kernel is normalized so the weights sum to 1."
        )
        s = float(max(gauss_sigma, 1e-6))
        k = int(gauss_ksize)
        if k <= 0:
            k = int(2 * round(3 * s) + 1)  # typical ~6*s + 1
        k = max(3, k | 1)
        g1 = cv2.getGaussianKernel(k, s, ktype=cv2.CV_64F)
        kern = g1 @ g1.T
        viz1 = _render_heatmap(kern, size=220)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _US:
        md = (
            "### Unsharp masking (sharpening)\n"
            "$$I_{sharp} = I + a\\,(I - G_{\\sigma} * I)$$\n\n"
            "Where **a** is *amount* and $G_{\\sigma}$ is Gaussian blur."
        )
        # show gaussian kernel that generates the blur (visual explanation)
        s = float(max(sigma, 1e-6))
        k = int(2 * round(3 * s) + 1)
        k = max(3, k | 1)
        g1 = cv2.getGaussianKernel(k, s, ktype=cv2.CV_64F)
        kern = g1 @ g1.T
        viz1 = _render_heatmap(kern, size=220)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _BT:
        t = int(thresh)
        md = (
            "### Binary threshold\n"
            "$$I'(x)=\\begin{cases}255,& I(x)\\ge T\\\\0,& I(x)\\lt T\\end{cases}$$\n\n"
            f"Current **T = {t}**."
        )
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            viz1 = _render_hist_with_vline(gray, t)
            _, bw = cv2.threshold(gray, max(0, min(255, t)), 255, cv2.THRESH_BINARY)
            viz2 = _render_before_after(_pick_patch(gray, 15), _pick_patch(bw, 15))
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _GM:
        g = float(gamma)
        md = (
            "### Gamma (power-law) transform\n"
            "$$I' = 255\\cdot (I/255)^{1/\\gamma}$$\n\n"
            f"Current **γ = {g:.2f}**."
        )
        viz1 = _render_curve(g)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _HE:
        md = (
            "### Histogram equalization\n"
            "Redistributes intensities using the cumulative distribution function (CDF):\n\n"
            "$$s = \\mathrm{CDF}(r)$$\n\n"
            "Below are the grayscale histograms (before/after) for the current image."
        )
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            # combine before/after histogram into one image; show CDF as second image
            h1 = _render_hist(gray, w=360, h=110)
            h2 = _render_hist(y_eq, w=360, h=110)
            viz1 = np.concatenate([h1, h2], axis=0)
            cv2.putText(viz1, "before", (32, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(viz1, "after", (32, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
            viz2 = _render_cdf(gray)
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _HP:
        cr = float(cutoff)
        md = (
            "### High-pass filtering (Fourier domain)\n"
            "$$F=\\mathcal{F}\\{I\\},\\; F' = H\\cdot F,\\; I' = \\mathcal{F}^{-1}\\{F'\\}$$\n\n"
            "Here **H** is a Gaussian high-pass mask. White = pass, dark = suppress."
        )
        # render the mask itself
        size = 128
        crow = size // 2
        ccol = size // 2
        d0 = max(cr * size * 0.5, 1.0)
        yy, xx = np.ogrid[:size, :size]
        dist = np.sqrt((yy - crow) ** 2 + (xx - ccol) ** 2).astype(np.float64)
        hp = 1.0 - np.exp(-(dist ** 2) / (2.0 * (d0 ** 2)))
        viz1 = _render_heatmap(hp, size=220)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _MD:
        md = (
            "### Median filter\n"
            "Replaces each pixel by the median value in its $k\\times k$ neighborhood.\n\n"
            "$$I'(x)=\\mathrm{median}\\{I(y): y\\in\\mathcal{N}(x)\\}$$"
        )
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            k = max(3, int(ksize) | 1)
            k = min(k, 15)
            patch = _pick_patch(gray, k)
            outp = cv2.medianBlur(patch, k)
            viz1 = _render_heatmap(patch, size=220)
            viz2 = _render_sorted_values(patch.reshape(-1))
            md += f"\n\nWorked example: center patch ({k}x{k}) → median of its values."
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _DI:
        md = (
            "### Morphological dilation\n"
            "Expands bright regions using a structuring element $B$:\n\n"
            "$$(I\\oplus B)(x)=\\max_{b\\in B} I(x-b)$$"
        )
        k = max(3, int(dilate_ksize) | 1)
        k = min(k, 31)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float64)
        viz1 = _render_heatmap(kernel, size=220)
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            # show dilation on a binarized patch for clarity
            t = 127
            _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            patch = _pick_patch(bw, 31)
            it = max(1, int(dilate_iter))
            outp = cv2.dilate(patch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it)
            viz2 = _render_before_after(patch, outp)
            md += f"\n\nWorked example: dilation on a binary patch (T=127), kernel {k}x{k}, iterations={it}."
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _ER:
        md = (
            "### Morphological erosion\n"
            "Shrinks bright regions using structuring element $B$:\n\n"
            "$$(I\\ominus B)(x)=\\min_{b\\in B} I(x+b)$$"
        )
        k = max(3, int(erode_ksize) | 1)
        k = min(k, 31)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float64)
        viz1 = _render_heatmap(kernel, size=220)
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            patch = _pick_patch(bw, 31)
            it = max(1, int(erode_iter))
            outp = cv2.erode(patch, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=it)
            viz2 = _render_before_after(patch, outp)
            md += f"\n\nWorked example: erosion on a binary patch (T=127), kernel {k}x{k}, iterations={it}."
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _WN:
        md = (
            "### Wiener deconvolution (frequency domain)\n"
            "$$\\hat{F} = \\frac{H^*}{|H|^2 + K}\\,G$$\n\n"
            "Where **H** is the blur transfer function (PSF in frequency), and **K** is the noise ratio."
        )
        p = int(psf)
        p = max(3, p | 1)
        ax = np.arange(p, dtype=np.float64) - (p // 2)
        gx, gy = np.meshgrid(ax, ax)
        sig = max(p / 6.0, 1e-6)
        h = np.exp(-(gx * gx + gy * gy) / (2.0 * sig * sig))
        h /= h.sum() + 1e-12
        viz1 = _render_heatmap(h, size=220)
        # Show Wiener gain in frequency domain (radial line through center)
        H = np.fft.fftshift(np.fft.fft2(h, s=(128, 128)))
        H2 = np.abs(H) ** 2
        K = max(float(wiener_ns), 1e-8)
        gain = H2 / (H2 + K)
        center = gain.shape[0] // 2
        radial = gain[center, :].astype(np.float64)
        viz2 = _render_sorted_values(radial)  # reuse plot helper for a simple curve-like view
        md += "\n\nTip: larger K (noise ratio) suppresses aggressive deblurring (more stable)."
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _DG:
        s = float(dodge_strength)
        md = (
            "### Dodge\n"
            "Brightens tones using a color-dodge style mapping:\n\n"
            "$$I' = \\frac{255\\,I}{255 - sI}$$\n\n"
            f"Current **strength = {s:.2f}**."
        )
        xs = np.arange(256, dtype=np.float64)
        ys = (255.0 * xs) / np.maximum(255.0 - s * xs, 1.0)
        ys = np.clip(ys, 0, 255)
        w, h = 360, 220
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
        pts = []
        for x, y in zip(xs, ys):
            px = int(30 + (x / 255.0) * (w - 40))
            py = int((h - 30) - (y / 255.0) * (h - 40))
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (180, 80, 20), 2)
        viz1 = canvas
        return _with_code(md, filter_name), viz1, None

    if filter_name == _SW:
        st = float(swirl_strength)
        rr = float(swirl_radius)
        md = (
            "### Swirl\n"
            "Geometric warp around image center with radius-controlled rotation:\n\n"
            "$$\\theta_{src} = \\theta - s\\,\\frac{R-r}{R},\\; r<R$$\n\n"
            f"Current **strength = {st:.2f}**, **radius ratio = {rr:.2f}**."
        )
        size = 128
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        cy = cx = (size - 1) * 0.5
        dx, dy = xx - cx, yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        R = rr * size * 0.5
        ang = np.zeros_like(r)
        inside = r < max(R, 1e-6)
        ang[inside] = st * (R - r[inside]) / max(R, 1e-6)
        viz1 = _render_heatmap(ang, size=220)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _BL:
        t = int(bloom_thresh)
        s = float(bloom_sigma)
        a = float(bloom_intensity)
        md = (
            "### Bloom\n"
            "Glow effect from bright regions:\n\n"
            "$$I' = I + \\alpha\\,\\mathrm{Blur}(I\\cdot\\mathbf{1}_{I\\ge T})$$\n\n"
            f"Current **threshold={t}**, **sigma={s:.2f}**, **intensity={a:.2f}**."
        )
        k = max(3, int(2 * round(3 * s) + 1) | 1)
        g1 = cv2.getGaussianKernel(k, s, ktype=cv2.CV_64F)
        viz1 = _render_heatmap(g1 @ g1.T, size=220)
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            viz2 = _render_hist_with_vline(gray, t)
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _PO:
        lv = int(poster_levels)
        md = (
            "### Posterize\n"
            "Reduces tones to a limited set of levels:\n\n"
            "$$I' = \\mathrm{round}\\left(\\frac{I}{255}(L-1)\\right)\\frac{255}{L-1}$$\n\n"
            f"Current **levels = {lv}**."
        )
        xs = np.arange(256, dtype=np.float64)
        ys = np.round((xs / 255.0) * (max(2, lv) - 1.0)) / (max(2, lv) - 1.0) * 255.0
        w, h = 360, 220
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
        pts = []
        for x, y in zip(xs, ys):
            px = int(30 + (x / 255.0) * (w - 40))
            py = int((h - 30) - (y / 255.0) * (h - 40))
            pts.append((px, py))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (120, 60, 180), 2)
        viz1 = canvas
        return _with_code(md, filter_name), viz1, None

    if filter_name == _CH:
        lv = int(hatch_levels)
        st = int(hatch_step)
        md = (
            "### Crosshatch threshold\n"
            "Maps luminance bands to line-hatching patterns for stylized shading.\n\n"
            f"Current **levels = {lv}**, **step = {st}**."
        )
        size = 128
        yy, xx = np.mgrid[0:size, 0:size]
        pat = np.full((size, size), 255, dtype=np.uint8)
        pat[((xx + yy) % max(3, st)) == 0] = 0
        pat[((xx - yy) % max(3, st)) == 0] = 0
        viz1 = cv2.cvtColor(pat, cv2.COLOR_GRAY2RGB)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _ZM:
        f = float(zoom_factor)
        md = (
            "### Zoom\n"
            "Geometric scaling around image center with interpolation.\n\n"
            "$$x_s = c_x + \\frac{x-c_x}{f},\\; y_s = c_y + \\frac{y-c_y}{f}$$\n\n"
            f"Current **factor = {f:.2f}**."
        )
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            patch = _pick_patch(gray, 31)
            zoomed = cv2.resize(patch, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
            zoomed = cv2.resize(zoomed, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
            viz1 = _render_before_after(patch, zoomed)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _LD:
        k = float(lens_strength)
        md = (
            "### Lens distortion\n"
            "Applies radial coordinate remapping (barrel/pincushion).\n\n"
            "$$x_s = \\frac{x}{1 + kr^2},\\; y_s = \\frac{y}{1 + kr^2}$$\n\n"
            f"Current **strength = {k:.2f}**."
        )
        size = 128
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        cx = cy = (size - 1) * 0.5
        xn = (xx - cx) / max(cx, 1.0)
        yn = (yy - cy) / max(cy, 1.0)
        r2 = xn * xn + yn * yn
        viz1 = _render_heatmap(1.0 + k * r2, size=220)
        return _with_code(md, filter_name), viz1, None

    return "### Filter\nNo theory available.", None, None


def run_filter(
    image: np.ndarray | None,
    filter_name: str,
    ksize: int,
    canny_t1: int,
    canny_t2: int,
    canny_ap: int,
    sigma: float,
    amount: float,
    gauss_sigma: float,
    gauss_ksize: int,
    thresh: int,
    gamma: float,
    cutoff: float,
    dilate_ksize: int,
    dilate_iter: int,
    erode_ksize: int,
    erode_iter: int,
    psf: int,
    wiener_ns: float,
    dodge_strength: float,
    swirl_strength: float,
    swirl_radius: float,
    bloom_thresh: int,
    bloom_sigma: float,
    bloom_intensity: float,
    poster_levels: int,
    hatch_levels: int,
    hatch_step: int,
    zoom_factor: float,
    lens_strength: float,
):
    if image is None:
        return None
    try:
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise FilterInputError("Invalid image array from upload.")

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        kwargs = {
            "ksize": int(ksize),
            "canny_t1": int(canny_t1),
            "canny_t2": int(canny_t2),
            "canny_ap": int(canny_ap),
            "sigma": float(sigma),
            "amount": float(amount),
            "gauss_sigma": float(gauss_sigma),
            "gauss_ksize": int(gauss_ksize),
            "thresh": int(thresh),
            "gamma": float(gamma),
            "cutoff": float(cutoff),
            "iter": int(dilate_iter),
            "erode_ksize": int(erode_ksize),
            "erode_iter": int(erode_iter),
            "psf": int(psf),
            "ns": float(wiener_ns),
            "dodge_strength": float(dodge_strength),
            "swirl_strength": float(swirl_strength),
            "swirl_radius": float(swirl_radius),
            "bloom_thresh": int(bloom_thresh),
            "bloom_sigma": float(bloom_sigma),
            "bloom_intensity": float(bloom_intensity),
            "poster_levels": int(poster_levels),
            "hatch_levels": int(hatch_levels),
            "hatch_step": int(hatch_step),
            "zoom_factor": float(zoom_factor),
            "lens_strength": float(lens_strength),
        }
        if filter_name == _DI:
            kwargs["ksize"] = int(dilate_ksize)

        out = apply_filter(filter_name, bgr, **kwargs)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except FilterInputError as e:
        raise gr.Error(str(e)) from e
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Filter failed: {e}") from e


def main():
    theme = gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="gray",
        neutral_hue="slate",
    ).set(
        block_background_fill="white",
        block_label_text_weight="600",
        block_title_text_weight="600",
    )

    with gr.Blocks(
        title="ZPOe 25/26L 2026: Filters On Human Faces",
    ) as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=12):
                gr.HTML('<div class="ff-title"><h2 style="margin:0;">ZPOe 25/26L 2026: Filters On Human Faces</h2></div>')
            with gr.Column(scale=2):
                if _LOGO_PATH.exists():
                    gr.HTML(_logo_html(68))
        with gr.Row():
            theme_btn = gr.Button("Light/Dark", size="sm", variant="secondary")
            gr.Markdown(
                '<p class="subtle" style="margin:0.15rem 0 0 0;">Upload an image, choose a filter, adjust parameters, then <strong>Apply</strong>.</p>',
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                _f0 = "All"
                function_dd = gr.Dropdown(
                    choices=["All"] + FUNCTION_NAMES,
                    value=_f0,
                    label="Function (what it does)",
                    container=False,
                )
                filter_dd = gr.Dropdown(
                    choices=_filter_dropdown_choices(
                        filters_for_methodology_and_function(
                            "All",
                            _f0,
                        )
                    ),
                    value=filters_for_methodology_and_function(
                        "All",
                        _f0,
                    )[0],
                    label="Filter",
                    container=False,
                )
                category_label = gr.Markdown("", elem_classes=["subtle"])
                gr.Markdown(
                    '<p class="subtle" style="margin:0.2rem 0 0.35rem 0;"><strong>Methodology</strong> is shown automatically for the selected filter. <strong>Function</strong> is used for filtering.</p>'
                )
                apply_btn = gr.Button("Apply", variant="primary", size="lg")

                gr.Markdown("**Parameters** (only for the selected filter)")
                param_hint = gr.Markdown("", visible=False, elem_classes=["subtle"])

                with gr.Group(elem_classes=["compact-params"]):
                    ksize = gr.Slider(
                        3, 15, value=3, step=2,
                        label="Kernel size (Sobel / Laplacian / Median)",
                        show_label=True,
                    )
                    canny_t1 = gr.Slider(0, 500, value=80, step=1, label="Canny threshold1", show_label=True)
                    canny_t2 = gr.Slider(0, 500, value=160, step=1, label="Canny threshold2", show_label=True)
                    canny_ap = gr.Slider(3, 7, value=3, step=2, label="Canny aperture (3/5/7)", show_label=True)
                    sigma = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Blur sigma (Unsharp)", show_label=True)
                    amount = gr.Slider(0.0, 3.0, value=1.5, step=0.1, label="Strength (Unsharp)", show_label=True)
                    gauss_sigma = gr.Slider(0.1, 15.0, value=2.0, step=0.1, label="Sigma (Gaussian blur)", show_label=True)
                    gauss_ksize = gr.Slider(
                        0, 31, value=0, step=2,
                        label="Kernel (Gaussian): 0 = auto",
                        show_label=True,
                    )
                    thresh = gr.Slider(0, 255, value=127, step=1, label="Threshold (Binary)", show_label=True)
                    gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.05, label="Gamma (1 = unchanged)", show_label=True)
                    cutoff = gr.Slider(0.02, 0.25, value=0.08, step=0.01, label="Cutoff (High-pass)", show_label=True)
                    dilate_ksize = gr.Slider(3, 21, value=5, step=2, label="Kernel (Dilation)", show_label=True)
                    dilate_iter = gr.Slider(1, 8, value=1, step=1, label="Iterations (Dilation)", show_label=True)
                    erode_ksize = gr.Slider(3, 21, value=5, step=2, label="Kernel (Erosion)", show_label=True)
                    erode_iter = gr.Slider(1, 8, value=1, step=1, label="Iterations (Erosion)", show_label=True)
                    psf = gr.Slider(5, 31, value=15, step=2, label="PSF size (Wiener)", show_label=True)
                    wiener_ns = gr.Slider(1e-5, 1e-1, value=1e-3, step=1e-5, label="Noise ratio (Wiener)", show_label=True)
                    dodge_strength = gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="Strength (Dodge)", show_label=True)
                    swirl_strength = gr.Slider(-6.0, 6.0, value=2.0, step=0.1, label="Strength (Swirl)", show_label=True)
                    swirl_radius = gr.Slider(0.05, 1.0, value=0.75, step=0.01, label="Radius ratio (Swirl)", show_label=True)
                    bloom_thresh = gr.Slider(0, 255, value=180, step=1, label="Threshold (Bloom)", show_label=True)
                    bloom_sigma = gr.Slider(0.1, 15.0, value=2.5, step=0.1, label="Sigma (Bloom glow)", show_label=True)
                    bloom_intensity = gr.Slider(0.0, 3.0, value=0.7, step=0.05, label="Intensity (Bloom)", show_label=True)
                    poster_levels = gr.Slider(2, 32, value=8, step=1, label="Levels (Posterize)", show_label=True)
                    hatch_levels = gr.Slider(2, 8, value=4, step=1, label="Tone levels (Crosshatch)", show_label=True)
                    hatch_step = gr.Slider(3, 24, value=8, step=1, label="Line spacing (Crosshatch)", show_label=True)
                    zoom_factor = gr.Slider(0.2, 4.0, value=1.2, step=0.05, label="Zoom factor", show_label=True)
                    lens_strength = gr.Slider(-0.8, 0.8, value=-0.25, step=0.01, label="Lens strength", show_label=True)

            with gr.Column(scale=2, min_width=400):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Input", height=360)
                    out = gr.Image(type="numpy", label="Output", height=360)

                gr.Markdown("### How it works")
                theory_md = gr.Markdown()
                with gr.Row():
                    theory_viz1 = gr.Image(type="numpy", label="Kernel / Mask / Curve", height=220)
                    theory_viz2 = gr.Image(type="numpy", label="Extra", height=220)

        sliders = (
            ksize,
            canny_t1,
            canny_t2,
            canny_ap,
            sigma,
            amount,
            gauss_sigma,
            gauss_ksize,
            thresh,
            gamma,
            cutoff,
            dilate_ksize,
            dilate_iter,
            erode_ksize,
            erode_iter,
            psf,
            wiener_ns,
            dodge_strength,
            swirl_strength,
            swirl_radius,
            bloom_thresh,
            bloom_sigma,
            bloom_intensity,
            poster_levels,
            hatch_levels,
            hatch_step,
            zoom_factor,
            lens_strength,
        )

        def _filter_info(name: str):
            function = FILTER_FUNCTIONS.get(name, "Other")
            methodology = FILTER_METHODOLOGIES.get(name, "Other")
            methodology_desc = METHODOLOGY_DESCRIPTIONS.get(methodology, "")
            tags = "[Convolution]" if name in _CONV_FILTER_SET else "[Non-convolution]"
            return gr.update(
                value=(
                    f"**Methodology:** {methodology} — {methodology_desc}"
                    f"<br>**Function:** {function}"
                    f"<br>**Tag:** {tags}"
                )
            )

        def _on_filter(name: str):
            vis = _param_row_updates(name)
            hint_md = gr.update(
                value="No adjustable parameters for this filter.",
                visible=(name == _HE),
            )
            return (*vis, hint_md, _filter_info(name))

        def _on_function(function: str):
            filters = filters_for_methodology_and_function("All", function)
            new_filter = filters[0] if filters else FILTER_NAMES[0]
            return (
                gr.update(choices=_filter_dropdown_choices(filters), value=new_filter),
                _filter_info(new_filter),
            )

        function_dd.change(fn=_on_function, inputs=[function_dd], outputs=[filter_dd, category_label])
        filter_dd.change(fn=_on_filter, inputs=[filter_dd], outputs=[*sliders, param_hint, category_label])

        demo.load(fn=_on_filter, inputs=[filter_dd], outputs=[*sliders, param_hint, category_label])

        inputs = [
            inp,
            filter_dd,
            ksize,
            canny_t1,
            canny_t2,
            canny_ap,
            sigma,
            amount,
            gauss_sigma,
            gauss_ksize,
            thresh,
            gamma,
            cutoff,
            dilate_ksize,
            dilate_iter,
            erode_ksize,
            erode_iter,
            psf,
            wiener_ns,
            dodge_strength,
            swirl_strength,
            swirl_radius,
            bloom_thresh,
            bloom_sigma,
            bloom_intensity,
            poster_levels,
            hatch_levels,
            hatch_step,
            zoom_factor,
            lens_strength,
        ]
        apply_btn.click(fn=run_filter, inputs=inputs, outputs=out)

        # Auto-apply: change filter or image => re-run filter.
        filter_dd.change(fn=run_filter, inputs=inputs, outputs=out)
        inp.change(fn=run_filter, inputs=inputs, outputs=out)
        for s in sliders:
            s.change(fn=run_filter, inputs=inputs, outputs=out)

        # Client-side theme toggle (no Python round-trip).
        theme_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              const html = document.documentElement;
              const cur = html.getAttribute('data-ff-theme') || 'dark';
              const next = (cur === 'dark') ? 'light' : 'dark';
              html.setAttribute('data-ff-theme', next);
              // Gradio also uses body classes in some builds
              document.body.classList.toggle('dark', next === 'dark');
              document.body.classList.toggle('light', next === 'light');
            }""",
        )

        # Ensure we always start in light mode unless toggled.
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              const html = document.documentElement;
              if (!html.getAttribute('data-ff-theme')) html.setAttribute('data-ff-theme', 'dark');
              const cur = html.getAttribute('data-ff-theme');
              document.body.classList.toggle('dark', cur === 'dark');
              document.body.classList.toggle('light', cur === 'light');
            }""",
        )

        # Theory updates (real-time): on filter / sliders / image change.
        theory_inputs = inputs  # same signature
        theory_outputs = [theory_md, theory_viz1, theory_viz2]
        filter_dd.change(fn=build_theory, inputs=theory_inputs, outputs=theory_outputs)
        inp.change(fn=build_theory, inputs=theory_inputs, outputs=theory_outputs)
        for s in sliders:
            s.change(fn=build_theory, inputs=theory_inputs, outputs=theory_outputs)
        demo.load(fn=build_theory, inputs=theory_inputs, outputs=theory_outputs)

    host = os.environ.get("FF_HOST", os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    port = int(os.environ.get("FF_PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    demo.launch(server_name=host, server_port=port, theme=theme, css=_CUSTOM_CSS)


if __name__ == "__main__":
    main()
