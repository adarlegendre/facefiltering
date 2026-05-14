"""
Official Gradio UI for this project — use **only** this `FaceFiltering/app.py` as the web entry point.

Filters: Bloom, Orton effect, Gamma, Dodge.

Run from the FaceFiltering folder:

    python app.py

`setup_and_run_ui.py` installs dependencies and then runs this same file.
"""
from __future__ import annotations

import base64
import os
import traceback
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from facefiltering import apply_filter
from facefiltering.validate import FilterInputError

# Fixed UI list — does not depend on whatever `FILTER_NAMES` resolves to at import time
# (avoids stale packages / wrong working directory showing old filters).
_FILTER_CHOICES: tuple[str, ...] = ("Bloom", "Orton effect", "Gamma", "Dodge")

_BL = "Bloom"
_OR = "Orton effect"
_GM = "Gamma"
_DG = "Dodge"

_LOGO_PATH = Path(__file__).resolve().parent / "facefiltering" / "logo" / "logo.png"
_GALLERY_DIR = Path(__file__).resolve().parent / "faces"
_FILTERS_DIR = Path(__file__).resolve().parent / "facefiltering" / "filters"

_FILTER_SOURCE_FILES: dict[str, str] = {
    _BL: "bloom.py",
    _OR: "orton_effect.py",
    _GM: "gamma.py",
    _DG: "dodge.py",
}

_filter_source_cache: dict[str, str] = {}


def _filter_source_for_ui(filter_name: str) -> tuple[str, str]:
    """Return (relative_path_for_label, source_text) for the filter implementation file."""
    fname = _FILTER_SOURCE_FILES.get(filter_name, "")
    if not fname:
        return "", ""
    rel = f"facefiltering/filters/{fname}"
    if filter_name in _filter_source_cache:
        return rel, _filter_source_cache[filter_name]
    path = _FILTERS_DIR / fname
    try:
        text = path.read_text(encoding="utf-8").rstrip() + "\n"
    except OSError:
        text = f"# Unable to read source: {path}\n"
    _filter_source_cache[filter_name] = text
    return f"facefiltering/filters/{fname}", text


_HOW_IT_WORKS_CODE: dict[str, str] = {
    _BL: (
        "bright = image * (gray >= T)\n"
        "glow = convolve(bright, gaussian_kernel(k, sigma))\n"
        "out = clip(image + alpha * glow)"
    ),
    _OR: (
        "blur = convolve(image, gaussian_kernel(k, sigma))\n"
        "screen = 255 - ((255-image)*(255-blur)/255)\n"
        "out = clip((1-alpha)*image + alpha*screen)"
    ),
    _GM: (
        "lut[i] = round(((i/255.0)**(1.0/gamma))*255)\n"
        "out = lut[image]"
    ),
    _DG: (
        "denom = 255 - strength*image\n"
        "out = clip((image*255) / max(denom, 1))"
    ),
}


_NOTATION_HTML: dict[str, str] = {
    _BL: (
        "<details>\n"
        '<summary><strong>Notation</strong></summary>\n\n'
        "<ul>\n"
        "<li><strong>I</strong> — input color image (each channel 0–255).</li>\n"
        "<li><strong>I′</strong> — output image after bloom.</li>\n"
        "<li><strong>T</strong> — grayscale threshold; pixels brighter than this feed the glow.</li>\n"
        "<li><strong>α</strong> — bloom intensity (how strongly glow is added back).</li>\n"
        "<li><strong>Blur</strong> — Gaussian blur; spread controlled by <strong>σ</strong> (sigma).</li>\n"
        "<li><strong>1<sub>I≥T</sub></strong> — mask (1 where luminance ≥ <strong>T</strong>, else 0).</li>\n"
        "</ul>\n\n"
        "</details>"
    ),
    _OR: (
        "<details>\n"
        '<summary><strong>Notation</strong></summary>\n\n'
        "<ul>\n"
        "<li><strong>I</strong> — input color image (each channel 0–255).</li>\n"
        "<li><strong>I′</strong> — output after mixing.</li>\n"
        "<li><strong>B</strong> — blurred copy of <strong>I</strong> (Gaussian, blur scale <strong>σ</strong>).</li>\n"
        "<li><strong>G<sub>σ</sub> * I</strong> — shorthand for that Gaussian blur.</li>\n"
        "<li><strong>S</strong> — “screen” blend of sharp <strong>I</strong> and blurred <strong>B</strong>.</li>\n"
        "<li><strong>α</strong> — strength from 0 to 1: 0 keeps <strong>I</strong>, 1 pushes toward full screen blend.</li>\n"
        "</ul>\n\n"
        "</details>"
    ),
    _GM: (
        "<details>\n"
        '<summary><strong>Notation</strong></summary>\n\n'
        "<ul>\n"
        "<li><strong>I</strong> — input intensity per channel (0–255).</li>\n"
        "<li><strong>I′</strong> — output after gamma correction.</li>\n"
        "<li><strong>γ</strong> — gamma; exponent 1/γ shapes mid-tones (<strong>γ = 1</strong> leaves the image unchanged).</li>\n"
        "</ul>\n\n"
        "</details>"
    ),
    _DG: (
        "<details>\n"
        '<summary><strong>Notation</strong></summary>\n\n'
        "<ul>\n"
        "<li><strong>I</strong> — input intensity per channel (0–255).</li>\n"
        "<li><strong>I′</strong> — output after dodge curve.</li>\n"
        "<li><strong>s</strong> — dodge strength (how aggressive brightening is).</li>\n"
        "<li><strong>max(255 − sI, 1)</strong> — denominator kept ≥ 1 so division stays finite.</li>\n"
        "<li><strong>clip</strong> — clamp values into valid 0–255 byte range.</li>\n"
        "</ul>\n\n"
        "</details>"
    ),
}


def _with_code(md: str, filter_name: str) -> str:
    snippet = _HOW_IT_WORKS_CODE.get(filter_name)
    rel_path, src = _filter_source_for_ui(filter_name)
    if not snippet and not src.strip():
        return md
    body = (
        "<details>\n"
        "<summary><strong>Core computation</strong></summary>\n\n"
    )
    if snippet:
        body += "**Algorithm (pseudocode)**\n\n"
        body += f"```python\n{snippet}\n```\n\n"
    if rel_path and src.strip():
        body += f"**Program (`{rel_path}`)**\n\n"
        body += f"```python\n{src}\n```\n"
    body += "</details>"
    return md + "\n\n" + body


def _logo_html(height_px: int = 68) -> str:
    if not _LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    return (
        '<div style="display:flex; justify-content:flex-end;">'
        f'<img src="data:image/png;base64,{b64}" alt="logo" style="height:{height_px}px; width:auto;" />'
        "</div>"
    )


def _gallery_items(limit: int = 24) -> list[str]:
    if not _GALLERY_DIR.exists():
        return []
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(sorted(_GALLERY_DIR.glob(ext)))
    if limit > 0:
        paths = paths[: max(0, int(limit))]
    return [str(p) for p in paths]


def _read_rgb_image(path: str) -> np.ndarray | None:
    if not path:
        return None
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _gallery_slice(items: list[str], start: int, page_size: int) -> list[str]:
    if not items:
        return []
    s = max(0, min(int(start), max(0, len(items) - 1)))
    e = min(len(items), s + max(1, int(page_size)))
    return items[s:e]


def _gallery_caption(items: list[str], start: int, page_size: int) -> str:
    total = len(items)
    if total == 0:
        return "No gallery images found."
    s = max(0, min(int(start), max(0, total - 1)))
    e = min(total, s + max(1, int(page_size)))
    return f"Showing {s + 1}-{e} of {total}"


_CUSTOM_CSS = """
.gradio-container { max-width: 980px !important; margin: auto !important; padding-top: 0.25rem !important; }
footer { display: none !important; }
.compact-params .wrap { gap: 0.22rem !important; }
.compact-params label { font-size: 0.80rem !important; }
.subtle { opacity: 0.75; font-size: 0.9rem !important; }
.ff-title { text-align: center; margin: 0.05rem 0 0.15rem 0; }
.ff-gallery { min-height: 72px !important; }
.ff-gallery img { object-fit: cover !important; }
.gradio-container .block { padding: 0.45rem 0.55rem !important; }
.gradio-container .form { gap: 0.3rem !important; }
.gradio-container .prose p, .gradio-container .prose h3 { margin-top: 0.35rem !important; margin-bottom: 0.35rem !important; }

:root{
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

body > *, #root, #app, .app, .gradio-container, .gradio-container > .main, .container, .main, .wrap, .svelte-1, .svelte-2 {
  background: var(--ff-bg) !important;
  color: var(--ff-fg) !important;
}

.block, .gr-panel, .gr-box, .gr-group, .gr-form, .gr-prose, .prose {
  background: var(--ff-card) !important;
  color: var(--ff-fg) !important;
  border-color: var(--ff-border) !important;
}

label, .label, .markdown, .gr-markdown, .prose, p, span, div { color: var(--ff-fg) !important; }
.prose h1, .prose h2, .prose h3, .prose h4, .prose strong, .prose b { color: var(--ff-fg) !important; }
.subtle { color: var(--ff-muted) !important; }

input, textarea, select {
  background: var(--ff-input) !important;
  color: var(--ff-input-fg) !important;
  border-color: var(--ff-border) !important;
}

.gr-dropdown, .wrap.svelte-*, .token, .item, .selected, .choices, .choice {
  color: var(--ff-fg) !important;
}

button, .gr-button, .primary, .secondary {
  border-color: var(--ff-border) !important;
}
"""


def _render_heatmap(mat: np.ndarray, size: int = 220) -> np.ndarray:
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
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = "Gaussian kernel (2D, center = peak)"
    ts = cv2.getTextSize(cap, font, 0.42, 1)[0]
    x0 = max(4, (size - ts[0]) // 2)
    cv2.putText(img, cap, (x0, size - 6), font, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _annotate_plot_axes(
    canvas_bgr: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    left: int = 30,
    top: int = 10,
    right_pad: int = 10,
    bottom_pad: int = 30,
    x_lo_tick: str = "0",
    x_hi_tick: str = "255",
    y_top_tick: str | None = "255",
) -> None:
    """Draw short axis captions for cv2-rendered plots (image is BGR for OpenCV)."""
    h, w = canvas_bgr.shape[:2]
    right = w - right_pad
    bottom = h - bottom_pad
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (55, 55, 55)
    scale = 0.45
    thickness = 1

    tw, th = cv2.getTextSize(x_label, font, scale, thickness)[0]
    cx = max(left, min(right - tw, (left + right) // 2 - tw // 2))
    cv2.putText(canvas_bgr, x_label, (cx, h - 7), font, scale, color, thickness, cv2.LINE_AA)

    y_mid = (top + bottom) // 2 + th // 2
    cv2.putText(canvas_bgr, y_label, (6, y_mid), font, scale, color, thickness, cv2.LINE_AA)

    cv2.putText(canvas_bgr, x_lo_tick, (left - 2, bottom + 12), font, 0.4, color, 1, cv2.LINE_AA)
    cv2.putText(canvas_bgr, x_hi_tick, (right - 26, bottom + 12), font, 0.4, color, 1, cv2.LINE_AA)
    if y_top_tick:
        cv2.putText(canvas_bgr, y_top_tick, (left - 14, top + 12), font, 0.4, color, 1, cv2.LINE_AA)


def _render_curve(gamma: float, w: int = 360, h: int = 220) -> np.ndarray:
    g = float(max(gamma, 1e-6))
    inv = 1.0 / g
    xs = np.arange(256, dtype=np.float64)
    ys = ((xs / 255.0) ** inv) * 255.0
    ys = np.clip(ys, 0, 255)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    pts = []
    for x, y in zip(xs, ys):
        px = int(30 + (x / 255.0) * (w - 40))
        py = int((h - 30) - (y / 255.0) * (h - 40))
        pts.append((px, py))
    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (20, 70, 200), 2)
    _annotate_plot_axes(canvas, x_label="Input level (0–255)", y_label="Output level")
    return canvas


def _render_hist(gray_u8: np.ndarray, w: int = 360, h: int = 220) -> np.ndarray:
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    hist = cv2.calcHist([gray_u8], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist / (hist.max() + 1e-9)
    cv2.rectangle(canvas, (30, 10), (w - 10, h - 30), (220, 220, 220), 1)
    for i in range(256):
        x = int(30 + (i / 255.0) * (w - 40))
        y = int((h - 30) - hist[i] * (h - 40))
        cv2.line(canvas, (x, h - 30), (x, y), (40, 40, 40), 1)
    _annotate_plot_axes(
        canvas,
        x_label="Gray level (0–255)",
        y_label="Count (norm.)",
        y_top_tick="1",
    )
    return canvas


def _render_hist_with_vline(gray_u8: np.ndarray, v: int, w: int = 360, h: int = 220) -> np.ndarray:
    img = _render_hist(gray_u8, w=w, h=h)
    v = int(max(0, min(255, v)))
    x = int(30 + (v / 255.0) * (w - 40))
    cv2.line(img, (x, 10), (x, h - 30), (220, 50, 50), 2)
    cv2.putText(img, f"T={v}", (max(32, x - 20), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 50, 50), 2, cv2.LINE_AA)
    return img


def _pick_patch(gray_u8: np.ndarray, k: int) -> np.ndarray:
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
    a = cv2.resize(a, (w_each, h), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(b, (w_each, h), interpolation=cv2.INTER_NEAREST)
    a_rgb = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    b_rgb = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
    body = np.concatenate([a_rgb, b_rgb], axis=1)
    header_h = 30
    header = np.full((header_h, body.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(header, "before (patch)", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(header, "after (patch)", (w_each + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def _param_visibility(filter_name: str):
    return (
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _BL),
        gr.update(visible=filter_name == _OR),
        gr.update(visible=filter_name == _OR),
        gr.update(visible=filter_name == _GM),
        gr.update(visible=filter_name == _DG),
    )


def build_theory(
    image: np.ndarray | None,
    filter_name: str,
    bloom_thresh: int,
    bloom_sigma: float,
    bloom_intensity: float,
    orton_sigma: float,
    orton_strength: float,
    gamma: float,
    dodge_strength: float,
):
    viz1 = None
    viz2 = None

    if filter_name == _BL:
        t = int(bloom_thresh)
        s = float(bloom_sigma)
        a = float(bloom_intensity)
        md = (
            "### Bloom\n"
            "Glow from bright regions:\n\n"
            "$$I' = I + \\alpha\\,\\mathrm{Blur}(I\\cdot\\mathbf{1}_{I\\ge T})$$\n\n"
            + _NOTATION_HTML[_BL]
            + "\n\n"
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

    if filter_name == _OR:
        s = float(max(orton_sigma, 1e-6))
        a = float(max(0.0, min(1.0, orton_strength)))
        md = (
            "### Orton effect\n"
            "Soft glow via blur + screen blend:\n\n"
            "$$B = G_{\\sigma} * I,\\quad S = 255 - \\frac{(255-I)(255-B)}{255},\\quad I'=(1-\\alpha)I+\\alpha S$$\n\n"
            + _NOTATION_HTML[_OR]
            + "\n\n"
            f"Current **sigma={s:.2f}**, **strength={a:.2f}**."
        )
        k = max(3, int(2 * round(3 * s) + 1) | 1)
        g1 = cv2.getGaussianKernel(k, s, ktype=cv2.CV_64F)
        viz1 = _render_heatmap(g1 @ g1.T, size=220)
        if image is not None and isinstance(image, np.ndarray) and image.ndim >= 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float64)
            blur = cv2.GaussianBlur(bgr, (k, k), s)
            screen = 255.0 - ((255.0 - bgr) * (255.0 - blur) / 255.0)
            mixed = np.clip((1.0 - a) * bgr + a * screen, 0, 255).astype(np.uint8)
            gray0 = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(mixed, cv2.COLOR_BGR2GRAY)
            viz2 = _render_before_after(_pick_patch(gray0, 31), _pick_patch(gray1, 31))
        return _with_code(md, filter_name), viz1, viz2

    if filter_name == _GM:
        g = float(gamma)
        md = (
            "### Gamma (power-law)\n"
            "$$I' = 255\\cdot (I/255)^{1/\\gamma}$$\n\n"
            + _NOTATION_HTML[_GM]
            + "\n\n"
            f"Current **γ = {g:.2f}**."
        )
        viz1 = _render_curve(g)
        return _with_code(md, filter_name), viz1, None

    if filter_name == _DG:
        s = float(dodge_strength)
        md = (
            "### Dodge\n"
            "Highlight brightening (stable mapping):\n\n"
            "$$I' = \\mathrm{clip}\\!\\left(\\frac{255\\,I}{\\max(255 - sI,\\,1)}\\right)$$\n\n"
            + _NOTATION_HTML[_DG]
            + "\n\n"
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
        _annotate_plot_axes(canvas, x_label="Input level (0–255)", y_label="Output level")
        viz1 = canvas
        return _with_code(md, filter_name), viz1, None

    return "### Filter\nSelect a filter.", None, None


def run_filter(
    image: np.ndarray | None,
    filter_name: str,
    bloom_thresh: int,
    bloom_sigma: float,
    bloom_intensity: float,
    orton_sigma: float,
    orton_strength: float,
    gamma: float,
    dodge_strength: float,
):
    if image is None:
        return None
    try:
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise FilterInputError("Invalid image array from upload.")

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        kwargs = {
            "bloom_thresh": int(bloom_thresh),
            "bloom_sigma": float(bloom_sigma),
            "bloom_intensity": float(bloom_intensity),
            "orton_sigma": float(orton_sigma),
            "orton_strength": float(orton_strength),
            "gamma": float(gamma),
            "dodge_strength": float(dodge_strength),
        }
        if filter_name not in _FILTER_CHOICES:
            raise FilterInputError(f"Unknown filter {filter_name!r}. Expected one of {_FILTER_CHOICES}.")
        out = apply_filter(filter_name, bgr, **kwargs)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    except FilterInputError as e:
        raise gr.Error(str(e)) from e
    except ValueError as e:
        raise gr.Error(str(e)) from e
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Filter failed: {e}") from e


def _merge_no_proxy_localhost() -> None:
    """Gradio checks 127.0.0.1 after bind; global HTTP(S)_PROXY without NO_PROXY breaks that on VPS/systemd."""
    for key in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(key, "")
        parts = [p.strip() for p in cur.split(",") if p.strip()]
        for item in ("127.0.0.1", "localhost", "::1"):
            if item not in parts:
                parts.append(item)
        os.environ[key] = ",".join(parts)


def main():
    print(f"[FaceFiltering] Running app from: {Path(__file__).resolve()}")
    print(f"[FaceFiltering] Filters in UI (fixed): {list(_FILTER_CHOICES)}")

    theme = (
        gr.themes.Soft(primary_hue="slate", secondary_hue="gray", neutral_hue="slate")
        .set(block_background_fill="white", block_label_text_weight="600", block_title_text_weight="600")
    )

    with gr.Blocks(title="ZPOe 25/26L 2026: Filters On Human Faces — XOTIENA00 Filters") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=12):
                gr.HTML(
                    '<div class="ff-title">'
                    '<h2 style="margin:0;">ZPOe 25/26L 2026: Filters On Human Faces</h2>'
                    '<div style="margin-top:0.2rem;font-size:1.05rem;font-weight:600;">XOTIENA00 Filters</div>'
                    "</div>"
                )
            with gr.Column(scale=2):
                if _LOGO_PATH.exists():
                    gr.HTML(_logo_html(68))

        with gr.Row():
            theme_btn = gr.Button("Light/Dark", size="sm", variant="secondary")
            gr.Markdown(
                '<p class="subtle" style="margin:0.15rem 0 0 0;">Upload an image, pick a filter, adjust parameters.</p>',
            )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=300):
                filter_pick = gr.Radio(
                    choices=list(_FILTER_CHOICES),
                    value=_FILTER_CHOICES[0],
                    label="Filter",
                    elem_id="ff-filter-pick",
                )
                apply_btn = gr.Button("Apply", variant="primary", size="lg")

                with gr.Accordion("Parameters", open=True):
                    with gr.Group(elem_classes=["compact-params"]):
                        bloom_thresh = gr.Slider(0, 255, value=180, step=1, label="Bloom: threshold", show_label=True)
                        bloom_sigma = gr.Slider(0.1, 15.0, value=2.5, step=0.1, label="Bloom: glow sigma", show_label=True)
                        bloom_intensity = gr.Slider(0.0, 3.0, value=0.7, step=0.05, label="Bloom: intensity", show_label=True)
                        orton_sigma = gr.Slider(0.1, 20.0, value=2.0, step=0.1, label="Orton: blur sigma", show_label=True)
                        orton_strength = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="Orton: strength", show_label=True)
                        gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.05, label="Gamma (1 = unchanged)", show_label=True)
                        dodge_strength = gr.Slider(0.0, 0.95, value=0.55, step=0.01, label="Dodge: strength", show_label=True)

            with gr.Column(scale=2, min_width=400):
                all_gallery_items = _gallery_items(0)
                gallery_page_size = 12
                gallery_start = gr.State(0)
                gallery_all = gr.State(all_gallery_items)
                gallery_page_items = gr.State(_gallery_slice(all_gallery_items, 0, gallery_page_size))
                with gr.Accordion("Image selection", open=True):
                    gallery_caption = gr.Markdown(
                        _gallery_caption(all_gallery_items, 0, gallery_page_size),
                        elem_classes=["subtle"],
                    )
                    with gr.Row():
                        gallery_prev_btn = gr.Button("◀", size="sm")
                        gallery_next_btn = gr.Button("▶", size="sm")
                    sample_gallery = gr.Gallery(
                        value=_gallery_slice(all_gallery_items, 0, gallery_page_size),
                        label="Sample gallery (click to load)",
                        show_label=False,
                        columns=6,
                        rows=2,
                        height=170,
                        object_fit="cover",
                        elem_classes=["ff-gallery"],
                        allow_preview=False,
                    )
                with gr.Row():
                    gr.Markdown("**Input**")
                    gr.Markdown("**Output**")
                with gr.Row():
                    inp = gr.Image(type="numpy", show_label=False, height=250)
                    out = gr.Image(type="numpy", show_label=False, height=250)

                with gr.Accordion("Optional: curve / kernel preview", open=False):
                    theory_md = gr.Markdown()
                    with gr.Row():
                        gr.Markdown("**Plot A**")
                        gr.Markdown("**Plot B**")
                    with gr.Row():
                        theory_viz1 = gr.Image(type="numpy", show_label=False, height=140)
                        theory_viz2 = gr.Image(type="numpy", show_label=False, height=140)

        sliders = (
            bloom_thresh,
            bloom_sigma,
            bloom_intensity,
            orton_sigma,
            orton_strength,
            gamma,
            dodge_strength,
        )

        def _on_filter(name: str):
            return _param_visibility(name)

        def _on_gallery_select(items: list[str] | None, evt: gr.SelectData):
            if not items:
                return None
            idx = evt.index[0] if isinstance(evt.index, tuple) else int(evt.index)
            if idx < 0 or idx >= len(items):
                return None
            return _read_rgb_image(str(items[idx]))

        def _gallery_prev(items: list[str], start: int):
            total = len(items) if items else 0
            if total == 0:
                return gr.update(value=[]), 0, _gallery_caption([], 0, gallery_page_size), []
            new_start = max(0, int(start) - gallery_page_size)
            page = _gallery_slice(items, new_start, gallery_page_size)
            return gr.update(value=page), new_start, _gallery_caption(items, new_start, gallery_page_size), page

        def _gallery_next(items: list[str], start: int):
            total = len(items) if items else 0
            if total == 0:
                return gr.update(value=[]), 0, _gallery_caption([], 0, gallery_page_size), []
            max_start = max(0, total - gallery_page_size)
            new_start = min(max_start, int(start) + gallery_page_size)
            page = _gallery_slice(items, new_start, gallery_page_size)
            return gr.update(value=page), new_start, _gallery_caption(items, new_start, gallery_page_size), page

        filter_pick.change(fn=_on_filter, inputs=[filter_pick], outputs=list(sliders))
        sample_gallery.select(fn=_on_gallery_select, inputs=[gallery_page_items], outputs=[inp])
        gallery_prev_btn.click(
            fn=_gallery_prev,
            inputs=[gallery_all, gallery_start],
            outputs=[sample_gallery, gallery_start, gallery_caption, gallery_page_items],
        )
        gallery_next_btn.click(
            fn=_gallery_next,
            inputs=[gallery_all, gallery_start],
            outputs=[sample_gallery, gallery_start, gallery_caption, gallery_page_items],
        )

        demo.load(fn=_on_filter, inputs=[filter_pick], outputs=list(sliders))

        inputs = [
            inp,
            filter_pick,
            bloom_thresh,
            bloom_sigma,
            bloom_intensity,
            orton_sigma,
            orton_strength,
            gamma,
            dodge_strength,
        ]
        apply_btn.click(fn=run_filter, inputs=inputs, outputs=out)
        filter_pick.change(fn=run_filter, inputs=inputs, outputs=out)
        inp.change(fn=run_filter, inputs=inputs, outputs=out)
        for s in sliders:
            s.change(fn=run_filter, inputs=inputs, outputs=out)

        theme_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js="""() => {
              const html = document.documentElement;
              const cur = html.getAttribute('data-ff-theme') || 'dark';
              const next = (cur === 'dark') ? 'light' : 'dark';
              html.setAttribute('data-ff-theme', next);
              document.body.classList.toggle('dark', next === 'dark');
              document.body.classList.toggle('light', next === 'light');
            }""",
        )
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

        theory_outputs = [theory_md, theory_viz1, theory_viz2]
        filter_pick.change(fn=build_theory, inputs=inputs, outputs=theory_outputs)
        inp.change(fn=build_theory, inputs=inputs, outputs=theory_outputs)
        for s in sliders:
            s.change(fn=build_theory, inputs=inputs, outputs=theory_outputs)
        demo.load(fn=build_theory, inputs=inputs, outputs=theory_outputs)

    _merge_no_proxy_localhost()
    host = os.environ.get("FF_HOST", os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"))
    port = int(os.environ.get("FF_PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    # Skip Gradio's post-launch localhost HEAD probe (breaks behind proxy / some VPS).
    demo.launch(
        server_name=host,
        server_port=port,
        theme=theme,
        css=_CUSTOM_CSS,
        inbrowser=False,
        _frontend=False,
    )


if __name__ == "__main__":
    main()
