# FaceFiltering

Classical filters for portrait demos (including **Gaussian blur**). Each filter lives in its own Python module under `facefiltering/filters/`. Shared validation is in `facefiltering/validate.py`; dispatch is in `facefiltering/registry.py`.

## Layout

- `facefiltering/validate.py` — `ensure_bgr_u8`, odd kernel sizes, clamps, `FilterInputError`
- `facefiltering/gray.py` — BGR ↔ grayscale
- `facefiltering/filters/*.py` — one file per filter (`apply(...)`)
- `facefiltering/registry.py` — `FILTER_NAMES`, `apply_filter(name, bgr, **kwargs)`
- `app.py` — Gradio UI (calls `apply_filter`)
- `main.py` — CLI (same implementations)

## Gradio UI

```bash
cd FaceFiltering
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the URL from the terminal, upload an image, choose a filter, click **Apply**.

### Theme (light / dark)

Use the **Light/Dark** button at the top-left of the UI.

## CLI (same code path as UI)

```bash
python main.py ..\data\lena.png out.png --filter "Sobel (magnitude)" --ksize 3
python main.py ..\data\lena.png out2.png --filter "Median" --ksize 5
```

Use `--help` for all options.

## C++ CLI (optional)

See `cpp/README` section below — small OpenCV-only tool for Sobel/median.

```bash
cd FaceFiltering/cpp
cmake -B build -DOpenCV_DIR=<path_to_opencv_build>
cmake --build build --config Release
build\Release\face_filter_cli.exe ..\..\data\lena.png out_sobel.png sobel 3
```
