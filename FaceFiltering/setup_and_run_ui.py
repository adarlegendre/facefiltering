#!/usr/bin/env python3
"""Make .venv in this folder, pip install, run app.py.

Run from FaceFiltering: python setup_and_run_ui.py

Windows uses .venv/Scripts/python.exe. Linux/macOS use .venv/bin/python or
python3. If .venv exists but has no pyvenv.cfg, it gets wiped with venv --clear
and rebuilt."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / ".venv"
_PYVENV_CFG = VENV_DIR / "pyvenv.cfg"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
APP = PROJECT_DIR / "app.py"


def venv_usable() -> bool:
    if not VENV_DIR.is_dir() or not _PYVENV_CFG.is_file():
        return False
    py = venv_python()
    return py.is_file()


def ensure_venv() -> int:
    if venv_usable():
        print("Using existing .venv:", VENV_DIR)
        return 0

    if VENV_DIR.is_dir():
        print(".venv folder exists but is not ok (missing pyvenv.cfg etc). Remaking it.")
        print(VENV_DIR)
        return subprocess.call(
            [sys.executable, "-m", "venv", "--clear", str(VENV_DIR)],
            cwd=str(PROJECT_DIR),
        )

    print("Creating .venv:", VENV_DIR)
    return subprocess.call(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        cwd=str(PROJECT_DIR),
    )


def platform_label() -> str:
    plat = sys.platform
    if plat == "win32":
        return "Windows"
    if plat == "darwin":
        return "macOS"
    if plat.startswith("linux"):
        return "Linux"
    return plat


def is_windows() -> bool:
    return sys.platform == "win32"


def venv_python() -> Path:
    if is_windows():
        return VENV_DIR / "Scripts" / "python.exe"
    bin_dir = VENV_DIR / "bin"
    for name in ("python", "python3"):
        path = bin_dir / name
        if path.is_file():
            return path
    return bin_dir / "python"


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(PROJECT_DIR))


def main() -> int:
    if not REQUIREMENTS.is_file():
        print(f"Missing {REQUIREMENTS}", file=sys.stderr)
        return 1
    if not APP.is_file():
        print(f"Missing {APP}", file=sys.stderr)
        return 1

    print("OS:", platform_label())
    print("Project folder:", PROJECT_DIR)

    rc = ensure_venv()
    if rc != 0:
        return rc

    if not venv_usable():
        print(
            "venv still broken after setup.",
            file=sys.stderr,
        )
        return 1

    py = venv_python()

    print("pip upgrade + install requirements.txt")
    rc = run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if rc != 0:
        return rc
    rc = run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    if rc != 0:
        return rc

    print("Starting app.py")
    return run([str(py), str(APP)])


if __name__ == "__main__":
    raise SystemExit(main())
