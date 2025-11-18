"""Utilities for selecting the localities base folder (shared between tools)."""
from __future__ import annotations

import os
from pathlib import Path
from tkinter import Tk, filedialog

LM_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = LM_ROOT / "cfg"
LAST_BASE_FILE = CFG_DIR / "last_base.txt"


def _read_last_base() -> Path | None:
    if not LAST_BASE_FILE.exists():
        return None
    try:
        text = LAST_BASE_FILE.read_text(encoding="utf-8").strip()
        if not text:
            return None
        path = Path(text)
        if not path.is_absolute():
            path = (LM_ROOT / path).resolve()
        return path
    except Exception:
        return None


def _save_last_base(path: Path) -> None:
    try:
        CFG_DIR.mkdir(parents=True, exist_ok=True)
        rel_path = Path(os.path.relpath(Path(path).resolve(), LM_ROOT))
        LAST_BASE_FILE.write_text(rel_path.as_posix(), encoding="utf-8")
    except Exception:
        # Non-critical: failure to persist last base should not block workflow
        pass


def choose_base_localities() -> Path | None:
    """Open a folder dialog to pick the localities base directory."""

    initial_dir = _read_last_base() or LM_ROOT

    root = Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(initialdir=str(initial_dir))
    finally:
        root.destroy()

    if not selected:
        print("Base localities selection cancelled. Exiting.")
        return None

    chosen = Path(selected).resolve()
    _save_last_base(chosen)
    return chosen
