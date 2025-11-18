"""Entry point for the interactive annotator workflow."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import Tk, filedialog

from scripts import init_structure
from scripts import rebuild_localities_status
from scripts.annotator_menu import run_annotator_menu

LM_ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = LM_ROOT / "cfg"
LAST_BASE_FILE = CFG_DIR / "last_base.txt"


def _ensure_structure() -> None:
    try:
        init_structure.ensure_directories()
        init_structure.ensure_status_file()
        init_structure.ensure_hrnet_config()
    except Exception as exc:  # noqa: BLE001 - safety for console UX
        print(f"[ERR] Не удалось инициализировать структуру проекта: {exc}")
        sys.exit(1)


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
        # Не критично для работы аннотатора
        pass


def _choose_base_localities() -> Path | None:
    initial_dir = _read_last_base() or LM_ROOT

    root = Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(initialdir=str(initial_dir))
    finally:
        root.destroy()

    if not selected:
        print("Выбор базы локальностей отменён. Выход.")
        return None

    chosen = Path(selected).resolve()
    _save_last_base(chosen)
    return chosen


def _rebuild_status(base_localities: Path) -> bool:
    try:
        count = rebuild_localities_status.rebuild_status(base_localities)
        print(f"Статус локальностей обновлён ({count}).")
        return True
    except Exception as exc:  # noqa: BLE001 - safety for console UX
        print(f"[ERR] Не удалось обновить статус локальностей: {exc}")
        return False


def main() -> int:
    _ensure_structure()
    base_localities = _choose_base_localities()
    if base_localities is None:
        return 0

    if not _rebuild_status(base_localities):
        return 1

    try:
        run_annotator_menu(base_localities=base_localities, lm_root=LM_ROOT)
    except KeyboardInterrupt:
        print("\nАннотатор остановлен пользователем.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] Неожиданная ошибка аннотатора: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
