"""Entry point for the HRNet trainer (stub version)."""
from __future__ import annotations

import logging
from pathlib import Path
import sys

LM_ROOT = Path(__file__).resolve().parent.parent
if str(LM_ROOT) not in sys.path:
    sys.path.insert(0, str(LM_ROOT))

from scripts import init_structure
from scripts import choose_localities
from scripts import rebuild_localities_status
from scripts.trainer_menu import run_trainer_menu


def _configure_logging() -> logging.Logger:
    logs_dir = LM_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "trainer_menu_last.log"

    logger = logging.getLogger("trainer_entry")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def main() -> int:
    try:
        init_structure.ensure_directories()
        init_structure.ensure_status_file()
        init_structure.ensure_hrnet_config()
    except Exception as exc:  # noqa: BLE001 - user-friendly console output
        print(f"[ERR] Failed to initialize project structure: {exc}")
        return 1

    base_localities = choose_localities.choose_base_localities()
    if base_localities is None:
        return 0

    logger = _configure_logging()
    logger.info("HRNet trainer started")
    logger.info("LM_ROOT: %s", LM_ROOT)
    logger.info("Base localities: %s", base_localities)

    try:
        rebuild_localities_status.rebuild_status(base_localities)
    except Exception as exc:  # noqa: BLE001 - user-facing simplicity
        logger.exception("Failed to rebuild localities status")
        print(f"ERROR: Failed to rebuild localities status: {exc}")
        return 1

    try:
        run_trainer_menu(base_localities=base_localities, lm_root=LM_ROOT)
    except KeyboardInterrupt:
        print("\nTrainer interrupted by user.")
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in trainer menu")
        print(f"ERROR: Unexpected trainer error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
