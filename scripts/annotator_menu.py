"""Console menu for launching the landmarking GUI per locality."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from annot_gui_custom import run_gui_for_locality
from scripts.hrnet_config_utils import HRNetConfigError, read_num_keypoints
from scripts.rebuild_localities_status import STATUS_HEADER, ensure_status_file, rebuild_status


def _read_status_rows(status_file: Path):
    if not status_file.exists():
        return []
    with status_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _write_status_rows(status_file: Path, rows) -> None:
    with status_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _print_menu(rows, base_localities: Path) -> None:
    if rows:
        max_name_len = max(len(row.get("locality", "")) for row in rows)
        max_name_len = min(max_name_len, 35)
    else:
        max_name_len = len("locality")

    print("=== GM Landmarking: Annotator ===")
    print("Base folder:")
    print(f"  {base_localities}")
    print()
    print(f"#  {'locality':{max_name_len}} {'status':8} {'labeled/total':13} autoQ")

    for idx, row in enumerate(rows, start=1):
        locality = row.get("locality", "")
        if len(locality) > max_name_len:
            locality_truncated = locality[: max_name_len - 1] + "…"
        else:
            locality_truncated = locality
        status = row.get("status", "")
        auto_q = row.get("auto_quality", "") or "-"
        n_images = _format_int(row.get("n_images", "0"))
        n_labeled = _format_int(row.get("n_labeled", "0"))
        print(
            f"{idx:>2}  {locality_truncated:{max_name_len}} {status:8} "
            f"{n_labeled:>3} / {n_images:<7} {auto_q}"
        )
    if not rows:
        print("(нет локальностей для отображения)")
    print()
    print("Enter locality number to open, or 0 to quit:", end=" ")


def run_annotator_menu(base_localities: Path, lm_root: Path) -> None:
    status_file = ensure_status_file()

    while True:
        rows = _read_status_rows(status_file)
        _print_menu(rows, base_localities)
        choice = input().strip()

        if choice == "0":
            print("Выход из аннотатора.")
            break

        try:
            idx = int(choice)
        except ValueError:
            print("Некорректный ввод. Требуется номер локальности или 0.")
            continue

        if idx < 1 or idx > len(rows):
            print("Некорректный номер локальности. Попробуйте снова.")
            continue

        row = rows[idx - 1]
        locality = row.get("locality", "").strip()
        if not locality:
            print("Не удалось определить имя локальности.")
            continue

        images_dir = Path(base_localities) / locality / "png"
        if not images_dir.is_dir():
            print(f"[ERR] Папка изображений не найдена: {images_dir}")
            continue

        # Обновляем статус до MANUAL перед запуском GUI
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_rows = []
        for r in rows:
            if (r.get("locality") or "") == locality:
                r = dict(r)
                r["status"] = "MANUAL"
                r["last_update"] = now
            updated_rows.append(r)
        try:
            _write_status_rows(status_file, updated_rows)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] Не удалось записать status/localities_status.csv: {exc}")
            continue

        # Определяем число ландмарков
        try:
            num_keypoints = read_num_keypoints()
        except HRNetConfigError as exc:
            print(f"[ERR] {exc}")
            continue

        try:
            run_gui_for_locality(images_dir=images_dir, lm_root=lm_root, num_keypoints=num_keypoints)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR] Ошибка при запуске GUI: {exc}")
            continue

        # После закрытия GUI пересобираем статусы
        try:
            rebuild_status(base_localities)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Не удалось обновить статус после аннотации: {exc}")

