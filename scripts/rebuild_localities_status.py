"""Rebuild status/localities_status.csv based on a localities base folder."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


LM_ROOT = Path(__file__).resolve().parent.parent
STATUS_HEADER = [
    "locality",
    "status",
    "auto_quality",
    "last_model_run",
    "n_images",
    "n_labeled",
    "last_update",
]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild localities status table.")
    parser.add_argument(
        "--base", "-b", dest="base", help="Path to base localities directory", required=False
    )
    return parser.parse_args()


def ensure_status_file() -> Path:
    status_file = LM_ROOT / "status" / "localities_status.csv"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    if not status_file.exists():
        with status_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(STATUS_HEADER)
    return status_file


def load_existing_status(status_file: Path) -> Dict[str, Dict[str, str]]:
    if not status_file.exists():
        return {}

    with status_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing = {}
        for row in reader:
            locality = row.get("locality")
            if not locality:
                continue
            existing[locality] = row
    return existing


def scan_localities(base_dir: Path) -> List[tuple[str, int, int]]:
    localities: List[tuple[str, int, int]] = []
    for locality_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        png_dir = locality_dir / "png"
        if not png_dir.is_dir():
            continue

        images = [p for p in png_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
        n_images = len(images)
        if n_images == 0:
            continue

        labeled = 0
        for image in images:
            if (png_dir / f"{image.stem}.csv").exists():
                labeled += 1

        localities.append((locality_dir.name, n_images, labeled))
    return localities


def build_rows(existing: Dict[str, Dict[str, str]], scanned: List[tuple[str, int, int]]):
    rows = []
    for locality, n_images, n_labeled in scanned:
        previous = existing.get(locality, {})
        status = (previous.get("status") or "").strip()
        auto_quality = (previous.get("auto_quality") or "").strip()
        last_model_run = (previous.get("last_model_run") or "").strip()
        last_update = (previous.get("last_update") or "").strip()

        if not status and n_images > 0 and n_images == n_labeled:
            status = "MANUAL"

        rows.append(
            {
                "locality": locality,
                "status": status,
                "auto_quality": auto_quality,
                "last_model_run": last_model_run,
                "n_images": str(n_images),
                "n_labeled": str(n_labeled),
                "last_update": last_update,
            }
        )
    rows.sort(key=lambda row: row["locality"])
    return rows


def write_status(status_file: Path, rows) -> None:
    with status_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if not args.base:
        print("Error: --base argument is required (path to localities base).", file=sys.stderr)
        return 1

    base_dir = Path(args.base)
    if not base_dir.exists():
        print(f"Error: base path does not exist: {base_dir}", file=sys.stderr)
        return 1
    if not base_dir.is_dir():
        print(f"Error: base path is not a directory: {base_dir}", file=sys.stderr)
        return 1

    status_file = ensure_status_file()
    existing = load_existing_status(status_file)
    scanned = scan_localities(base_dir)
    rows = build_rows(existing, scanned)
    write_status(status_file, rows)
    print(f"Updated status for {len(rows)} localities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
