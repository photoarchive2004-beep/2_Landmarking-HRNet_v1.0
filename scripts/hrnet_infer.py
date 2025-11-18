from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from scripts import hrnet_config_utils
from scripts.hrnet_dataset import _resize_image
from scripts.hrnet_model import build_model_from_config


def _setup_logging(lm_root: Path) -> Path:
    log_dir = lm_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "infer_hrnet_last.log"
    log_path.write_text("", encoding="utf-8")
    return log_path


def _log(message: str, log_path: Path) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _read_status_rows(status_file: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not status_file.exists():
        return rows
    with status_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    return rows


def _write_status_rows(status_file: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = rows[0].keys()
    with status_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _select_locality(candidates: List[Dict[str, str]]) -> str | None:
    if not candidates:
        return None
    print("Available localities for autolabel:")
    for idx, row in enumerate(candidates, start=1):
        print(f"  {idx}) {row['locality']}  labeled {row['n_labeled']}/{row['n_images']}")
    print("  0) Cancel")
    while True:
        choice = input("Select locality number: ").strip()
        if choice == "0":
            return None
        try:
            num = int(choice)
        except ValueError:
            print("Invalid input. Try again.")
            continue
        if 1 <= num <= len(candidates):
            return candidates[num - 1]["locality"]
        print("Invalid choice. Try again.")


def _prepare_candidates(base_localities: Path, status_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    candidates: List[Dict[str, str]] = []
    for row in status_rows:
        locality = row.get("locality") or ""
        png_dir = base_localities / locality / "png"
        if not png_dir.is_dir():
            continue
        images = list(png_dir.glob("*.png"))
        if not images:
            continue
        labeled = sum(1 for img in images if img.with_suffix(".csv").exists())
        if labeled < len(images):
            row = dict(row)
            row["n_images"] = str(len(images))
            row["n_labeled"] = str(labeled)
            candidates.append(row)
    return candidates


def _heatmaps_to_coords(heatmaps: torch.Tensor, downscale: int, threshold: float) -> np.ndarray:
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // w).float() * float(downscale)
    xs = (idx % w).float() * float(downscale)
    coords = torch.stack([xs, ys], dim=-1)
    coords[conf < threshold] = torch.tensor([-1.0, -1.0], device=heatmaps.device)
    return coords[0].cpu().numpy()


def _save_csv(csv_path: Path, coords: np.ndarray) -> None:
    flat = []
    for x, y in coords:
        flat.extend([f"{float(x):.2f}", f"{float(y):.2f}"])
    csv_path.write_text(",".join(flat), encoding="utf-8")


def autolabel_locality(base_localities: Path, lm_root: Path) -> None:
    log_path = _setup_logging(lm_root)
    cfg = hrnet_config_utils.load_hrnet_config()
    model_path = lm_root / "models" / "current" / "hrnet_best.pth"
    quality_path = lm_root / "models" / "current" / "quality.json"
    if not model_path.exists():
        _log("No trained model found (models/current/hrnet_best.pth missing).", log_path)
        input("Press Enter to return to menu...")
        return

    model = build_model_from_config(cfg)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    status_file = lm_root / "status" / "localities_status.csv"
    status_rows = _read_status_rows(status_file)
    candidates = _prepare_candidates(base_localities, status_rows)
    locality = _select_locality(candidates)
    if locality is None:
        return

    png_dir = base_localities / locality / "png"
    images = sorted(png_dir.glob("*.png"))
    unlabeled = [img for img in images if not img.with_suffix(".csv").exists()]
    if not unlabeled:
        _log("All images already labeled for this locality.", log_path)
        input("Press Enter to return to menu...")
        return

    resize_cfg = cfg.get("resize", {})
    threshold = float(cfg.get("infer", {}).get("threshold", 0.2))

    for img_path in unlabeled:
        image = Image.open(img_path).convert("RGB")
        resized, prep = _resize_image(image, resize_cfg)
        tensor = TF.to_tensor(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            heatmaps = model(tensor)
            if heatmaps.shape[2:] != (resized.height // 4, resized.width // 4):
                heatmaps = torch.nn.functional.interpolate(
                    heatmaps, size=(resized.height // 4, resized.width // 4), mode="bilinear", align_corners=False
                )
            coords_resized = _heatmaps_to_coords(heatmaps, 4, threshold)

        coords_orig = coords_resized.copy()
        coords_orig[:, 0] = (coords_resized[:, 0] - prep.pad_x) / prep.scale
        coords_orig[:, 1] = (coords_resized[:, 1] - prep.pad_y) / prep.scale
        _save_csv(img_path.with_suffix(".csv"), coords_orig)
        _log(f"Autolabeled {img_path.name}", log_path)

    run_id = None
    if quality_path.exists():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            run_id = quality.get("run_id")
        except Exception:  # noqa: BLE001 - не мешаем авторазметке
            run_id = None

    updated = []
    for row in status_rows:
        if (row.get("locality") or "") == locality:
            row = dict(row)
            row["status"] = row.get("status") or "AUTO"
            if run_id:
                row["last_model_run"] = str(run_id)
            row["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
            png_dir = base_localities / locality / "png"
            images = list(png_dir.glob("*.png"))
            row["n_images"] = str(len(images))
            row["n_labeled"] = str(sum(1 for img in images if img.with_suffix(".csv").exists()))
        updated.append(row)
    _write_status_rows(status_file, updated)

    _log(f"Autolabel finished for locality {locality}.", log_path)
    input("Press Enter to return to menu...")
