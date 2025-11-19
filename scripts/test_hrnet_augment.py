"""Diagnostics for HRNet dataset augmentations."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from scripts import hrnet_config_utils
from scripts.hrnet_dataset import build_train_val_datasets


def _read_last_base() -> Path:
    base_txt = hrnet_config_utils.PROJECT_ROOT / "cfg" / "last_base.txt"
    if not base_txt.exists():
        raise RuntimeError(
            "cfg/last_base.txt не найден. Сначала выбери базу локальностей через choose_localities.py."
        )
    raw = base_txt.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError("cfg/last_base.txt пуст. Укажи путь к BASE_LOCALITIES.")
    base_path = Path(raw)
    if not base_path.is_dir():
        raise RuntimeError(f"Папка с локальностями недоступна: {base_path}")
    return base_path


def _format_keypoints(keypoints: torch.Tensor, limit: int = 3) -> str:
    preview = []
    for idx in range(min(limit, keypoints.shape[0])):
        x, y = keypoints[idx].tolist()
        preview.append(f"kp{idx}=(x={x:.1f}, y={y:.1f})")
    return ", ".join(preview) if preview else "<no keypoints>"


def _describe_aug(aug_meta: Dict) -> str:
    if not aug_meta:
        return "augmentations: <none>"
    flip = "flip" if aug_meta.get("horizontal_flip") else "no-flip"
    rot = aug_meta.get("rotation_deg", 0.0)
    scale = aug_meta.get("scale_factor", 1.0)
    color = aug_meta.get("color_jitter", {})
    cj = "color" if color.get("applied") else "no-color"
    return f"augmentations: {flip}, rot={rot:.2f} deg, scale={scale:.3f}, {cj}"


def main(iterations: int = 5, sample_index: int = 0) -> None:
    cfg = hrnet_config_utils.load_hrnet_config()
    base_localities = _read_last_base()
    train_ds, _, ds_meta = build_train_val_datasets(base_localities, cfg)
    if len(train_ds) == 0:
        print("Train dataset is empty. Проверь статус локальностей MANUAL.")
        return

    print(f"Train dataset size: {len(train_ds)} images")
    print(f"Localities in split: {', '.join(ds_meta.get('localities', []))}")
    print(f"Testing sample index: {sample_index}")

    sample_index = max(0, min(sample_index, len(train_ds) - 1))
    for step in range(1, iterations + 1):
        sample = train_ds[sample_index]
        image_tensor: torch.Tensor = sample["image"]
        keypoints: torch.Tensor = sample["keypoints"]
        meta: Dict[str, Dict | float | int] = sample["meta"]
        aug_meta = meta.get("augmentations", {})

        print(f"\nIteration {step}:")
        print(f"  image tensor shape: {tuple(image_tensor.shape)}")
        print(f"  first keypoints: {_format_keypoints(keypoints)}")
        print(f"  {_describe_aug(aug_meta)}")


if __name__ == "__main__":
    main()
