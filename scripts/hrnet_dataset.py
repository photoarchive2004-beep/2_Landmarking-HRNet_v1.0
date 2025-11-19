from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from scripts import hrnet_config_utils
from scripts import hrnet_geom


def _load_keypoints(csv_path: Path, num_keypoints: int) -> np.ndarray:
    coords = np.full((num_keypoints, 2), -1.0, dtype=np.float32)
    if not csv_path.exists():
        return coords
    raw = csv_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return coords
    first_line = raw.splitlines()[0]
    parts = [p for p in first_line.split(",") if p.strip()]
    values: List[float] = []
    for cell in parts:
        try:
            values.append(float(cell))
        except ValueError:
            values.append(-1.0)
    for idx in range(num_keypoints):
        j = idx * 2
        if j + 1 >= len(values):
            break
        coords[idx, 0] = values[j]
        coords[idx, 1] = values[j + 1]
    return coords
def _apply_affine_to_points(
    points: torch.Tensor,
    angle_deg: float,
    scale: float,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    if points.numel() == 0:
        return points
    valid_mask = (points[:, 0] >= 0) & (points[:, 1] >= 0)
    if not torch.any(valid_mask):
        return points
    cx = image_size[0] / 2.0
    cy = image_size[1] / 2.0
    rad = math.radians(angle_deg)
    rot = torch.tensor(
        [[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=torch.float32
    )
    shifted = points[valid_mask] - torch.tensor([cx, cy])
    shifted = shifted @ rot.T * scale
    transformed = shifted + torch.tensor([cx, cy])
    transformed[:, 0].clamp_(0.0, max(0.0, image_size[0] - 1))
    transformed[:, 1].clamp_(0.0, max(0.0, image_size[1] - 1))
    result = points.clone()
    result[valid_mask] = transformed
    return result


def _maybe_augment(
    image: torch.Tensor, keypoints: torch.Tensor, cfg: Dict
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    aug_cfg = cfg
    _, h, w = image.shape
    info: Dict[str, Any] = {
        "horizontal_flip": False,
        "rotation_deg": 0.0,
        "scale_factor": 1.0,
        "color_jitter": {"applied": False},
    }

    rot_prob = float(aug_cfg.get("rotation_augmentation_prob", aug_cfg.get("rotation_prob", 0.0)))
    rot_max = float(aug_cfg.get("rotation_augmentation_deg", aug_cfg.get("rotation_deg", 0.0)))
    if rot_max > 0.0 and random.random() < rot_prob:
        angle = random.uniform(-rot_max, rot_max)
        image = TF.rotate(image, angle, fill=0)
        keypoints = _apply_affine_to_points(keypoints, angle, 1.0, (w, h))
        info["rotation_deg"] = angle

    scale_margin = float(aug_cfg.get("scale_augmentation", aug_cfg.get("scale_min", 0.0)))
    scale_prob = float(aug_cfg.get("scale_augmentation_prob", aug_cfg.get("scale_prob", 0.0)))
    if scale_margin > 0.0 and random.random() < scale_prob:
        scale_min = max(0.01, 1.0 - scale_margin)
        scale_max = 1.0 + scale_margin
        scale = random.uniform(scale_min, scale_max)
        image = TF.affine(image, angle=0.0, translate=[0, 0], scale=scale, shear=[0.0, 0.0], fill=0)
        keypoints = _apply_affine_to_points(keypoints, 0.0, scale, (w, h))
        info["scale_factor"] = scale

    flip_enabled = bool(aug_cfg.get("flip_augmentation", aug_cfg.get("horizontal_flip", False)))
    if flip_enabled and random.random() < 0.5:
        image = TF.hflip(image)
        keypoints = keypoints.clone()
        valid_mask = (keypoints[:, 0] >= 0) & (keypoints[:, 1] >= 0)
        keypoints[valid_mask, 0] = float(w - 1) - keypoints[valid_mask, 0]
        info["horizontal_flip"] = True

    color_cfg = aug_cfg.get("color_jitter", {})
    color_prob = float(color_cfg.get("prob", 0.0))
    if color_cfg.get("enabled", False) and random.random() < color_prob:
        brightness = float(color_cfg.get("brightness", 0.0))
        contrast = float(color_cfg.get("contrast", 0.0))
        saturation = float(color_cfg.get("saturation", 0.0))
        cj_info: Dict[str, Any] = {"applied": True}
        if brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
            image = TF.adjust_brightness(image, factor)
            cj_info["brightness_factor"] = factor
        if contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            image = TF.adjust_contrast(image, factor)
            cj_info["contrast_factor"] = factor
        if saturation > 0:
            factor = random.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
            image = TF.adjust_saturation(image, factor)
            cj_info["saturation_factor"] = factor
        info["color_jitter"] = cj_info

    return image, keypoints, info


def _build_heatmaps(keypoints: torch.Tensor, visible: torch.Tensor, height: int, width: int, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    yy = yy.unsqueeze(0)
    xx = xx.unsqueeze(0)
    keypoints = keypoints.unsqueeze(-1).unsqueeze(-1)
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    exp = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
    heatmaps = exp * visible.view(-1, 1, 1)
    return heatmaps


class LandmarkDataset(Dataset):
    def __init__(self, items: List[Dict], num_keypoints: int, cfg: Dict, resize_cfg: Dict, augment_cfg: Dict, phase: str = "train") -> None:
        self.items = items
        self.num_keypoints = num_keypoints
        self.cfg = cfg
        self.resize_cfg = resize_cfg
        self.augment_cfg = augment_cfg
        self.phase = phase

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        image, prep = hrnet_geom.resize_with_config(image, self.resize_cfg)

        keypoints_np = _load_keypoints(item["csv_path"], self.num_keypoints)
        keypoints_np = hrnet_geom.transform_keypoints(keypoints_np, prep, skip_invalid=True)
        visible = (keypoints_np[:, 0] >= 0) & (keypoints_np[:, 1] >= 0)

        image_tensor = TF.to_tensor(image)
        keypoints = torch.tensor(keypoints_np, dtype=torch.float32)
        visible_tensor = torch.tensor(visible.astype(np.float32))

        aug_info: Dict[str, Any] = {}
        if self.phase == "train":
            image_tensor, keypoints, aug_info = _maybe_augment(image_tensor, keypoints, self.augment_cfg)

        _, H, W = image_tensor.shape
        down = hrnet_geom.HEATMAP_DOWNSCALE
        Hh = max(1, H // down)
        Wh = max(1, W // down)
        sigma = float(self.cfg.get("data", {}).get("heatmap_sigma_px", 4)) / down
        keypoints_hm = keypoints / float(down)
        heatmaps = _build_heatmaps(keypoints_hm, visible_tensor, Hh, Wh, sigma)

        return {
            "image": image_tensor,
            "heatmaps": heatmaps,
            "visible": visible_tensor,
            "keypoints": keypoints,
            "meta": {
                "locality": item["locality"],
                "orig_size": prep.orig_size,
                "resized_size": prep.resized_size,
                "scale": prep.scale,
                "pad_x": prep.pad_x,
                "pad_y": prep.pad_y,
                "downscale": down,
                "augmentations": aug_info,
            },
        }


def _collect_items(base_localities: Path, status_rows: List[Dict[str, str]]) -> List[Dict]:
    items: List[Dict] = []
    for row in status_rows:
        if (row.get("status") or "").strip().upper() != "MANUAL":
            continue
        locality = row.get("locality") or ""
        png_dir = base_localities / locality / "png"
        if not png_dir.is_dir():
            continue
        for img_path in sorted(png_dir.glob("*.png")):
            csv_path = img_path.with_suffix(".csv")
            if not csv_path.exists():
                continue
            items.append({"image_path": img_path, "csv_path": csv_path, "locality": locality})
    return items


def _read_status(base_localities: Path) -> List[Dict[str, str]]:
    status_file = hrnet_config_utils.PROJECT_ROOT / "status" / "localities_status.csv"
    if not status_file.exists():
        return []
    rows: List[Dict[str, str]] = []
    with status_file.open("r", encoding="utf-8", newline="") as handle:
        headers = handle.readline().strip().split(",")
        for line in handle:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or len(parts) < len(headers):
                continue
            rows.append({h: parts[i] for i, h in enumerate(headers)})
    return rows


def build_train_val_datasets(base_localities: Path, cfg: Dict) -> Tuple[Dataset, Dataset, Dict]:
    rows = _read_status(base_localities)
    items = _collect_items(base_localities, rows)
    random.shuffle(items)

    split = float(cfg.get("train", {}).get("train_val_split", 0.9))
    pivot = int(len(items) * split)
    train_items = items[:pivot]
    val_items = items[pivot:]

    num_keypoints = hrnet_config_utils.read_num_keypoints()
    resize_cfg = cfg.get("resize", {})
    augment_cfg = cfg.get("augment", {})

    train_ds = LandmarkDataset(train_items, num_keypoints, cfg, resize_cfg, augment_cfg, phase="train")
    val_ds = LandmarkDataset(val_items, num_keypoints, cfg, resize_cfg, augment_cfg, phase="val")

    metadata = {
        "localities": sorted({item["locality"] for item in items}),
        "n_train_images": len(train_items),
        "n_val_images": len(val_items),
        "train_share": float(len(train_items)) / float(max(1, len(items))),
        "val_share": float(len(val_items)) / float(max(1, len(items))),
    }
    return train_ds, val_ds, metadata
