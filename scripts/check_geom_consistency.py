"""Utility to ensure train/infer geometry stay in sync."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

from scripts import hrnet_config_utils, hrnet_geom
from scripts.hrnet_dataset import _load_keypoints


def _compute_errors(image_path: Path, csv_path: Path) -> Tuple[float, float]:
    cfg = hrnet_config_utils.load_hrnet_config()
    resize_cfg = cfg.get("resize", {})
    num_keypoints = hrnet_config_utils.read_num_keypoints()

    image = Image.open(image_path).convert("RGB")
    _resized, prep = hrnet_geom.resize_with_config(image, resize_cfg)

    keypoints = _load_keypoints(csv_path, num_keypoints)
    projected = hrnet_geom.transform_keypoints(keypoints, prep, skip_invalid=True)
    heatmap_coords = projected / float(hrnet_geom.HEATMAP_DOWNSCALE)
    restored = hrnet_geom.inverse_transform_keypoints(
        heatmap_coords * float(hrnet_geom.HEATMAP_DOWNSCALE), prep, skip_invalid=False
    )

    valid = (keypoints[:, 0] >= 0) & (keypoints[:, 1] >= 0)
    if not np.any(valid):
        return 0.0, 0.0
    diff = np.abs(restored - keypoints)[valid]
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    return max_err, mean_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Check train/infer geometry consistency")
    parser.add_argument("image", type=Path, help="Path to sample PNG image")
    parser.add_argument("csv", type=Path, help="Path to CSV with the same landmarks")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Max allowed error in pixels")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}")

    max_err, mean_err = _compute_errors(args.image, args.csv)
    print(f"Max error: {max_err:.4f} px")
    print(f"Mean error: {mean_err:.4f} px")

    if max_err > args.tolerance:
        print(
            "[WARN] Geometry mismatch exceeds tolerance. ",
            "Check resize/padding settings.",
        )
        return 1
    print("[OK] Geometry consistency within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
