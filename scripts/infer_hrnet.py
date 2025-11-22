from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional heavy dependency
    import torch
    from torchvision import transforms
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]

from scripts import hrnet_config_utils, hrnet_geom
from scripts.hrnet_dataset import _collect_items, _read_status
from scripts.hrnet_model import HRNetW32GM
from scripts.train_hrnet import heatmaps_to_keypoints, read_last_base


def _log(msg: str) -> None:
    print(msg)


def _load_model(model_path: Path, num_keypoints: int) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetW32GM(num_keypoints)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _prepare_items(base_localities: Path, mode: str) -> List[dict]:
    status_rows = _read_status(base_localities)
    require_csv = mode == "debug"
    items = _collect_items(base_localities, status_rows, require_csv=require_csv)
    if mode == "debug":
        items = items[: min(32, len(items))]
    return items


def _predict_coordinates(
    model: torch.nn.Module,
    image: Image.Image,
    resize_cfg: dict,
    threshold: float,
) -> Tuple[np.ndarray, hrnet_geom.ResizePadResult]:
    resized, prep = hrnet_geom.resize_with_config(image, resize_cfg)
    tensor = transforms.ToTensor()(resized).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        heatmaps = model(tensor)
        target_h = resized.height // hrnet_geom.HEATMAP_DOWNSCALE
        target_w = resized.width // hrnet_geom.HEATMAP_DOWNSCALE
        if tuple(heatmaps.shape[2:]) != (target_h, target_w):
            heatmaps = torch.nn.functional.interpolate(
                heatmaps,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        coords_hm = heatmaps_to_keypoints(heatmaps)
        conf, _ = heatmaps.view(heatmaps.shape[0], heatmaps.shape[1], -1).max(dim=2)
        coords_hm[conf < threshold] = -1.0

    coords_resized = coords_hm.cpu().numpy() * float(hrnet_geom.HEATMAP_DOWNSCALE)
    coords_orig = hrnet_geom.inverse_transform_keypoints(coords_resized[0], prep, skip_invalid=False)
    return coords_orig, prep


def _save_csv(csv_path: Path, coords: np.ndarray) -> None:
    """Сохраняем ландмарки в формате аннотатора:
    одна строка: x1,y1,x2,y2,...,xN,yN
    """
    flat = np.asarray(coords, dtype=float).reshape(-1)
    line = ",".join(f"{v:.2f}" for v in flat)
    csv_path.write_text(line + "\n", encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inference for HRNet landmarking")
    parser.add_argument(
        "--mode",
        choices=["debug", "all", "full"],
        default="debug",
        help="Inference mode: debug (small subset) or all/full dataset.",
    )
    args = parser.parse_args(argv)

    root = hrnet_config_utils.PROJECT_ROOT
    cfg = hrnet_config_utils.load_hrnet_config()
    num_keypoints = hrnet_config_utils.read_num_keypoints()

    base_dir = read_last_base(root)
    if base_dir is None:
        _log("[ERR] cfg/last_base.txt не найден или папка с локальностями отсутствует.")
        return 1

    model_path = root / "models" / "current" / "hrnet_best.pth"
    if not model_path.is_file():
        _log(f"[ERR] Model file not found: {model_path}")
        return 1

    if torch is None:
        _log("[ERR] PyTorch is required for inference but is not installed.")
        return 1

    _log(f"Loading model from {model_path}")
    model = _load_model(model_path, num_keypoints)

    resize_cfg = cfg.get("resize", {})
    infer_cfg = cfg.get("infer", {}) or {}
    threshold = float(infer_cfg.get("threshold", 0.2))

    items = _prepare_items(base_dir, args.mode)
    if not items:
        _log("[ERR] No images found for inference.")
        return 1

    _log(f"Found {len(items)} images for mode {args.mode!r}")
    for item in items:
        img_path: Path = item["image_path"]
        image = Image.open(img_path).convert("RGB")
        coords, prep = _predict_coordinates(model, image, resize_cfg, threshold)
        csv_path = img_path.with_suffix(".hrnet.csv")
        _save_csv(csv_path, coords)
        finite_mask = np.isfinite(coords).all(axis=1)
        min_xy = coords[finite_mask].min(axis=0) if np.any(finite_mask) else np.array([-1.0, -1.0])
        max_xy = coords[finite_mask].max(axis=0) if np.any(finite_mask) else np.array([-1.0, -1.0])
        _log(
            f"{img_path} -> {csv_path} | orig_size={prep.orig_size} resized={prep.resized_size} "
            f"min={min_xy.tolist()} max={max_xy.tolist()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
