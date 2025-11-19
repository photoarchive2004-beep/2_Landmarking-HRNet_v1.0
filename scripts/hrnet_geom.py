"""Utilities for geometry preprocessing shared between train/infer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from PIL import Image

HEATMAP_DOWNSCALE = 4


@dataclass
class ResizePadResult:
    """Information about resize+pad transform applied to an image."""

    scale: float
    pad_x: int
    pad_y: int
    orig_size: Tuple[int, int]
    resized_size: Tuple[int, int]


def _identity_result(image: Image.Image) -> ResizePadResult:
    w, h = image.size
    return ResizePadResult(scale=1.0, pad_x=0, pad_y=0, orig_size=(w, h), resized_size=(w, h))


def _resize_keep_aspect(image: Image.Image, target: int) -> Tuple[Image.Image, ResizePadResult]:
    w, h = image.size
    if target <= 0 or max(w, h) == 0:
        return image, _identity_result(image)

    scale = float(target) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target, target), (0, 0, 0))
    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    info = ResizePadResult(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_size=(w, h),
        resized_size=(target, target),
    )
    return canvas, info


def _resize_without_pad(image: Image.Image, target: int) -> Tuple[Image.Image, ResizePadResult]:
    w, h = image.size
    if target <= 0 or max(w, h) == 0:
        return image, _identity_result(image)
    scale = float(target) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    info = ResizePadResult(
        scale=scale,
        pad_x=0,
        pad_y=0,
        orig_size=(w, h),
        resized_size=(new_w, new_h),
    )
    return resized, info


def resize_with_config(image: Image.Image, resize_cfg: Dict) -> Tuple[Image.Image, ResizePadResult]:
    """Apply resize/padding according to hrnet_config ``resize`` block."""

    if not resize_cfg or not resize_cfg.get("enabled", False):
        return image, _identity_result(image)

    target = int(resize_cfg.get("long_side", max(image.size)))
    keep_aspect = bool(resize_cfg.get("keep_aspect_ratio", True))
    if keep_aspect:
        return _resize_keep_aspect(image, target)
    return _resize_without_pad(image, target)


def transform_keypoints(
    keypoints: np.ndarray,
    transform: ResizePadResult,
    skip_invalid: bool = True,
) -> np.ndarray:
    """Project keypoints from original image into resized coordinates."""

    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape (N, 2)")
    result = np.full_like(points, -1.0, dtype=np.float32)
    if points.size == 0:
        return result

    mask = np.ones(points.shape[0], dtype=bool)
    if skip_invalid:
        mask &= (points[:, 0] >= 0) & (points[:, 1] >= 0)

    result[mask, 0] = points[mask, 0] * transform.scale + float(transform.pad_x)
    result[mask, 1] = points[mask, 1] * transform.scale + float(transform.pad_y)
    return result


def inverse_transform_keypoints(
    keypoints: np.ndarray,
    transform: ResizePadResult,
    skip_invalid: bool = False,
) -> np.ndarray:
    """Map keypoints from resized image coordinates back to the original image."""

    points = np.asarray(keypoints, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("keypoints must have shape (N, 2)")
    result = np.full_like(points, -1.0, dtype=np.float32)
    if points.size == 0 or transform.scale == 0:
        return result

    mask = np.ones(points.shape[0], dtype=bool)
    if skip_invalid:
        mask &= (points[:, 0] >= 0) & (points[:, 1] >= 0)

    result[mask, 0] = (points[mask, 0] - float(transform.pad_x)) / transform.scale
    result[mask, 1] = (points[mask, 1] - float(transform.pad_y)) / transform.scale
    return result
