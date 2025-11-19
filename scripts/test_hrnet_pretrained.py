"""Utility script to verify HRNet-W32 pretrained backbone weights."""
from __future__ import annotations

import torch

from scripts.train_hrnet import HRNetW32GM


def _get_first_conv_weight(model: HRNetW32GM) -> torch.Tensor:
    if not getattr(model, "use_mmpose", False):
        raise RuntimeError("MMPose HRNet backbone is required for this test.")
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model does not expose a backbone attribute.")
    state_dict = backbone.state_dict()
    for key, tensor in state_dict.items():
        if tensor.ndim == 4 and "conv" in key.lower():
            return tensor.clone().cpu()
    return next(iter(state_dict.values())).clone().cpu()


def main() -> None:
    model_random = HRNetW32GM(num_keypoints=1, load_imagenet_pretrained=False)
    random_weight = _get_first_conv_weight(model_random)

    model_pretrained = HRNetW32GM(num_keypoints=1, load_imagenet_pretrained=True)
    pretrained_weight = _get_first_conv_weight(model_pretrained)

    mse = torch.mean((random_weight - pretrained_weight) ** 2).item()
    print(f"Conv1 weight MSE difference (random vs pretrained): {mse:.6f}")

if __name__ == "__main__":
    main()
