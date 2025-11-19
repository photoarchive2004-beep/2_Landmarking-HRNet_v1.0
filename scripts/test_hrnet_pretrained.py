"""Utility script to verify HRNet-W32 pretrained backbone weights."""
from __future__ import annotations

from typing import Optional

import torch

from scripts.hrnet_model import HRNetW32GM


def _get_first_conv_weight(model: HRNetW32GM) -> Optional[torch.Tensor]:
    """Return weights of the first Conv2d layer inside the HRNet backbone."""

    if not getattr(model, "use_mmpose", False):
        return None

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model does not expose a backbone attribute.")

    for module in backbone.modules():
        if isinstance(module, torch.nn.Conv2d):
            return module.weight.detach().clone().cpu()

    state_dict = backbone.state_dict()
    for key, tensor in state_dict.items():
        if tensor.ndim == 4:
            return tensor.detach().clone().cpu()

    raise RuntimeError("Unable to locate convolutional weights in HRNet backbone.")


def main() -> None:
    model_random = HRNetW32GM(num_keypoints=1, load_imagenet_pretrained=False)
    random_weight = _get_first_conv_weight(model_random)

    model_pretrained = HRNetW32GM(num_keypoints=1, load_imagenet_pretrained=True)
    pretrained_weight = _get_first_conv_weight(model_pretrained)

    if random_weight is None or pretrained_weight is None:
        print("skip: fallback backbone in use, HRNet pretrained check is not applicable.")
        return

    mse = torch.mean((random_weight - pretrained_weight) ** 2).item()
    print(f"Conv1 weight MSE difference (random vs pretrained): {mse:.6f}")

if __name__ == "__main__":
    main()
