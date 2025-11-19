"""Check how many HRNet parameters remain trainable after freezing low stages."""
from __future__ import annotations

from typing import Tuple

from scripts import hrnet_config_utils
from scripts.hrnet_model import HRNetW32GM


def _collect_param_examples(
    model: HRNetW32GM, max_examples: int = 5
) -> Tuple[list[str], list[str]]:
    frozen: list[str] = []
    trainable: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(trainable) < max_examples:
                trainable.append(name)
        else:
            if len(frozen) < max_examples:
                frozen.append(name)
        if len(frozen) >= max_examples and len(trainable) >= max_examples:
            break
    return frozen, trainable


def main() -> None:
    num_keypoints = hrnet_config_utils.read_num_keypoints()
    model = HRNetW32GM(num_keypoints=num_keypoints)

    if not getattr(model, "use_mmpose", False):
        print("[WARN] HRNet backbone from MMPose is unavailable; freeze test is skipped.")
        return

    model.freeze_low_stages()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    frozen_names, trainable_names = _collect_param_examples(model)

    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params}")
    print("Frozen parameter examples:")
    if frozen_names:
        for name in frozen_names:
            print(f"  {name}")
    else:
        print("  <none>")

    print("Trainable parameter examples:")
    if trainable_names:
        for name in trainable_names:
            print(f"  {name}")
    else:
        print("  <none>")


if __name__ == "__main__":
    main()
