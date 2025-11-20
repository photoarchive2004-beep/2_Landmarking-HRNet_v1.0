from __future__ import annotations

import torch

from scripts import hrnet_config_utils
from scripts.hrnet_model import HRNetW32GM


def main() -> int:
    num_kpts_txt = hrnet_config_utils.read_num_keypoints()
    cfg = hrnet_config_utils.load_hrnet_config()
    num_kpts_cfg = cfg["model"]["num_keypoints"]
    train_cfg = cfg.get("train", {})
    max_epochs = train_cfg.get("max_epochs", 200)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetW32GM(num_keypoints=num_kpts_cfg)
    model = model.to(device)

    output_channels = None
    head = getattr(model, "head", None)
    if head is not None:
        output_channels = getattr(head, "out_channels", None)
    if output_channels is None and hasattr(model, "fallback"):
        fallback = getattr(model, "fallback", None)
        if fallback is not None:
            output_channels = getattr(getattr(fallback, "head", None), "out_channels", None)

    print(f"LM_number.txt num_keypoints: {num_kpts_txt}")
    print(f"config model.num_keypoints: {num_kpts_cfg}")
    print(f"model output channels: {output_channels}")
    print(f"train.max_epochs: {max_epochs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
