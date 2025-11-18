from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn

from scripts import hrnet_config_utils


class HRNetKeypointModel(nn.Module):
    def __init__(self, num_keypoints: int, model_type: str = "hrnet_w32") -> None:
        super().__init__()
        channels = 64 if model_type.lower() == "hrnet_w48" else 48
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for _ in range(4):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Conv2d(channels, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            residual = x
            x = self.relu(block(x) + residual)
        return self.head(x)


def build_model_from_config(cfg: Dict) -> HRNetKeypointModel:
    model_cfg = cfg.get("model", {})
    num_keypoints = hrnet_config_utils.read_num_keypoints()
    model_type = str(model_cfg.get("model_type", "hrnet_w32"))

    model = HRNetKeypointModel(num_keypoints=num_keypoints, model_type=model_type)
    weights_path_str = str(model_cfg.get("pretrained_weights") or "").strip()
    if weights_path_str:
        weights_path = Path(weights_path_str)
        if not weights_path.is_absolute():
            weights_path = (hrnet_config_utils.PROJECT_ROOT / weights_path).resolve()
        if weights_path.exists():
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
    return model
