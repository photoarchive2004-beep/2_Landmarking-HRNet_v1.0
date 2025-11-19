"""Utility script to inspect the LR scheduler behaviour for HRNet training."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from scripts.train_hrnet import HRNetConfig


def main() -> None:
    cfg = HRNetConfig()
    max_epochs = max(1, int(cfg.max_epochs))
    param = nn.Parameter(torch.ones(1))
    optimizer = Adam([param], lr=float(cfg.learning_rate))
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(max_epochs * 0.5), int(max_epochs * 0.75)],
        gamma=0.1,
    )

    milestone_1 = int(max_epochs * 0.5)
    milestone_2 = int(max_epochs * 0.75)
    interesting_epochs = sorted(
        {
            0,
            milestone_1 - 1,
            milestone_1,
            milestone_2 - 1,
            milestone_2,
            max_epochs - 1,
        }
    )
    interesting_epochs = [e for e in interesting_epochs if 0 <= e < max_epochs]

    for epoch in range(max_epochs):
        if epoch in interesting_epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch} lr={current_lr:.6f}")
        scheduler.step()


if __name__ == "__main__":
    main()
