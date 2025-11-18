from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts import hrnet_config_utils
from scripts.hrnet_dataset import build_train_val_datasets
from scripts.hrnet_model import build_model_from_config


def _setup_logging(lm_root: Path) -> Path:
    logs_dir = lm_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "train_hrnet_last.log"
    log_path.write_text("", encoding="utf-8")
    return log_path


def _log(message: str, log_path: Path) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _prepare_dataloaders(base_localities: Path, cfg: Dict) -> Tuple[DataLoader, DataLoader, Dict]:
    train_ds, val_ds, meta = build_train_val_datasets(base_localities, cfg)
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 8))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, meta


def _pck_at_r(pred: torch.Tensor, gt: torch.Tensor, visible: torch.Tensor, radius: float) -> float:
    mask = visible > 0.5
    if mask.sum() == 0:
        return 0.0
    dist = torch.norm(pred - gt, dim=-1)
    correct = (dist <= radius) & mask
    return float(correct.sum().item()) / float(mask.sum().item())


def _extract_keypoints_from_heatmaps(heatmaps: torch.Tensor, downscale: int) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    idx = flat.argmax(dim=-1)
    ys = (idx // w).float() * float(downscale)
    xs = (idx % w).float() * float(downscale)
    return torch.stack([xs, ys], dim=-1)


def train_hrnet_model(base_localities: Path, lm_root: Path) -> None:
    log_path = _setup_logging(lm_root)
    try:
        cfg = hrnet_config_utils.load_hrnet_config()
    except Exception as exc:  # noqa: BLE001 - пользовательское сообщение
        _log(f"Failed to read HRNet config: {exc}", log_path)
        input("Press Enter to return to menu...")
        return

    try:
        train_loader, val_loader, meta = _prepare_dataloaders(base_localities, cfg)
    except Exception as exc:  # noqa: BLE001
        _log(f"Failed to prepare datasets: {exc}", log_path)
        input("Press Enter to return to menu...")
        return

    if len(train_loader.dataset) == 0:
        _log("No MANUAL localities with labeled images found. Nothing to train.", log_path)
        input("Press Enter to return to menu...")
        return

    model = build_model_from_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("train", {}).get("learning_rate", 5e-4)),
        weight_decay=float(cfg.get("train", {}).get("weight_decay", 1e-4)),
    )
    criterion = nn.MSELoss(reduction="none")

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 50))
    patience = int(cfg.get("train", {}).get("early_stop_patience", 10))
    best_pck = -1.0
    epochs_without_improve = 0

    run_id = time.strftime("%Y%m%d_%H%M%S")
    history_dir = lm_root / "models" / "history" / run_id
    history_dir.mkdir(parents=True, exist_ok=True)
    current_dir = lm_root / "models" / "current"
    current_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["heatmaps"].to(device)
            visible = batch["visible"].to(device)

            preds = model(images)
            if preds.shape != targets.shape:
                preds = torch.nn.functional.interpolate(preds, size=targets.shape[2:], mode="bilinear", align_corners=False)

            mask = visible.view(visible.shape[0], visible.shape[1], 1, 1)
            loss = criterion(preds, targets)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1
        avg_loss = running_loss / max(1, batches)
        _log(f"Epoch {epoch + 1}/{max_epochs} - train loss: {avg_loss:.6f}", log_path)

        model.eval()
        with torch.no_grad():
            all_pck = []
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["heatmaps"].to(device)
                visible = batch["visible"].to(device)
                keypoints = batch["keypoints"].to(device)
                downscale = int(batch["meta"]["downscale"][0]) if "downscale" in batch["meta"] else 4

                preds = model(images)
                if preds.shape != targets.shape:
                    preds = torch.nn.functional.interpolate(preds, size=targets.shape[2:], mode="bilinear", align_corners=False)
                pred_kp = _extract_keypoints_from_heatmaps(preds, downscale)
                radius = 0.05 * max(images.shape[2], images.shape[3])
                all_pck.append(_pck_at_r(pred_kp, keypoints, visible, radius))

            val_pck = float(sum(all_pck) / max(1, len(all_pck)))
        _log(f"Epoch {epoch + 1} - val PCK@R: {val_pck:.4f}", log_path)

        if val_pck > best_pck:
            best_pck = val_pck
            epochs_without_improve = 0
            best_path = history_dir / "hrnet_best.pth"
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), current_dir / "hrnet_best.pth")
            _log(f"New best model saved with PCK@R={best_pck:.4f}", log_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                _log("Early stopping triggered.", log_path)
                break

    quality = {
        "run_id": run_id,
        "num_keypoints": hrnet_config_utils.read_num_keypoints(),
        "n_train_images": meta.get("n_train_images", 0),
        "n_val_images": meta.get("n_val_images", 0),
        "localities": meta.get("localities", []),
        "pck_r": best_pck,
        "pck_r_percent": round(best_pck * 100, 2) if best_pck >= 0 else 0,
        "resize_enabled": cfg.get("resize", {}).get("enabled", False),
        "resize_long_side": cfg.get("resize", {}).get("long_side", None),
    }

    (current_dir / "quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")
    (history_dir / "metrics.json").write_text(
        json.dumps({"best_pck": best_pck}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (history_dir / "train_config.yaml").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        (history_dir / "train_log.txt").write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    except FileNotFoundError:
        pass

    _log("Training finished.", log_path)
    input("Press Enter to return to menu...")
