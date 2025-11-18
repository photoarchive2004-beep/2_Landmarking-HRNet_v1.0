"""Utility to initialize project folder structure and default configs."""
from __future__ import annotations

from pathlib import Path


LM_ROOT = Path(__file__).resolve().parent.parent

DIRECTORIES = [
    "cfg",
    "config",
    "logs",
    "models",
    "models/current",
    "models/history",
    "datasets",
    "status",
]

STATUS_HEADER = "locality,status,auto_quality,last_model_run,n_images,n_labeled,last_update\n"

DEFAULT_HRNET_CONFIG = """# Default HRNet config for 2_Landmarking-HRNet_v1.0 (auto-created)
model:
  pretrained_weights: ""     # путь к предобученным весам или пусто
  model_type: "hrnet_w32"    # тип модели
  num_keypoints: 20          # число ландмарков, можно править вручную

resize:
  enabled: true
  long_side: 1280
  keep_aspect_ratio: true

train:
  train_val_split: 0.9
  max_epochs: 80
  batch_size: 8
  learning_rate: 0.0005
  weight_decay: 0.0001
  early_stop_patience: 10

augment:
  rotation_deg: 15
  rotation_prob: 0.5
  scale_min: 0.8
  scale_max: 1.2
  scale_prob: 0.5
  brightness: 0.2
  contrast: 0.2
  color_prob: 0.3
  horizontal_flip: false
  vertical_flip: false

data:
  heatmap_sigma_px: 4

infer:
  threshold: 0.2
"""


def ensure_directories() -> None:
    """Create required project directories under LM_ROOT."""

    for subdir in DIRECTORIES:
        (LM_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def ensure_status_file() -> Path:
    """Ensure the status CSV exists with the expected header."""

    status_file = LM_ROOT / "status" / "localities_status.csv"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    if not status_file.exists():
        status_file.write_text(STATUS_HEADER, encoding="utf-8")
    return status_file


def ensure_hrnet_config() -> Path:
    """Ensure the default HRNet config exists."""

    config_file = LM_ROOT / "config" / "hrnet_config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        config_file.write_text(DEFAULT_HRNET_CONFIG, encoding="utf-8")
    return config_file


def main() -> int:
    ensure_directories()
    ensure_status_file()
    ensure_hrnet_config()
    print("Project structure initialized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
