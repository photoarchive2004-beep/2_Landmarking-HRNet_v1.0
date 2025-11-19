from __future__ import annotations

"""Diagnostic script to verify HRNet and MMPose imports."""

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("[ERR] PyTorch is required for test_hrnet_import.") from exc

from scripts import hrnet_config_utils
from scripts.hrnet_model import HRNetW32GM
from scripts.train_hrnet import HRNetConfig, get_landmark_root, load_yaml_config, read_lm_number


def _determine_num_keypoints(root: Path) -> int:
    """Return number of landmarks using same priority as training script."""

    try:
        lm_from_txt = read_lm_number(root)
        if lm_from_txt > 0:
            return lm_from_txt
    except Exception as exc:
        print(f"[WARN] Failed to read LM_number.txt: {exc}")

    try:
        return hrnet_config_utils.read_num_keypoints()
    except Exception as exc:
        print(f"[WARN] Could not read num_keypoints from config: {exc}")

    print("[INFO] Falling back to 5 keypoints for diagnostics.")
    return 5


def _load_config(cfg_path: Path) -> HRNetConfig:
    try:
        return load_yaml_config(cfg_path)
    except Exception as exc:
        print(f"[WARN] Could not load hrnet_config.yaml: {exc}")
        return HRNetConfig()


def main() -> int:
    root = get_landmark_root()
    cfg_path = root / "config" / "hrnet_config.yaml"
    cfg = _load_config(cfg_path)
    num_keypoints = _determine_num_keypoints(root)

    model_type = (cfg.model_type or "").lower()
    if not model_type.startswith("hrnet_w32"):
        print(
            f"[WARN] model_type={cfg.model_type!r} не поддерживается. "
            "Используем HRNet-W32 (как и train_hrnet)."
        )

    model = HRNetW32GM(num_keypoints=num_keypoints)
    model.eval()

    use_mmpose = bool(getattr(model, "use_mmpose", False))
    total_params = sum(p.numel() for p in model.parameters())
    backbone_name = "MMPose HRNet-W32" if use_mmpose else "fallback SimpleHRNet"

    print(f"Project root: {root}")
    print(f"Config file: {cfg_path}")
    print(f"Requested keypoints: {num_keypoints}")
    print(f"Use MMPose HRNet backbone: {use_mmpose}")
    print(f"Backbone in use: {backbone_name}")
    print(f"Total parameters: {total_params}")

    if not use_mmpose:
        print("WARNING: using simplified fallback backbone")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
