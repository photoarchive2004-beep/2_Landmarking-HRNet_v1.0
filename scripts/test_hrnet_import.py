from __future__ import annotations

"""Diagnostic script to verify HRNet and MMPose imports."""

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("[ERR] PyTorch is required for test_hrnet_import.") from exc

try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
    MMPPOSE_HRNET_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"[WARN] Unable to import MMPose HRNet: {exc!r}")
    MMPoseHRNet = None  # type: ignore
    MMPPOSE_HRNET_AVAILABLE = False

from scripts import hrnet_config_utils
from scripts.train_hrnet import HRNetConfig, HRNetW32GM, get_landmark_root, load_yaml_config, read_lm_number


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

    _ = cfg  # reserved for future use; ensures config is parsed like in training

    model = HRNetW32GM(num_keypoints=num_keypoints)
    model.eval()

    use_mmpose = bool(getattr(model, "use_mmpose", False))
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Project root: {root}")
    print(f"Config file: {cfg_path}")
    print(f"Requested keypoints: {num_keypoints}")
    print(f"MMPose HRNet module importable: {MMPPOSE_HRNET_AVAILABLE}")
    print(f"Use MMPose HRNet backbone: {use_mmpose}")
    print(f"Total parameters: {total_params}")

    if not use_mmpose:
        print("WARNING: using simplified fallback backbone")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
