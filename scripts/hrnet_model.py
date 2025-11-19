from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - heavy optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency that might be missing during tests
    from mmpose.models.backbones import HRNet as MMPoseHRNet  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"[WARN] Unable to import MMPose HRNet: {exc!r}")
    MMPoseHRNet = None  # type: ignore

from scripts import hrnet_config_utils

__all__ = ["HRNetW32GM", "build_model_from_config"]


def _build_hrnet_extra_config() -> Dict:
    return {
        "stage1": dict(
            num_modules=1,
            num_branches=1,
            block="BOTTLENECK",
            num_blocks=(4,),
            num_channels=(64,),
        ),
        "stage2": dict(
            num_modules=1,
            num_branches=2,
            block="BASIC",
            num_blocks=(4, 4),
            num_channels=(32, 64),
        ),
        "stage3": dict(
            num_modules=4,
            num_branches=3,
            block="BASIC",
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128),
        ),
        "stage4": dict(
            num_modules=3,
            num_branches=4,
            block="BASIC",
            num_blocks=(4, 4, 4, 4),
            num_channels=(32, 64, 128, 256),
        ),
    }


if torch is not None:

    class SimpleHRNet(nn.Module):
        """Очень упрощённая сеть на случай отсутствия MMPose."""

        def __init__(self, num_keypoints: int) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            blocks = []
            in_channels = 64
            for _ in range(3):
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_channels),
                )
                blocks.append(block)
            self.blocks = nn.ModuleList(blocks)
            self.relu = nn.ReLU(inplace=True)
            self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.stem(x)
            for block in self.blocks:
                residual = x
                out = block(x)
                x = self.relu(out + residual)
            return self.head(x)


    class HRNetW32GM(nn.Module):
        """HRNet-W32 backbone из MMPose + heatmap head."""

        def __init__(self, num_keypoints: int, load_imagenet_pretrained: bool = True) -> None:
            super().__init__()
            if torch is None:  # pragma: no cover
                raise RuntimeError("PyTorch is required for HRNetW32GM but is not installed.")

            self.num_keypoints = int(num_keypoints)
            self.use_mmpose = MMPoseHRNet is not None
            self._load_pretrained = bool(load_imagenet_pretrained)

            if self.use_mmpose:
                print("[INFO] Using MMPose HRNet backbone for HRNetW32GM")
                extra = _build_hrnet_extra_config()
                self.backbone = MMPoseHRNet(extra=extra, in_channels=3)  # type: ignore[call-arg]
                if self._load_pretrained:
                    self._load_imagenet_pretrained_backbone()
                self.head = nn.Conv2d(32, self.num_keypoints, kernel_size=1)
                self.fallback = None
            else:
                print(
                    "[WARN] MMPose HRNet backbone unavailable. Using simplified fallback network."
                )
                self.fallback = SimpleHRNet(num_keypoints)
                self.backbone = None
                self.head = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_mmpose:
                feats = self.backbone(x)
                feats0 = feats[0] if isinstance(feats, (list, tuple)) else feats
                return self.head(feats0)
            return self.fallback(x)

        def _load_imagenet_pretrained_backbone(self) -> None:
            try:  # pragma: no cover - optional dependency
                import timm
            except ImportError:
                print("[WARN] timm is not installed. Skipping ImageNet pretrained weights.")
                return

            timm_model = timm.create_model("hrnet_w32", pretrained=True)
            timm_state = timm_model.state_dict()
            backbone_state = self.backbone.state_dict()
            partial_state = {}
            matched = 0
            for key, tensor in timm_state.items():
                if key in backbone_state and backbone_state[key].shape == tensor.shape:
                    partial_state[key] = tensor.to(dtype=backbone_state[key].dtype)
                    matched += 1

            total = len(backbone_state)
            self.backbone.load_state_dict(partial_state, strict=False)
            print(
                f"Loaded ImageNet pretrained HRNet-W32 backbone weights (matched = {matched}, total = {total})"
            )


else:  # torch is None

    class SimpleHRNet:  # type: ignore[misc]
        def __init__(self, num_keypoints: int) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for SimpleHRNet but is not installed.")

    class HRNetW32GM:  # type: ignore[misc]
        def __init__(self, num_keypoints: int, load_imagenet_pretrained: bool = True) -> None:  # pragma: no cover
            raise RuntimeError("PyTorch is required for HRNetW32GM but is not installed.")


def build_model_from_config(
    cfg: Dict, *, load_imagenet_pretrained: Optional[bool] = None
) -> HRNetW32GM:
    """Создаём HRNet-модель по hrnet_config.yaml и, при необходимости, подгружаем веса."""

    model_cfg = cfg.get("model", {})
    num_keypoints = hrnet_config_utils.read_num_keypoints()
    if load_imagenet_pretrained is None:
        load_imagenet_pretrained = bool(model_cfg.get("load_imagenet_pretrained", True))

    model = HRNetW32GM(
        num_keypoints=num_keypoints,
        load_imagenet_pretrained=load_imagenet_pretrained,
    )

    weights_path_str = str(model_cfg.get("pretrained_weights") or "").strip()
    if weights_path_str:
        weights_path = Path(weights_path_str)
        if not weights_path.is_absolute():
            weights_path = (hrnet_config_utils.PROJECT_ROOT / weights_path).resolve()
        if weights_path.exists():
            if torch is None:  # pragma: no cover
                raise RuntimeError("PyTorch is required to load pretrained weights.")
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
    return model
