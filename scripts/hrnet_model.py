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

try:  # pragma: no cover - optional dependency
    import timm
except Exception as exc:  # pragma: no cover
    print(f"[WARN] Unable to import timm HRNet models: {exc!r}")
    timm = None  # type: ignore

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
        """HRNet-W32 backbone с поддержкой timm/MMPose и fallback-сеткой."""

        def __init__(self, num_keypoints: int, load_imagenet_pretrained: bool = True) -> None:
            super().__init__()
            if torch is None:  # pragma: no cover
                raise RuntimeError("PyTorch is required for HRNetW32GM but is not installed.")

            self.num_keypoints = int(num_keypoints)
            self._load_pretrained = bool(load_imagenet_pretrained)
            self.backbone_type = "unknown"
            self.backbone: Optional[nn.Module] = None
            self.head: Optional[nn.Module] = None
            self.fallback: Optional[nn.Module] = None
            self._timm_selected_position: Optional[int] = None

            if self._init_timm_backbone():
                pass
            elif self._init_mmpose_backbone():
                pass
            else:
                self._init_simple_fallback()

            self.use_mmpose = self.backbone_type == "mmpose_hrnet"

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.backbone_type in {"timm_hrnet", "mmpose_hrnet"}:
                if self.backbone is None or self.head is None:
                    raise RuntimeError("Backbone/head are not initialized.")
                feats = self.backbone(x)
                if self.backbone_type == "timm_hrnet":
                    if not isinstance(feats, (list, tuple)):
                        raise RuntimeError("timm HRNet backbone is expected to return a list of features.")
                    if self._timm_selected_position is None:
                        raise RuntimeError("timm HRNet backbone is missing feature selection info.")
                    feature_map = feats[self._timm_selected_position]
                    return self.head(feature_map)

                feats0 = feats[0] if isinstance(feats, (list, tuple)) else feats
                return self.head(feats0)

            if self.fallback is None:
                raise RuntimeError("Fallback network is not initialized.")
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

        def freeze_low_stages(self, freeze_stem: bool = True) -> None:
            """Freeze низкие стадии активного бекбона (или стем SimpleHRNet)."""

            if self.backbone_type in {"timm_hrnet", "mmpose_hrnet"} and self.backbone is not None:
                modules_to_freeze = []
                if freeze_stem:
                    for attr_name in ("conv1", "bn1", "conv2", "bn2"):
                        stem_module = getattr(self.backbone, attr_name, None)
                        if stem_module is not None:
                            modules_to_freeze.append(stem_module)

                stage_attr_names = [
                    "layer1",
                    "stage1",
                    "transition1",
                    "stage2",
                    "transition2",
                ]
                seen_ids = set()
                for attr_name in stage_attr_names:
                    module = getattr(self.backbone, attr_name, None)
                    if module is None:
                        continue
                    module_id = id(module)
                    if module_id not in seen_ids:
                        modules_to_freeze.append(module)
                        seen_ids.add(module_id)

                frozen_tensors = 0
                for module in modules_to_freeze:
                    for param in module.parameters():
                        param.requires_grad = False
                        frozen_tensors += 1
                    module.apply(self._set_batchnorm_eval)

                if frozen_tensors:
                    print(
                        f"[INFO] Frozen {frozen_tensors} parameter tensors in HRNet backbone (Stage1/Stage2)."
                    )
                else:
                    print(
                        "[WARN] freeze_low_stages() did not find matching modules. "
                        "Check HRNet backbone structure."
                    )
                return

            if self.backbone_type == "simple_fallback" and self.fallback is not None:
                for module in [self.fallback.stem]:
                    module.apply(self._set_batchnorm_eval)
                    for param in module.parameters():
                        param.requires_grad = False
                print("[INFO] Frozen fallback stem parameters.")
                return

            print("[WARN] Cannot freeze stages: backbone is unavailable.")

        def _init_timm_backbone(self) -> bool:
            if timm is None:
                print("[WARN] timm HRNet backbone unavailable. Falling back to alternatives.")
                return False

            try:
                out_indices = (0, 1, 2, 3)
                backbone = timm.create_model(
                    "hrnet_w32",
                    pretrained=self._load_pretrained,
                    features_only=True,
                    out_indices=out_indices,
                )
            except Exception as exc:  # pragma: no cover - runtime dependency
                print(f"[WARN] Failed to create timm HRNet backbone: {exc!r}")
                return False

            reductions = list(backbone.feature_info.reduction())
            channels = list(backbone.feature_info.channels())
            selected_pos = min(range(len(reductions)), key=lambda idx: reductions[idx])
            self._timm_selected_position = selected_pos
            in_channels = channels[selected_pos]
            self.backbone = backbone
            self.head = nn.Conv2d(in_channels, self.num_keypoints, kernel_size=1)
            self.backbone_type = "timm_hrnet"
            print("[INFO] Using timm HRNet-W32 backbone for HRNetW32GM")
            return True

        def _init_mmpose_backbone(self) -> bool:
            if MMPoseHRNet is None:
                return False

            try:
                extra = _build_hrnet_extra_config()
                backbone = MMPoseHRNet(extra=extra, in_channels=3)  # type: ignore[call-arg]
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to initialize MMPose HRNet backbone: {exc!r}")
                return False

            self.backbone = backbone
            self.head = nn.Conv2d(32, self.num_keypoints, kernel_size=1)
            self.backbone_type = "mmpose_hrnet"
            print("[INFO] Using MMPose HRNet backbone for HRNetW32GM")
            if self._load_pretrained:
                self._load_imagenet_pretrained_backbone()
            return True

        def _init_simple_fallback(self) -> None:
            print("[WARN] Neither timm nor MMPose HRNet backbones are available. Using fallback HRNet.")
            self.fallback = SimpleHRNet(self.num_keypoints)
            self.backbone = None
            self.head = None
            self.backbone_type = "simple_fallback"

        @staticmethod
        def _set_batchnorm_eval(module: nn.Module) -> None:
            """Переводим BatchNorm замороженных блоков в eval-режим."""

            bn_cls = getattr(nn.modules.batchnorm, "_BatchNorm", None)
            if bn_cls is not None and isinstance(module, bn_cls):
                module.eval()


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
