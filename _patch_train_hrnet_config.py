from pathlib import Path

train_path = Path("scripts") / "train_hrnet.py"
text = train_path.read_text(encoding="utf-8")

old_dc = """@dataclass
class HRNetConfig:
    model_type: str = "hrnet_w32"
    input_size: int = 256
    resize_mode: str = "resize"  # "resize" или "original"
    keep_aspect_ratio: bool = True
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_epochs: int = 200
    train_val_split: float = 0.9
    flip_augmentation: bool = False
    rotation_augmentation_deg: float = 15.0
    scale_augmentation: float = 0.3
    weight_decay: float = 1e-4
    heatmap_sigma_px: float = 2.5
"""

new_dc = """@dataclass
class HRNetConfig:
    model_type: str = "hrnet_w32"
    input_size: int = 256
    resize_mode: str = "resize"  # "resize" или "original"
    keep_aspect_ratio: bool = True
    batch_size: int = 8
    learning_rate: float = 5e-4
    max_epochs: int = 200
    train_val_split: float = 0.9
    flip_augmentation: bool = False
    rotation_augmentation_deg: float = 15.0
    scale_augmentation: float = 0.3
    weight_decay: float = 1e-4
    heatmap_sigma_px: float = 2.5
    num_workers: int = 0
    num_keypoints: int = 0
"""

old_load = """def load_yaml_config(cfg_path: Path) -> HRNetConfig:
    cfg = HRNetConfig()
    if not cfg_path.is_file() or yaml is None:
        return cfg
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for field in cfg.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if field in data:
            setattr(cfg, field, data[field])
    return cfg
"""

new_load = """def load_yaml_config(cfg_path: Path) -> HRNetConfig:
    \"\"\"Загрузка HRNet-конфига из YAML + LM_number.txt.

    Поддерживает вложенные блоки:
      - model: model_type, num_keypoints
      - train: batch_size, learning_rate, max_epochs, train_val_split, weight_decay, num_workers
      - resize: long_side -> input_size, keep_aspect_ratio
      - data: heatmap_sigma_px

    Число ландмарок всегда берётся из LM_number.txt (если там > 0) и
    синхронизируется с model.num_keypoints в hrnet_config.yaml.
    \"\"\"
    cfg = HRNetConfig()

    # Если нет yaml или файла конфига — просто возвращаем дефолты
    if yaml is None or not cfg_path.is_file():
        # Но всё равно попробуем подхватить число ландмарок
        root = get_landmark_root()
        lm_number = read_lm_number(root)
        if lm_number > 0:
            cfg.num_keypoints = lm_number
        return cfg

    from typing import Any, Dict

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        return cfg

    flat: Dict[str, Any] = {}
    flat.update(raw)

    model_block = raw.get("model") or {}
    train_block = raw.get("train") or {}
    resize_block = raw.get("resize") or {}
    data_block = raw.get("data") or {}

    if isinstance(model_block, dict):
        if "model_type" in model_block:
            flat["model_type"] = model_block["model_type"]
        if "num_keypoints" in model_block:
            flat["num_keypoints"] = model_block["num_keypoints"]

    if isinstance(train_block, dict):
        for key in ("batch_size", "learning_rate", "max_epochs", "train_val_split", "weight_decay", "num_workers"):
            if key in train_block:
                flat[key] = train_block[key]

    if isinstance(resize_block, dict):
        if "long_side" in resize_block:
            flat["input_size"] = resize_block["long_side"]
        if "keep_aspect_ratio" in resize_block:
            flat["keep_aspect_ratio"] = resize_block["keep_aspect_ratio"]

    if isinstance(data_block, dict) and "heatmap_sigma_px" in data_block:
        flat["heatmap_sigma_px"] = data_block["heatmap_sigma_px"]

    # Число ландмарок читаем из LM_number.txt и перезаписываем num_keypoints
    root = get_landmark_root()
    lm_number = read_lm_number(root)
    if lm_number > 0:
        flat["num_keypoints"] = lm_number
        # Попробуем синхронизировать YAML на диске
        if not isinstance(model_block, dict):
            model_block = {}
        model_block["num_keypoints"] = lm_number
        raw["model"] = model_block
        try:
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(raw, f, allow_unicode=True, sort_keys=False)
        except Exception:
            pass

    for field in cfg.__dataclass_fields__.keys():  # type: ignore[attr-defined]
        if field in flat:
            setattr(cfg, field, flat[field])

    return cfg
"""

if old_dc not in text:
    raise SystemExit("Не найден блок HRNetConfig для замены")

if old_load not in text:
    raise SystemExit("Не найдена функция load_yaml_config для замены")

text = text.replace(old_dc, new_dc).replace(old_load, new_load)

# включаем num_workers из конфига в оба DataLoader
text = text.replace(
    "num_workers=0,",
    "num_workers=max(0, int(getattr(cfg, \"num_workers\", 0))),",
)

train_path.write_text(text, encoding="utf-8")
print("OK: train_hrnet.py patched")
