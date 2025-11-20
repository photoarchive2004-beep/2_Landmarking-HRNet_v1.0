from __future__ import annotations

"""Проверка согласованности LM_number.txt, hrnet_config.yaml и настроек train_hrnet."""

from typing import Any, Dict

from scripts import hrnet_config_utils


def main() -> int:
    print("=== HRNet config consistency check ===")

    # 1. LM_number.txt
    try:
        n_txt = hrnet_config_utils.read_num_keypoints()
    except Exception as e:  # noqa: BLE001
        print("LM_number.txt error:", repr(e))
        return 1

    print(f"LM_number.txt num_keypoints: {n_txt}")

    # 2. Конфиг
    try:
        cfg: Dict[str, Any] = hrnet_config_utils.load_hrnet_config()
    except Exception as e:  # noqa: BLE001
        print("hrnet_config.yaml error:", repr(e))
        return 1

    model_cfg = (cfg.get("model") or {})
    train_cfg = (cfg.get("train") or {})

    n_cfg = model_cfg.get("num_keypoints")
    max_epochs = train_cfg.get("max_epochs")

    print(f"config model.num_keypoints: {n_cfg}")
    print(f"train.max_epochs: {max_epochs}")

    if n_cfg != n_txt:
        print("[WARN] LM_number.txt и config.model.num_keypoints не совпадают")

    if max_epochs is None:
        print("[WARN] train.max_epochs не задан в конфиге (используется дефолт в коде).")

    print("=== DONE config consistency check ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
