"""
Вспомогательные функции для работы с конфигурацией HRNet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Корень проекта: один уровень выше папки scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]

LM_NUMBER_FILE = PROJECT_ROOT / "LM_number.txt"
HRNET_CONFIG_FILE = PROJECT_ROOT / "config" / "hrnet_config.yaml"


class HRNetConfigError(RuntimeError):
    """Ошибки, связанные с конфигом HRNet или LM_number.txt."""


def read_num_keypoints() -> int:
    """Читает количество ландмарок из ``LM_number.txt``."""
    if not LM_NUMBER_FILE.exists():
        raise HRNetConfigError(
            f"LM_number.txt не найден: {LM_NUMBER_FILE}\n"
            "Проверь, что файл существует в корне модуля 2_Landmarking_v1.0."
        )

    text = LM_NUMBER_FILE.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = int(line)
        except ValueError:
            continue
        if value <= 0:
            raise HRNetConfigError(
                f"Некорректное количество ландмарок в LM_number.txt: {value}. "
                "Должно быть положительное целое число."
            )
        return value

    raise HRNetConfigError(
        "Не удалось прочитать количество ландмарок из LM_number.txt.\n"
        "Убедись, что в файле есть строка с целым числом (например: 18)."
    )


def read_hrnet_config() -> dict:
    """Читает словарь конфигурации HRNet из ``config/hrnet_config.yaml``."""
    if not HRNET_CONFIG_FILE.exists():
        raise HRNetConfigError(
            f"Файл конфигурации HRNet не найден: {HRNET_CONFIG_FILE}\n"
            "Сначала создай config/hrnet_config.yaml."
        )

    with HRNET_CONFIG_FILE.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise HRNetConfigError(
            f"Ожидался словарь в hrnet_config.yaml, но получено: {type(cfg).__name__}"
        )

    return cfg


def load_hrnet_config() -> Dict[str, Any]:
    """Возвращает конфиг HRNet, гарантируя наличие основных блоков."""
    cfg = read_hrnet_config()
    cfg.setdefault("model", {})
    cfg.setdefault("resize", {})
    cfg.setdefault("train", {})
    cfg.setdefault("augment", {})
    cfg.setdefault("data", {})
    cfg.setdefault("infer", {})
    return cfg


def get_resize_long_side(cfg: dict) -> int:
    """Достать параметр ``resize_long_side`` (совместимость со старым кодом)."""
    value = cfg.get("resize_long_side", cfg.get("resize", {}).get("long_side", 1280))
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # noqa: PERF203 - явное сообщение об ошибке
        raise HRNetConfigError(
            f"Некорректное значение resize_long_side в hrnet_config.yaml: {value!r}"
        ) from exc


if __name__ == "__main__":
    print("=== HRNet config utils: diagnostic mode ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")

    try:
        n_kpts = read_num_keypoints()
        print(f"LM_number.txt -> num_keypoints = {n_kpts}")
    except HRNetConfigError as e:  # pragma: no cover - консольная диагностика
        print("\n[ERR] Проблема с LM_number.txt:")
        print(e)

    try:
        cfg = load_hrnet_config()
        resize = get_resize_long_side(cfg)
        print(f"hrnet_config.yaml -> resize_long_side = {resize}")
        print("Ключи конфига:", ", ".join(sorted(cfg.keys())))
    except HRNetConfigError as e:  # pragma: no cover
        print("\n[ERR] Проблема с hrnet_config.yaml:")
        print(e)
