from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import json
import math

import numpy as np

from scripts.train_hrnet import get_landmark_root, _load_keypoints
from scripts import hrnet_config_utils


def load_num_keypoints() -> int:
    cfg_dict = hrnet_config_utils.load_hrnet_config()
    return hrnet_config_utils.get_num_keypoints(cfg_dict)


def load_run_id(root: Path) -> str:
    """Берём run_id из models/current/quality.json."""
    quality_path = root / "models" / "current" / "quality.json"
    if not quality_path.is_file():
        raise FileNotFoundError(f"Не найден {quality_path}")
    data = json.loads(quality_path.read_text(encoding="utf-8"))
    run_id = data.get("run_id", "")
    if not run_id:
        raise RuntimeError(f"В {quality_path} нет поля run_id")
    return str(run_id)


def read_val_samples(root: Path, run_id: str) -> List[Dict[str, Any]]:
    """
    Читаем список валидационных кадров из datasets/hrnet_val_<run_id>.txt.
    Формат строки: img_path;csv_path;locality
    """
    ds_file = root / "datasets" / f"hrnet_val_{run_id}.txt"
    if not ds_file.is_file():
        raise FileNotFoundError(f"Не найден файл валидации: {ds_file}")

    samples: List[Dict[str, Any]] = []
    for line in ds_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        if len(parts) < 2:
            continue
        img_str = parts[0].strip()
        csv_str = parts[1].strip()
        loc = parts[2].strip() if len(parts) > 2 else ""
        img_path = Path(img_str)
        csv_path = Path(csv_str)
        samples.append(
            {
                "image": img_path,
                "csv_manual": csv_path,
                "locality": loc,
            }
        )
    return samples


def compute_refspan(gt: np.ndarray) -> float:
    """RefSpan = максимум попарного расстояния между валидными точками."""
    if gt.shape[0] == 0:
        return 1.0
    # валидные точки: gt > 0 по хоть одной координате
    mask = (gt[:, 0] > 0) | (gt[:, 1] > 0)
    pts = gt[mask]
    if pts.shape[0] < 2:
        return 1.0
    diff = pts[None, :, :] - pts[:, None, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    refspan = float(dists.max())
    return refspan if refspan > 0 else 1.0


def main() -> None:
    root = get_landmark_root()
    print(f"[EVAL] root = {root}")

    num_kp = load_num_keypoints()
    print(f"[EVAL] num_keypoints = {num_kp}")

    run_id = load_run_id(root)
    print(f"[EVAL] run_id = {run_id}")

    samples = read_val_samples(root, run_id)
    print(f"[EVAL] val samples listed: {len(samples)}")

    all_err_px: List[float] = []
    all_err_rel: List[float] = []

    per_kp_err_px: Dict[int, List[float]] = {}
    per_kp_err_rel: Dict[int, List[float]] = {}

    bad_samples: List[Dict[str, Any]] = []

    processed = 0

    for s in samples:
        img_path: Path = s["image"]
        csv_manual: Path = s["csv_manual"]
        loc: str = s["locality"]

        if not img_path.is_file():
            print(f"[WARN] Нет изображения: {img_path}")
            continue
        if not csv_manual.is_file():
            print(f"[WARN] Нет ручного CSV: {csv_manual}")
            continue

        # авто-CSV: img_XXXX.hrnet.csv рядом с картинкой
        csv_auto = img_path.with_suffix(".hrnet.csv")
        if not csv_auto.is_file():
            print(f"[WARN] Нет авто CSV: {csv_auto}, пропускаем кадр")
            continue

        gt = _load_keypoints(csv_manual, num_kp)
        pred = _load_keypoints(csv_auto, num_kp)

        if gt.shape != pred.shape:
            print(f"[WARN] Разный размер kps для {img_path}, gt={gt.shape}, pred={pred.shape}")
            K = min(gt.shape[0], pred.shape[0])
            gt = gt[:K]
            pred = pred[:K]
        else:
            K = gt.shape[0]

        if K == 0:
            continue

        # маска валидных точек (как в compute_pck_at_r)
        mask = (gt[:, 0] > 0) | (gt[:, 1] > 0)
        if not np.any(mask):
            print(f"[WARN] Нет валидных GT-точек для {img_path}")
            continue

        refspan = compute_refspan(gt)

        dists_px: List[float] = []
        dists_rel: List[float] = []

        for k in range(K):
            if not mask[k]:
                continue
            dx = float(pred[k, 0] - gt[k, 0])
            dy = float(pred[k, 1] - gt[k, 1])
            dist = math.sqrt(dx * dx + dy * dy)
            dist_rel = dist / refspan

            dists_px.append(dist)
            dists_rel.append(dist_rel)

            per_kp_err_px.setdefault(k, []).append(dist)
            per_kp_err_rel.setdefault(k, []).append(dist_rel)

            all_err_px.append(dist)
            all_err_rel.append(dist_rel)

        if dists_rel:
            mean_rel = float(np.mean(dists_rel))
            mean_px = float(np.mean(dists_px))
            bad_samples.append(
                {
                    "image": str(img_path),
                    "locality": loc,
                    "mean_err_px": mean_px,
                    "mean_err_rel": mean_rel,
                }
            )
        processed += 1

    if not all_err_px:
        print("[EVAL] Не удалось посчитать ни одной ошибки. Проверь, запускался ли 8_HRNET_INFER_ALL.ps1.")
        return

    mean_err_px = float(np.mean(all_err_px))
    median_err_px = float(np.median(all_err_px))
    mean_err_rel = float(np.mean(all_err_rel))
    median_err_rel = float(np.median(all_err_rel))

    print(f"[EVAL] mean_err_px      = {mean_err_px:.2f}")
    print(f"[EVAL] median_err_px    = {median_err_px:.2f}")
    print(f"[EVAL] mean_err_rel     = {mean_err_rel:.4f}")
    print(f"[EVAL] median_err_rel   = {median_err_rel:.4f}")
    print(f"[EVAL] processed samples = {processed}")

    # собираем метрики
    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "num_keypoints": num_kp,
        "processed_samples": processed,
        "mean_err_px": mean_err_px,
        "median_err_px": median_err_px,
        "mean_err_rel": mean_err_rel,
        "median_err_rel": median_err_rel,
        "per_kp": {},
    }

    for k, vals in per_kp_err_px.items():
        v_px = np.array(vals, dtype=np.float32)
        v_rel = np.array(per_kp_err_rel.get(k, []), dtype=np.float32)
        metrics["per_kp"][int(k)] = {
            "mean_px": float(v_px.mean()),
            "median_px": float(np.median(v_px)),
            "mean_rel": float(v_rel.mean()),
            "median_rel": float(np.median(v_rel)),
            "count": int(v_px.shape[0]),
        }

    # топ-20 самых плохих по относительной ошибке
    worst = sorted(bad_samples, key=lambda d: d["mean_err_rel"], reverse=True)[:20]

    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)

    metrics_json = logs_dir / f"metrics_val_{run_id}.json"
    metrics_csv = logs_dir / f"metrics_val_per_kp_{run_id}.csv"

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": metrics,
                "worst_samples": worst,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # CSV по ландмаркам
    import csv

    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            ["kp_index", "mean_px", "median_px", "mean_rel", "median_rel", "count"]
        )
        for k, stat in metrics["per_kp"].items():
            writer.writerow(
                [
                    k,
                    stat["mean_px"],
                    stat["median_px"],
                    stat["mean_rel"],
                    stat["median_rel"],
                    stat["count"],
                ]
            )

    print("[EVAL] Метрики записаны в:")
    print("  ", metrics_json)
    print("  ", metrics_csv)


if __name__ == "__main__":
    main()
