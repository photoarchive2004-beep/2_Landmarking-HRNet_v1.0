"""Console menu for HRNet training and inference (stub version)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.annotator_menu import run_annotator_menu
from scripts.hrnet_train import train_hrnet_model
from scripts.hrnet_infer import autolabel_locality


def run_trainer_menu(base_localities: Path, lm_root: Path) -> None:
    while True:
        print("=== GM Landmarking: HRNet Trainer (v1.0) ===")
        print("Base folder:")
        print(f"  {base_localities}")
        print()
        print("Actions:")
        print()
        print("  1) Train / finetune HRNet model on MANUAL localities")
        print("  2) Autolabel locality with current HRNet model")
        print("  3) Open locality in annotator")
        print("  4) Info about current model")
        print("  5) View / edit HRNet settings")
        print("  0) Quit")
        print()

        choice = input("Select action: ").strip()

        if choice == "0":
            print("Exiting HRNet trainer.")
            break
        if choice == "1":
            try:
                train_hrnet_model(base_localities, lm_root)
            except Exception as exc:  # noqa: BLE001 - пользовательский вывод
                print(f"Training failed: {exc}")
                input("Press Enter to return to menu...")
            continue
        if choice == "2":
            try:
                autolabel_locality(base_localities, lm_root)
            except Exception as exc:  # noqa: BLE001
                print(f"Autolabel failed: {exc}")
                input("Press Enter to return to menu...")
            continue
        if choice == "3":
            run_annotator_menu(base_localities, lm_root)
            continue
        if choice == "4":
            quality_path = lm_root / "models" / "current" / "quality.json"
            if not quality_path.exists():
                print("No trained model found.")
                input("Press Enter to return to menu...")
                continue
            try:
                data = quality_path.read_text(encoding="utf-8")
                info = json.loads(data)
                print("=== Current HRNet model ===")
                print(f"run_id: {info.get('run_id')}")
                print(f"num_keypoints: {info.get('num_keypoints')}")
                print(f"localities: {info.get('localities')}")
                print(f"n_train_images: {info.get('n_train_images')}")
                print(f"n_val_images: {info.get('n_val_images')}")
                print(f"PCK@R: {info.get('pck_r_percent')}%")
                print(f"resize.enabled: {info.get('resize_enabled')}")
                print(f"resize.long_side: {info.get('resize_long_side')}")
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to read model info: {exc}")
            input("Press Enter to return to menu...")
            continue
        if choice == "5":
            print("Opening HRNet config in Notepad (model / resize / train / augment / infer)...")
            subprocess.Popen(["notepad.exe", str(lm_root / "config" / "hrnet_config.yaml")])
            input("Press Enter to return to menu...")
            continue

        print("Invalid choice, please try again.")
