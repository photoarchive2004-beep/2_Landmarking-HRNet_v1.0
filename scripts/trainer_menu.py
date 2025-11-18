"""Console menu for HRNet training and inference (stub version)."""
from __future__ import annotations

from pathlib import Path

from scripts.annotator_menu import run_annotator_menu


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
            print("[TODO] HRNet training is not implemented yet in this step.")
            input("Press Enter to return to menu...")
            continue
        if choice == "2":
            print("[TODO] HRNet autolabel is not implemented yet in this step.")
            input("Press Enter to return to menu...")
            continue
        if choice == "3":
            run_annotator_menu(base_localities, lm_root)
            continue
        if choice == "4":
            print("No model info yet. HRNet training will be added in a later step.")
            input("Press Enter to return to menu...")
            continue
        if choice == "5":
            print("HRNet config editing will be added in a later step.")
            input("Press Enter to return to menu...")
            continue

        print("Invalid choice, please try again.")
