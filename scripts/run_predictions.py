#!/usr/bin/env python3
"""
Batch prediction script pro YOLO modely.

Pouziti:
    python scripts/run_predictions.py --all
    python scripts/run_predictions.py --finetuned
    python scripts/run_predictions.py --pretrained
    python scripts/run_predictions.py --dry-run
    python scripts/run_predictions.py --models yolo11l_v11 yolov8n_v12
    python scripts/run_predictions.py --videos vid16.mp4 vid17.mp4
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_DIR / "config.yaml"
CONFIG_BACKUP = PROJECT_DIR / "config.yaml.backup"

# Videa pro predikci
VIDEOS = ["vid16.mp4", "vid17.mp4", "vid18.mp4"]

# Finetuned modely
FINETUNED_MODELS = {
    "yolo11l_v11": "models/yolo/v1/yolo11l_v11/weights/best.pt",
    "yolov8n_v12": "models/yolo/v1/yolov8n_v12/weights/best.pt",
    "yolov8s_v13": "models/yolo/v1/yolov8s_v13/weights/best.pt",
}

# Pretrained modely
PRETRAINED_MODELS = {
    "yolo11l_pretrained": "yolo11l.pt",
    "yolov8l_pretrained": "yolov8l.pt",
    "yolov8m_pretrained": "yolov8m.pt",
}


# =============================================================================
# Utility funkce
# =============================================================================

class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"


def print_header(msg: str):
    print(f"\n{Colors.BLUE}{'=' * 70}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 70}{Colors.NC}")


def print_info(msg: str):
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def load_config() -> dict:
    """Load YAML config."""
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def save_config(config: dict):
    """Save YAML config."""
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def backup_config():
    """Backup config file."""
    if CONFIG_FILE.exists():
        shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
        print_info(f"Config backed up to {CONFIG_BACKUP}")


def restore_config():
    """Restore config from backup."""
    if CONFIG_BACKUP.exists():
        shutil.copy(CONFIG_BACKUP, CONFIG_FILE)
        CONFIG_BACKUP.unlink()
        print_info("Config restored from backup")


# =============================================================================
# Hlavni logika
# =============================================================================

def update_config(
    config: dict,
    model_path: str,
    video_file: str,
    mode: str,
    output_suffix: str,
) -> dict:
    """Update config for current run."""
    config["predict"]["paths"]["model_path"] = model_path
    config["predict"]["paths"]["video_filename"] = video_file
    config["predict"]["parameters"]["mode"] = mode
    config["predict"]["paths"]["output_folder"] = f"./predictions/mp4/{output_suffix}"
    config["predict"]["paths"]["results_folder"] = f"./predictions/csv/{output_suffix}"
    return config


def run_prediction(
    model_name: str,
    model_path: str,
    video: str,
    mode: str,
    dry_run: bool = False,
) -> bool:
    """Run single prediction."""
    print_info(f"Model: {model_name} | Video: {video} | Mode: {mode}")

    if dry_run:
        print(f"  [DRY-RUN] python -m src.predict")
        print(f"    model_path: {model_path}")
        print(f"    video: {video}")
        print(f"    mode: {mode}")
        return True

    # Create output folders
    (PROJECT_DIR / "predictions" / "mp4" / model_name).mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "predictions" / "csv" / model_name).mkdir(parents=True, exist_ok=True)

    # Load and update config
    config = load_config()
    config = update_config(config, model_path, video, mode, model_name)
    save_config(config)

    # Run prediction
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.predict"],
            cwd=PROJECT_DIR,
            check=True,
        )
        print_info(f"Done: {model_name} / {video}\n")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Prediction error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction script for YOLO models"
    )
    parser.add_argument(
        "--finetuned", action="store_true", help="Run only finetuned models"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Run only pretrained models"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all models (default)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only show what would be executed"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to run"
    )
    parser.add_argument(
        "--videos", nargs="+", help="Specific videos to process"
    )

    args = parser.parse_args()

    # Default: run all
    run_finetuned = args.finetuned or args.all or (not args.finetuned and not args.pretrained)
    run_pretrained = args.pretrained or args.all or (not args.finetuned and not args.pretrained)

    # Videos
    videos = args.videos if args.videos else VIDEOS

    # Models
    finetuned_to_run = {}
    pretrained_to_run = {}

    if args.models:
        for model in args.models:
            if model in FINETUNED_MODELS:
                finetuned_to_run[model] = FINETUNED_MODELS[model]
            elif model in PRETRAINED_MODELS:
                pretrained_to_run[model] = PRETRAINED_MODELS[model]
            else:
                print_warning(f"Unknown model: {model}")
    else:
        if run_finetuned:
            finetuned_to_run = FINETUNED_MODELS
        if run_pretrained:
            pretrained_to_run = PRETRAINED_MODELS

    print_header("YOLO Batch Prediction Script")

    # Check config exists
    if not CONFIG_FILE.exists():
        print_error(f"Config file not found: {CONFIG_FILE}")
        sys.exit(1)

    # Backup
    backup_config()

    try:
        total_runs = (len(finetuned_to_run) + len(pretrained_to_run)) * len(videos)
        completed = 0

        print_info(f"Total predictions: {total_runs}")

        # Finetuned models
        if finetuned_to_run:
            print_header("Finetuned models (mode: custom)")

            for model_name, model_path in finetuned_to_run.items():
                full_path = PROJECT_DIR / model_path
                if not full_path.exists() and not args.dry_run:
                    print_warning(f"Model not found: {model_path}, skipping...")
                    continue

                for video in videos:
                    completed += 1
                    print_info(f"[{completed}/{total_runs}] Processing...")
                    run_prediction(model_name, model_path, video, "custom", args.dry_run)

        # Pretrained models
        if pretrained_to_run:
            print_header("Pretrained models (mode: pretrained)")

            for model_name, model_path in pretrained_to_run.items():
                full_path = PROJECT_DIR / model_path
                if not full_path.exists() and not args.dry_run:
                    print_warning(f"Model not found: {model_path}, skipping...")
                    continue

                for video in videos:
                    completed += 1
                    print_info(f"[{completed}/{total_runs}] Processing...")
                    run_prediction(model_name, model_path, video, "pretrained", args.dry_run)

    finally:
        # Always restore config
        restore_config()

    print_header("Done!")
    print_info("Results saved to:")
    print_info("  - predictions/mp4/<model_name>/")
    print_info("  - predictions/csv/<model_name>/")


if __name__ == "__main__":
    main()
