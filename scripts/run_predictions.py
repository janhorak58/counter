#!/usr/bin/env python3
"""
Batch prediction script pro YOLO a RF-DETR modely.

Pouziti:
    python scripts/run_predictions.py --all
    python scripts/run_predictions.py --finetuned
    python scripts/run_predictions.py --pretrained
    python scripts/run_predictions.py --dry-run
    python scripts/run_predictions.py --models yolo11l_v11 rfdetr_base_v1
    python scripts/run_predictions.py --videos vid16.mp4 vid17.mp4
    python scripts/run_predictions.py --yolo-only
    python scripts/run_predictions.py --rfdetr-only
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

# =============================================================================
# Finetuned modely
# =============================================================================

YOLO_MODELS = {
    "yolo26l_v11": "models/yolo/v2/yolo26l_v11/weights/best.pt",
    "yolo26x_v11": "models/yolo/v2/yolo26x_v11/weights/best.pt",
    "yolo26s_v11": "models/yolo/v2/yolo26s_v11/weights/best.pt",
    "yolov8n_v12": "models/yolo/v2/yolov8n_v12/weights/best.pt",
}

RFDETR_MODELS = {
    "rfdetr_large_v3": "models/rfdetr/v1/rfdetr_large_v3/checkpoint_best_ema.pth",
    "rfdetr_medium_v2": "models/rfdetr/v1/rfdetr_medium_v2/checkpoint_best_ema.pth",
    "rfdetr_small_v6": "models/rfdetr/v1/rfdetr_small_v6/checkpoint_best_ema.pth",
}

# =============================================================================
# Pretrained modely
# =============================================================================

PRETRAINED_YOLO_MODELS = {
    "yolo26l_pretrained": "yolo26l.pt",
    "yolo26x_pretrained": "yolo26x.pt",
}

PRETRAINED_RFDETR_MODELS = {
    "rfdetr_large_pretrained": "rf-detr-large.pth",
    "rfdetr_medium_pretrained": "rf-detr-medium.pth",
    "rfdetr_small_pretrained": "rf-detr-small.pth",
}

# =============================================================================
# Combined dictionaries
# =============================================================================

FINETUNED_MODELS = {
    **{k: (v, "yolo") for k, v in YOLO_MODELS.items()},
    **{k: (v, "rfdetr") for k, v in RFDETR_MODELS.items()},
}

PRETRAINED_MODELS = {
    **{k: (v, "yolo") for k, v in PRETRAINED_YOLO_MODELS.items()},
    **{k: (v, "rfdetr") for k, v in PRETRAINED_RFDETR_MODELS.items()},
}

ALL_MODELS = {**FINETUNED_MODELS, **PRETRAINED_MODELS}


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
    model_type: str,
    output_suffix: str,
) -> dict:
    """Update config for current run."""
    config["predict"]["paths"]["model_path"] = model_path
    config["predict"]["paths"]["video_filename"] = video_file
    config["predict"]["parameters"]["mode"] = mode
    config["predict"]["parameters"]["model_type"] = model_type
    config["predict"]["paths"]["output_folder"] = f"./predictions/mp4/{output_suffix}"
    config["predict"]["paths"]["results_folder"] = f"./predictions/csv/{output_suffix}"
    return config


def run_prediction(
    model_name: str,
    model_path: str,
    model_type: str,
    video: str,
    mode: str,
    dry_run: bool = False,
) -> bool:
    """Run single prediction."""
    print_info(f"Model: {model_name} | Type: {model_type} | Video: {video} | Mode: {mode}")

    if dry_run:
        print(f"  [DRY-RUN] python -m src.predict")
        print(f"    model_path: {model_path}")
        print(f"    model_type: {model_type}")
        print(f"    video: {video}")
        print(f"    mode: {mode}")
        return True

    # Create output folders
    (PROJECT_DIR / "predictions" / "mp4" / model_name).mkdir(parents=True, exist_ok=True)
    (PROJECT_DIR / "predictions" / "csv" / model_name).mkdir(parents=True, exist_ok=True)

    # Load and update config
    config = load_config()
    config = update_config(config, model_path, video, mode, model_type, model_name)
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


def filter_models_by_type(models: dict, model_type: str | None) -> dict:
    """Filter models by type (yolo/rfdetr)."""
    if model_type is None:
        return models
    return {k: v for k, v in models.items() if v[1] == model_type}


def main():
    parser = argparse.ArgumentParser(
        description="Batch prediction script for YOLO and RF-DETR models"
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
        "--yolo-only", action="store_true", help="Run only YOLO models"
    )
    parser.add_argument(
        "--rfdetr-only", action="store_true", help="Run only RF-DETR models"
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

    # Determine model type filter
    model_type_filter = None
    if args.yolo_only and not args.rfdetr_only:
        model_type_filter = "yolo"
    elif args.rfdetr_only and not args.yolo_only:
        model_type_filter = "rfdetr"

    # Default: run all
    run_finetuned = args.finetuned or args.all or (not args.finetuned and not args.pretrained)
    run_pretrained = args.pretrained or args.all or (not args.finetuned and not args.pretrained)

    # Videos
    videos = args.videos if args.videos else VIDEOS

    # Models
    finetuned_to_run = {}
    pretrained_to_run = {}

    if args.models:
        # Specific models requested
        for model in args.models:
            if model in FINETUNED_MODELS:
                model_data = FINETUNED_MODELS[model]
                # Apply type filter if specified
                if model_type_filter is None or model_data[1] == model_type_filter:
                    finetuned_to_run[model] = model_data
            elif model in PRETRAINED_MODELS:
                model_data = PRETRAINED_MODELS[model]
                # Apply type filter if specified
                if model_type_filter is None or model_data[1] == model_type_filter:
                    pretrained_to_run[model] = model_data
            else:
                print_warning(f"Unknown model: {model}")
    else:
        # Use all models based on flags
        if run_finetuned:
            finetuned_to_run = filter_models_by_type(FINETUNED_MODELS, model_type_filter)
        if run_pretrained:
            pretrained_to_run = filter_models_by_type(PRETRAINED_MODELS, model_type_filter)

    print_header("YOLO & RF-DETR Batch Prediction Script")

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
        print_info(f"Finetuned models: {len(finetuned_to_run)}")
        print_info(f"Pretrained models: {len(pretrained_to_run)}")
        print_info(f"Videos: {len(videos)}")

        # Finetuned models
        if finetuned_to_run:
            print_header("Finetuned models (mode: custom)")

            for model_name, (model_path, model_type) in finetuned_to_run.items():
                full_path = PROJECT_DIR / model_path
                if not full_path.exists() and not args.dry_run:
                    print_warning(f"Model not found: {model_path}, skipping...")
                    continue

                for video in videos:
                    completed += 1
                    print_info(f"[{completed}/{total_runs}] Processing...")
                    run_prediction(
                        model_name, model_path, model_type, video, "custom", args.dry_run
                    )

        # Pretrained models
        if pretrained_to_run:
            print_header("Pretrained models (mode: pretrained)")

            for model_name, (model_path, model_type) in pretrained_to_run.items():
                full_path = PROJECT_DIR / model_path
                if not full_path.exists() and not args.dry_run:
                    print_warning(f"Model not found: {model_path}, skipping...")
                    continue

                for video in videos:
                    completed += 1
                    print_info(f"[{completed}/{total_runs}] Processing...")
                    run_prediction(
                        model_name, model_path, model_type, video, "pretrained", args.dry_run
                    )

    finally:
        # Always restore config
        restore_config()

    print_header("Done!")
    print_info("Results saved to:")
    print_info("  - predictions/mp4/<model_name>/")
    print_info("  - predictions/csv/<model_name>/")


if __name__ == "__main__":
    main()