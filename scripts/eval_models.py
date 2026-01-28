#!/usr/bin/env python3
"""
Batch evaluation script pro YOLO a RF-DETR modely.

Automaticky najde vsechny modely v models/ slozce a spusti evaluaci.

Pouziti:
    python scripts/eval_models.py --all
    python scripts/eval_models.py --yolo-only
    python scripts/eval_models.py --rfdetr-only
    python scripts/eval_models.py --models yolo26l_v11 rfdetr_large_v3
    python scripts/eval_models.py --dry-run
    python scripts/eval_models.py --split test
"""

import argparse
import subprocess
import sys
from pathlib import Path

# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"

# YOLO config
YOLO_MODELS_DIR = MODELS_DIR / "yolo"
YOLO_DATA_YAML = PROJECT_DIR / "data" / "yolo_dataset" / "data.yaml"
YOLO_WEIGHTS_SUBPATH = "weights/best.pt"

# RF-DETR config
RFDETR_MODELS_DIR = MODELS_DIR / "rfdetr"
RFDETR_DATASET_DIR = PROJECT_DIR / "data" / "coco_dataset"
RFDETR_WEIGHTS_FILENAME = "checkpoint_best_ema.pth"

# Default eval settings
DEFAULT_SPLIT = "val"
DEFAULT_DEVICE = "0"
DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.6
DEFAULT_IMGSZ = 640
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results" / "eval"


# =============================================================================
# Utility
# =============================================================================

class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
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


def print_model(msg: str):
    print(f"{Colors.CYAN}[MODEL]{Colors.NC} {msg}")


# =============================================================================
# Model discovery
# =============================================================================

def find_yolo_models() -> dict[str, Path]:
    """Find all YOLO models in models/yolo/v1/"""
    models = {}
    
    version_dir = YOLO_MODELS_DIR / "v1"
    if not version_dir.exists():
        return models
    
    for model_dir in version_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        weights_path = model_dir / YOLO_WEIGHTS_SUBPATH
        if weights_path.exists():
            model_name = model_dir.name
            models[model_name] = weights_path
    
    return models


def find_rfdetr_models() -> dict[str, Path]:
    """Find all RF-DETR models in models/rfdetr/v1/"""
    models = {}
    
    version_dir = RFDETR_MODELS_DIR / "v1"
    if not version_dir.exists():
        return models
    
    for model_dir in version_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        weights_path = model_dir / RFDETR_WEIGHTS_FILENAME
        if weights_path.exists():
            model_name = model_dir.name
            models[model_name] = weights_path
    
    return models


def find_all_models() -> tuple[dict[str, Path], dict[str, Path]]:
    """Find all models."""
    yolo_models = find_yolo_models()
    rfdetr_models = find_rfdetr_models()
    return yolo_models, rfdetr_models


# =============================================================================
# Evaluation
# =============================================================================

def run_eval(
    model_name: str,
    model_path: Path,
    model_type: str,
    split: str,
    device: str,
    conf: float,
    iou: float,
    imgsz: int,
    output_dir: Path,
    coco_eval: bool,
    visuals: bool,
    dry_run: bool = False,
) -> bool:
    """Run evaluation for a single model."""
    
    print_model(f"{model_name} ({model_type})")
    print_info(f"  Weights: {model_path}")
    print_info(f"  Split: {split}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "src.eval_model",
        "--model-type", model_type,
        "--model-path", str(model_path),
        "--split", split,
        "--device", device,
        "--conf", str(conf),
        "--iou", str(iou),
        "--imgsz", str(imgsz),
        "--output-dir", str(output_dir),
        "--run-name", model_name,
    ]
    
    # Add dataset path based on model type
    if model_type == "yolo":
        cmd.extend(["--data-yaml", str(YOLO_DATA_YAML)])
    else:  # rfdetr
        cmd.extend(["--dataset-dir", str(RFDETR_DATASET_DIR)])
    
    # Add optional flags
    if coco_eval:
        cmd.append("--coco-eval")
    else:
        cmd.append("--no-coco-eval")
    
    if visuals:
        cmd.append("--visuals")
    else:
        cmd.append("--no-visuals")
    
    if dry_run:
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return True
    
    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            check=True,
        )
        print_info(f"  Done: {model_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"  Evaluation failed: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation script for YOLO and RF-DETR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/eval_models.py --all
    python scripts/eval_models.py --yolo-only
    python scripts/eval_models.py --rfdetr-only
    python scripts/eval_models.py --models yolo26l_v11 rfdetr_large_v3
    python scripts/eval_models.py --split test
    python scripts/eval_models.py --dry-run
        """
    )
    
    # Model selection
    parser.add_argument("--all", action="store_true", help="Evaluate all models (default)")
    parser.add_argument("--yolo-only", action="store_true", help="Evaluate only YOLO models")
    parser.add_argument("--rfdetr-only", action="store_true", help="Evaluate only RF-DETR models")
    parser.add_argument("--models", nargs="+", help="Specific models to evaluate")
    
    # Eval settings
    parser.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split [default: {DEFAULT_SPLIT}]")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"GPU device [default: {DEFAULT_DEVICE}]")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help=f"Confidence threshold [default: {DEFAULT_CONF}]")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help=f"IoU threshold [default: {DEFAULT_IOU}]")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help=f"Image size [default: {DEFAULT_IMGSZ}]")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    
    # Flags
    parser.add_argument("--coco-eval", action="store_true", default=True, help="Run COCO evaluation")
    parser.add_argument("--no-coco-eval", action="store_false", dest="coco_eval")
    parser.add_argument("--visuals", action="store_true", default=False, help="Generate visualizations")
    parser.add_argument("--no-visuals", action="store_false", dest="visuals")
    
    # Dry run
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be executed")
    parser.add_argument("--list", action="store_true", help="List available models and exit")
    
    args = parser.parse_args()
    
    # Discover models
    yolo_models, rfdetr_models = find_all_models()
    
    # List models and exit
    if args.list:
        print_header("Available Models")
        print("\nYOLO models:")
        for name, path in sorted(yolo_models.items()):
            print(f"  - {name}")
        print(f"\nRF-DETR models:")
        for name, path in sorted(rfdetr_models.items()):
            print(f"  - {name}")
        print(f"\nTotal: {len(yolo_models)} YOLO, {len(rfdetr_models)} RF-DETR")
        return 0
    
    # Filter models
    models_to_eval = []  # List of (name, path, type)
    
    if args.models:
        # Specific models requested
        for model_name in args.models:
            if model_name in yolo_models:
                models_to_eval.append((model_name, yolo_models[model_name], "yolo"))
            elif model_name in rfdetr_models:
                models_to_eval.append((model_name, rfdetr_models[model_name], "rfdetr"))
            else:
                print_warning(f"Unknown model: {model_name}")
    else:
        # Use filters
        include_yolo = args.yolo_only or args.all or (not args.yolo_only and not args.rfdetr_only)
        include_rfdetr = args.rfdetr_only or args.all or (not args.yolo_only and not args.rfdetr_only)
        
        if include_yolo:
            for name, path in yolo_models.items():
                models_to_eval.append((name, path, "yolo"))
        
        if include_rfdetr:
            for name, path in rfdetr_models.items():
                models_to_eval.append((name, path, "rfdetr"))
    
    if not models_to_eval:
        print_error("No models found to evaluate")
        print_info("Use --list to see available models")
        return 1
    
    # Print summary
    print_header("Batch Model Evaluation")
    print_info(f"Models to evaluate: {len(models_to_eval)}")
    print_info(f"Split: {args.split}")
    print_info(f"Device: {args.device}")
    print_info(f"Output: {args.output_dir}")
    print_info(f"COCO eval: {args.coco_eval}")
    print_info(f"Visuals: {args.visuals}")
    
    if args.dry_run:
        print_warning("DRY RUN - no actual evaluation will be performed")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    print_header("Running Evaluations")
    
    successful = 0
    failed = 0
    
    for i, (model_name, model_path, model_type) in enumerate(models_to_eval, 1):
        print_info(f"[{i}/{len(models_to_eval)}] Evaluating...")
        
        success = run_eval(
            model_name=model_name,
            model_path=model_path,
            model_type=model_type,
            split=args.split,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            output_dir=output_dir,
            coco_eval=args.coco_eval,
            visuals=args.visuals,
            dry_run=args.dry_run,
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print_header("Evaluation Complete")
    print_info(f"Successful: {successful}")
    if failed > 0:
        print_warning(f"Failed: {failed}")
    print_info(f"Results saved to: {args.output_dir}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())