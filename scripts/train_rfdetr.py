#!/usr/bin/env python3
"""
Training script pro RF-DETR modely.

Pouziti:
    python scripts/train_rfdetr.py --model base --epochs 100
    python scripts/train_rfdetr.py --model large --epochs 200 --batch 4
    python scripts/train_rfdetr.py --model small --lr 1e-4 --grad-accum 8
    python scripts/train_rfdetr.py --list-models
    python scripts/train_rfdetr.py --resume models/rfdetr/v1/rfdetr_base_v1/best.pt
    
    # S scratch dir (Metacentrum)
    python scripts/train_rfdetr.py --model base --scratch $SCRATCHDIR
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
COCO_DATASET = PROJECT_DIR / "data" / "coco_dataset"
DEFAULT_PROJECT = PROJECT_DIR / "models" / "rfdetr" / "v1"

# Dostupne RF-DETR modely
AVAILABLE_MODELS = {
    "small": "rf-detr-small.pth",
    "base": "rf-detr-base.pth",
    "medium": "rf-detr-medium.pth",  
    "large": "rf-detr-large.pth",
}

# =============================================================================
# Utility
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


def list_models():
    """Print available models."""
    print_header("Available RF-DETR Models")
    print("")
    for name, weights in AVAILABLE_MODELS.items():
        status = "OK" if (PROJECT_DIR / weights).exists() else "NOT FOUND"
        print(f"  - {name:8} ({weights}) [{status}]")
    print("")
    print("Size guide: small < base < medium < large")
    print("Larger models = better accuracy, more VRAM, slower training")


def generate_run_name(model_name: str, project_dir: Path) -> str:
    """Generate unique run name based on existing runs."""
    base_name = f"rfdetr_{model_name}"
    existing = list(project_dir.glob(f"{base_name}_v*"))
    
    if not existing:
        return f"{base_name}_v1"
    
    versions = []
    for p in existing:
        try:
            v = int(p.name.split("_v")[-1])
            versions.append(v)
        except ValueError:
            continue
    
    next_version = max(versions) + 1 if versions else 1
    return f"{base_name}_v{next_version}"


# =============================================================================
# Training
# =============================================================================

def train(
    model_name: str,
    epochs: int = 100,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    lr: float = 1e-4,
    lr_encoder: float = 1.5e-4,
    weight_decay: float = 1e-4,
    imgsz: int = 640,
    device: str = "0",
    workers: int = 4,
    project: Path = DEFAULT_PROJECT,
    name: str | None = None,
    resume: str | None = None,
    dataset_dir: Path = COCO_DATASET,
    warmup_epochs: int = 5,
    checkpoint_interval: int = 10,
    use_ema: bool = True,
    gradient_checkpointing: bool = False,
    # Early stopping
    early_stopping: bool = False,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001,
    early_stopping_use_ema: bool = False,
    # Logging
    tensorboard: bool = True,
    wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run: str | None = None,
    seed: int = 42,
    scratch_dir: Path | None = None,
):
    """Run RF-DETR training."""
    
    print_header("RF-DETR Training")
    
    # Import rfdetr
    try:
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRSmall
        # Try importing medium if available
        try:
            from rfdetr import RFDETRMedium
            has_medium = True
        except ImportError:
            has_medium = False
    except ImportError:
        print_error("rfdetr not installed. Run: pip install rfdetr")
        sys.exit(1)
    
    # Model selection
    model_classes = {
        "small": RFDETRSmall,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    }
    if has_medium:
        model_classes["medium"] = RFDETRMedium
    
    if model_name not in model_classes:
        print_error(f"Unknown model: {model_name}")
        print_info(f"Available: {', '.join(model_classes.keys())}")
        sys.exit(1)
    
    # Generate run name
    if name is None:
        name = generate_run_name(model_name, project)
    
    output_dir = project / name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle scratch directory
    working_dataset_dir = dataset_dir
    working_output_dir = output_dir
    
    if scratch_dir:
        scratch_dir = Path(scratch_dir)
        print_info(f"Using scratch directory: {scratch_dir}")
        
        # Copy dataset to scratch
        scratch_dataset = scratch_dir / "dataset"
        if not scratch_dataset.exists():
            print_info(f"Copying dataset to scratch...")
            shutil.copytree(dataset_dir, scratch_dataset)
            print_info(f"Dataset copied to {scratch_dataset}")
        else:
            print_info(f"Dataset already exists in scratch")
        
        working_dataset_dir = scratch_dataset
        
        # Use scratch for output during training
        scratch_output = scratch_dir / "output" / name
        scratch_output.mkdir(parents=True, exist_ok=True)
        working_output_dir = scratch_output
        print_info(f"Training output will be saved to scratch, then copied back")
    
    # Print config
    print_info(f"Model: RF-DETR {model_name}")
    print_info(f"Run name: {name}")
    print_info(f"Final output: {output_dir}")
    print_info(f"Working output: {working_output_dir}")
    print_info(f"Dataset: {working_dataset_dir}")
    print_info(f"Epochs: {epochs}")
    print_info(f"Batch size: {batch_size}")
    print_info(f"Gradient accumulation: {grad_accum_steps}")
    print_info(f"Effective batch size: {batch_size * grad_accum_steps}")
    print_info(f"Learning rate: {lr}")
    print_info(f"Image size: {imgsz}")
    print_info(f"Device: cuda:{device}")
    print_info(f"Workers: {workers}")
    
    # Validate dataset
    if not working_dataset_dir.exists():
        print_error(f"Dataset not found: {working_dataset_dir}")
        sys.exit(1)
    
    # RF-DETR expects _annotations.coco.json in each split folder
    required_files = [
        working_dataset_dir / "train" / "_annotations.coco.json",
        working_dataset_dir / "valid" / "_annotations.coco.json",
    ]
    
    for f in required_files:
        if not f.exists():
            print_error(f"Required file not found: {f}")
            print_error("Run: python scripts/convert_yolo_to_coco.py")
            sys.exit(1)
    
    print_info("Dataset validated OK")
    
    # Initialize model
    print_info("Initializing model...")
    model_class = model_classes[model_name]
    
    if resume:
        print_info(f"Resuming from: {resume}")
        model = model_class(pretrained_weights=resume)
    else:
        model = model_class()
    
    # Train
    print_info("Starting training...")
    print_info("Note: RF-DETR uses built-in 'architecture augmentation' for regularization")
    print("")

    try:
        model.train(
            dataset_dir=str(working_dataset_dir),
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            output_dir=str(working_output_dir),
            device=f"cuda:{device}" if device.isdigit() else device,
            num_workers=workers,
            use_ema=use_ema,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_interval=checkpoint_interval,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_use_ema=early_stopping_use_ema,
            tensorboard=tensorboard,
            wandb=wandb,
            project=wandb_project,
            run=wandb_run,
            seed=seed,
        )
    except TypeError as e:
        # Fallback for older rfdetr versions with different API
        print_warning(f"Trying alternative API due to: {e}")
        model.train(
            dataset_dir=str(working_dataset_dir),
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            weight_decay=weight_decay,
            output_dir=str(working_output_dir),
        )
    
    # Copy results back from scratch if used
    if scratch_dir and working_output_dir != output_dir:
        print_info(f"Copying results from scratch to {output_dir}...")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(working_output_dir, output_dir)
        print_info("Results copied successfully")
    
    print_header("Training Complete!")
    print_info(f"Results saved to: {output_dir}")
    
    # List output files
    print_info("Output files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_rfdetr.py --model base --epochs 100
  python scripts/train_rfdetr.py --model large --epochs 200 --batch 4
  python scripts/train_rfdetr.py --model small --lr 1e-4 --grad-accum 8
  python scripts/train_rfdetr.py --list-models
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["small", "base", "medium", "large"],
        help="Model size: small, base, medium, large"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    # Training parameters
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", "-b", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Model learning rate (default: 1e-4)")
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4, help="Encoder learning rate (default: 1.5e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", "-d", type=str, default="0", help="GPU device (default: 0)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs (default: 5)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use Exponential Moving Average")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001, help="Min improvement delta")
    parser.add_argument("--early-stopping-use-ema", action="store_true", help="Use EMA for early stopping")

    # Logging
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-run", type=str, default=None, help="W&B run name")
    
    # Output
    parser.add_argument("--project", "-p", type=str, default=str(DEFAULT_PROJECT), help="Project directory")
    parser.add_argument("--name", "-n", type=str, default=None, help="Run name (auto-generated if not set)")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default=str(COCO_DATASET), help="COCO dataset directory")
    
    # Resume
    parser.add_argument("--resume", "-r", type=str, default=None, help="Resume from checkpoint")
    
    # Scratch directory (Metacentrum)
    parser.add_argument(
        "--scratch", "-s",
        type=str,
        default=os.environ.get("SCRATCHDIR"),
        help="Scratch directory for faster I/O (default: $SCRATCHDIR)"
    )
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # List models and exit
    if args.list_models:
        list_models()
        sys.exit(0)
    
    # Check model is provided
    if args.model is None and args.resume is None:
        print_error("--model is required (or use --resume)")
        parser.print_help()
        sys.exit(1)
    
    # Run training
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=Path(args.project),
        name=args.name,
        resume=args.resume,
        dataset_dir=Path(args.dataset),
        warmup_epochs=args.warmup_epochs,
        checkpoint_interval=args.checkpoint_interval,
        use_ema=args.use_ema,
        gradient_checkpointing=args.gradient_checkpointing,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_use_ema=args.early_stopping_use_ema,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        seed=args.seed,
        scratch_dir=Path(args.scratch) if args.scratch else None,
    )


if __name__ == "__main__":
    main()
