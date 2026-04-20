#!/usr/bin/env python3
"""
Training script pro YOLO modely.

Pouziti:
    python scripts/train_yolo.py --model yolov8n --epochs 100
    python scripts/train_yolo.py --model yolo11l --epochs 200 --batch 16
    python scripts/train_yolo.py --model yolov8s --resume models/yolo/v1/yolov8s_v11/weights/last.pt
    python scripts/train_yolo.py --model yolov8m --name my_experiment --project models/yolo/v2
    python scripts/train_yolo.py --list-models
"""

import argparse
import sys
from pathlib import Path


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()
    def fileno(self):
        return self._streams[0].fileno()

# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
DATA_YAML = PROJECT_DIR / "data" / "dataset_yolo_v4s" / "data.yaml"
DEFAULT_PROJECT = PROJECT_DIR / "models" / "yolo" / "v1"

# COCO class index pro každou vlastní třídu (tourist, skier, cyclist, tourist_dog)
# COCO 0-indexed: 0=person, 1=bicycle, 16=dog, 30=skis
DEFAULT_COCO_MAPPING = [0, 30, 1, 16]

# Dostupne YOLO modely
AVAILABLE_MODELS = {
    # YOLOv8 varianty
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLO11 varianty
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
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


def list_models():
    """Print available models."""
    print_header("Available YOLO Models")
    print("\nYOLOv8 family:")
    for name in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
        print(f"  - {name}")
    print("\nYOLO11 family:")
    for name in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"]:
        print(f"  - {name}")
    print("\nSize guide: n=nano, s=small, m=medium, l=large, x=xlarge")


def generate_run_name(model_name: str, project_dir: Path) -> str:
    """Generate unique run name based on existing runs."""
    # Find existing runs for this model
    existing = list(project_dir.glob(f"{model_name}_v*"))
    if not existing:
        return f"{model_name}_v1"
    
    # Extract version numbers
    versions = []
    for p in existing:
        try:
            v = int(p.name.split("_v")[-1])
            versions.append(v)
        except ValueError:
            continue
    
    next_version = max(versions) + 1 if versions else 1
    return f"{model_name}_v{next_version}"


# =============================================================================
# COCO weight initialization
# =============================================================================

def init_coco_class_weights(
    model_path: str,
    nc: int,
    coco_mapping: list[int],
    save_dir: Path | None = None,
) -> str:
    """
    Vytvoří YOLO model s detection headem inicializovaným z COCO vah.

    Namísto náhodné inicializace class head použije váhy z odpovídajících COCO tříd.
    Funguje pro YOLOv8 a YOLO11 (cv3-based Detect head).

    Returns: cesta k uloženému dočasnému modelu .pt
    """
    import torch
    import torch.nn as nn
    from copy import deepcopy

    print_info(f"COCO-INIT: načítám {model_path}")

    from ultralytics import YOLO
    yolo = YOLO(model_path)
    yolo_model = yolo.model

    detect = yolo_model.model[-1]
    old_nc = getattr(detect, "nc", None)
    if old_nc is None:
        raise ValueError("Detect head nemá atribut 'nc' — nepodporovaná architektura")
    if not hasattr(detect, "cv3"):
        raise ValueError(
            f"Detect head nemá 'cv3' — architektura není YOLOv8/YOLO11. "
            f"Typ: {type(detect).__name__}"
        )
    if max(coco_mapping) >= old_nc:
        raise ValueError(
            f"COCO mapping {coco_mapping} vyžaduje index až {max(coco_mapping)}, "
            f"ale model má pouze {old_nc} tříd"
        )

    print_info(f"COCO-INIT: {old_nc} → {nc} tříd, mapování COCO indexů: {coco_mapping}")

    for i, seq in enumerate(detect.cv3):
        # Najdi poslední nn.Conv2d (ne Ultralytics Conv wrapper)
        last_idx, last_conv = None, None
        for j, layer in enumerate(seq):
            if isinstance(layer, nn.Conv2d):
                last_idx, last_conv = j, layer

        if last_conv is None:
            raise ValueError(f"cv3[{i}]: nenalezena nn.Conv2d")
        if last_conv.out_channels != old_nc:
            raise ValueError(
                f"cv3[{i}]: výstupní kanály={last_conv.out_channels}, očekáváno {old_nc}"
            )

        new_conv = nn.Conv2d(last_conv.in_channels, nc, 1, bias=True)
        new_conv.weight.data = last_conv.weight.data[coco_mapping].clone()
        if last_conv.bias is not None:
            new_conv.bias.data = last_conv.bias.data[coco_mapping].clone()
        seq[last_idx] = new_conv
        print_info(f"COCO-INIT: cv3[{i}]  {old_nc}→{nc} (COCO indices {coco_mapping})")

    detect.nc = nc
    yolo_model.nc = nc
    if hasattr(yolo_model, "yaml") and isinstance(yolo_model.yaml, dict):
        yolo_model.yaml["nc"] = nc

    # Ulož do trvalého souboru vedle ostatních modelů
    if save_dir is None:
        save_dir = PROJECT_DIR / "models" / "_temp"
    save_dir.mkdir(parents=True, exist_ok=True)

    base = Path(model_path).stem
    save_path = save_dir / f"{base}_coco_init_{nc}cls.pt"
    torch.save({"model": deepcopy(yolo_model).half()}, save_path)
    print_info(f"COCO-INIT: uloženo → {save_path}")
    return str(save_path)


# =============================================================================
# Training
# =============================================================================

def train(
    model_name: str,
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "0",
    workers: int = 8,
    project: Path = DEFAULT_PROJECT,
    name: str | None = None,
    resume: str | None = None,
    patience: int = 50,
    data_yaml: Path | None = None,
    optimizer: str = "auto",
    lr0: float = 0.01,
    lrf: float = 0.01,
    momentum: float = 0.937,
    weight_decay: float = 0.0005,
    warmup_epochs: float = 3.0,
    coco_init: bool = False,
    coco_mapping: list[int] | None = None,
    # Color/HSV augmentations
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    # Geometric augmentations
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
    # Flip augmentations
    flipud: float = 0.0,
    fliplr: float = 0.5,
    # Advanced mixing augmentations
    mosaic: float = 1.0,
    mixup: float = 0.0,
    copy_paste: float = 0.0,
    # Other augmentations
    bgr: float = 0.0,
    erasing: float = 0.4,
    close_mosaic: int = 10,
    auto_augment: str = "randaugment",
    augment: bool = True,
    cache: bool = False,
    exist_ok: bool = False,
    pretrained: bool = True,
    freeze: int | None = None,
    amp: bool = True,
    cos_lr: bool = False,
    extra_args: dict | None = None,
):
    """Run YOLO training."""
    from ultralytics import YOLO
    
    print_header("YOLO Training")
    
    # Resolve model
    if resume:
        model_path = resume
        print_info(f"Resuming from: {resume}")
    else:
        if model_name not in AVAILABLE_MODELS:
            print_error(f"Unknown model: {model_name}")
            print_info(f"Available: {', '.join(AVAILABLE_MODELS.keys())}")
            sys.exit(1)
        model_path = AVAILABLE_MODELS[model_name]
        print_info(f"Model: {model_name} ({model_path})")

    # COCO weight initialization
    if coco_init and not resume:
        mapping = coco_mapping if coco_mapping is not None else DEFAULT_COCO_MAPPING
        model_path = init_coco_class_weights(
            model_path=model_path,
            nc=4,
            coco_mapping=mapping,
        )

    # Generate run name if not provided
    if name is None:
        name = generate_run_name(model_name, project)
    
    print_info(f"Run name: {name}")
    print_info(f"Project: {project}")
    resolved_data = data_yaml if data_yaml is not None else DATA_YAML
    print_info(f"Dataset: {resolved_data}")
    print_info(f"Epochs: {epochs}")
    print_info(f"Batch size: {batch_size}")
    print_info(f"Image size: {imgsz}")
    print_info(f"Device: {device}")
    print_info(f"Workers: {workers}")
    print_info(f"Patience: {patience}")

    # Check data.yaml exists
    if not resolved_data.exists():
        print_error(f"Dataset config not found: {resolved_data}")
        sys.exit(1)

    # Patch data.yaml: ensure path field is absolute (Ultralytics resolves relative path from CWD)
    import yaml as _yaml
    with open(resolved_data) as _f:
        _cfg = _yaml.safe_load(_f)
    _p = _cfg.get("path", ".")
    if not Path(_p).is_absolute():
        _cfg["path"] = str((resolved_data.parent / _p).resolve())
        with open(resolved_data, "w") as _f:
            _yaml.dump(_cfg, _f, default_flow_style=False, allow_unicode=True)
        print_info(f"data.yaml path patched → {_cfg['path']}")

    # Load model
    model = YOLO(model_path)

    # Build training arguments
    train_args = {
        "data": str(resolved_data),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "project": str(project),
        "name": name,
        "exist_ok": exist_ok,
        "pretrained": pretrained,
        "patience": patience,
        "optimizer": optimizer,
        "lr0": lr0,
        "lrf": lrf,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        # Color/HSV augmentations
        "hsv_h": hsv_h,
        "hsv_s": hsv_s,
        "hsv_v": hsv_v,
        # Geometric augmentations
        "degrees": degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "perspective": perspective,
        # Flip augmentations
        "flipud": flipud,
        "fliplr": fliplr,
        # Advanced mixing augmentations
        "mosaic": mosaic,
        "mixup": mixup,
        "copy_paste": copy_paste,
        # Other augmentations
        "bgr": bgr,
        "erasing": erasing,
        "close_mosaic": close_mosaic,
        "auto_augment": auto_augment,
        "augment": augment,
        "cache": cache,
        "amp": amp,
        "cos_lr": cos_lr,
        "verbose": True,
        "save": True,
        "save_period": -1,  # save only best and last
        "plots": True,
    }
    
    if resume:
        train_args["resume"] = True
    
    if freeze is not None:
        train_args["freeze"] = freeze
    
    # Add extra args if provided
    if extra_args:
        train_args.update(extra_args)
    
    print_info("Starting training...")
    print("")

    # Redirect stdout+stderr to log file in the run directory
    import sys as _sys
    log_dir = Path(project) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    print_info(f"Log: {log_path}")
    _log_fh = open(log_path, "w", buffering=1)
    _tee_out = _TeeStream(_sys.stdout, _log_fh)
    _tee_err = _TeeStream(_sys.stderr, _log_fh)
    _orig_out, _orig_err = _sys.stdout, _sys.stderr
    _sys.stdout, _sys.stderr = _tee_out, _tee_err

    try:
        results = model.train(**train_args)
    finally:
        _sys.stdout, _sys.stderr = _orig_out, _orig_err
        _log_fh.close()
    
    print_header("Training Complete!")
    print_info(f"Results saved to: {project / name}")
    print_info(f"Best weights: {project / name / 'weights' / 'best.pt'}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_yolo.py --model yolov8n --epochs 100
  python scripts/train_yolo.py --model yolo11l --epochs 200 --batch 8
  python scripts/train_yolo.py --model yolov8s --resume models/yolo/v1/yolov8s_v1/weights/last.pt
  python scripts/train_yolo.py --list-models
        """
    )
    
    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to train (e.g., yolov8n, yolo11l)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    # Dataset
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (default: data/dataset_yolo_v4s/data.yaml)")

    # Training parameters
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", "-d", type=str, default="0", help="Device: 0, 1, cpu (default: 0)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Dataloader workers (default: 8)")
    
    # Output
    parser.add_argument("--project", "-p", type=str, default=str(DEFAULT_PROJECT), help="Project directory")
    parser.add_argument("--name", "-n", type=str, default=None, help="Run name (auto-generated if not set)")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing run")
    
    # Resume
    parser.add_argument("--resume", "-r", type=str, default=None, help="Resume from checkpoint")
    
    # Optimizer & Training
    parser.add_argument("--optimizer", type=str, default="auto", help="Optimizer: SGD, Adam, AdamW, auto")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Optimizer weight decay")
    parser.add_argument("--warmup-epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR scheduler")
    parser.add_argument("--amp", action="store_true", default=True, help="Automatic Mixed Precision")

    # Color/HSV Augmentation
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-Hue augmentation (0.0-1.0)")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-Saturation augmentation (0.0-1.0)")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-Value augmentation (0.0-1.0)")

    # Geometric Augmentation
    parser.add_argument("--degrees", type=float, default=0.0, help="Image rotation (+/- deg)")
    parser.add_argument("--translate", type=float, default=0.1, help="Image translation (+/- fraction)")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale (+/- gain)")
    parser.add_argument("--shear", type=float, default=0.0, help="Image shear (+/- deg)")
    parser.add_argument("--perspective", type=float, default=0.0, help="Image perspective (0.0-0.001)")

    # Flip Augmentation
    parser.add_argument("--flipud", type=float, default=0.0, help="Image flip up-down (probability)")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Image flip left-right (probability)")

    # Advanced Mixing Augmentation
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation (0.0-1.0)")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation (0.0-1.0)")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste augmentation (0.0-1.0)")

    # Other Augmentation
    parser.add_argument("--bgr", type=float, default=0.0, help="BGR channel swap probability")
    parser.add_argument("--erasing", type=float, default=0.4, help="Random erasing probability (0-0.9)")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic last N epochs")
    parser.add_argument("--auto-augment", type=str, default="randaugment", help="Auto augment policy")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    
    # Other
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from scratch")

    # COCO weight initialization
    parser.add_argument(
        "--coco-init",
        action="store_true",
        help=(
            "Inicializuj class head z COCO vah místo náhodné inicializace. "
            "Funguje pouze s COCO-pretrained modely (80 tříd)."
        ),
    )
    parser.add_argument(
        "--coco-mapping",
        type=int,
        nargs=4,
        default=None,
        metavar=("TOURIST", "SKIER", "CYCLIST", "DOG"),
        help=(
            f"COCO class index pro každou ze 4 tříd "
            f"(výchozí: {DEFAULT_COCO_MAPPING} = person,person,person,dog)"
        ),
    )
    
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
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=Path(args.project),
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        # Color/HSV augmentations
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        # Geometric augmentations
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        # Flip augmentations
        flipud=args.flipud,
        fliplr=args.fliplr,
        # Advanced mixing augmentations
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        # Other augmentations
        bgr=args.bgr,
        erasing=args.erasing,
        close_mosaic=args.close_mosaic,
        auto_augment=args.auto_augment,
        augment=not args.no_augment,
        cache=args.cache,
        exist_ok=args.exist_ok,
        pretrained=not args.no_pretrained,
        freeze=args.freeze,
        amp=args.amp,
        cos_lr=args.cos_lr,
        data_yaml=Path(args.data) if args.data else None,
        coco_init=args.coco_init,
        coco_mapping=args.coco_mapping,
    )


if __name__ == "__main__":
    main()
