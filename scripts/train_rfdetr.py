#!/usr/bin/env python3
"""
Training script pro RF-DETR modely + maximalni augmentace (data aug).

Pouziti:
  python scripts/train_rfdetr.py --model medium --epochs 100
  python scripts/train_rfdetr.py --model large --epochs 200 --batch 4 --grad-accum 8
  python scripts/train_rfdetr.py --model small --lr 1e-4 --grad-accum 8
  python scripts/train_rfdetr.py --list-models
  python scripts/train_rfdetr.py --resume models/rfdetr/v1/rfdetr_medium_v7/best.pt

  # Scratch dir (Metacentrum)
  python scripts/train_rfdetr.py --model medium --scratch $SCRATCHDIR

  # Vytiskne realne pouzite transformy (po patchi)
  python scripts/train_rfdetr.py --model medium --print-augs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Callable


# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_DIR / "data" / "coco_dataset"
DEFAULT_PROJECT = PROJECT_DIR / "models" / "rfdetr" / "v1"


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


def seed_everything(seed: int):
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def infer_model_from_checkpoint(path: str) -> str:
    p = path.lower()
    for k in ("nano", "small", "medium", "base", "large"):
        if k in p:
            return k
    return "medium"


def get_available_model_classes() -> dict[str, type]:
    try:
        import rfdetr
    except ImportError:
        print_error("rfdetr neni nainstalovany. Instalace: pip install rfdetr")
        sys.exit(1)

    name_map = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "base": "RFDETRBase",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    }

    out: dict[str, type] = {}
    for key, cls_name in name_map.items():
        cls = getattr(rfdetr, cls_name, None)
        if cls is not None:
            out[key] = cls
    return out


def list_models():
    print_header("Available RF-DETR Models (detekce podle nainstalovane verze)")
    model_classes = get_available_model_classes()
    if not model_classes:
        print("  (zadne modely nenalezeny v rfdetr importu)")
        return
    for k in ("nano", "small", "base", "medium", "large"):
        if k in model_classes:
            print(f"  - {k}")
    print("")


def generate_run_name(model_name: str, project_dir: Path) -> str:
    base_name = f"rfdetr_{model_name}"
    existing = list(project_dir.glob(f"{base_name}_v*"))
    if not existing:
        return f"{base_name}_v1"

    versions: list[int] = []
    for p in existing:
        try:
            v = int(p.name.split("_v")[-1])
            versions.append(v)
        except ValueError:
            continue

    next_version = max(versions) + 1 if versions else 1
    return f"{base_name}_v{next_version}"


# =============================================================================
# RF-DETR augmentation patch (monkey-patch coco make_*_transforms)
# =============================================================================

def install_rfdetr_aug_patch_max(
    *,
    enable: bool,
    hflip_p: float,
    color_p: float,
    erasing_p: float,
    expand_prob: float,
    expand_ratio: float,
    pad_max: int,
    square_resize_div_64: bool,
):
    """
    "Max" preset: pridava vsechny bbox-aware augmentace, ktere jsou v rfdetr.datasets.transforms:
      - RandomHorizontalFlip(p)
      - RandomResize / SquareResize (uz default)
      - RandomSizeCrop (uz default)
      - RandomPad (extra)
      - RandomExpand (extra; PIL->ndarray->PIL)
      - RandomErasing (extra; po ToTensor)
    + image-only augmentace (bbox se nemeni):
      - ColorJitter, RandomGrayscale, RandomAutocontrast, RandomEqualize,
        RandomAdjustSharpness, GaussianBlur (pokud je dostupne v torchvision).
    """
    if not enable:
        return

    try:
        import torchvision.transforms as tvT  # type: ignore
    except Exception:
        tvT = None  # type: ignore

    try:
        import rfdetr.datasets.coco as coco  # type: ignore
        import rfdetr.datasets.transforms as T  # type: ignore
    except Exception as e:
        print_warning(f"Aug patch preskocen (import rfdetr.datasets.* selhal): {e}")
        return

    # ---- wrappers
    class ImgOnlyRandomApply:
        def __init__(self, t: Callable[[Any], Any], p: float):
            self.t = t
            self.p = p

        def __call__(self, img, target):
            if random.random() < self.p:
                return self.t(img), target
            return img, target

        def __repr__(self):
            return f"{self.__class__.__name__}(p={self.p}, t={self.t})"

    def _maybe_add_img_only(transforms: list, ctor_name: str, apply_p: float, *args, **kwargs):
        if tvT is None:
            return
        ctor = getattr(tvT, ctor_name, None)
        if ctor is None:
            return
        try:
            transforms.append(ImgOnlyRandomApply(ctor(*args, **kwargs), p=apply_p))
        except Exception:
            return

    def build_pre_geom_extras() -> list:
        extras: list = []

        # Image-only color a "photometric" augmentace (bez zmen bbox)
        # Tyhle jsou bezpecne, protoze nehybou geometrii.
        if color_p > 0:
            _maybe_add_img_only(extras, "ColorJitter", color_p, brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)
            _maybe_add_img_only(extras, "RandomGrayscale", min(0.2, color_p), p=1.0)  # wrapper ma vlastni p
            _maybe_add_img_only(extras, "RandomAutocontrast", min(0.2, color_p), p=1.0)
            _maybe_add_img_only(extras, "RandomEqualize", min(0.2, color_p), p=1.0)
            _maybe_add_img_only(extras, "RandomAdjustSharpness", min(0.15, color_p), sharpness_factor=2.0, p=1.0)
            _maybe_add_img_only(extras, "GaussianBlur", min(0.15, color_p), kernel_size=3, sigma=(0.1, 2.0))

        # BBox-aware expand (vyzaduje numpy pipeline)
        # POZOR: v rfdetr RandomExpand ma "prob" semantics: pokud rand < prob -> NO-OP, jinak expand.
        # Chceme expand_prob jako pravdepodobnost provest expand => nastavime prob = 1 - expand_prob.
        if expand_prob > 0:
            try:
                extras.extend([
                    T.PILtoNdArray(),
                    T.RandomExpand(ratio=float(expand_ratio), prob=float(max(0.0, min(1.0, 1.0 - expand_prob)))),
                    T.NdArraytoPIL(),
                ])
            except Exception as e:
                print_warning(f"RandomExpand nelze pridat: {e}")

        # BBox-safe pad (pridava padding jen doprava/dolu, bbox se nemeni)
        if pad_max and pad_max > 0:
            try:
                extras.append(T.RandomPad(int(pad_max)))
            except Exception as e:
                print_warning(f"RandomPad nelze pridat: {e}")

        return extras

    def build_normalize_with_erasing() -> Any:
        # Erasing je tensor-only, takze musi byt po ToTensor.
        t_list = [T.ToTensor()]
        if erasing_p and erasing_p > 0:
            try:
                t_list.append(T.RandomErasing(p=float(erasing_p), scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"))
            except Exception as e:
                print_warning(f"RandomErasing nelze pridat: {e}")
        t_list.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        return T.Compose(t_list)

    # ---- patch functions (reimplement na zaklade rfdetr.datasets.coco)
    orig_make = getattr(coco, "make_coco_transforms", None)
    orig_make_sq64 = getattr(coco, "make_coco_transforms_square_div_64", None)
    compute_scales = getattr(coco, "compute_multi_scale_scales", None)

    def _compute_scales(resolution: int, multi_scale: bool, expanded_scales: bool, patch_size: int, num_windows: int) -> list[int]:
        if not multi_scale:
            return [int(resolution)]
        if callable(compute_scales):
            try:
                return list(compute_scales(int(resolution), bool(expanded_scales), int(patch_size), int(num_windows)))
            except Exception:
                pass
        # fallback
        return [int(resolution)]

    def make_coco_transforms_patched(
        image_set: str,
        resolution: int,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        skip_random_resize: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
    ):
        normalize = build_normalize_with_erasing()
        scales = _compute_scales(resolution, multi_scale, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]

        if image_set == "train":
            return T.Compose([
                T.RandomHorizontalFlip(p=float(hflip_p)),
                *build_pre_geom_extras(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ]),
                    p=0.5,
                ),
                normalize,
            ])

        if image_set == "val":
            return T.Compose([
                T.RandomResize([int(resolution)], max_size=1333),
                normalize,
            ])

        if image_set == "val_speed":
            return T.Compose([
                T.SquareResize([int(resolution)]),
                normalize,
            ])

        raise ValueError(f"unknown image_set={image_set}")

    def make_coco_transforms_square_div_64_patched(
        image_set: str,
        resolution: int,
        multi_scale: bool = False,
        expanded_scales: bool = False,
        skip_random_resize: bool = False,
        patch_size: int = 16,
        num_windows: int = 4,
    ):
        normalize = build_normalize_with_erasing()
        scales = _compute_scales(resolution, multi_scale, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]

        if image_set == "train":
            return T.Compose([
                T.RandomHorizontalFlip(p=float(hflip_p)),
                *build_pre_geom_extras(),
                T.RandomSelect(
                    T.SquareResize(scales),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.SquareResize(scales),
                    ]),
                    p=0.5,
                ),
                normalize,
            ])

        if image_set in ("val", "test", "val_speed"):
            return T.Compose([
                T.SquareResize([int(resolution)]),
                normalize,
            ])

        raise ValueError(f"unknown image_set={image_set}")

    # Instalace patchu do modulu
    coco.make_coco_transforms = make_coco_transforms_patched  # type: ignore[attr-defined]
    coco.make_coco_transforms_square_div_64 = make_coco_transforms_square_div_64_patched  # type: ignore[attr-defined]

    # Guard: kdyby nekdo chtel revert v runtime, nechavame orig v __dict__
    coco.__dict__["_orig_make_coco_transforms"] = orig_make
    coco.__dict__["_orig_make_coco_transforms_square_div_64"] = orig_make_sq64

    # Preferovani square_resize_div_64 v treningu muze vyzadovat flag v TrainConfig; to se resi v args a model.train.
    _ = square_resize_div_64


# =============================================================================
# Training
# =============================================================================

def train(
    model_name: str | None,
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
    dataset_dir: Path = DEFAULT_DATASET,
    warmup_epochs: int = 5,
    checkpoint_interval: int = 10,
    use_ema: bool = True,
    gradient_checkpointing: bool = False,
    # multi-scale / resizing knobs (TrainConfig)
    multi_scale: bool = True,
    expanded_scales: bool = True,
    do_random_resize_via_padding: bool = True,
    square_resize_div_64: bool = True,
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
    # Augmentation
    augs: str = "max",  # none|default|max
    print_augs: bool = False,
    aug_hflip_p: float = 0.5,
    aug_color_p: float = 0.8,
    aug_erasing_p: float = 0.25,
    aug_expand_p: float = 0.5,
    aug_expand_ratio: float = 4.0,
    aug_pad_max: int = 32,
):
    print_header("RF-DETR Training")

    seed_everything(seed)

    model_classes = get_available_model_classes()
    if not model_classes:
        print_error("Nelze nacist zadnou RFDETR* tridu z rfdetr. Zkontroluj instalaci baliku.")
        sys.exit(1)

    if resume and not model_name:
        model_name = infer_model_from_checkpoint(resume)

    if not model_name:
        print_error("--model je povinny (nebo --resume)")
        sys.exit(1)

    if model_name not in model_classes:
        print_error(f"Unknown/Unavailable model: {model_name}")
        print_info(f"Available: {', '.join(model_classes.keys())}")
        sys.exit(1)

    # Run name/output
    project.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = generate_run_name(model_name, project)
    output_dir = project / name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scratch handling
    working_dataset_dir = dataset_dir
    working_output_dir = output_dir

    if scratch_dir:
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Using scratch directory: {scratch_dir}")

        scratch_dataset = scratch_dir / "dataset"
        if not scratch_dataset.exists():
            print_info("Copying dataset to scratch...")
            shutil.copytree(dataset_dir, scratch_dataset)
            print_info(f"Dataset copied to {scratch_dataset}")
        else:
            print_info("Dataset already exists in scratch")

        working_dataset_dir = scratch_dataset

        scratch_output = scratch_dir / "output" / name
        scratch_output.mkdir(parents=True, exist_ok=True)
        working_output_dir = scratch_output
        print_info("Training output will be saved to scratch, then copied back")

    # Validate dataset (Roboflow COCO structure)
    if not working_dataset_dir.exists():
        print_error(f"Dataset not found: {working_dataset_dir}")
        sys.exit(1)

    required_files = [
        working_dataset_dir / "train" / "_annotations.coco.json",
        working_dataset_dir / "valid" / "_annotations.coco.json",
    ]
    for f in required_files:
        if not f.exists():
            print_error(f"Required file not found: {f}")
            print_error("Ocekavam Roboflow COCO export: train/valid/_annotations.coco.json")
            sys.exit(1)

    # Aug patch
    if augs == "max":
        install_rfdetr_aug_patch_max(
            enable=True,
            hflip_p=aug_hflip_p,
            color_p=aug_color_p,
            erasing_p=aug_erasing_p,
            expand_prob=aug_expand_p,
            expand_ratio=aug_expand_ratio,
            pad_max=aug_pad_max,
            square_resize_div_64=square_resize_div_64,
        )
    elif augs in ("default", "none"):
        pass
    else:
        print_error(f"Unknown --augs preset: {augs} (valid: none, default, max)")
        sys.exit(1)

    if print_augs and augs == "max":
        try:
            import rfdetr.datasets.coco as coco  # type: ignore
            # vytisknout obe varianty, at je videt rozdil
            print_header("Patched train transforms (make_coco_transforms)")
            print(coco.make_coco_transforms("train", imgsz, multi_scale=multi_scale, expanded_scales=expanded_scales,
                                           skip_random_resize=not do_random_resize_via_padding))
            print_header("Patched train transforms (make_coco_transforms_square_div_64)")
            print(coco.make_coco_transforms_square_div_64("train", imgsz, multi_scale=multi_scale, expanded_scales=expanded_scales,
                                                         skip_random_resize=not do_random_resize_via_padding))
        except Exception as e:
            print_warning(f"print-augs selhal: {e}")

    # Save config to JSON
    config = {
        "model": model_name,
        "run_name": name,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "lr": lr,
        "lr_encoder": lr_encoder,
        "weight_decay": weight_decay,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "warmup_epochs": warmup_epochs,
        "checkpoint_interval": checkpoint_interval,
        "use_ema": use_ema,
        "gradient_checkpointing": gradient_checkpointing,
        "multi_scale": multi_scale,
        "expanded_scales": expanded_scales,
        "do_random_resize_via_padding": do_random_resize_via_padding,
        "square_resize_div_64": square_resize_div_64,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_use_ema": early_stopping_use_ema,
        "augs": augs,
        "aug_hflip_p": aug_hflip_p,
        "aug_color_p": aug_color_p,
        "aug_erasing_p": aug_erasing_p,
        "aug_expand_p": aug_expand_p,
        "aug_expand_ratio": aug_expand_ratio,
        "aug_pad_max": aug_pad_max,
        "seed": seed,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
    }

    config_path = output_dir / "train_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Compact config print
    print_header(f"RF-DETR {model_name.upper()} Training - {name}")
    print(f"{'='*70}")
    print(f"Model:        RF-DETR {model_name}")
    print(f"Output:       {output_dir}")
    print(f"Dataset:      {working_dataset_dir}")
    print(f"{'='*70}")
    print(f"Epochs:       {epochs}  |  Batch: {batch_size}x{grad_accum_steps} = {batch_size * grad_accum_steps}")
    print(f"LR:           {lr:.2e}  |  LR Enc: {lr_encoder:.2e}  |  WD: {weight_decay:.2e}")
    print(f"Image size:   {imgsz}  |  Workers: {workers}  |  Device: cuda:{device}")
    print(f"{'='*70}")
    print(f"Augmentations: {augs}")
    if augs == "max":
        print(f"  hflip={aug_hflip_p} color={aug_color_p} erasing={aug_erasing_p}")
        print(f"  expand={aug_expand_p} (ratio={aug_expand_ratio}) pad_max={aug_pad_max}")
    print(f"{'='*70}")
    if early_stopping:
        print(f"Early Stop:   patience={early_stopping_patience}  min_delta={early_stopping_min_delta}")
        print(f"{'='*70}")
    print(f"Config saved: {config_path}")
    print(f"{'='*70}\n")

    # Reduce verbosity from rfdetr library
    logging.getLogger("rfdetr").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Init model
    print(f"Initializing model...")
    model_class = model_classes[model_name]
    if resume:
        print(f"Resuming from: {resume}")
        try:
            model = model_class(pretrained_weights=resume)  # older/newer variants
        except TypeError:
            model = model_class(weights=resume)  # fallback
    else:
        model = model_class()

    # Train
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}\n")

    train_kwargs: dict[str, Any] = dict(
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
        warmup_epochs=warmup_epochs,
        checkpoint_interval=checkpoint_interval,
        use_ema=use_ema,
        gradient_checkpointing=gradient_checkpointing,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        tensorboard=tensorboard,
        wandb=wandb,
        project=wandb_project,
        run=wandb_run,
        seed=seed,
        # TrainConfig knobs pro transformy/scales
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        square_resize_div_64=square_resize_div_64,
    )

    try:
        model.train(**train_kwargs)
    except TypeError as e:
        # Fallback pro starsi API
        print_warning(f"Falling back to minimal train() args due to TypeError: {e}")
        minimal = dict(
            dataset_dir=str(working_dataset_dir),
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            lr=lr,
            weight_decay=weight_decay,
            output_dir=str(working_output_dir),
        )
        model.train(**minimal)

    # Copy results back from scratch
    if scratch_dir and working_output_dir != output_dir:
        print(f"\n{'='*70}")
        print(f"Copying results from scratch to {output_dir}...")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.copytree(working_output_dir, output_dir)
        print(f"Results copied successfully")

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results: {output_dir}")

    # Print key metrics if available
    results_json = output_dir / "results.json"
    if results_json.exists():
        try:
            with open(results_json) as f:
                results = json.load(f)
            if "test" in results:
                test_results = results["test"]
                print(f"\nFinal Metrics:")
                if "coco_eval_bbox" in test_results:
                    bbox = test_results["coco_eval_bbox"]
                    print(f"  mAP:       {bbox[0]:.4f}")
                    print(f"  mAP@50:    {bbox[1]:.4f}")
                    print(f"  mAP@75:    {bbox[2]:.4f}")
        except Exception:
            pass

    print(f"{'='*70}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RF-DETR Training Script (max augmentations patch)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--list-models", action="store_true", help="List available model classes and exit")

    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model size (depends on installed rfdetr): nano|small|base|medium|large")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Resume/pretrained checkpoint path")

    # Training params
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch", "-b", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", "-d", type=str, default="0")
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)

    # Bool toggles (Python 3.10+)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable EMA")
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run", type=str, default=None)

    # TrainConfig transform/scales knobs
    parser.add_argument("--multi-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expanded-scales", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-random-resize-via-padding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--square-resize-div-64", action=argparse.BooleanOptionalAction, default=True)

    # Early stopping
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)
    parser.add_argument("--early-stopping-use-ema", action=argparse.BooleanOptionalAction, default=False)

    # Augmentace
    parser.add_argument("--augs", type=str, choices=["none", "default", "max"], default="max",
                        help="Aug preset: none(default rfdetr), default(default rfdetr), max(patch + extra augs)")
    parser.add_argument("--print-augs", action="store_true", default=False, help="Print patched transforms")
    parser.add_argument("--aug-hflip-p", type=float, default=0.5)
    parser.add_argument("--aug-color-p", type=float, default=0.8)
    parser.add_argument("--aug-erasing-p", type=float, default=0.25)
    parser.add_argument("--aug-expand-p", type=float, default=0.5)
    parser.add_argument("--aug-expand-ratio", type=float, default=4.0)
    parser.add_argument("--aug-pad-max", type=int, default=32)

    # Output/dataset
    parser.add_argument("--project", "-p", type=str, default=str(DEFAULT_PROJECT))
    parser.add_argument("--name", "-n", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))

    # Scratch
    parser.add_argument("--scratch", "-s", type=str, default=os.environ.get("SCRATCHDIR"))

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if args.model is None and args.resume is None:
        print_error("--model is required (or use --resume)")
        sys.exit(1)

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
        use_ema=args.ema,
        gradient_checkpointing=args.gradient_checkpointing,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        do_random_resize_via_padding=args.do_random_resize_via_padding,
        square_resize_div_64=args.square_resize_div_64,
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
        augs=args.augs,
        print_augs=args.print_augs,
        aug_hflip_p=args.aug_hflip_p,
        aug_color_p=args.aug_color_p,
        aug_erasing_p=args.aug_erasing_p,
        aug_expand_p=args.aug_expand_p,
        aug_expand_ratio=args.aug_expand_ratio,
        aug_pad_max=args.aug_pad_max,
    )


if __name__ == "__main__":
    main()