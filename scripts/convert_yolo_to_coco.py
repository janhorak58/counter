#!/usr/bin/env python3
"""
Konverze YOLO datasetu do COCO formatu pro RF-DETR.

Pouziti:
    python scripts/convert_yolo_to_coco.py
    python scripts/convert_yolo_to_coco.py --val-split 0.15 --test-split 0.15
    python scripts/convert_yolo_to_coco.py --output data/coco_dataset
    python scripts/convert_yolo_to_coco.py --dry-run
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Konfigurace
# =============================================================================

PROJECT_DIR = Path(__file__).parent.parent
YOLO_DATASET = PROJECT_DIR / "data" / "yolo_dataset"
COCO_DATASET = PROJECT_DIR / "data" / "coco_dataset"

# Tridy z YOLO data.yaml
CLASSES = {
    0: "tourist",
    1: "skier",
    2: "cyclist",
    3: "tourist_dog",
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


# =============================================================================
# Konverzni funkce
# =============================================================================

def get_image_size(image_path: Path) -> tuple[int, int]:
    """Get image width and height."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def yolo_to_coco_bbox(yolo_bbox: list[float], img_width: int, img_height: int) -> list[float]:
    """
    Convert YOLO bbox to COCO bbox.
    
    YOLO: [x_center, y_center, width, height] (normalized 0-1)
    COCO: [x_min, y_min, width, height] (absolute pixels)
    """
    x_center, y_center, w, h = yolo_bbox
    
    # Denormalize
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    
    # Convert center to top-left
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    
    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]


def parse_yolo_label(label_path: Path) -> list[tuple[int, list[float]]]:
    """Parse YOLO label file and return list of (class_id, bbox)."""
    annotations = []
    
    if not label_path.exists():
        return annotations
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                annotations.append((class_id, bbox))
    
    return annotations


def create_coco_structure(output_dir: Path):
    """Create COCO directory structure for RF-DETR."""
    # RF-DETR expects:
    # dataset/
    #   train/
    #     _annotations.coco.json
    #     image1.jpg
    #     image2.jpg
    #   valid/
    #     _annotations.coco.json
    #     ...
    #   test/
    #     _annotations.coco.json
    #     ...
    for split in ["train", "valid", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)


def build_coco_categories() -> list[dict]:
    """Build COCO categories list."""
    return [
        {"id": class_id, "name": class_name, "supercategory": "person"}
        for class_id, class_name in CLASSES.items()
    ]


def convert_split(
    image_files: list[Path],
    labels_dir: Path,
    output_dir: Path,
    split_name: str,
    start_image_id: int = 0,
    start_ann_id: int = 0,
    copy_images: bool = True,
    dry_run: bool = False,
) -> tuple[dict, int, int]:
    """
    Convert a split (train/val/test) to COCO format.
    
    Returns:
        (coco_dict, next_image_id, next_ann_id)
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": build_coco_categories(),
    }
    
    image_id = start_image_id
    ann_id = start_ann_id
    
    images_output = output_dir / split_name
    
    skipped = 0
    total_anns = 0
    
    for img_path in tqdm(image_files, desc=f"Converting {split_name}"):
        # Find corresponding label file
        label_name = img_path.stem + ".txt"
        label_path = labels_dir / label_name
        
        # Get image size
        try:
            img_width, img_height = get_image_size(img_path)
        except Exception as e:
            print_warning(f"Cannot read image {img_path}: {e}")
            skipped += 1
            continue
        
        # Add image info
        coco["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": img_width,
            "height": img_height,
        })
        
        # Copy image
        if copy_images and not dry_run:
            shutil.copy(img_path, images_output / img_path.name)
        
        # Parse and convert annotations
        annotations = parse_yolo_label(label_path)
        
        for class_id, yolo_bbox in annotations:
            coco_bbox = yolo_to_coco_bbox(yolo_bbox, img_width, img_height)
            area = coco_bbox[2] * coco_bbox[3]
            
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": coco_bbox,
                "area": round(area, 2),
                "iscrowd": 0,
            })
            
            ann_id += 1
            total_anns += 1
        
        image_id += 1
    
    print_info(f"{split_name}: {len(coco['images'])} images, {total_anns} annotations")
    if skipped > 0:
        print_warning(f"{split_name}: skipped {skipped} images")
    
    return coco, image_id, ann_id


def split_dataset(
    yolo_train_images: list[Path],
    yolo_val_images: list[Path],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split dataset into train/val/test.
    
    Strategy:
    - Use existing YOLO val as base for val+test
    - If not enough, take from train
    - Randomly split val portion into val and test
    """
    random.seed(seed)
    
    all_images = yolo_train_images + yolo_val_images
    total = len(all_images)
    
    n_val = int(total * val_split)
    n_test = int(total * test_split)
    n_train = total - n_val - n_test
    
    # Shuffle all
    random.shuffle(all_images)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    return train_images, val_images, test_images


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to COCO format for RF-DETR"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(YOLO_DATASET),
        help=f"Input YOLO dataset directory (default: {YOLO_DATASET})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(COCO_DATASET),
        help=f"Output COCO dataset directory (default: {COCO_DATASET})"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Don't copy images, only create annotations (symlinks not supported)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    print_header("YOLO to COCO Conversion")
    
    # Validate input
    if not input_dir.exists():
        print_error(f"Input directory not found: {input_dir}")
        return 1
    
    # Find images and labels
    yolo_train_images_dir = input_dir / "images" / "train"
    yolo_val_images_dir = input_dir / "images" / "val"
    yolo_train_labels_dir = input_dir / "labels" / "train"
    yolo_val_labels_dir = input_dir / "labels" / "val"
    
    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    yolo_train_images = [
        p for p in yolo_train_images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]
    yolo_val_images = [
        p for p in yolo_val_images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ] if yolo_val_images_dir.exists() else []
    
    print_info(f"Found {len(yolo_train_images)} train images")
    print_info(f"Found {len(yolo_val_images)} val images")
    print_info(f"Total: {len(yolo_train_images) + len(yolo_val_images)} images")
    
    # Split dataset
    print_info(f"Splitting: train={1-args.val_split-args.test_split:.0%}, val={args.val_split:.0%}, test={args.test_split:.0%}")
    
    train_images, val_images, test_images = split_dataset(
        yolo_train_images,
        yolo_val_images,
        args.val_split,
        args.test_split,
        args.seed,
    )
    
    print_info(f"Split result: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    if args.dry_run:
        print_info("Dry run - no files will be created")
        return 0
    
    # Create output structure
    print_info(f"Creating output directory: {output_dir}")
    create_coco_structure(output_dir)
    
    # Determine labels directory for each image
    def get_labels_dir(img_path: Path) -> Path:
        if str(yolo_train_images_dir) in str(img_path.parent):
            return yolo_train_labels_dir
        return yolo_val_labels_dir
    
    # Convert each split
    image_id = 0
    ann_id = 0
    
    # Train
    train_coco, image_id, ann_id = convert_split(
        train_images,
        yolo_train_labels_dir,  # Most images from train
        output_dir,
        "train",
        image_id,
        ann_id,
        copy_images=not args.no_copy,
    )
    
    # Val -> valid (RF-DETR naming)
    val_coco, image_id, ann_id = convert_split(
        val_images,
        yolo_train_labels_dir,
        output_dir,
        "valid",
        image_id,
        ann_id,
        copy_images=not args.no_copy,
    )
    
    # Test
    test_coco, image_id, ann_id = convert_split(
        test_images,
        yolo_train_labels_dir,
        output_dir,
        "test",
        image_id,
        ann_id,
        copy_images=not args.no_copy,
    )
    
    # Save annotations (RF-DETR expects _annotations.coco.json in each split folder)
    print_info("Saving annotation files...")
    
    with open(output_dir / "train" / "_annotations.coco.json", "w") as f:
        json.dump(train_coco, f)
    
    with open(output_dir / "valid" / "_annotations.coco.json", "w") as f:
        json.dump(val_coco, f)
    
    with open(output_dir / "test" / "_annotations.coco.json", "w") as f:
        json.dump(test_coco, f)
    
    # Summary
    print_header("Conversion Complete!")
    
    print_info(f"Output directory: {output_dir}")
    print("")
    print("Structure (RF-DETR format):")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── _annotations.coco.json  ({len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations)")
    print(f"  │   └── *.jpg/png               ({len(train_images)} files)")
    print(f"  ├── valid/")
    print(f"  │   ├── _annotations.coco.json  ({len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations)")
    print(f"  │   └── *.jpg/png               ({len(val_images)} files)")
    print(f"  └── test/")
    print(f"      ├── _annotations.coco.json  ({len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations)")
    print(f"      └── *.jpg/png               ({len(test_images)} files)")
    print("")
    print("Categories:")
    for cat in train_coco["categories"]:
        print(f"  {cat['id']}: {cat['name']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
