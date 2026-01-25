from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw


def _load_coco(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_annotations(
    annotations: List[dict],
) -> Dict[int, List[dict]]:
    by_image: Dict[int, List[dict]] = {}
    for ann in annotations:
        image_id = int(ann["image_id"])
        by_image.setdefault(image_id, []).append(ann)
    return by_image


def _build_cat_map(categories: List[dict]) -> Dict[int, str]:
    return {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in categories}


def _resolve_image_path(image_entry: dict, images_root: Path) -> Path:
    file_name = Path(str(image_entry["file_name"]))
    if file_name.is_absolute():
        return file_name
    return images_root / file_name


def _draw_annotations(
    image: Image.Image,
    annotations: List[dict],
    cat_map: Dict[int, str],
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for ann in annotations:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x2 = x + w
        y2 = y + h
        draw.rectangle([x, y, x2, y2], outline="#ff6b35", width=3)
        label = cat_map.get(int(ann.get("category_id", -1)), "unknown")
        draw.text((x + 3, y + 3), label, fill="#ff6b35")
    return image


def _save_grid(images: List[Tuple[Image.Image, str]], output_path: Path) -> None:
    if not images:
        return
    cols = 5
    rows = (len(images) + cols - 1) // cols
    thumb_w, thumb_h = images[0][0].size
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), color="#111111")
    for idx, (img, _) in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * thumb_w, row * thumb_h))
    grid.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize random COCO annotations.")
    parser.add_argument("--coco", required=True, help="Path to COCO _annotations.coco.json")
    parser.add_argument("--images-root", default=None, help="Images root (defaults to JSON parent)")
    parser.add_argument("--output-dir", default="outputs/coco_samples", help="Output directory")
    parser.add_argument("--count", type=int, default=100, help="Number of random images to visualize")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    coco_path = Path(args.coco)
    coco = _load_coco(coco_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    images_root = Path(args.images_root) if args.images_root else coco_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_index = _index_annotations(annotations)
    cat_map = _build_cat_map(categories)

    rng = random.Random(args.seed)
    chosen = rng.sample(images, min(args.count, len(images)))

    saved: List[Tuple[Image.Image, str]] = []
    for image_entry in chosen:
        image_id = int(image_entry["id"])
        image_path = _resolve_image_path(image_entry, images_root)
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image = _draw_annotations(image, ann_index.get(image_id, []), cat_map)
        out_path = output_dir / image_path.name
        image.save(out_path)
        saved.append((image.copy(), image_path.name))

    if saved:
        _save_grid(saved, output_dir / "grid.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
