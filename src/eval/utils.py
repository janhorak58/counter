import os
import re
from typing import Dict, Optional

import yaml

EPS = 1e-9
KNOWN_CLASSES = {0, 1, 2, 3}
CLASS_NAME_MAP = {
    0: "tourist",
    1: "skier",
    2: "cyclist",
    3: "tourist_dog",
}

DEFAULT_PRETRAINED_COUNT_MAP = {
    4: 0,  # person -> tourist
    5: 1,  # skis -> skier
    6: 2,  # bicycle -> cyclist
    7: 3,  # dog -> tourist_dog
}

COCO_CLASS_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    16: "dog",
    30: "skis",
}

COCO_NAME_TO_CUSTOM = {
    "person": "tourist",
    "bicycle": "cyclist",
    "skis": "skier",
    "dog": "tourist_dog",
}


def parse_video_num(filename: str) -> Optional[str]:
    m = re.search(r"vid(\d+)", filename, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"data_(\d+)", filename, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def parse_model_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r"^vid\d+_", "", base, flags=re.IGNORECASE)
    base = re.sub(r"_results$", "", base, flags=re.IGNORECASE)
    return base or "unknown_model"


def load_data_yaml(data_yaml: str) -> Dict:
    with open(data_yaml, "r") as f:
        return yaml.safe_load(f)


def dataset_name_to_id(data_yaml: str) -> Dict[str, int]:
    data = load_data_yaml(data_yaml)
    names = data.get("names", {})
    if isinstance(names, dict):
        return {v: int(k) for k, v in names.items()}
    if isinstance(names, list):
        return {name: i for i, name in enumerate(names)}
    return {}


def dataset_class_ids(data_yaml: str) -> Dict[int, int]:
    name_to_id = dataset_name_to_id(data_yaml)
    return {idx: idx for idx in name_to_id.values()}


def coco_to_dataset_map(data_yaml: str) -> Dict[int, int]:
    name_to_id = dataset_name_to_id(data_yaml)
    mapping: Dict[int, int] = {}
    for coco_id, coco_name in COCO_CLASS_ID_TO_NAME.items():
        custom_name = COCO_NAME_TO_CUSTOM.get(coco_name)
        if custom_name is None:
            continue
        if custom_name in name_to_id:
            mapping[coco_id] = name_to_id[custom_name]
    return mapping
