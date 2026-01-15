import os
from ast import literal_eval
from typing import Dict, List, Tuple

import pandas as pd

from src.eval.utils import DEFAULT_PRETRAINED_COUNT_MAP, parse_model_name, parse_video_num


def load_yolo_results_from_csv(file_path: str) -> List[Tuple[Dict[int, int], Dict[int, int]]]:
    df = pd.read_csv(file_path)
    results = []
    for _, row in df.iterrows():
        in_count = literal_eval(row["in_count"])
        out_count = literal_eval(row["out_count"])
        results.append((in_count, out_count))
    return results


def load_gt_results_from_csv(file_path: str) -> List[Tuple[Dict[int, int], Dict[int, int]]]:
    def class_name_mapping(name: str):
        mapping = {
            "dogs left": 3,
            "dogs right": 3,
            "people left": 0,
            "pople right": 0,
            "cyclist left": 2,
            "cyclist right": 2,
            "skiers left": 1,
            "skiers right": 1,
        }
        return mapping.get(name.lower(), name)

    df = pd.read_csv(file_path, header=None)
    in_count: Dict[int, int] = {}
    out_count: Dict[int, int] = {}
    for index, row in df.iterrows():
        if index == 0:
            continue
        class_id = class_name_mapping(row[0])
        count = int(row[1])
        if "left" in row[0].lower():
            in_count[class_id] = count
        elif "right" in row[0].lower():
            out_count[class_id] = count
    return [(in_count, out_count)]


def _apply_class_map(
    counts: List[Tuple[Dict[int, int], Dict[int, int]]],
    class_map: Dict[int, int],
) -> List[Tuple[Dict[int, int], Dict[int, int]]]:
    mapped = []
    for in_counts, out_counts in counts:
        in_new: Dict[int, int] = {}
        out_new: Dict[int, int] = {}
        for k, v in in_counts.items():
            new_k = class_map.get(int(k), int(k))
            in_new[new_k] = in_new.get(new_k, 0) + int(v)
        for k, v in out_counts.items():
            new_k = class_map.get(int(k), int(k))
            out_new[new_k] = out_new.get(new_k, 0) + int(v)
        mapped.append((in_new, out_new))
    return mapped


def compare_counts(
    gt_counts: List[Tuple[Dict[int, int], Dict[int, int]]],
    pred_counts: List[Tuple[Dict[int, int], Dict[int, int]]],
) -> List[Dict[str, int]]:
    comparison_results = []
    if len(gt_counts) == 1 and len(pred_counts) > 1:
        pred_in: Dict[int, int] = {}
        pred_out: Dict[int, int] = {}
        for in_c, out_c in pred_counts:
            for k, v in in_c.items():
                pred_in[k] = pred_in.get(k, 0) + int(v)
            for k, v in out_c.items():
                pred_out[k] = pred_out.get(k, 0) + int(v)
        pred_counts = [(pred_in, pred_out)]

    for (gt_in, gt_out), (pred_in, pred_out) in zip(gt_counts, pred_counts):
        all_classes = set(gt_in.keys()) | set(gt_out.keys()) | set(pred_in.keys()) | set(pred_out.keys())
        for class_id in all_classes:
            gt_in_count = int(gt_in.get(class_id, 0))
            gt_out_count = int(gt_out.get(class_id, 0))
            pred_in_count = int(pred_in.get(class_id, 0))
            pred_out_count = int(pred_out.get(class_id, 0))
            comparison_results.append(
                {
                    "class_id": class_id,
                    "gt_in": gt_in_count,
                    "gt_out": gt_out_count,
                    "pred_in": pred_in_count,
                    "pred_out": pred_out_count,
                    "in_diff": pred_in_count - gt_in_count,
                    "out_diff": pred_out_count - gt_out_count,
                }
            )
    return comparison_results


def build_complete_results(
    gt_folder: str,
    pred_folder: str,
    map_pretrained_counts: bool = False,
) -> pd.DataFrame:
    gt_files = [f for f in os.listdir(gt_folder) if f.lower().endswith(".csv")]
    pred_files = [f for f in os.listdir(pred_folder) if f.lower().endswith(".csv")]

    complete_rows = []
    for pred_file in pred_files:
        video_num = parse_video_num(pred_file)
        if not video_num:
            print(f"Skipping {pred_file}, cannot parse video number.")
            continue

        gt_file = None
        for gt in gt_files:
            if parse_video_num(gt) == video_num:
                gt_file = gt
                break
        if not gt_file:
            print(f"No GT file found for video {video_num}, skipping.")
            continue

        yolo_model = parse_model_name(pred_file)

        gt_results = load_gt_results_from_csv(os.path.join(gt_folder, gt_file))
        pred_results = load_yolo_results_from_csv(os.path.join(pred_folder, pred_file))

        if map_pretrained_counts:
            pred_results = _apply_class_map(pred_results, DEFAULT_PRETRAINED_COUNT_MAP)

        comparison = compare_counts(gt_results, pred_results)
        for res in comparison:
            complete_rows.append(
                {
                    "video_num": video_num,
                    "yolo_model": yolo_model,
                    **res,
                }
            )
    return pd.DataFrame(complete_rows)
