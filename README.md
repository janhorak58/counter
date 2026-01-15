# Counter

Video object counting with YOLO detection and tracking.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Modules
- `predict`: run counting on a video, save annotated video + CSV with counts.
- `eval`: compare predictions vs ground truth, compute metrics, optional YOLO validation, plots.
- `train`: train a YOLO model from config.

## Config
Edit `config.yaml` (see `config.example.yaml`).

### prediction
- `paths.video_folder`: input videos folder.
- `paths.output_folder`: annotated video output folder.
- `paths.results_folder`: CSV counts output folder.
- `paths.model_path`: YOLO weights to use.
- `paths.video_filename`: input video filename.
- `parameters.confidence_threshold`: detection confidence threshold.
- `parameters.iou_threshold`: IOU threshold.
- `parameters.grey_zone_size`: dead-zone width around the line.
- `parameters.device`: `cpu` or GPU id (`0`, `0,1`).
- `parameters.mode`: `custom` or `pretrained` (class mapping logic).

### eval
- `gt_folder`: ground truth CSV folder.
- `pred_folder`: predicted CSV folder.
- `out_dir`: base output folder (timestamped run folders).
- `plots`: enable plot generation.
- `map_pretrained_counts`: map COCO count classes to custom classes.
- `run_yolo_eval`: run YOLO validation.
- `yolo_mode`: `custom` or `pretrained`.
- `model_path`: YOLO model for eval.
- `data_yaml`: dataset yaml for val.
- `device`: `cpu` or GPU id.
- `conf`: confidence threshold for val.
- `iou`: IOU threshold for val.
- `split`: dataset split (`val`).

### train
- `model`: base model (e.g., `yolov8n.pt`).
- `data_yaml`: dataset yaml.
- `epochs`: number of epochs.
- `imgsz`: image size.
- `batch`: batch size.
- `workers`: dataloader workers.
- `device`: `cpu`, `gpu`, or GPU id (`0`).
- `patience`: early stopping patience.
- `project`: output folder for runs.
- `name`: run name.
- `plots`: save training plots.
- `save`: save checkpoints.
- `cos_lr`: cosine LR schedule.

## Run
Prediction:
```bash
python -m src.predict
```

Evaluation:
```bash
python -m src.eval
```

Training:
```bash
python -m src.train
```

## Metrics (eval)
From `complete_results.csv`:
- `gt_in`, `gt_out`: ground truth counts.
- `pred_in`, `pred_out`: predicted counts.
- `in_diff`, `out_diff`: prediction minus GT (positive = overcount).

From `scores_micro_macro.csv`:
- `E_micro`: total relative error across all classes.
- `Score_micro`: `1 - E_micro`.
- `E_macro`: average relative error per class (equal weight).
- `Score_macro`: `1 - E_macro`.

From `tracking_miss_rate.csv`:
- `tmr`: average relative error per class using `max(gt_total, pred_total)` as denom.
- `tracking_accuracy`: `1 - tmr`.

From `diff_stats.csv`:
- `mean`, `std` for `in_diff`/`out_diff` per class.

YOLO val (when `run_yolo_eval: true`):
- `map50`, `map50_95`, `precision`, `recall` in `yolo_val_metrics.csv`.
- Per-class metrics in `yolo_class_metrics.csv`.
- Confusion matrix plots in `yolo_plots/confusion_matrix.png` and `yolo_plots/confusion_matrix_norm.png`.
