# Counter

Video object counting with YOLO or RF-DETR detection and tracking.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Optional RF-DETR support:
```bash
pip install rf-detr
```

## Modules
- `predict`: run counting on a video, save annotated video + CSV with counts.
- `eval`: compare predictions vs ground truth, compute metrics, optional detector validation, plots.
- `eval_model`: evaluate detection models on dataset_yolo (YOLO) or dataset_coco (RF-DETR).
- `train`: train a detector model from config.

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
- `parameters.model_type`: `yolo` or `rfdetr`.
- `parameters.track_iou_threshold`: IOU threshold for RF-DETR tracking.
- `parameters.track_max_lost`: max frames to keep a lost track (RF-DETR).
- `parameters.track_match_classes`: require same class for matching (RF-DETR).
- `parameters.rfdetr_box_format`: `xyxy`, `xywh`, or `cxcywh`.
- `parameters.rfdetr_box_normalized`: `auto`, `true`, or `false`.
- `parameters.use_interactive_lines`: `true` to pick lines in UI, `false` to use config lines.
- `parameters.num_lines`: number of lines to draw in UI when interactive selection is on.
- `parameters.lines`: list of line dicts with `start`, `end`, `name` when not interactive.
- `parameters.show_window`: `true` to show OpenCV window, `false` to disable display.
- `parameters.progress_every_n_frames`: print progress every N frames when display is off.

### eval
- `gt_folder`: ground truth CSV folder (default: `predictions/csv_gt`).
- `pred_folder`: predicted CSV folder (default: `predictions/csv`).
- `out_dir`: base output folder (timestamped run folders).
- `plots`: enable plot generation.
- `map_pretrained_counts`: map COCO count classes to custom classes.
- `run_model_eval`: run detector evaluation.
- `run_yolo_eval`: legacy alias for YOLO eval.
- `model_type`: `yolo` or `rfdetr`.
- `model_mode`: `custom` or `pretrained`.
- `yolo_mode`: legacy alias for model_mode.
- `model_path`: detector model for eval.
- `data_yaml`: dataset yaml for val.
- `device`: `cpu` or GPU id.
- `conf`: confidence threshold for val.
- `iou`: IOU threshold for val.
- `split`: dataset split (`val`).
- `rfdetr_box_format`: `xyxy`, `xywh`, or `cxcywh`.
- `rfdetr_box_normalized`: `auto`, `true`, or `false`.

### train
- `model_type`: `yolo` or `rfdetr`.
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

Model evaluation:
```bash
python -m src.eval_model --model-type yolo --model-path models/yolo/v2/yolov8n_v12/weights/best.pt
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

YOLO val (when `run_model_eval: true` and `model_type: yolo`):
- `map50`, `map50_95`, `precision`, `recall` in `yolo_val_metrics.csv`.
- Per-class metrics in `yolo_class_metrics.csv`.
- Confusion matrix plots in `yolo_plots/confusion_matrix.png` and `yolo_plots/confusion_matrix_norm.png`.

RF-DETR eval (when `run_model_eval: true` and `model_type: rfdetr`):
- Per-class metrics in `rfdetr_class_metrics.csv`.
- Summary metrics in `rfdetr_metrics.csv`.
- Confusion matrix plots in `rfdetr_plots/confusion_matrix.png` and `rfdetr_plots/confusion_matrix_norm.png`.
