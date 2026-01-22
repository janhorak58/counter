# People Counter - Scripts Documentation

Kompletni dokumentace skriptu pro trenovani, inferenci a evaluaci object detection modelu pro pocitani lidi.

## Quick Start

```bash
# 1. Aktivace prostredi
mamba activate ~/.conda/envs/counter-yolo/

# 2. Konverze datasetu (pro RF-DETR)
python scripts/convert_yolo_to_coco.py

# 3. Trenovani modelu
python scripts/train_yolo.py --model yolov8s --epochs 100
python scripts/train_rfdetr.py --model base --epochs 100

# 4. Batch predikce
python scripts/run_predictions.py --all

# 5. Evaluace
python scripts/evaluate.py --predictions predictions/csv --ground-truth data/ground_truth
```

---

## Obsah

| Skript | Popis |
|--------|-------|
| [train_yolo.py](#train_yolopy) | Trenovani YOLO modelu (YOLOv8, YOLO11) |
| [train_rfdetr.py](#train_rfdetrpy) | Trenovani RF-DETR modelu |
| [convert_yolo_to_coco.py](#convert_yolo_to_cocopy) | Konverze YOLO datasetu do COCO formatu |
| [run_predictions.py](#run_predictionspy) | Batch predikce na videich |
| [evaluate.py](#evaluatepy) | Benchmark a evaluace modelu |

---

## train_yolo.py

Trenovani YOLO modelu (YOLOv8, YOLO11) s plnou kontrolou hyperparametru.

### Pouziti

```bash
# Zakladni trenovani
python scripts/train_yolo.py --model yolov8s --epochs 100

# Velky model, maly batch
python scripts/train_yolo.py --model yolo11l --epochs 200 --batch 4

# Seznam dostupnych modelu
python scripts/train_yolo.py --list-models
```

### Dostupne modely

| Model | Size | Parameters | Use Case |
|-------|------|------------|----------|
| yolov8n | Nano | 3.2M | Edge devices, real-time |
| yolov8s | Small | 11.2M | Balanced speed/accuracy |
| yolov8m | Medium | 25.9M | General purpose |
| yolov8l | Large | 43.7M | High accuracy |
| yolov8x | XLarge | 68.2M | Maximum accuracy |
| yolo11n | Nano | 2.6M | Latest architecture, fast |
| yolo11s | Small | 9.4M | YOLO11 balanced |
| yolo11m | Medium | 20.1M | YOLO11 general |
| yolo11l | Large | 25.3M | YOLO11 accurate |
| yolo11x | XLarge | 56.9M | YOLO11 maximum |

### Parametry

```
Model:
  --model, -m          Model name (yolov8n, yolo11l, etc.)
  --project, -p        Output directory [models/yolo/v1]
  --name, -n           Run name [auto-generated]

Training:
  --epochs, -e         Number of epochs [100]
  --batch, -b          Batch size [16]
  --imgsz              Image size [640]
  --device, -d         GPU device [0]
  --workers, -w        Dataloader workers [8]
  --cache              Cache images in RAM

Optimizer:
  --optimizer          SGD, Adam, AdamW, auto [auto]
  --lr0                Initial learning rate [0.01]
  --lrf                Final LR factor [0.01]
  --patience           Early stopping patience [50]

Augmentation:
  --mosaic             Mosaic augmentation 0-1 [1.0]
  --mixup              Mixup augmentation 0-1 [0.0]
  --copy-paste         Copy-paste augmentation 0-1 [0.0]
  --no-augment         Disable all augmentation

Advanced:
  --resume, -r         Resume from checkpoint
  --freeze             Freeze first N layers
  --no-pretrained      Train from scratch
```

### Priklady

```bash
# Fine-tuning s nizkym LR
python scripts/train_yolo.py --model yolov8m --epochs 50 --lr0 0.001 --freeze 10

# Pokracovani preruseneho trenovani
python scripts/train_yolo.py --model yolov8s --resume models/yolo/v1/yolov8s_v1/weights/last.pt

# Silne augmentace
python scripts/train_yolo.py --model yolov8m --mosaic 1.0 --mixup 0.5 --copy-paste 0.3

# Trenovani bez augmentaci
python scripts/train_yolo.py --model yolov8n --epochs 200 --no-augment
```

### Vystup

```
models/yolo/v1/{model}_v{N}/
├── weights/
│   ├── best.pt              # Best weights (by mAP)
│   └── last.pt              # Last checkpoint
├── args.yaml                # Training arguments
├── results.csv              # Metrics per epoch
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
├── BoxP_curve.png           # Precision curve
├── BoxR_curve.png           # Recall curve
└── BoxPR_curve.png          # PR curve
```

---

## train_rfdetr.py

Trenovani RF-DETR modelu (Real-time Detection Transformer).

### Pouziti

```bash
# Zakladni trenovani
python scripts/train_rfdetr.py --model base --epochs 100

# S vyuzitim scratch (Metacentrum)
python scripts/train_rfdetr.py --model base --epochs 100 --scratch $SCRATCHDIR

# Seznam modelu
python scripts/train_rfdetr.py --list-models
```

### Dostupne modely

| Model | Parameters | Resolution | Use Case |
|-------|------------|------------|----------|
| small | 32M | 512 | Fast inference |
| base | 55M | 560 | Balanced |
| medium | 85M | 640 | Higher accuracy |
| large | 128M | 728 | Maximum accuracy |

### Parametry

```
Model:
  --model, -m          Model size: small, base, medium, large
  --project, -p        Output directory [models/rfdetr/v1]
  --name, -n           Run name [auto-generated]

Training:
  --epochs, -e         Number of epochs [100]
  --batch, -b          Batch size [8]
  --grad-accum         Gradient accumulation steps [4]
  --lr                 Learning rate [1e-4]
  --weight-decay       Weight decay [1e-4]
  --imgsz              Image size [640]
  --device, -d         GPU device [0]
  --workers, -w        Dataloader workers [4]

Dataset:
  --dataset            COCO dataset directory [data/coco_dataset]

I/O:
  --scratch, -s        Scratch directory [$SCRATCHDIR]
  --resume, -r         Resume from checkpoint
  --seed               Random seed [42]
```

### Scratch Directory

RF-DETR training benefits from fast I/O. Na Metacentru pouzij scratch:

```bash
# Automaticky z $SCRATCHDIR
python scripts/train_rfdetr.py --model base --epochs 100

# Explicitni cesta
python scripts/train_rfdetr.py --model base --scratch /scratch/user/job_123

# Bez scratch (pomalejsi)
python scripts/train_rfdetr.py --model base --scratch ""
```

Skript automaticky:
1. Zkopiruje dataset do scratch
2. Trenuje ze scratch
3. Po dokonceni zkopiruje vysledky zpet

### Vystup

```
models/rfdetr/v1/rfdetr_{model}_v{N}/
├── best.pt              # Best weights
├── last.pt              # Last checkpoint
├── config.yaml          # Training config
└── logs/                # Training logs
```

---

## convert_yolo_to_coco.py

Konverze YOLO datasetu do COCO formatu (potrebne pro RF-DETR).

### Pouziti

```bash
# Zakladni konverze (70/15/15 split)
python scripts/convert_yolo_to_coco.py

# Vlastni split ratio
python scripts/convert_yolo_to_coco.py --val-split 0.1 --test-split 0.1

# Dry-run
python scripts/convert_yolo_to_coco.py --dry-run
```

### Parametry

```
  --input, -i          Input YOLO dataset [data/yolo_dataset]
  --output, -o         Output COCO dataset [data/coco_dataset]
  --val-split          Validation split ratio [0.15]
  --test-split         Test split ratio [0.15]
  --seed               Random seed [42]
  --no-copy            Don't copy images (only annotations)
  --dry-run            Show what would be done
```

### Format comparison

**YOLO format (input):**
```
data/yolo_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

**COCO format (output):**
```
data/coco_dataset/
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg
```

---

## run_predictions.py

Batch predikce na videich pomoci natrenovanych modelu.

### Pouziti

```bash
# Vsechny modely
python scripts/run_predictions.py --all

# Pouze finetuned modely
python scripts/run_predictions.py --finetuned

# Pouze pretrained modely
python scripts/run_predictions.py --pretrained

# Dry-run
python scripts/run_predictions.py --dry-run
```

### Konfigurace

Uprav modely a videa primo ve skriptu:

```python
# Finetuned modely (mode: custom)
FINETUNED_MODELS = {
    "yolo11l_v11": "models/yolo/v1/yolo11l_v11/weights/best.pt",
    "yolov8n_v12": "models/yolo/v1/yolov8n_v12/weights/best.pt",
}

# Pretrained modely (mode: pretrained)
PRETRAINED_MODELS = {
    "yolo11l_pretrained": "yolo11l.pt",
    "yolov8l_pretrained": "yolov8l.pt",
}

# Videa
VIDEOS = ["vid16.mp4", "vid17.mp4", "vid18.mp4"]
```

### Parametry

```
  --all                Run all models (default)
  --finetuned          Run only finetuned models
  --pretrained         Run only pretrained models
  --models             Specific models to run
  --videos             Specific videos to process
  --dry-run            Show what would be executed
```

### Priklady

```bash
# Konkretni modely
python scripts/run_predictions.py --models yolo11l_v11 yolov8l_pretrained

# Konkretni videa
python scripts/run_predictions.py --videos vid18.mp4

# Kombinace
python scripts/run_predictions.py --models yolo11l_v11 --videos vid16.mp4 vid17.mp4
```

### Vystup

```
predictions/
├── csv/
│   ├── yolo11l_v11/
│   │   ├── vid16_results.csv
│   │   ├── vid17_results.csv
│   │   └── vid18_results.csv
│   └── yolov8l_pretrained/
│       └── ...
└── mp4/
    └── ... (same structure)
```

---

## evaluate.py

Benchmark a evaluace counting modelu.

### Pouziti

```bash
# Zakladni evaluace
python scripts/evaluate.py

# Vlastni cesty
python scripts/evaluate.py \
    --predictions predictions/csv \
    --ground-truth data/ground_truth \
    --output results/benchmark

# Pouze specificke modely
python scripts/evaluate.py --models yolo11l_v11 yolov8s_v13
```

### Metriky

| Metrika | Popis |
|---------|-------|
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| RMSE | Root Mean Square Error |
| Total Error | Error in total count |
| Direction Accuracy | In/Out direction accuracy |
| Per-class MAE | MAE for each class |

### Vystup

```
results/benchmark/
├── summary.csv              # Overall metrics per model
├── per_video.csv            # Metrics per video
├── per_class.csv            # Metrics per class
├── comparison_bar.png       # Model comparison chart
├── error_distribution.png   # Box plots
├── class_heatmap.png        # Model x Class heatmap
└── radar_chart.png          # Multi-metric radar
```

---

## Metacentrum PBS Scripts

### Training job

```bash
#!/bin/bash
#PBS -N yolo_train
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb:scratch_local=50gb
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -m ae

module load mambaforge
source activate ~/.conda/envs/counter-yolo/

cd $HOME/counter

# YOLO training
python scripts/train_yolo.py --model yolo11l --epochs 200 --batch 8 --cache

# Or RF-DETR training (uses scratch automatically)
# python scripts/train_rfdetr.py --model base --epochs 100
```

### Prediction job

```bash
#!/bin/bash
#PBS -N yolo_predict
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb:scratch_local=20gb
#PBS -l walltime=4:00:00
#PBS -q gpu
#PBS -m ae

module load mambaforge
source activate ~/.conda/envs/counter-yolo/

cd $HOME/counter
python scripts/run_predictions.py --all
```

### Submit jobs

```bash
qsub scripts/pbs_train_yolo.sh
qsub scripts/pbs_train_rfdetr.sh
qsub scripts/pbs_predict.sh
```

---

## Troubleshooting

### CUDA out of memory

```bash
# Sniz batch size
python scripts/train_yolo.py --model yolo11l --batch 4
python scripts/train_rfdetr.py --model base --batch 4 --grad-accum 8
```

### Model not found

```bash
# Zkontroluj pretrained vahy
ls *.pt *.pth

# Stahni chybejici
# YOLO: automaticky se stahne pri prvnim pouziti
# RF-DETR: manualne stahnout z roboflow
```

### Dataset not found

```bash
# YOLO
ls data/yolo_dataset/data.yaml

# COCO (pro RF-DETR)
ls data/coco_dataset/train/_annotations.coco.json

# Pokud COCO chybi, spust konverzi
python scripts/convert_yolo_to_coco.py
```

### Scratch directory issues

```bash
# Smaz stary dataset ve scratch
rm -rf $SCRATCHDIR/dataset

# Nebo spust bez scratch
python scripts/train_rfdetr.py --model base --scratch ""
```

### Permission denied

```bash
chmod +x scripts/*.py
chmod +x scripts/*.sh
```

---

## Project Structure

```
counter/
├── config.yaml                 # Main config for predictions
├── data/
│   ├── videos/                 # Input videos
│   │   ├── vid16.mp4
│   │   ├── vid17.mp4
│   │   └── vid18.mp4
│   ├── yolo_dataset/           # YOLO format dataset
│   │   ├── data.yaml
│   │   ├── images/
│   │   └── labels/
│   ├── coco_dataset/           # COCO format dataset (RF-DETR)
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── ground_truth/           # Ground truth for evaluation
│       ├── vid16.csv
│       ├── vid17.csv
│       └── vid18.csv
├── models/
│   ├── yolo/
│   │   └── v1/                 # Trained YOLO models
│   │       ├── yolov8s_v1/
│   │       └── yolo11l_v1/
│   └── rfdetr/
│       └── v1/                 # Trained RF-DETR models
│           └── rfdetr_base_v1/
├── predictions/
│   ├── csv/                    # Counting results
│   └── mp4/                    # Annotated videos
├── results/
│   └── benchmark/              # Evaluation results
├── scripts/
│   ├── README.md               # This file
│   ├── train_yolo.py
│   ├── train_rfdetr.py
│   ├── convert_yolo_to_coco.py
│   ├── run_predictions.py
│   └── evaluate.py
├── src/
│   └── predict.py              # Main prediction module
├── yolov8n.pt                  # Pretrained YOLO weights
├── yolov8s.pt
├── yolo11l.pt
├── rf-detr-base.pth            # Pretrained RF-DETR weights
├── rf-detr-small.pth
└── rf-detr-large.pth
```

---

## Workflow

### 1. Prepare dataset

```bash
# YOLO format uz mas
ls data/yolo_dataset/

# Pro RF-DETR preved do COCO
python scripts/convert_yolo_to_coco.py
```

### 2. Train models

```bash
# YOLO
python scripts/train_yolo.py --model yolov8s --epochs 100
python scripts/train_yolo.py --model yolo11l --epochs 200 --batch 8

# RF-DETR
python scripts/train_rfdetr.py --model base --epochs 100
python scripts/train_rfdetr.py --model large --epochs 150 --batch 4
```

### 3. Run predictions

```bash
# Update run_predictions.py with your trained models
# Then run:
python scripts/run_predictions.py --all
```

### 4. Evaluate

```bash
# Prepare ground truth files in data/ground_truth/
python scripts/evaluate.py --output results/benchmark
```

### 5. Compare results

```bash
# Check results
cat results/benchmark/summary.csv
open results/benchmark/comparison_bar.png
```
