# Counter - Simple Video Object Counter

Jednoduchý counter pro počítání objektů přecházejících přes čáru ve videu.

Podporuje:
- YOLO modely (tuned i pretrained)
- RF-DETR modely (tuned i pretrained)
- Debug mód pro zobrazení všech detekovaných tříd

## Instalace

```bash
pip install -e .
```

## Použití

### Základní použití

```bash
python -m counter predict --config configs/predict.yaml
```

### S debug módem (vidět všechny detekované třídy)

```bash
python -m counter predict --config configs/predict.yaml --debug
```

### Jedno video

```bash
python -m counter single --config configs/predict.yaml --video data/videos/vid16.mp4
```

### S GUI oknem

```bash
python -m counter predict --config configs/predict.yaml --show
```

## CLI argumenty

```
--config, -c    Config file (default: configs/predict.yaml)
--video, -v     Override video path
--model, -m     Override model path
--output, -o    Override output path
--device, -d    Device (cpu/cuda)
--show          Show preview window
--no-save       Don't save video
--debug         Show ALL detected classes (no filtering)
--quiet, -q     Less output
```

## Konfigurace

`configs/predict.yaml`:
```yaml
model_id: yolo11m_v11

videos_dir: data/videos
videos:
  - vid16.mp4
  - vid17.mp4

line:
  name: Line_1
  start: [846, 404]
  end: [1328, 456]

thresholds:
  conf: 0.35
  iou: 0.35

export:
  out_dir: runs/predict
  save_video: true

# Debug mode - show all detected classes
debug: false
```

`configs/models.yaml`:
```yaml
models:
  # Tuned models (0-3 classes)
  yolo11m_v11:
    backend: yolo
    variant: tuned
    weights: models/yolo/v1/yolo11m_v11/weights/best.pt

  # RF-DETR tuned
  rfdetr_large_v3:
    backend: rfdetr
    variant: tuned
    weights: models/rfdetr/v1/rfdetr_large_v3/checkpoint_best_total.pth

  # Pretrained COCO (person, bicycle, dog, skis -> 4,5,6,7)
  yolov8m_pretrained:
    backend: yolo
    variant: pretrained
    weights: yolov8m.pt
```

## Výstup

- `{model_id}/{video_name}.mp4` - Video s vykreslením
- `{model_id}/{video_name}.json` - Počty ve formátu JSON

## Progress logy

```
[18:30:15] [   0.0s] [INFO] PREDICTION START
[18:30:15] [   0.0s] [INFO] Model: models/yolo/v1/yolo11m_v11/weights/best.pt
[18:30:15] [   0.0s] [INFO] Video: data/videos/vid16.mp4
[18:30:16] [   1.2s] [INFO] Model loaded in 1.2s
[18:30:16] [   1.2s] [INFO] Video: 1920x1080 @ 24fps, 79493 frames
[18:30:16] [   1.2s] [INFO] Processing frames...
[==============----------------] 38000/79493 (47.8%) | FPS: 45 | IN: 12 OUT: 8 | ETA: 920s
```
