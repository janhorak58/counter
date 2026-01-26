# counter (refactor scaffold)

This repository contains a clean, OOP-first implementation of:
- `predict`: video -> `counts.json` (+ optional overlay mp4)
- `eval`: compare predicted counts vs GT and produce metrics + charts
- `benchmark`: run predict+eval across multiple models and videos
- Streamlit UI for comfortable runs

## Quick start (uv)

Create venv:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[ui,eval]"
# on a machine that runs inference:
uv pip install -e ".[predict]"
```

Run UI:

```bash
streamlit run src/counter/ui/app.py
```

Run CLI:

```bash
counter predict --config configs/predict.yaml
counter eval --config configs/eval.yaml --predict-run-dir runs/<id>/predict
counter benchmark --model-ids yolo11m_tuned rfdetr_large_tuned --videos-dir data/videos --gt-dir predictions/csv_gt
```
