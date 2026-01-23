# Objects Counting Benchmark - Evaluation Guide

This folder contains timestamped evaluation runs with summary tables and charts.
Each run compares predicted in/out counts against GT counts for the same videos.

## Benchmark Setup
- Input: per-video CSV files with per-line counts of in/out per class.
- Pairing: prediction file `vidXX_*_results.csv` is matched with GT `data_XX.csv`.
- Aggregation: metrics are computed per line/class and aggregated per model and per video.

## How counts are formed
- For each line and class, we read `in_count` and `out_count`.
- Total count per line/class is `total = in + out`.
- Errors are computed on totals and on directions separately.

## Key Metrics (primary)
- MAE: core accuracy measure for total counts.
- MAPE In/Out: relative error for each direction, useful across videos with different volumes.
- Total Count Error: indicates systematic over/under-counting bias.

## Primary Benchmark: MAE
This is the main number to compare models.

| Metric | Meaning | Why it is primary |
| --- | --- | --- |
| MAE | "In avarage the model is off by X objects." | Standard v counting literature (ShanghaiTech, UCF-CC-50, Mall), snadno porovnatelne, nepenalizuje outliers tak extremne jako RMSE. |

If you want a second number, use RMSE to see if the model makes occasional large errors.

## Counting Accuracy Metrics (details)
- MAE (Mean Absolute Error)
  - Formula: mean of `|pred_total - gt_total|` over all line/class rows.
  - Meaning: average absolute counting mistake in raw counts.
  - Good when: you care about absolute objects counts.
- MAPE In/Out
  - Formula: mean of `|pred_in - gt_in| / gt_in` and `|pred_out - gt_out| / gt_out`.
  - Meaning: relative error per direction, normalized by GT volume.
  - Note: when `gt_in` or `gt_out` is 0, that direction's MAPE is undefined and left blank.
- RMSE (Root Mean Squared Error)
  - Formula: `sqrt(mean((pred_total - gt_total)^2))`.
  - Meaning: penalizes large mistakes more than MAE; highlights outliers.
- Total Count Error
  - Formula: `sum(pred_total) - sum(gt_total)` per model.
  - Meaning: signed bias; negative = under-counting, positive = over-counting.

## Per-class Metrics (details)
- Per-class MAE/MAPE/RMSE
  - Same formulas as above, computed only for a given class.
  - Meaning: shows which categories are systematically harder to count.
- Weighted MAE
  - Formula: `sum(abs_error * gt_total) / sum(gt_total)`.
  - Meaning: overall MAE weighted by how frequent the class/line is.
  - Why: avoids rare classes dominating the overall score.

## Robustness Metrics (details)
- Std deviation across videos
  - Computed on per-video MAE values.
  - Meaning: stability across different scenes.
- Worst-case error
  - Max per-video MAE for each model.
  - Meaning: how bad it gets in the hardest video.
- Percentiles (P50, P90, P95)
  - Computed on per-video MAE distribution.
  - Meaning: typical error (P50) vs tail error (P90/P95).

## How to interpret results
- Start with MAE and Total Count Error for overall accuracy and bias.
- Use MAPE In/Out to compare videos with different traffic volume.
- Check per-class metrics to diagnose class-specific failures.
- Use robustness metrics to judge stability and worst-case behavior.

## Outputs per run
- README.md: list of artifacts for the run.
- ANALYSIS.md: formatted tables and embedded plots for quick inspection.
- CSV/JSON: machine-readable metrics for further analysis.
