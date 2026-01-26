from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt

Counts = Union[Dict[int, int], Sequence[int]]

# Windows forbidden + control chars
_INVALID_WIN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def _sanitize_path_component(s: str) -> str:
    s = _INVALID_WIN.sub("_", s)
    # Windows also hates trailing dots/spaces
    s = s.strip().strip(".")
    return s or "x"


def _sanitize_path(p: Path) -> Path:
    parts = list(p.parts)
    if parts and (parts[0].endswith(":") or parts[0].endswith(":\\") ):
        start = 1
    else:
        start = 0
    clean = parts[:start] + [_sanitize_path_component(x) for x in parts[start:]]
    return Path(*clean)


def bar_counts(
    path: str | Path,
    title: str,
    gt: Counts,
    pred: Counts,
    labels: Optional[List[str]] = None,
) -> None:
    """Bar chart GT vs Pred (per-class counts)."""
    p = _sanitize_path(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(gt, dict) or isinstance(pred, dict):
        gt_d = gt if isinstance(gt, dict) else {i: int(v) for i, v in enumerate(gt)}
        pr_d = pred if isinstance(pred, dict) else {i: int(v) for i, v in enumerate(pred)}
        keys = sorted(set(gt_d.keys()) | set(pr_d.keys()))
        gt_vals = [int(gt_d.get(k, 0)) for k in keys]
        pr_vals = [int(pr_d.get(k, 0)) for k in keys]
        x_labels = labels if (labels and len(labels) == len(keys)) else [str(k) for k in keys]
        x = list(range(len(keys)))
    else:
        gt_list = list(gt)
        pr_list = list(pred)
        n = max(len(gt_list), len(pr_list))
        gt_vals = [int(gt_list[i]) if i < len(gt_list) else 0 for i in range(n)]
        pr_vals = [int(pr_list[i]) if i < len(pr_list) else 0 for i in range(n)]
        x = list(range(n))
        x_labels = labels if (labels and len(labels) == n) else [str(i) for i in x]

    plt.figure()
    plt.bar([i - 0.2 for i in x], gt_vals, width=0.4, label="GT")
    plt.bar([i + 0.2 for i in x], pr_vals, width=0.4, label="Pred")
    plt.xticks(x, x_labels, rotation=0)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def bar_metric(
    path: str | Path,
    title: str,
    labels: List[str],
    values: List[float],
    ylabel: str = "",
) -> None:
    """Simple bar chart: label -> value. Sorted ascending."""
    p = _sanitize_path(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    pairs = sorted(zip(labels, values), key=lambda t: t[1])
    labels2 = [a for a, _ in pairs]
    values2 = [float(b) for _, b in pairs]

    w = max(6.0, min(18.0, 0.6 * len(labels2) + 3.0))
    plt.figure(figsize=(w, 4.2))
    plt.bar(list(range(len(labels2))), values2)
    plt.xticks(list(range(len(labels2))), labels2, rotation=45, ha="right")
    if ylabel:
        plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def scatter_xy(
    path: str | Path,
    title: str,
    x: List[float],
    y: List[float],
    labels: Optional[List[str]] = None,
    xlabel: str = "x",
    ylabel: str = "y",
) -> None:
    """Scatter plot, optionally with point labels."""
    p = _sanitize_path(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if labels:
        for xi, yi, lab in zip(x, y, labels):
            plt.text(float(xi), float(yi), str(lab), fontsize=8)

    plt.tight_layout()
    plt.savefig(p)
    plt.close()


def heatmap_matrix(
    path: str | Path,
    title: str,
    x_labels: List[str],
    y_labels: List[str],
    matrix: List[List[float]],
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = "{:.1f}",
) -> None:
    """Heatmap with numeric annotations. NaNs are shown as blank."""
    p = _sanitize_path(Path(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    m: List[List[float]] = []
    for row in matrix:
        m.append([0.0 if (isinstance(v, float) and math.isnan(v)) else float(v) for v in row])

    plt.figure(figsize=(max(6.0, 0.6 * len(x_labels) + 2.0), max(3.5, 0.5 * len(y_labels) + 2.0)))
    plt.imshow(m, aspect="auto")
    plt.title(title)
    plt.xticks(list(range(len(x_labels))), x_labels, rotation=45, ha="right")
    plt.yticks(list(range(len(y_labels))), y_labels)

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            v = matrix[i][j]
            if isinstance(v, float) and math.isnan(v):
                continue
            plt.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(p)
    plt.close()
