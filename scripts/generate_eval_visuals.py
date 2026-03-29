#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "models" / "eval"
PLOTS_DIR = EVAL_DIR / "plots"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def short_label(model_rel: str) -> str:
    return model_rel.split("/")[-1]


def fmt(x: float | str | None, digits: int = 4) -> str:
    if x in ("", None):
        return "-"
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    return f"{x:.{digits}f}"


def color_scale(value: float, min_value: float, max_value: float) -> str:
    if max_value <= min_value:
        t = 0.5
    else:
        t = (value - min_value) / (max_value - min_value)
    t = max(0.0, min(1.0, t))
    # Green -> yellow -> red
    if t < 0.5:
        a = t / 0.5
        r = int(46 + (253 - 46) * a)
        g = int(204 + (224 - 204) * a)
        b = int(113 + (71 - 113) * a)
    else:
        a = (t - 0.5) / 0.5
        r = int(253 + (220 - 253) * a)
        g = int(224 + (38 - 224) * a)
        b = int(71 + (38 - 71) * a)
    return f"rgb({r},{g},{b})"


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        "<style>",
        "text { fill: #1f2937; font-family: Arial, sans-serif; }",
        ".title { font-size: 22px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #4b5563; }",
        ".axis { stroke: #9ca3af; stroke-width: 1; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; fill: #4b5563; }",
        ".value { font-size: 12px; font-weight: 700; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff" />',
        f'<text x="24" y="36" class="title">{escape(title)}</text>',
    ]


def write_svg(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float]],
    value_label: str,
    higher_is_better: bool = True,
    bar_color: str = "#2563eb",
) -> None:
    width = 1100
    left = 260
    right = 70
    top = 90
    row_h = 34
    bar_h = 22
    height = top + len(rows) * row_h + 70
    chart_w = width - left - right
    max_value = max(v for _, v in rows) if rows else 1.0
    lines = svg_header(width, height, title)
    lines.append(f'<text x="24" y="58" class="subtitle">{escape(subtitle)}</text>')
    for i in range(6):
        x = left + chart_w * i / 5
        lines.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{height - 40}" class="grid" />')
        tick_value = max_value * i / 5
        lines.append(f'<text x="{x:.1f}" y="{height - 16}" text-anchor="middle" class="small">{fmt(tick_value, 3)}</text>')
    lines.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{height - 40}" class="axis" />')
    for idx, (label, value) in enumerate(rows):
        y = top + idx * row_h
        bar_w = 0 if max_value == 0 else chart_w * value / max_value
        lines.append(f'<text x="{left - 12}" y="{y + 15}" text-anchor="end" class="label">{escape(label)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="4" fill="{bar_color}" />')
        lines.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 15}" class="value">{fmt(value, 4)}</text>')
    foot = f"{value_label}. {'Higher is better.' if higher_is_better else 'Lower is better.'}"
    lines.append(f'<text x="24" y="{height - 16}" class="small">{escape(foot)}</text>')
    lines.append("</svg>")
    write_svg(path, lines)


def grouped_bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    groups: list[tuple[str, float, float]],
    left_name: str,
    right_name: str,
) -> None:
    width = 980
    left = 180
    right = 60
    top = 90
    row_h = 58
    bar_h = 16
    height = top + len(groups) * row_h + 90
    chart_w = width - left - right
    max_value = max(max(a, b) for _, a, b in groups) if groups else 1.0
    lines = svg_header(width, height, title)
    lines.append(f'<text x="24" y="58" class="subtitle">{escape(subtitle)}</text>')
    for i in range(6):
        x = left + chart_w * i / 5
        lines.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{height - 50}" class="grid" />')
        lines.append(f'<text x="{x:.1f}" y="{height - 26}" text-anchor="middle" class="small">{fmt(max_value * i / 5, 3)}</text>')
    lines.append(f'<rect x="{width - 230}" y="24" width="14" height="14" fill="#2563eb" rx="3" />')
    lines.append(f'<text x="{width - 210}" y="36" class="small">{escape(left_name)}</text>')
    lines.append(f'<rect x="{width - 130}" y="24" width="14" height="14" fill="#f59e0b" rx="3" />')
    lines.append(f'<text x="{width - 110}" y="36" class="small">{escape(right_name)}</text>')
    for idx, (label, a, b) in enumerate(groups):
        y = top + idx * row_h
        aw = 0 if max_value == 0 else chart_w * a / max_value
        bw = 0 if max_value == 0 else chart_w * b / max_value
        lines.append(f'<text x="{left - 12}" y="{y + 20}" text-anchor="end" class="label">{escape(label)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{aw:.1f}" height="{bar_h}" rx="4" fill="#2563eb" />')
        lines.append(f'<rect x="{left}" y="{y + 22}" width="{bw:.1f}" height="{bar_h}" rx="4" fill="#f59e0b" />')
        lines.append(f'<text x="{left + aw + 8:.1f}" y="{y + 12}" class="small">{fmt(a, 4)}</text>')
        lines.append(f'<text x="{left + bw + 8:.1f}" y="{y + 34}" class="small">{fmt(b, 4)}</text>')
    lines.append(f'<text x="24" y="{height - 16}" class="small">mAP50-95 comparison across RF-DETR valid and test splits.</text>')
    lines.append("</svg>")
    write_svg(path, lines)


def heatmap(
    path: Path,
    title: str,
    subtitle: str,
    row_labels: list[str],
    col_labels: list[str],
    values: list[list[float]],
    low_label: str,
    high_label: str,
    reverse_scale: bool = False,
) -> None:
    cell_w = 110
    cell_h = 42
    left = 220
    top = 110
    width = left + len(col_labels) * cell_w + 80
    height = top + len(row_labels) * cell_h + 100
    flat = [v for row in values for v in row]
    min_value = min(flat) if flat else 0.0
    max_value = max(flat) if flat else 1.0
    lines = svg_header(width, height, title)
    lines.append(f'<text x="24" y="58" class="subtitle">{escape(subtitle)}</text>')
    for c, col in enumerate(col_labels):
        x = left + c * cell_w + cell_w / 2
        lines.append(f'<text x="{x:.1f}" y="{top - 18}" text-anchor="middle" class="label">{escape(col)}</text>')
    for r, row_label in enumerate(row_labels):
        y = top + r * cell_h + 26
        lines.append(f'<text x="{left - 12}" y="{y:.1f}" text-anchor="end" class="label">{escape(row_label)}</text>')
        for c, value in enumerate(values[r]):
            x = left + c * cell_w
            scaled = max_value - value if reverse_scale else value
            fill = color_scale(scaled, min_value if not reverse_scale else max_value - max_value, max_value if not reverse_scale else max_value - min_value)
            lines.append(f'<rect x="{x}" y="{top + r * cell_h}" width="{cell_w - 6}" height="{cell_h - 6}" rx="6" fill="{fill}" />')
            text_color = "#111827" if value < (min_value + max_value) / 2 else "#ffffff"
            lines.append(
                f'<text x="{x + (cell_w - 6) / 2:.1f}" y="{top + r * cell_h + 23:.1f}" text-anchor="middle" style="font-size:12px; font-weight:700; fill:{text_color}">{fmt(value, 4)}</text>'
            )
    legend_x = 24
    legend_y = height - 46
    for i in range(100):
        v = min_value + (max_value - min_value) * i / 99 if max_value > min_value else min_value
        scaled = max_value - v if reverse_scale else v
        fill = color_scale(scaled, min_value if not reverse_scale else 0.0, max_value if not reverse_scale else max_value - min_value)
        lines.append(f'<rect x="{legend_x + i * 3}" y="{legend_y}" width="3" height="14" fill="{fill}" />')
    lines.append(f'<text x="{legend_x}" y="{legend_y - 6}" class="small">{escape(low_label)}</text>')
    lines.append(f'<text x="{legend_x + 300}" y="{legend_y - 6}" text-anchor="end" class="small">{escape(high_label)}</text>')
    lines.append("</svg>")
    write_svg(path, lines)


def main() -> None:
    ensure_dir(PLOTS_DIR)

    detection_rows = read_csv(EVAL_DIR / "detection_summary.csv")
    yolo_rows = [r for r in detection_rows if r["backend"] == "yolo"]
    yolo_rows.sort(key=lambda r: float(r["primary"]), reverse=True)
    bar_chart(
        PLOTS_DIR / "yolo_detection_ranking.svg",
        "YOLO Detection Ranking",
        "Best validation mAP50-95 from stored results.csv files.",
        [(short_label(r["model_rel"]), float(r["primary"])) for r in yolo_rows],
        "Metric: mAP50-95",
        higher_is_better=True,
        bar_color="#2563eb",
    )

    rfdetr_rows = [r for r in detection_rows if r["backend"] == "rfdetr"]
    valid = {r["model_rel"]: float(r["primary"]) for r in rfdetr_rows if r["basis"] == "valid"}
    test = {r["model_rel"]: float(r["primary"]) for r in rfdetr_rows if r["basis"] == "test"}
    groups = [(short_label(model_rel), valid[model_rel], test[model_rel]) for model_rel in sorted(test, key=lambda k: test[k], reverse=True)]
    grouped_bar_chart(
        PLOTS_DIR / "rfdetr_valid_test_compare.svg",
        "RF-DETR Valid vs Test",
        "Split-level mAP50-95 from results.json.",
        groups,
        "valid",
        "test",
    )

    rfdetr_pc = read_csv(EVAL_DIR / "rfdetr_per_class_detection.csv")
    test_rows = [r for r in rfdetr_pc if r["split"] == "test"]
    model_order = []
    for row in sorted(test_rows, key=lambda r: (r["model_rel"], r["class_name"])):
        if row["model_rel"] not in model_order:
            model_order.append(row["model_rel"])
    class_order = ["tourist", "skier", "cyclist", "tourist_dog"]
    values = []
    for model_rel in model_order:
        by_class = {r["class_name"]: float(r["mAP50-95"]) for r in test_rows if r["model_rel"] == model_rel}
        values.append([by_class[c] for c in class_order])
    heatmap(
        PLOTS_DIR / "rfdetr_test_per_class_heatmap.svg",
        "RF-DETR Test Per-Class Detection",
        "mAP50-95 by model and class. Higher is better.",
        [short_label(m) for m in model_order],
        [c.upper() for c in class_order],
        values,
        low_label="lower",
        high_label="higher",
        reverse_scale=False,
    )

    count_overview = read_csv(EVAL_DIR / "count_benchmark_overview.csv")
    count_rows = [r for r in count_overview if r["empty_benchmark"] == "False" and r["score_total_micro_wape"]]
    count_rows.sort(key=lambda r: float(r["score_total_micro_wape"]))
    bar_chart(
        PLOTS_DIR / "count_micro_wape_ranking.svg",
        "Counting Benchmark Ranking",
        "Non-empty per-class count benchmarks ranked by micro WAPE.",
        [(short_label(r["model_rel"]), float(r["score_total_micro_wape"])) for r in count_rows],
        "Metric: micro WAPE",
        higher_is_better=False,
        bar_color="#dc2626",
    )

    count_pc = read_csv(EVAL_DIR / "count_benchmark_per_class.csv")
    count_pc = [r for r in count_pc if r["empty_benchmark"] == "False"]
    count_model_order = []
    for row in sorted(count_pc, key=lambda r: (r["model_rel"], r["direction"], int(r["class_id"]))):
        if row["model_rel"] not in count_model_order:
            count_model_order.append(row["model_rel"])
    count_cols = [
        ("IN", "TOURIST"),
        ("IN", "SKIER"),
        ("IN", "CYCLIST"),
        ("IN", "TOURIST_DOG"),
        ("OUT", "TOURIST"),
        ("OUT", "SKIER"),
        ("OUT", "CYCLIST"),
        ("OUT", "TOURIST_DOG"),
    ]
    count_values = []
    for model_rel in count_model_order:
        by_key = {
            (row["direction"], row["class_name"]): float(row["mae"])
            for row in count_pc
            if row["model_rel"] == model_rel
        }
        count_values.append([by_key[key] for key in count_cols])
    heatmap(
        PLOTS_DIR / "count_per_class_mae_heatmap.svg",
        "Per-Class Count Error Heatmap",
        "MAE per model, split by IN/OUT and class. Lower is better.",
        [short_label(m) for m in count_model_order],
        [f"{d}-{c.split('_')[0]}" for d, c in count_cols],
        count_values,
        low_label="lower error",
        high_label="higher error",
        reverse_scale=True,
    )

    visuals_md = [
        "# Evaluation Visuals",
        "",
        "These plots are generated from the CSV summaries in `models/eval`.",
        "",
        "## Files",
        "",
        f"- [YOLO detection ranking]({(PLOTS_DIR / 'yolo_detection_ranking.svg').name}): best validation mAP50-95 across all YOLO result files.",
        f"- [RF-DETR valid vs test]({(PLOTS_DIR / 'rfdetr_valid_test_compare.svg').name}): side-by-side split comparison for RF-DETR.",
        f"- [RF-DETR per-class heatmap]({(PLOTS_DIR / 'rfdetr_test_per_class_heatmap.svg').name}): test mAP50-95 by class.",
        f"- [Counting micro WAPE ranking]({(PLOTS_DIR / 'count_micro_wape_ranking.svg').name}): lower is better.",
        f"- [Per-class count MAE heatmap]({(PLOTS_DIR / 'count_per_class_mae_heatmap.svg').name}): lower is better; empty benchmarks excluded.",
        "",
        "## Reading Notes",
        "",
        "- The two all-zero latest benchmarks for `yolo11l_v11` and `yolo11m_v11` are excluded from the counting plots because their source benchmark has zero GT totals.",
        "- YOLO per-class detection is still unavailable directly from `results.csv`, so the per-class detection heatmap is RF-DETR-only.",
    ]
    (PLOTS_DIR / "VISUALS.md").write_text("\n".join(visuals_md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
