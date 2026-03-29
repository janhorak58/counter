#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def fmt(x, digits=4):
    if x is None or x == "":
        return "-"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def line(sep="-", junction="+"):
        return junction + junction.join(sep * (w + 2) for w in widths) + junction

    print(line())
    print(
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    )
    print(line("="))
    for row in rows:
        print(
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
            + " |"
        )
    print(line())


def summarize_rf_detr(path: Path, per_class: bool):
    data = json.loads(path.read_text(encoding="utf-8"))
    class_map = data["class_map"]

    print(f"\nSouhrn modelu RF-DETR: {path}\n")

    overall_rows = []
    for split in ("valid", "test"):
        if split not in class_map:
            continue

        overall = next((r for r in class_map[split] if r["class"] == "all"), None)
        if overall is None:
            continue

        p = float(overall["precision"])
        r = float(overall["recall"])
        f1 = f1_score(p, r)

        overall_rows.append([
            split,
            fmt(overall["map@50:95"]),
            fmt(overall["map@50"]),
            fmt(p),
            fmt(r),
            fmt(f1),
        ])

    print("Hlavní metriky:")
    print_table(
        ["split", "mAP50-95", "mAP50", "precision", "recall", "F1"],
        overall_rows,
    )

    if per_class:
        for split in ("valid", "test"):
            if split not in class_map:
                continue

            rows = []
            for r in class_map[split]:
                if r["class"] == "all":
                    continue
                p = float(r["precision"])
                rec = float(r["recall"])
                rows.append([
                    r["class"],
                    fmt(r["map@50:95"]),
                    fmt(r["map@50"]),
                    fmt(p),
                    fmt(rec),
                    fmt(f1_score(p, rec)),
                ])

            if rows:
                print(f"\nPer-class metriky [{split}]:")
                print_table(
                    ["class", "mAP50-95", "mAP50", "precision", "recall", "F1"],
                    rows,
                )


def summarize_yolo(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("Prázdný CSV soubor.")

    numeric_rows = []
    for row in rows:
        parsed = {}
        for k, v in row.items():
            try:
                parsed[k] = float(v)
            except (ValueError, TypeError):
                parsed[k] = v
        numeric_rows.append(parsed)

    best = max(numeric_rows, key=lambda r: r["metrics/mAP50-95(B)"])
    last = numeric_rows[-1]

    def make_row(name, r):
        p = float(r["metrics/precision(B)"])
        rec = float(r["metrics/recall(B)"])
        return [
            name,
            int(r["epoch"]),
            fmt(r["metrics/mAP50-95(B)"]),
            fmt(r["metrics/mAP50(B)"]),
            fmt(p),
            fmt(rec),
            fmt(f1_score(p, rec)),
            fmt(float(r["time"]) / 3600.0, 2),
        ]

    print(f"\nSouhrn modelu YOLO: {path}\n")
    print("Hlavní metriky:")
    print_table(
        ["výběr", "epoch", "mAP50-95", "mAP50", "precision", "recall", "F1", "čas[h]"],
        [
            make_row("best", best),
            make_row("last", last),
        ],
    )


def detect_type(path: Path):
    if path.suffix.lower() == ".csv":
        return "yolo"

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if "class_map" in data:
            return "rf_detr"

    raise ValueError("Nepodporovaný formát. Očekávám YOLO results.csv nebo RF-DETR JSON.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Cesta k results.csv nebo RF-DETR JSON souboru")
    parser.add_argument("--per-class", action="store_true", help="U RF-DETR vypíše i metriky po třídách")
    args = parser.parse_args()

    path = Path(args.file).expanduser().resolve()
    file_type = detect_type(path)

    if file_type == "yolo":
        summarize_yolo(path)
    elif file_type == "rf_detr":
        summarize_rf_detr(path, per_class=args.per_class)


if __name__ == "__main__":
    main()