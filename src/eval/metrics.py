import pandas as pd

from src.eval.utils import EPS, KNOWN_CLASSES


def scores_micro_macro(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["class_id"].isin(KNOWN_CLASSES)].copy()
    d["abs_err"] = d["in_diff"].abs() + d["out_diff"].abs()
    d["sum_tot"] = (d["gt_in"] + d["gt_out"]) + (d["pred_in"] + d["pred_out"])
    d["e_class"] = d["abs_err"] / (d["sum_tot"] + EPS)

    g = d.groupby(["video_num", "yolo_model"], as_index=False).agg(
        err=("abs_err", "sum"),
        denom=("sum_tot", "sum"),
        E_macro=("e_class", "mean"),
    )
    g["E_micro"] = g["err"] / (g["denom"] + EPS)
    g["Score_micro"] = 1 - g["E_micro"]
    g["Score_macro"] = 1 - g["E_macro"]
    return g.sort_values("Score_micro", ascending=False)


def tracking_miss_rate(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["class_id"].isin(KNOWN_CLASSES)].copy()
    d["abs_err"] = d["in_diff"].abs() + d["out_diff"].abs()
    d["gt_total"] = d["gt_in"] + d["gt_out"]
    d["pred_total"] = d["pred_in"] + d["pred_out"]
    denom = d["gt_total"].where(d["gt_total"] > 0, d["pred_total"])
    d["tmr_class"] = d["abs_err"] / (denom + EPS)

    g = d.groupby(["video_num", "yolo_model"], as_index=False).agg(
        tmr=("tmr_class", "mean"),
    )
    g["tracking_accuracy"] = 1 - g["tmr"]
    return g.sort_values("tmr", ascending=True)


def diff_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = ["in_diff", "out_diff"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    stats = df.groupby(["class_id"])[numeric_columns].agg(["mean", "std"])
    return stats
