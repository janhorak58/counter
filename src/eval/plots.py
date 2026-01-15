import matplotlib.pyplot as plt
import pandas as pd

from src.eval.utils import CLASS_NAME_MAP


def save_counts_plot(df: pd.DataFrame, out_path: str, title: str) -> None:
    classes = [CLASS_NAME_MAP.get(int(cid), str(cid)) for cid in df["class_id"]]
    gt_in = df["gt_in"].tolist()
    pred_in = df["pred_in"].tolist()
    gt_out = df["gt_out"].tolist()
    pred_out = df["pred_out"].tolist()

    x = list(range(len(classes)))
    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=14)

    plt.subplot(2, 1, 1)
    plt.bar(x, gt_in, width=0.4, label="GT In", align="center")
    plt.bar([i + 0.4 for i in x], pred_in, width=0.4, label="Pred In", align="center")
    plt.xticks([i + 0.2 for i in x], classes, rotation=30)
    plt.ylabel("In")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.bar(x, gt_out, width=0.4, label="GT Out", align="center")
    plt.bar([i + 0.4 for i in x], pred_out, width=0.4, label="Pred Out", align="center")
    plt.xticks([i + 0.2 for i in x], classes, rotation=30)
    plt.ylabel("Out")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_diff_plot(df: pd.DataFrame, out_path: str, title: str) -> None:
    classes = [CLASS_NAME_MAP.get(int(cid), str(cid)) for cid in df["class_id"]]
    in_diff = df["in_diff"].tolist()
    out_diff = df["out_diff"].tolist()

    x = list(range(len(classes)))
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(x, in_diff, width=0.4, label="In diff", align="center")
    plt.bar([i + 0.4 for i in x], out_diff, width=0.4, label="Out diff", align="center")
    plt.xticks([i + 0.2 for i in x], classes, rotation=30)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Diff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_scores_plot(df: pd.DataFrame, out_path: str, title: str) -> None:
    labels = [f"v{v}-{m}" for v, m in zip(df["video_num"], df["yolo_model"])]
    x = list(range(len(labels)))
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(x, df["Score_micro"].tolist(), width=0.4, label="Score micro")
    plt.bar([i + 0.4 for i in x], df["Score_macro"].tolist(), width=0.4, label="Score macro")
    plt.xticks([i + 0.2 for i in x], labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_tmr_plot(df: pd.DataFrame, out_path: str, title: str) -> None:
    labels = [f"v{v}-{m}" for v, m in zip(df["video_num"], df["yolo_model"])]
    x = list(range(len(labels)))
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(x, df["tmr"].tolist(), width=0.5, color="orange")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Tracking miss rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_diff_stats_plot(stats_df: pd.DataFrame, out_path: str, title: str) -> None:
    class_ids = stats_df.index.tolist()
    classes = [CLASS_NAME_MAP.get(int(cid), str(cid)) for cid in class_ids]
    in_mean = stats_df[("in_diff", "mean")].tolist()
    in_std = stats_df[("in_diff", "std")].tolist()
    out_mean = stats_df[("out_diff", "mean")].tolist()
    out_std = stats_df[("out_diff", "std")].tolist()

    x = list(range(len(classes)))
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.errorbar(x, in_mean, yerr=in_std, fmt="o", capsize=4, label="In diff")
    plt.errorbar([i + 0.2 for i in x], out_mean, yerr=out_std, fmt="o", capsize=4, label="Out diff")
    plt.xticks([i + 0.1 for i in x], classes, rotation=30)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Diff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(cm, labels, out_path: str, title: str, normalize: bool = False) -> None:
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / (row_sums + 1e-9)
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    tick_marks = list(range(len(labels)))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("GT")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

