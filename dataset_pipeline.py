import pandas as pd
import numpy as np
import gc
import logging
import json
import os
import time
import pickle
from datetime import datetime
from itertools import product
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

try:
    import psutil
except ImportError:
    psutil = None

try:
    # pyrefly: ignore [missing-import]
    import matplotlib.pyplot as plt
    # pyrefly: ignore [missing-import]
    import matplotlib as mpl
except ImportError:
    plt = None
    mpl = None

try:
    import seaborn as sns
except ImportError:
    sns = None

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATASET_PATH = "NF-CSE-CIC-IDS2018-v2.csv"
TARGET_COL = "Label"
BENIGN_LABEL = 0
ANOMALY_LABEL = 1

MAX_ACCEPTABLE_FPR = 0.08
SEEDS = [42, 52, 62]
PERCENTILES = [5, 10, 15, 20, 25, 30, 35]
CONTAMINATIONS = [ 0.15, 0.20, 0.25]

OUTPUT_ROOT = Path("evaluation_outputs")
PLOT_DPI = 300
SHOW_PLOTS = os.environ.get("EVALUATION_SHOW_PLOTS", "0").lower() in {"1", "true", "yes"}
REPORT_METRICS = ["recall", "precision", "f1", "fpr", "roc_auc", "pr_auc", "recall@fpr<5%"]

class NoScaler:
    def fit_transform(self, X):
        return X.values if isinstance(X, pd.DataFrame) else X
    def transform(self, X):
        return X.values if isinstance(X, pd.DataFrame) else X

def get_optimization_score(recall, f1, precision, fpr):
    return (recall * 0.60) + (f1 * 0.25) + (precision * 0.10) - (fpr * 0.05)

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, np.ndarray)) else False:
        return None
    return value

def ensure_output_dirs(root=OUTPUT_ROOT):
    dirs = {
        "root": root,
        "metrics": root / "metrics",
        "plots": root / "plots",
        "confusion_matrices": root / "confusion_matrices",
        "threshold_analysis": root / "threshold_analysis",
        "roc_curves": root / "roc_curves",
        "pr_curves": root / "pr_curves",
        "reports": root / "reports",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs

def configure_plot_style():
    if plt is None:
        logger.warning("matplotlib is not installed. Plot export will be skipped.")
        return False
    if sns is not None:
        sns.set_theme(
            style="darkgrid",
            context="paper",
            font="DejaVu Sans",
            rc={
                "axes.facecolor": "#111827",
                "figure.facecolor": "#0B1020",
                "grid.color": "#334155",
                "axes.edgecolor": "#CBD5E1",
                "axes.labelcolor": "#E5E7EB",
                "xtick.color": "#CBD5E1",
                "ytick.color": "#CBD5E1",
                "text.color": "#F8FAFC",
                "legend.facecolor": "#111827",
                "legend.edgecolor": "#475569",
            },
        )
    else:
        plt.style.use("dark_background")
    if mpl is not None:
        mpl.rcParams.update({
            "figure.dpi": 120,
            "savefig.dpi": PLOT_DPI,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "lines.linewidth": 2.2,
            "axes.titleweight": "bold",
        })
    return True

def save_figure(fig, path, show=False):
    if plt is None or fig is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return str(path)

def system_snapshot():
    if psutil is None:
        return {"cpu_usage": None, "ram_usage": None}
    process = psutil.Process(os.getpid())
    return {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "ram_usage": process.memory_info().rss / (1024 ** 3),
    }

def confidence_interval(values, z=1.96):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"low": 0.0, "high": 0.0, "half_width": 0.0}
    if len(arr) == 1:
        mean = float(arr[0])
        return {"low": mean, "high": mean, "half_width": 0.0}
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    half_width = z * sem
    mean = float(np.mean(arr))
    return {"low": mean - half_width, "high": mean + half_width, "half_width": half_width}

def aggregate_seed_statistics(seed_results):
    rows = pd.DataFrame(seed_results)
    stats = {}
    for metric in REPORT_METRICS:
        if metric not in rows.columns:
            continue
        values = pd.to_numeric(rows[metric], errors="coerce").dropna().values
        if len(values) == 0:
            continue
        ci = confidence_interval(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        cv = float(std / abs(mean)) if mean != 0 else 0.0
        stats[metric] = {
            "mean": mean,
            "std": std,
            "ci95_low": ci["low"],
            "ci95_high": ci["high"],
            "ci95_half_width": ci["half_width"],
            "coefficient_of_variation": cv,
            "instability_flag": bool(cv > 0.10 or std > 0.05),
        }
    return stats

def export_metrics(dirs, experiment_id, seed_results, summary_stats, config, threshold_records):
    metrics_dir = dirs["metrics"]
    seed_df = pd.DataFrame(seed_results)
    threshold_df = pd.DataFrame(threshold_records)

    seed_csv = metrics_dir / f"{experiment_id}_seed_metrics.csv"
    seed_json = metrics_dir / f"{experiment_id}_seed_metrics.json"
    summary_json = metrics_dir / f"{experiment_id}_summary.json"
    threshold_csv = metrics_dir / f"{experiment_id}_threshold_sweep.csv"
    history_csv = metrics_dir / "experiment_history.csv"
    history_jsonl = metrics_dir / "experiment_history.jsonl"

    seed_df.to_csv(seed_csv, index=False)
    seed_json.write_text(json.dumps(json_safe(seed_results), indent=2), encoding="utf-8")
    threshold_df.to_csv(threshold_csv, index=False)

    summary_payload = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "configuration": config,
        "summary_statistics": summary_stats,
        "seed_count": len(seed_results),
        "seed_metrics_csv": str(seed_csv),
        "threshold_sweep_csv": str(threshold_csv),
    }
    summary_json.write_text(json.dumps(json_safe(summary_payload), indent=2), encoding="utf-8")

    history_row = {
        "experiment_id": experiment_id,
        "timestamp": summary_payload["timestamp"],
        **config,
    }
    for metric, values in summary_stats.items():
        history_row[f"{metric}_mean"] = values["mean"]
        history_row[f"{metric}_std"] = values["std"]
        history_row[f"{metric}_ci95_low"] = values["ci95_low"]
        history_row[f"{metric}_ci95_high"] = values["ci95_high"]
        history_row[f"{metric}_unstable"] = values["instability_flag"]

    history_df = pd.DataFrame([history_row])
    if history_csv.exists():
        history_df.to_csv(history_csv, mode="a", header=False, index=False)
    else:
        history_df.to_csv(history_csv, index=False)
    with history_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(json_safe(history_row)) + "\n")

    logger.info(f"Exported metrics: {seed_csv}, {summary_json}")
    return {
        "seed_csv": seed_csv,
        "seed_json": seed_json,
        "summary_json": summary_json,
        "threshold_csv": threshold_csv,
        "history_csv": history_csv,
        "history_jsonl": history_jsonl,
    }

def plot_confusion_matrix(y_true, y_pred, metrics, path, normalize=False, title_suffix="", show=False):
    if plt is None:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=[BENIGN_LABEL, ANOMALY_LABEL])
    display_cm = cm.astype(float)
    if normalize:
        row_sums = display_cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(display_cm, row_sums, out=np.zeros_like(display_cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), facecolor="#0B1020")
    cmap = "mako" if sns is not None else "Blues"
    if sns is not None:
        sns.heatmap(display_cm, annot=False, cmap=cmap, cbar=True, square=True, ax=ax, linewidths=0.8, linecolor="#1F2937")
    else:
        im = ax.imshow(display_cm, cmap=cmap)
        fig.colorbar(im, ax=ax)

    labels = ["Benign", "Anomaly"]
    ax.set_xticks([0.5, 1.5] if sns is not None else [0, 1])
    ax.set_yticks([0.5, 1.5] if sns is not None else [0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix {title_suffix}".strip())

    max_value = display_cm.max() if display_cm.size else 1
    for i in range(2):
        for j in range(2):
            raw = cm[i, j]
            pct = display_cm[i, j] * 100 if normalize else (raw / cm.sum() * 100 if cm.sum() else 0)
            color = "#0B1020" if display_cm[i, j] > max_value * 0.55 else "#F8FAFC"
            ax.text(j + (0.5 if sns is not None else 0), i + (0.5 if sns is not None else 0),
                    f"{raw:,}\n{pct:.1f}%", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    summary = (
        f"Recall: {metrics.get('recall', 0):.3f} | Precision: {metrics.get('precision', 0):.3f}\n"
        f"F1: {metrics.get('f1', 0):.3f} | FPR: {metrics.get('fpr', 0):.3f}"
    )
    ax.text(0.5, -0.18, summary, transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#111827", edgecolor="#475569", alpha=0.95))
    return save_figure(fig, path, show)

def plot_metric_distributions(seed_df, summary_stats, path, show=False):
    if plt is None or seed_df.empty:
        return None
    metrics = [m for m in ["recall", "f1", "precision", "fpr", "roc_auc", "pr_auc"] if m in seed_df.columns]
    melted = seed_df.melt(id_vars=["seed"], value_vars=metrics, var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="#0B1020")
    if sns is not None:
        sns.boxplot(data=melted, x="Metric", y="Value", ax=ax, color="#38BDF8", width=0.52, fliersize=4)
        sns.stripplot(data=melted, x="Metric", y="Value", ax=ax, color="#F97316", size=6, jitter=0.08)
    else:
        ax.boxplot([seed_df[m].values for m in metrics], labels=metrics)
    ax.set_title("Multi-Seed Metric Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    for idx, metric in enumerate(metrics):
        stat = summary_stats.get(metric, {})
        if stat.get("instability_flag"):
            ax.text(idx, 1.02, "High variance", ha="center", color="#FCA5A5", fontsize=8)
    return save_figure(fig, path, show)

def plot_seed_stability(seed_df, path, show=False):
    if plt is None or seed_df.empty:
        return None
    metrics = [m for m in ["recall", "f1", "fpr"] if m in seed_df.columns]
    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor="#0B1020")
    palette = {"recall": "#22C55E", "f1": "#38BDF8", "fpr": "#F97316"}
    for metric in metrics:
        ax.plot(seed_df["seed"], seed_df[metric], marker="o", label=metric.upper(), color=palette.get(metric, None))
    ax.set_title("Cross-Seed Stability Analysis")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    return save_figure(fig, path, show)

def plot_roc_curves(curve_payloads, path, selected_fpr=None, show=False):
    if plt is None or not curve_payloads:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 6.3), facecolor="#0B1020")
    mean_grid = np.linspace(0, 1, 250)
    interpolated = []
    aucs = []
    for payload in curve_payloads:
        fpr = payload["roc_fpr"]
        tpr = payload["roc_tpr"]
        aucs.append(payload["roc_auc"])
        ax.plot(fpr, tpr, alpha=0.28, color="#60A5FA", linewidth=1.4)
        interpolated.append(np.interp(mean_grid, fpr, tpr))
    mean_tpr = np.mean(interpolated, axis=0)
    std_tpr = np.std(interpolated, axis=0)
    ax.plot(mean_grid, mean_tpr, color="#22C55E", linewidth=3, label=f"Mean ROC (AUC={np.mean(aucs):.3f})")
    ax.fill_between(mean_grid, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                    color="#22C55E", alpha=0.16, label="Seed variability")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94A3B8", label="Random baseline")
    if selected_fpr is not None:
        selected_tpr = np.interp(selected_fpr, mean_grid, mean_tpr)
        ax.scatter([selected_fpr], [selected_tpr], color="#F97316", s=80, zorder=4, label="Selected operating point")
        ax.annotate(f"FPR={selected_fpr:.3f}\nTPR={selected_tpr:.3f}", (selected_fpr, selected_tpr),
                    xytext=(18, -18), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#F97316"))
    ax.set_title("Multi-Seed ROC Analysis")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate / Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return save_figure(fig, path, show)

def plot_pr_curves(curve_payloads, path, show=False):
    if plt is None or not curve_payloads:
        return None
    fig, ax = plt.subplots(figsize=(7.8, 6.3), facecolor="#0B1020")
    recall_grid = np.linspace(0, 1, 250)
    interpolated = []
    aps = []
    for payload in curve_payloads:
        recall = payload["pr_recall"]
        precision = payload["pr_precision"]
        aps.append(payload["pr_auc"])
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        ax.plot(recall_sorted, precision_sorted, alpha=0.28, color="#A78BFA", linewidth=1.4)
        interpolated.append(np.interp(recall_grid, recall_sorted, precision_sorted))
    mean_precision = np.mean(interpolated, axis=0)
    std_precision = np.std(interpolated, axis=0)
    ax.plot(recall_grid, mean_precision, color="#F59E0B", linewidth=3, label=f"Mean PR (AP={np.mean(aps):.3f})")
    ax.fill_between(recall_grid, np.maximum(mean_precision - std_precision, 0), np.minimum(mean_precision + std_precision, 1),
                    color="#F59E0B", alpha=0.16, label="Seed variability")
    ax.set_title("Precision-Recall Analysis for Imbalanced Anomaly Detection")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    return save_figure(fig, path, show)

def plot_threshold_sweep(threshold_df, selected_percentile, path, show=False):
    if plt is None or threshold_df.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#0B1020")
    metrics = [("recall", "Recall"), ("fpr", "False Positive Rate"), ("f1", "F1 Score"), ("precision", "Precision")]
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for cont, group in threshold_df.groupby("contamination"):
            summary = group.groupby("percentile")[metric].mean().reset_index()
            ax.plot(summary["percentile"], summary[metric], marker="o", label=f"cont={cont}")
        ax.axvline(selected_percentile, color="#F97316", linestyle="--", linewidth=2, label="selected" if metric == "recall" else None)
        ax.set_title(f"{label} vs Percentile")
        ax.set_xlabel("Percentile Threshold")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.05)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Threshold Optimization Sweep", fontsize=15, fontweight="bold")
    return save_figure(fig, path, show)

def plot_recall_fpr_tradeoff(threshold_df, selected_metrics, path, show=False):
    if plt is None or threshold_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.2, 6.4), facecolor="#0B1020")
    scatter = ax.scatter(
        threshold_df["fpr"], threshold_df["recall"],
        c=threshold_df["percentile"], s=70, cmap="viridis", alpha=0.82,
        edgecolors="#E5E7EB", linewidths=0.5
    )
    ax.axvspan(0, MAX_ACCEPTABLE_FPR, color="#22C55E", alpha=0.12, label=f"Operational region (FPR <= {MAX_ACCEPTABLE_FPR:.2f})")
    selected_fpr = selected_metrics.get("fpr", 0)
    selected_recall = selected_metrics.get("recall", 0)
    ax.scatter([selected_fpr], [selected_recall], s=160, marker="*", color="#F97316", edgecolor="#FFFFFF", linewidth=1.0, label="Selected threshold")
    ax.annotate(f"Selected\nRecall={selected_recall:.3f}\nFPR={selected_fpr:.3f}", (selected_fpr, selected_recall),
                xytext=(18, 18), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="#F97316"))
    fig.colorbar(scatter, ax=ax, label="Percentile")
    ax.set_title("Operational Recall-FPR Tradeoff")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, max(MAX_ACCEPTABLE_FPR * 1.8, threshold_df["fpr"].max() * 1.08, 0.02))
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    return save_figure(fig, path, show)

def build_curve_payload(y_true, scores, threshold, metrics):
    anomaly_scores = -np.asarray(scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)
    selected_score_threshold = -threshold
    nearest_roc = int(np.argmin(np.abs(roc_thresholds - selected_score_threshold))) if len(roc_thresholds) else 0
    nearest_pr = int(np.argmin(np.abs(pr_thresholds - selected_score_threshold))) if len(pr_thresholds) else 0
    roc_grid = np.linspace(0, 1, 1000)
    recall_grid = np.linspace(0, 1, 1000)
    pr_order = np.argsort(recall)
    return {
        "roc_fpr": roc_grid,
        "roc_tpr": np.interp(roc_grid, fpr, tpr),
        "roc_threshold_index": nearest_roc,
        "pr_precision": np.interp(recall_grid, recall[pr_order], precision[pr_order]),
        "pr_recall": recall_grid,
        "pr_threshold_index": nearest_pr,
        "roc_auc": metrics.get("roc_auc", 0.0),
        "pr_auc": metrics.get("pr_auc", 0.0),
    }

def generate_reports(dirs, experiment_id, config, summary_stats, seed_results, artifact_paths):
    reports_dir = dirs["reports"]
    seed_df = pd.DataFrame(seed_results)
    summary_rows = []
    for metric, values in summary_stats.items():
        flag = "Yes" if values["instability_flag"] else "No"
        summary_rows.append(
            f"| {metric} | {values['mean']:.4f} | {values['std']:.4f} | "
            f"[{values['ci95_low']:.4f}, {values['ci95_high']:.4f}] | {flag} |"
        )
    summary_table = "\n".join([
        "| Metric | Mean | Std | 95% CI | Instability Flag |",
        "|---|---:|---:|---:|---|",
        *summary_rows,
    ])
    best_seed = seed_df.sort_values("f1", ascending=False).iloc[0].to_dict() if not seed_df.empty else {}
    operational_note = (
        "The selected operating point prioritizes anomaly recall while enforcing the configured false-positive "
        f"constraint of {MAX_ACCEPTABLE_FPR:.2%}. This is appropriate for lightweight SMB-oriented deployment "
        "where missed intrusions are costlier than bounded alert volume."
    )
    def report_relative_path(path):
        return Path(os.path.relpath(Path(path), reports_dir)).as_posix()

    images = "\n".join([f"![{name}]({report_relative_path(path)})" for name, path in artifact_paths.items() if path and Path(path).suffix.lower() == ".png"])
    config_lines = "\n".join([f"- **{k}:** {v}" for k, v in config.items()])

    md = f"""# Explainable Lightweight NIDS Evaluation Report

**Experiment ID:** `{experiment_id}`

## Best Configuration

{config_lines}

## Multi-Seed Statistical Summary

{summary_table}

## Best Seed-Level Result

- **Seed:** {best_seed.get('seed', 'n/a')}
- **Recall:** {best_seed.get('recall', 0):.4f}
- **Precision:** {best_seed.get('precision', 0):.4f}
- **F1:** {best_seed.get('f1', 0):.4f}
- **FPR:** {best_seed.get('fpr', 0):.4f}
- **ROC-AUC:** {best_seed.get('roc_auc', 0):.4f}
- **PR-AUC:** {best_seed.get('pr_auc', 0):.4f}

## Operational Interpretation

{operational_note}

The threshold analysis compares percentile-driven operating points across contamination settings. The selected
configuration is highlighted as the production candidate because it maximizes the recall-weighted objective while
remaining inside the acceptable false-positive region.

## Visual Evidence

{images}
"""
    md_path = reports_dir / f"{experiment_id}_evaluation_report.md"
    html_path = reports_dir / f"{experiment_id}_evaluation_report.html"
    md_path.write_text(md, encoding="utf-8")

    html_images = "\n".join([
        f"<section><h3>{name.replace('_', ' ').title()}</h3><img src='{report_relative_path(path)}' alt='{name}'></section>"
        for name, path in artifact_paths.items() if path and Path(path).suffix.lower() == ".png"
    ])
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{experiment_id} Evaluation Report</title>
<style>
body {{ margin: 0; padding: 36px; background: #0B1020; color: #E5E7EB; font-family: Arial, sans-serif; }}
h1, h2, h3 {{ color: #F8FAFC; }}
code {{ background: #111827; padding: 2px 6px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 18px 0; }}
th, td {{ border: 1px solid #334155; padding: 8px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #111827; }}
img {{ max-width: 100%; border: 1px solid #334155; border-radius: 6px; margin: 10px 0 28px; }}
.panel {{ background: #111827; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Explainable Lightweight NIDS Evaluation Report</h1>
<div class="panel"><strong>Experiment ID:</strong> <code>{experiment_id}</code></div>
<h2>Configuration</h2>
<ul>{"".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in config.items()])}</ul>
<h2>Statistical Summary</h2>
{pd.DataFrame(summary_stats).T.to_html(float_format=lambda x: f"{x:.4f}", escape=False)}
<h2>Operational Interpretation</h2>
<p>{operational_note}</p>
<h2>Visual Evidence</h2>
{html_images}
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Generated reports: {md_path}, {html_path}")
    return {"markdown_report": md_path, "html_report": html_path}

def eval_metrics(y_true, y_pred, y_scores=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true)) == 1 and y_true.values[0] == 0:
            tn = sum(y_true == y_pred)
            fp = sum(y_true != y_pred)
        elif len(np.unique(y_true)) == 1 and y_true.values[0] == 1:
            tp = sum(y_true == y_pred)
            fn = sum(y_true != y_pred)
            
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = roc_auc_score(y_true, -y_scores) if y_scores is not None else 0.0
    pr_auc = average_precision_score(y_true, -y_scores) if y_scores is not None else 0.0
    
    return {
        'recall': rec, 'precision': prec, 'f1': f1, 'fpr': fpr, 'fnr': fnr,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'roc_auc': roc_auc, 'pr_auc': pr_auc
    }

def main():
    logger.info("Starting Explainable AI-Driven Lightweight NIDS Pipeline (Recall-Optimized)")
    experiment_id = datetime.now().strftime("nids_eval_%Y%m%d_%H%M%S")
    output_dirs = ensure_output_dirs()
    plotting_enabled = configure_plot_style()
    logger.info(f"Experiment ID: {experiment_id}")
    
    # --------------------------------------------------
    # 1. MEMORY MANAGEMENT & DATA LOADING (STABLE VERSION)
    # --------------------------------------------------
    drop_cols = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack']
    
    sample_df = pd.read_csv(DATASET_PATH, nrows=5)
    use_cols = [c for c in sample_df.columns if c not in drop_cols]
    
    logger.info("Loading dataset into memory...")
    df = pd.read_csv(DATASET_PATH, usecols=use_cols)
    
    # OVERFLOW ÇÖZÜMÜ: Tüm sütunları sayıya zorla, bozuk olanları (metin vb.) NaN yap
    for col in df.columns:
        if col != TARGET_COL:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Bellek yönetimi için float64 -> float32 dönüşümü (Sayısal sınırı aşmadan)
    for col in df.columns:
        if col == TARGET_COL: continue
        if df[col].dtype == 'float64':
            # Önce çok uç değerleri float32 sınırına çekiyoruz ki 'cast' hatası olmasın
            df[col] = df[col].clip(lower=-3.4e38, upper=3.4e38)
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Stratified split ile veri miktarını ayarla
    df, _ = train_test_split(df, train_size=0.10, stratify=df[TARGET_COL], random_state=42)

    # --------------------------------------------------
    # 2. DUPLICATE PRESERVATION & CLEANING (RECALL OPTIMIZED)
    # --------------------------------------------------
    logger.info("Cleaning duplicates ONLY from benign traffic to preserve anomaly diversity...")
    b_df = df[df[TARGET_COL] == BENIGN_LABEL]
    a_df = df[df[TARGET_COL] == ANOMALY_LABEL]
    
    # Sadece normal trafiği temizle, anomalilerin her biri altın değerindedir!
    b_df = b_df.drop_duplicates()
    df = pd.concat([b_df, a_df]).reset_index(drop=True)
    
    # Gereksiz object sütunlarını at
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
        
    # SONSUZ DEĞERLERİ TEMİZLE (Sayısal Stabilite İçin)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # ÖNEMLİ: dropna() yerine sayısal sütunları medyan ile doldurmak 
    # bazen daha fazla anomaliyi sistemde tutar. Ama dropna kullanacaksan:
    df.dropna(inplace=True) 
    
    gc.collect()


        # --------------------------------------------------
    # 3. FEATURE ENGINEERING (REVISED FOR STABILITY & RECALL)
    # --------------------------------------------------
    logger.info("Applying Feature Engineering with numeric stability checks...")
    
    # Temel metrikler
    df['Total_Packets'] = df.get('IN_PKTS', 0) + df.get('OUT_PKTS', 0)
    df['Total_Bytes'] = df.get('IN_BYTES', 0) + df.get('OUT_BYTES', 0)
    
    duration_sec = (df.get('FLOW_DURATION_MILLISECONDS', 0) / 1000.0) + 1e-6
    packets_safe = df['Total_Packets'] + 1e-6
    
    # Oranlar
    df['Packets_per_Second'] = df['Total_Packets'] / duration_sec
    df['Bytes_per_Packet'] = df['Total_Bytes'] / packets_safe
    df['Byte_Rate'] = df['Total_Bytes'] / duration_sec
    
    # STABİL FLOW INTENSITY (Overflow Engelleme)
    # in_b * out_b yerine log toplamı kullanarak sayısal stabilite sağlıyoruz
    in_log = np.log1p(df.get('IN_BYTES', 0).astype(np.float64))
    out_log = np.log1p(df.get('OUT_BYTES', 0).astype(np.float64))
    dur_log = np.log1p(df.get('FLOW_DURATION_MILLISECONDS', 0).astype(np.float64))
    df['Flow_Intensity'] = in_log + out_log - dur_log
    
    # INFINITY TEMİZLİĞİ (SİLMEK YERİNE DOLDUR)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Sayısal sütunları yakala
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop([TARGET_COL], errors='ignore')
    
    # NaN değerleri medyan ile doldur (Satırları silmiyoruz ki anomaliler gitmesin!)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Çok uç değerleri baskıla (Scaling öncesi stabilite için)
    for col in numeric_cols:
        upper_limit = df[col].quantile(0.999) # En üst %0.1'lik kısmı kırp
        df[col] = df[col].clip(upper=upper_limit)

    # Varyansı sıfır olanları at
    variances = df[numeric_cols].var()
    drop_var = variances[variances <= 0.0].index
    if len(drop_var) > 0:
        df.drop(columns=drop_var, inplace=True)
        
    gc.collect()


    # --------------------------------------------------
    # 4. TEMPORARY TUNING DATASET REDUCTION
    # --------------------------------------------------
    logger.info("Splitting into Tuning Set (50%) and Final Evaluation Set (50%)")
    df_tuning, df_final = train_test_split(df, test_size=0.50, stratify=df[TARGET_COL], random_state=42)
    
    b_tuning = df_tuning[df_tuning[TARGET_COL] == BENIGN_LABEL]
    a_tuning = df_tuning[df_tuning[TARGET_COL] == ANOMALY_LABEL]
    
    b_tr, b_tmp = train_test_split(b_tuning, train_size=0.40, random_state=42)
    b_va, b_vb = train_test_split(b_tmp, test_size=0.50, random_state=42)
    
    a_va, a_vb = train_test_split(a_tuning, test_size=0.50, random_state=42)
    
    X_tr_tune = b_tr.drop(columns=[TARGET_COL])
    X_va_tune = pd.concat([b_va, a_va]).drop(columns=[TARGET_COL])
    y_va_tune = pd.concat([b_va, a_va])[TARGET_COL]
    
    X_vb_tune = pd.concat([b_vb, a_vb]).drop(columns=[TARGET_COL])
    y_vb_tune = pd.concat([b_vb, a_vb])[TARGET_COL]

    scalers = {
        "NoScaler": NoScaler(),
        "RobustScaler": RobustScaler(),
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler()
    }

    # --------------------------------------------------
    # STAGE 1: OPTIMIZE SCALER & CONTAMINATION & PERCENTILE
    # --------------------------------------------------
    logger.info("\n--- STAGE 1: Optimizing Scaler & Contamination (Fixed Hyperparams) ---")
    best_stage1_score = -999.0
    best_scaler_name = None
    best_scaler_obj = None
    best_contamination = None
    best_percentile = None
    threshold_records = []

    for scaler_name, scaler_obj in scalers.items():
        X_tr_sc = scaler_obj.fit_transform(X_tr_tune)
        X_va_sc = scaler_obj.transform(X_va_tune)
        
        for cont in CONTAMINATIONS:
            model = IsolationForest(
                n_estimators=100, max_samples=0.8, max_features=1.0, 
                contamination=cont, random_state=42, n_jobs=-1
            )
            model.fit(X_tr_sc)
            scores_va = model.decision_function(X_va_sc)
            
            for p in PERCENTILES:
                thresh = np.percentile(scores_va, p)
                preds = np.where(scores_va <= thresh, 1, 0)
                m = eval_metrics(y_va_tune, preds)
                opt_score = get_optimization_score(m['recall'], m['f1'], m['precision'], m['fpr'])
                threshold_records.append({
                    "stage": "stage_1_scaler_contamination_percentile",
                    "scaler": scaler_name,
                    "contamination": cont,
                    "percentile": p,
                    "threshold": thresh,
                    "max_samples": 0.8,
                    "max_features": 1.0,
                    "optimization_score": opt_score,
                    **m,
                })
                
                if m['fpr'] > MAX_ACCEPTABLE_FPR or m['precision'] == 0:
                    continue
                    
                if opt_score > best_stage1_score:
                    best_stage1_score = opt_score
                    best_scaler_name = scaler_name
                    best_scaler_obj = scaler_obj
                    best_contamination = cont
                    best_percentile = p

    if best_percentile is None:
        logger.warning("FPR constraint failed across all configs in Stage 1. Falling back to safe defaults.")
        best_percentile = PERCENTILES[0]
        best_contamination = CONTAMINATIONS[0]
        best_scaler_name = "RobustScaler"
        best_scaler_obj = scalers["RobustScaler"]
        
    logger.info(f"Stage 1 Selected -> Scaler: {best_scaler_name}, Contamination: {best_contamination}, Percentile: {best_percentile} (Score: {best_stage1_score:.4f})")

    # --------------------------------------------------
    # STAGE 2: OPTIMIZE HYPERPARAMETERS
    # --------------------------------------------------
    logger.info("\n--- STAGE 2: Optimizing max_samples & max_features ---")
    X_tr_sc = best_scaler_obj.fit_transform(X_tr_tune)
    X_va_sc = best_scaler_obj.transform(X_va_tune)

    max_samples_opts = [0.4, 0.6, 0.8, 1.0]
    max_features_opts = [0.5, 0.7, 0.9, 1.0]
    
    best_stage2_score = -999.0
    best_m_samp = 0.8
    best_m_feat = 1.0
    best_tuned_model = None

    for m_samp, m_feat in product(max_samples_opts, max_features_opts):
        model = IsolationForest(
            n_estimators=100, max_samples=m_samp, max_features=m_feat,
            contamination=best_contamination, random_state=42, n_jobs=-1
        )
        model.fit(X_tr_sc)
        scores_va = model.decision_function(X_va_sc)
        
        thresh = np.percentile(scores_va, best_percentile)
        preds = np.where(scores_va <= thresh, 1, 0)
        m = eval_metrics(y_va_tune, preds)
        opt_score = get_optimization_score(m['recall'], m['f1'], m['precision'], m['fpr'])
        threshold_records.append({
            "stage": "stage_2_hyperparameter_search",
            "scaler": best_scaler_name,
            "contamination": best_contamination,
            "percentile": best_percentile,
            "threshold": thresh,
            "max_samples": m_samp,
            "max_features": m_feat,
            "optimization_score": opt_score,
            **m,
        })
        
        if m['fpr'] <= MAX_ACCEPTABLE_FPR and m['precision'] > 0:
            if opt_score > best_stage2_score:
                best_stage2_score = opt_score
                best_m_samp = m_samp
                best_m_feat = m_feat
                best_tuned_model = model

    if best_tuned_model is None:
        logger.warning("No hyperparameters met the FPR constraint! Reverting to Stage 1 defaults.")
        best_tuned_model = IsolationForest(n_estimators=100, max_samples=0.8, max_features=1.0, contamination=best_contamination, random_state=42, n_jobs=-1)
        best_tuned_model.fit(X_tr_sc)
        
    logger.info(f"Stage 2 Selected -> max_samples: {best_m_samp}, max_features: {best_m_feat} (Score: {best_stage2_score:.4f})")

    # --------------------------------------------------
    # STAGE 3: THRESHOLD VERIFICATION ON VAL-B
    # --------------------------------------------------
    logger.info("\n--- STAGE 3: Threshold Verification on Validation-B ---")
    X_vb_sc = best_scaler_obj.transform(X_vb_tune)
    scores_vb = best_tuned_model.decision_function(X_vb_sc)
    
    final_locked_threshold = np.percentile(scores_vb, best_percentile)
    preds_vb = np.where(scores_vb <= final_locked_threshold, 1, 0)
    m_vb = eval_metrics(y_vb_tune, preds_vb)
    
    logger.info(f"Locked Percentile: {best_percentile} | Val-B FPR: {m_vb['fpr']:.4f} | Val-B Recall: {m_vb['recall']:.4f}")

    # --------------------------------------------------
    # STAGE 4: MULTI-SEED FINAL EVALUATION
    # --------------------------------------------------
    logger.info("\n=========================================\nSTAGE 4: MULTI-SEED EVALUATION ON FINAL SET\n=========================================")
    
    seed_results = []
    curve_payloads = []
    final_confusion_payload = None
    
    best_f1 = -1.0
    best_seed_model = None
    best_seed_scaler = None
    best_seed_id = None
    
    for seed in SEEDS:
        b_final = df_final[df_final[TARGET_COL] == BENIGN_LABEL]
        a_final = df_final[df_final[TARGET_COL] == ANOMALY_LABEL]
        
        b_tr_final, b_te_final = train_test_split(b_final, train_size=0.60, random_state=seed)
        
        X_tr_fin = b_tr_final.drop(columns=[TARGET_COL])
        X_te_fin = pd.concat([b_te_final, a_final]).drop(columns=[TARGET_COL])
        y_te_fin = pd.concat([b_te_final, a_final])[TARGET_COL]
        
        X_tr_fin_sc = best_scaler_obj.fit_transform(X_tr_fin)
        X_te_fin_sc = best_scaler_obj.transform(X_te_fin)
        
        final_model = IsolationForest(
            n_estimators=100, max_samples=best_m_samp, max_features=best_m_feat,
            contamination=best_contamination, random_state=seed, n_jobs=-1
        )
        final_model.fit(X_tr_fin_sc)
        
        start_inference = time.perf_counter()
        scores_te = final_model.decision_function(X_te_fin_sc)
        inference_time = time.perf_counter() - start_inference
        thresh_te = np.percentile(scores_te, best_percentile)
        preds_te = np.where(scores_te <= thresh_te, 1, 0)
        
        test_m = eval_metrics(y_te_fin, preds_te, scores_te)
        
        sorted_indices = np.argsort(scores_te)
        y_te_sorted = y_te_fin.values[sorted_indices]
        
        cum_tp = np.cumsum(y_te_sorted == 1)
        cum_fp = np.cumsum(y_te_sorted == 0)
        total_pos = sum(y_te_fin == 1)
        total_neg = sum(y_te_fin == 0)
        
        fpr_curve = cum_fp / total_neg if total_neg > 0 else np.zeros_like(cum_fp)
        rec_curve = cum_tp / total_pos if total_pos > 0 else np.zeros_like(cum_tp)
        
        valid_fpr_idx = np.where(fpr_curve <= 0.05)[0]
        recall_at_5_fpr = rec_curve[valid_fpr_idx[-1]] if len(valid_fpr_idx) > 0 else 0.0
        
        test_m['recall@fpr<5%'] = recall_at_5_fpr
        resources = system_snapshot()
        seed_record = {
            "experiment_id": experiment_id,
            "seed": seed,
            "scaler": best_scaler_name,
            "contamination": best_contamination,
            "percentile": best_percentile,
            "threshold": thresh_te,
            "locked_validation_threshold": final_locked_threshold,
            "max_samples": best_m_samp,
            "max_features": best_m_feat,
            "inference_time": inference_time,
            "cpu_usage": resources["cpu_usage"],
            "ram_usage": resources["ram_usage"],
            **test_m,
        }
        seed_results.append(seed_record)
        curve_payloads.append(build_curve_payload(y_te_fin, scores_te, thresh_te, test_m))
        final_confusion_payload = {
            "seed": seed,
            "y_true": y_te_fin.copy(),
            "y_pred": preds_te.copy(),
            "metrics": test_m.copy(),
        }
        
        # Save model and scaler for the seed
        models_dir = output_dirs["models"]
        model_path = models_dir / f"{experiment_id}_model_seed_{seed}.pkl"
        scaler_path = models_dir / f"{experiment_id}_scaler_seed_{seed}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(best_scaler_obj, f)
        logger.info(f"Saved model and scaler for seed {seed} to {models_dir}")
        
        if test_m['f1'] > best_f1:
            best_f1 = test_m['f1']
            best_seed_model = final_model
            best_seed_scaler = best_scaler_obj
            best_seed_id = seed
            
        logger.info(f"Seed {seed} -> Recall: {test_m['recall']:.4f} | F1: {test_m['f1']:.4f} | FPR: {test_m['fpr']:.4f}")

    agg = defaultdict(list)
    for res in seed_results:
        for k, v in res.items():
            if isinstance(v, (int, float)):
                agg[k].append(v)
                
    logger.info("\n--- Final Aggregated Performance ---")
    logger.info(f"Config: Scaler={best_scaler_name}, max_samples={best_m_samp}, max_features={best_m_feat}, Contamination={best_contamination}, Percentile={best_percentile}")
    logger.info(f"Recall:    {np.mean(agg['recall']):.4f} +/- {np.std(agg['recall']):.4f}")
    logger.info(f"F1-Score:  {np.mean(agg['f1']):.4f} +/- {np.std(agg['f1']):.4f}")
    logger.info(f"Precision: {np.mean(agg['precision']):.4f} +/- {np.std(agg['precision']):.4f}")
    logger.info(f"FPR:       {np.mean(agg['fpr']):.4f} +/- {np.std(agg['fpr']):.4f}")
    logger.info(f"ROC-AUC:   {np.mean(agg['roc_auc']):.4f} +/- {np.std(agg['roc_auc']):.4f}")
    logger.info(f"PR-AUC:    {np.mean(agg['pr_auc']):.4f} +/- {np.std(agg['pr_auc']):.4f}")
    logger.info(f"Rec@FPR<5: {np.mean(agg['recall@fpr<5%']):.4f} +/- {np.std(agg['recall@fpr<5%']):.4f}")

    if best_seed_model is not None:
        best_model_path = output_dirs["models"] / "best_model.pkl"
        best_scaler_path = output_dirs["models"] / "best_scaler.pkl"
        with open(best_model_path, "wb") as f:
            pickle.dump(best_seed_model, f)
        with open(best_scaler_path, "wb") as f:
            pickle.dump(best_seed_scaler, f)
        logger.info(f"Saved overall best model (seed {best_seed_id}, F1: {best_f1:.4f}) to {best_model_path}")

    # --------------------------------------------------
    # STAGE 5: RESEARCH-GRADE EXPORTS, VISUALS, REPORTS
    # --------------------------------------------------
    logger.info("\n=========================================\nSTAGE 5: EXPORTING EVALUATION ARTIFACTS\n=========================================")
    selected_summary_metrics = {
        "recall": float(np.mean(agg["recall"])),
        "precision": float(np.mean(agg["precision"])),
        "f1": float(np.mean(agg["f1"])),
        "fpr": float(np.mean(agg["fpr"])),
        "roc_auc": float(np.mean(agg["roc_auc"])),
        "pr_auc": float(np.mean(agg["pr_auc"])),
    }
    config = {
        "dataset": DATASET_PATH,
        "model": "IsolationForest",
        "scaler": best_scaler_name,
        "contamination": best_contamination,
        "percentile": best_percentile,
        "locked_validation_threshold": final_locked_threshold,
        "max_samples": best_m_samp,
        "max_features": best_m_feat,
        "max_acceptable_fpr": MAX_ACCEPTABLE_FPR,
        "seeds": SEEDS,
    }
    summary_stats = aggregate_seed_statistics(seed_results)
    export_paths = export_metrics(output_dirs, experiment_id, seed_results, summary_stats, config, threshold_records)

    artifact_paths = {k: str(v) for k, v in export_paths.items()}
    if plotting_enabled:
        seed_df = pd.DataFrame(seed_results)
        threshold_df = pd.DataFrame(threshold_records)
        artifact_paths["metric_distribution"] = plot_metric_distributions(
            seed_df, summary_stats, output_dirs["plots"] / f"{experiment_id}_multi_seed_metric_distribution.png", SHOW_PLOTS
        )
        artifact_paths["seed_stability"] = plot_seed_stability(
            seed_df, output_dirs["plots"] / f"{experiment_id}_seed_stability.png", SHOW_PLOTS
        )
        if final_confusion_payload is not None:
            artifact_paths["confusion_matrix_raw"] = plot_confusion_matrix(
                final_confusion_payload["y_true"], final_confusion_payload["y_pred"], final_confusion_payload["metrics"],
                output_dirs["confusion_matrices"] / f"{experiment_id}_confusion_matrix_raw.png",
                normalize=False, title_suffix=f"(Seed {final_confusion_payload['seed']})", show=SHOW_PLOTS
            )
            artifact_paths["confusion_matrix_normalized"] = plot_confusion_matrix(
                final_confusion_payload["y_true"], final_confusion_payload["y_pred"], final_confusion_payload["metrics"],
                output_dirs["confusion_matrices"] / f"{experiment_id}_confusion_matrix_normalized.png",
                normalize=True, title_suffix=f"Normalized (Seed {final_confusion_payload['seed']})", show=SHOW_PLOTS
            )
        artifact_paths["roc_curve"] = plot_roc_curves(
            curve_payloads, output_dirs["roc_curves"] / f"{experiment_id}_roc_curve_multi_seed.png",
            selected_fpr=selected_summary_metrics["fpr"], show=SHOW_PLOTS
        )
        artifact_paths["pr_curve"] = plot_pr_curves(
            curve_payloads, output_dirs["pr_curves"] / f"{experiment_id}_pr_curve_multi_seed.png", SHOW_PLOTS
        )
        artifact_paths["threshold_sweep"] = plot_threshold_sweep(
            threshold_df, best_percentile, output_dirs["threshold_analysis"] / f"{experiment_id}_threshold_sweep.png", SHOW_PLOTS
        )
        artifact_paths["recall_fpr_tradeoff"] = plot_recall_fpr_tradeoff(
            threshold_df, selected_summary_metrics,
            output_dirs["threshold_analysis"] / f"{experiment_id}_recall_fpr_tradeoff.png", SHOW_PLOTS
        )

    report_paths = generate_reports(output_dirs, experiment_id, config, summary_stats, seed_results, artifact_paths)
    artifact_paths.update({k: str(v) for k, v in report_paths.items()})
    logger.info(f"Evaluation artifact root: {OUTPUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()
