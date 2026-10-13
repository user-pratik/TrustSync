"""Plotting utilities with high-contrast dark theme for TrustSync reports."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Dark theme palette tokens
COLOR_BG = "#0A0E14"
COLOR_TEXT = "#E6EDF3"
COLOR_GRID = "#FFFFFF"
COLOR_ACCEPT = "#2DD4BF"
COLOR_REJECT = "#F87171"
COLOR_LISTEN = "#FBBF24"
COLOR_INFO = "#A78BFA"
COLOR_IDLE = "#64748B"


def _setup_dark_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Configure dark theme styling for figure and axes."""
    fig = ax.figure
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax.xaxis.label.set_color(COLOR_TEXT)
    ax.yaxis.label.set_color(COLOR_TEXT)
    ax.set_xlabel(xlabel, fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_title(title, color=COLOR_TEXT, fontsize=12, pad=10)

    for spine in ax.spines.values():
        spine.set_color(COLOR_IDLE)
        spine.set_alpha(0.3)

    ax.grid(True, color=COLOR_GRID, alpha=0.06, linestyle="-", linewidth=0.8)


def plot_roc(
    pos: list[float], neg: list[float], title: str, path: Path | str
) -> Path:
    """Generate ROC curve plot on dark theme."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, title, "False Positive Rate (FPR)", "True Positive Rate (TPR)")

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_IDLE, alpha=0.6, label="Chance")

    if pos and neg:
        all_vals = np.array(pos + neg)
        labels = np.array([1] * len(pos) + [0] * len(neg))
        thresholds = np.sort(np.unique(all_vals))[::-1]
        thresholds = np.concatenate(([np.inf], thresholds, [-np.inf]))

        tpr_list = []
        fpr_list = []
        n_pos = len(pos)
        n_neg = len(neg)

        for th in thresholds:
            preds = (all_vals >= th).astype(int)
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        ax.plot(fpr_list, tpr_list, color=COLOR_ACCEPT, linewidth=2, label="ROC")
        ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_IDLE, labelcolor=COLOR_TEXT)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path


def plot_box_by_scenario(
    values_by_scenario: dict[str, list[float]],
    ylabel: str,
    threshold: float | None,
    path: Path | str,
) -> Path:
    """Generate scenario box/scatter plot with optional threshold reference."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, f"{ylabel} by Scenario", "Scenario", ylabel)

    scenarios = list(values_by_scenario.keys())
    data = [values_by_scenario[sc] for sc in scenarios]

    if any(len(d) > 0 for d in data):
        # Filter for valid boxplot data
        plot_data = [d if len(d) > 0 else [0.0] for d in data]
        bp = ax.boxplot(
            plot_data,
            patch_artist=True,
            labels=scenarios,
            medianprops=dict(color=COLOR_ACCEPT, linewidth=1.5),
            whiskerprops=dict(color=COLOR_TEXT),
            capprops=dict(color=COLOR_TEXT),
            flierprops=dict(marker="o", markerfacecolor=COLOR_INFO, markeredgecolor=COLOR_INFO, markersize=4),
        )
        for box in bp["boxes"]:
            box.set_facecolor(COLOR_IDLE)
            box.set_alpha(0.3)
            box.set_edgecolor(COLOR_TEXT)

    if threshold is not None:
        ax.axhline(
            threshold,
            color=COLOR_LISTEN,
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({threshold:.2f})",
        )
        ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_IDLE, labelcolor=COLOR_TEXT)

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path


def plot_confusion(
    confusion: dict[str, dict[str, int]], path: Path | str
) -> Path:
    """Generate confusion matrix heatmap."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, "Confusion Matrix", "Predicted Decision", "Expected Label")

    rows = sorted(confusion.keys())
    # Collect all unique predicted decisions
    cols_set = set()
    for row in rows:
        cols_set.update(confusion[row].keys())
    cols = sorted(cols_set)

    if rows and cols:
        mat = np.zeros((len(rows), len(cols)), dtype=int)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                mat[i, j] = confusion[r].get(c, 0)

        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=25, ha="right")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows)

        # Annotate counts
        for i in range(len(rows)):
            for j in range(len(cols)):
                val = mat[i, j]
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color=COLOR_TEXT,
                    fontweight="bold",
                )

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path


def plot_ablation(rows: list[dict[str, Any]], path: Path | str) -> Path:
    """Generate grouped bar chart showing accuracy and FAR per ablation variant."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, "Ablation Study: Accuracy vs FAR", "Variant", "Rate [0..1]")

    if rows:
        names = [r["name"] for r in rows]
        accs = [r["accuracy"] for r in rows]
        fars = [r["far_all_reject"] for r in rows]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width / 2, accs, width, label="Accuracy", color=COLOR_ACCEPT, alpha=0.85)
        ax.bar(x + width / 2, fars, width, label="FAR (Spoof + Rejects)", color=COLOR_REJECT, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0.0, 1.05)
        ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_IDLE, labelcolor=COLOR_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path


def plot_lag_hist(lags: list[float], max_lag_s: float, path: Path | str) -> Path:
    """Generate audio-visual latency histogram."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, "Audio-Visual Latency Distribution", "Latency [seconds]", "Count")

    if lags:
        arr = np.array(lags)
        bins = np.linspace(-max_lag_s, max_lag_s, 25)
        ax.hist(arr, bins=bins, color=COLOR_INFO, edgecolor=COLOR_BG, alpha=0.85)
        ax.axvline(0.0, color=COLOR_TEXT, linestyle="--", linewidth=1.2, label="Zero Lag")
        ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_IDLE, labelcolor=COLOR_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path


def plot_timeline(frames_csv: Path | str, title: str, path: Path | str) -> Path:
    """Plot temporal signals (envelope, sync, addressee) with speech periods and decisions."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    _setup_dark_axes(ax, title, "Time [seconds]", "Signal Level")

    csv_path = Path(frames_csv)
    if csv_path.is_file():
        times: list[float] = []
        envs: list[float] = []
        liveness_list: list[float] = []
        addressee_list: list[float] = []
        speech_active_list: list[bool] = []
        decisions: list[str] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = float(row.get("t", 0.0))
                times.append(t)
                envs.append(float(row.get("env", 0.0)))
                liveness_list.append(float(row.get("liveness", 0.0)))
                addressee_list.append(float(row.get("addressee", 0.0)))
                sa = row.get("speech_active", "False").lower() in ("true", "1")
                speech_active_list.append(sa)
                decisions.append(row.get("decision", "IDLE"))

        if times:
            t0 = times[0]
            rel_times = np.array([t - t0 for t in times])

            # Draw signals
            ax.plot(rel_times, envs, color=COLOR_LISTEN, linewidth=1.5, label="Audio Env")
            ax.plot(rel_times, liveness_list, color=COLOR_ACCEPT, linewidth=1.5, label="Liveness (Sync)")
            ax.plot(rel_times, addressee_list, color=COLOR_INFO, linewidth=1.5, label="Addressee")

            # Shade speech active
            in_speech = False
            start_t = 0.0
            for i, sa in enumerate(speech_active_list):
                if sa and not in_speech:
                    in_speech = True
                    start_t = rel_times[i]
                elif not sa and in_speech:
                    in_speech = False
                    ax.axvspan(start_t, rel_times[i], color=COLOR_LISTEN, alpha=0.12)
            if in_speech:
                ax.axvspan(start_t, rel_times[-1], color=COLOR_LISTEN, alpha=0.12)

            # Decision strip at bottom
            for i in range(len(rel_times) - 1):
                dec = decisions[i]
                color = (
                    COLOR_ACCEPT if dec == "ACCEPT"
                    else COLOR_REJECT if dec.startswith("REJECT_")
                    else COLOR_LISTEN if dec == "LISTENING"
                    else COLOR_IDLE
                )
                ax.axvspan(rel_times[i], rel_times[i + 1], ymin=0.0, ymax=0.04, color=color, alpha=0.8)

            ax.legend(facecolor=COLOR_BG, edgecolor=COLOR_IDLE, labelcolor=COLOR_TEXT, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=COLOR_BG)
    plt.close(fig)
    return out_path
