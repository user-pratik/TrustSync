"""Tests for report plotting functions under normal and edge conditions."""
from __future__ import annotations

import csv
from pathlib import Path
import pytest

from trustsync.eval.plots import (
    plot_ablation,
    plot_box_by_scenario,
    plot_confusion,
    plot_lag_hist,
    plot_roc,
    plot_timeline,
)


def test_plots_tiny_inputs_produce_png(tmp_path: Path) -> None:
    # 1. plot_roc
    p_roc = plot_roc([0.8, 0.9], [0.1, 0.2], "ROC Test", tmp_path / "roc.png")
    assert p_roc.is_file() and p_roc.stat().st_size > 5000

    # 2. plot_box_by_scenario
    p_box = plot_box_by_scenario(
        {"live_single": [0.7, 0.8], "replay_spoof": [0.1, 0.2]},
        "Sync Score",
        0.35,
        tmp_path / "box.png",
    )
    assert p_box.is_file() and p_box.stat().st_size > 5000

    # 3. plot_confusion
    p_conf = plot_confusion(
        {"ACCEPT": {"ACCEPT": 5, "REJECT_SPOOF": 1}, "REJECT_SPOOF": {"REJECT_SPOOF": 4}},
        tmp_path / "conf.png",
    )
    assert p_conf.is_file() and p_conf.stat().st_size > 5000

    # 4. plot_ablation
    p_abl = plot_ablation(
        [
            {"name": "B0", "accuracy": 0.5, "far_all_reject": 1.0},
            {"name": "A1", "accuracy": 0.7, "far_all_reject": 0.3},
        ],
        tmp_path / "abl.png",
    )
    assert p_abl.is_file() and p_abl.stat().st_size > 5000

    # 5. plot_lag_hist
    p_lag = plot_lag_hist([0.02, 0.04, -0.03, 0.05], 0.2, tmp_path / "lag.png")
    assert p_lag.is_file() and p_lag.stat().st_size > 5000

    # 6. plot_timeline
    csv_file = tmp_path / "frames.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t", "decision", "speech_active", "env", "speaker_id", "liveness", "addressee", "n_faces", "fps"])
        writer.writerow([0, 100.0, "IDLE", "False", 0.002, "", 0.0, 0.0, 1, 30.0])
        writer.writerow([1, 100.033, "LISTENING", "True", 0.05, "0", 0.6, 0.8, 1, 30.0])
        writer.writerow([2, 100.066, "ACCEPT", "False", 0.002, "0", 0.7, 0.9, 1, 30.0])

    p_time = plot_timeline(csv_file, "Timeline Test", tmp_path / "timeline.png")
    assert p_time.is_file() and p_time.stat().st_size > 5000


def test_plots_empty_inputs_do_not_raise(tmp_path: Path) -> None:
    p_roc = plot_roc([], [], "Empty ROC", tmp_path / "empty_roc.png")
    assert p_roc.is_file()

    p_box = plot_box_by_scenario({}, "Score", None, tmp_path / "empty_box.png")
    assert p_box.is_file()

    p_conf = plot_confusion({}, tmp_path / "empty_conf.png")
    assert p_conf.is_file()

    p_abl = plot_ablation([], tmp_path / "empty_abl.png")
    assert p_abl.is_file()

    p_lag = plot_lag_hist([], 0.2, tmp_path / "empty_lag.png")
    assert p_lag.is_file()

    p_time = plot_timeline(tmp_path / "nonexistent.csv", "Empty Timeline", tmp_path / "empty_time.png")
    assert p_time.is_file()
