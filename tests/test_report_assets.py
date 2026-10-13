"""Tests for report assets generator, Wilson confidence intervals, and report skeleton."""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pytest

from tools.make_report_assets import generate_assets, wilson
from tools.make_report_skeleton import generate_skeleton


def test_wilson_score_interval_values() -> None:
    """Verify wilson(22, 24) is within 0.01 of (0.742, 0.968)."""
    low, high = wilson(22, 24)
    assert abs(low - 0.742) <= 0.01, f"Wilson lower bound {low} not within 0.01 of 0.742"
    assert abs(high - 0.968) <= 0.01, f"Wilson upper bound {high} not within 0.01 of 0.968"


def test_make_report_assets_and_skeleton(tmp_path: Path) -> None:
    """Verify report assets generation from synthetic results.json and digit matching."""
    synthetic_results = {
        "generated_at": "2026-10-13T00:00:00Z",
        "dataset": {
            "n_clips": 24,
            "per_scenario_counts": {
                "live_single": 8,
                "replay_spoof": 8,
                "multi_one_speaks": 4,
                "not_addressed": 4,
            },
            "subjects": ["alice", "bob"],
        },
        "config": {
            "gate": {"min_sync": 0.35, "min_margin": 0.10, "min_directed": 0.50, "min_quality": 0.50},
            "min_lip_std": 0.004,
            "addressee": {"max_yaw_deg": 25, "max_pitch_deg": 20},
        },
        "current": {
            "n_clips": 24,
            "decision_accuracy": 0.9167,
            "far_spoof": 0.0,
            "far_all_reject": 0.0,
            "frr": 0.0833,
            "false_trigger_rate": 0.0,
            "missed_utterance_rate": 0.0,
            "liveness": {"precision": 1.0, "recall": 0.9167, "f1": 0.9565},
            "attribution_accuracy": 1.0,
            "addressee_accuracy": 1.0,
            "reason_accuracy": 1.0,
            "per_scenario": {
                "live_single": {"n": 8, "accuracy": 1.0, "accept_rate": 1.0},
                "replay_spoof": {"n": 8, "accuracy": 1.0, "accept_rate": 0.0},
                "multi_one_speaks": {"n": 4, "accuracy": 1.0, "accept_rate": 1.0},
                "not_addressed": {"n": 4, "accuracy": 1.0, "accept_rate": 0.0},
            },
            "confusion": {
                "ACCEPT": {"ACCEPT": 11, "REJECT_SPOOF": 1, "REJECT_AMBIGUOUS": 0, "REJECT_NOT_ADDRESSED": 0, "REJECT_LOW_QUALITY": 0, "REJECT_NO_FACE": 0, "NONE": 0},
                "REJECT_SPOOF": {"ACCEPT": 0, "REJECT_SPOOF": 8, "REJECT_AMBIGUOUS": 0, "REJECT_NOT_ADDRESSED": 0, "REJECT_LOW_QUALITY": 0, "REJECT_NO_FACE": 0, "NONE": 0},
                "REJECT_NOT_ADDRESSED": {"ACCEPT": 0, "REJECT_SPOOF": 0, "REJECT_AMBIGUOUS": 0, "REJECT_NOT_ADDRESSED": 4, "REJECT_LOW_QUALITY": 0, "REJECT_NO_FACE": 0, "NONE": 0},
            },
        },
        "calibrated": None,
        "loso_rule": None,
        "loso_learned": None,
        "learned_kind": None,
        "ablation": [
            {"name": "B0", "accuracy": 0.50, "far_all_reject": 1.0, "frr": 0.0, "liveness_f1": 0.67},
            {"name": "A1", "accuracy": 0.75, "far_all_reject": 0.25, "frr": 0.0, "liveness_f1": 0.85},
            {"name": "A2", "accuracy": 0.80, "far_all_reject": 0.20, "frr": 0.0, "liveness_f1": 0.88},
            {"name": "A3", "accuracy": 0.85, "far_all_reject": 0.10, "frr": 0.0, "liveness_f1": 0.91},
            {"name": "A4", "accuracy": 0.92, "far_all_reject": 0.0, "frr": 0.08, "liveness_f1": 0.95},
            {"name": "A5", "accuracy": 0.95, "far_all_reject": 0.0, "frr": 0.05, "liveness_f1": 0.97},
        ],
        "av_lag": {"median": 0.015, "p10": -0.01, "p90": 0.04, "n": 24, "saturated_fraction": 0.0},
        "fps": {"mean": 28.5, "p5": 14.2, "source": "photo", "machine": "TestCPU"},
        "figures": [],
        "notes": ["Synthetic test dataset."],
    }

    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(synthetic_results, indent=2), encoding="utf-8")

    report_dir = tmp_path / "report"

    # Generate assets
    generate_assets(results_path=results_path, report_dir=report_dir)

    # Verify files created
    assert (report_dir / "tables.md").is_file()
    assert (report_dir / "numbers.json").is_file()
    assert (report_dir / "table_headline.csv").is_file()
    assert (report_dir / "table_ablation.csv").is_file()
    assert (report_dir / "table_per_scenario.csv").is_file()
    assert (report_dir / "table_confusion.csv").is_file()
    assert (report_dir / "table_loso_folds.csv").is_file()
    assert (report_dir / "table_dataset.csv").is_file()

    numbers = json.loads((report_dir / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["n_clips"] == 24
    assert np.isclose(numbers["decision_accuracy"], 0.9167)
    assert numbers["far_spoof"] == 0.0

    # Generate skeleton
    generate_skeleton(report_dir=report_dir)
    assert (report_dir / "REPORT.md").is_file()
    assert (report_dir / "SLIDES_OUTLINE.md").is_file()
    assert (report_dir / "DEMO_SHOTLIST.md").is_file()

    report_text = (report_dir / "REPORT.md").read_text(encoding="utf-8")

    # Extract Results section
    results_match = re.search(r"## Results\s+(.*?)\s+## Ablation", report_text, re.DOTALL)
    assert results_match is not None, "Results section not found in REPORT.md"
    results_section = results_match.group(1)

    # Collect all numbers appearing in numbers.json
    all_num_values: set[str] = set()
    for k, v in numbers.items():
        if isinstance(v, (int, float)):
            all_num_values.add(str(v))
            if isinstance(v, float):
                all_num_values.add(f"{v:.1f}")
                all_num_values.add(f"{v:.2f}")
                all_num_values.add(f"{v*100.0:.1f}")
                all_num_values.add(f"{v*100.0:.0f}")

    # Also add numbers from tables.md
    for token in re.findall(r"\b\d+(?:\.\d+)?\b", (report_dir / "tables.md").read_text(encoding="utf-8")):
        all_num_values.add(token)

    # Add standard confidence values (95, 95%)
    all_num_values.update(["95", "95.0", "0", "1", "2"])

    # Verify each number token in Results section prose is represented in numbers.json
    # Only check the prose before the tables (tables are verified separately)
    prose_before_tables = results_section.split("## Headline Metrics")[0]
    tokens = re.findall(r"\b\d+(?:\.\d+)?\b", prose_before_tables)
    for tok in tokens:
        assert tok in all_num_values, f"Token {tok} in Results section absent from numbers.json"
