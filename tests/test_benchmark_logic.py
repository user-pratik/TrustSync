"""Unit tests for benchmark utility functions and results updating."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from tools.benchmark import compute_fps_from_ms, compute_percentiles, format_table, update_results_fps


def test_benchmark_percentiles_fixed_array() -> None:
    """Verify percentiles on fixed array 1..100 (p50, p95, p99) and FPS computation."""
    data = np.arange(1, 101, dtype=np.float64)
    stats = compute_percentiles(data)

    assert np.isclose(stats["p50_ms"], 50.5, atol=1e-3), f"p50 mismatch: {stats['p50_ms']}"
    assert np.isclose(stats["p95_ms"], 95.05, atol=1e-3), f"p95 mismatch: {stats['p95_ms']}"
    assert np.isclose(stats["p99_ms"], 99.01, atol=1e-3), f"p99 mismatch: {stats['p99_ms']}"

    # FPS from mean ms 25 is 40.0
    fps_stats = compute_fps_from_ms(25.0, 50.0)
    assert np.isclose(fps_stats["mean"], 40.0, atol=1e-3), f"FPS mean mismatch: {fps_stats['mean']}"
    assert np.isclose(fps_stats["p5"], 20.0, atol=1e-3), f"FPS p5 mismatch: {fps_stats['p5']}"


def test_benchmark_format_table() -> None:
    """Verify table formatter produces aligned table containing all stages."""
    stages = {
        "face_tracker.process": {"mean_ms": 12.34, "p50_ms": 12.0, "p95_ms": 15.0, "p99_ms": 18.0},
        "draw_hud": {"mean_ms": 1.5, "p50_ms": 1.4, "p95_ms": 2.1, "p99_ms": 2.8},
    }
    table = format_table(stages)
    assert "face_tracker.process" in table
    assert "draw_hud" in table
    assert "mean ms" in table
    assert "p50" in table
    assert "p95" in table
    assert "p99" in table


def test_benchmark_update_results_fps_isolated(tmp_path: Path) -> None:
    """Verify update_results_fps modifies ONLY the 'fps' key in results.json."""
    sample_data = {
        "generated_at": "2026-10-13T00:00:00Z",
        "dataset": {"n_clips": 12, "subjects": ["alice", "bob"]},
        "config": {"gate": {"mode": "rule"}},
        "current": {"decision_accuracy": 0.95},
        "fps": None,
        "notes": ["Sample evaluation"],
    }
    json_path = tmp_path / "results.json"
    json_path.write_text(json.dumps(sample_data, indent=2), encoding="utf-8")

    fps_payload = {
        "mean": 38.5,
        "p5": 32.1,
        "source": "photo",
        "machine": "TestCPU",
    }
    update_results_fps(json_path, fps_payload)

    updated_data = json.loads(json_path.read_text(encoding="utf-8"))

    # fps key must be updated
    assert updated_data["fps"] == fps_payload

    # All other keys must be identical to original sample
    for key, val in sample_data.items():
        if key != "fps":
            assert updated_data[key] == val, f"Key {key} was unexpectedly modified"
