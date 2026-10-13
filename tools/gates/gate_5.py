"""TrustSync Gate 5: Lip-audio synchrony scorer and distribution separation."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import numpy as np

from trustsync.config import load_config
from trustsync.synth import make_env, make_lip
from trustsync.sync.scorer import sync_score


def _make_pair(
    rng: np.random.Generator,
    lag_s: float,
    noise: float,
    dur_s: float = 2.0,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    raw_env = make_env(dur_s + 1.0, 100.0, [(0.3, dur_s + 0.7)], rng)
    env_times = np.arange(len(raw_env), dtype=np.float64) / 100.0
    n = round(dur_s * fps)
    t_grid = 0.5 + np.arange(n, dtype=np.float64) / fps
    env_fps = np.interp(t_grid, env_times, raw_env)

    lip_full = make_lip(raw_env, 100.0, fps, lag_s, 0.05, noise, rng)
    start_idx = round(0.5 * fps)
    lip_fps = lip_full[start_idx : start_idx + n]

    return lip_fps, env_fps


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 5.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # 1. Run unit tests subprocess
    test_file = root / "tests" / "test_sync_scorer.py"
    if not test_file.is_file():
        results.append(("test_sync_scorer", "FAIL", f"File not found: {test_file}"))
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", str(test_file)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "passed"
            results.append(("tests/test_sync_scorer.py", "PASS", summary))
        else:
            err = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-1]
            results.append(("tests/test_sync_scorer.py", "FAIL", f"Exit {proc.returncode}: {err}"))

    # 2. Recompute distributions (500 seeds each)
    cfg = load_config(root / "tests" / "fixtures" / "config_spec_defaults.yaml")
    null_scores: list[float] = []
    pos_scores: list[float] = []

    for s in range(500):
        rng1 = np.random.default_rng(4000 + s)
        rng2 = np.random.default_rng(5000 + s)
        lip_null, _ = _make_pair(rng1, 0.0, 0.003)
        _, env_null = _make_pair(rng2, 0.0, 0.003)
        null_scores.append(sync_score(lip_null, env_null, 30.0, cfg.sync).score)

        lip_pos, env_pos = _make_pair(rng1, 0.0, 0.003)
        pos_scores.append(sync_score(lip_pos, env_pos, 30.0, cfg.sync).score)

    pos_arr = np.array(pos_scores)
    null_arr = np.array(null_scores)

    pos_p5 = float(np.percentile(pos_arr, 5))
    pos_med = float(np.median(pos_arr))
    null_med = float(np.median(null_arr))
    null_p95 = float(np.percentile(null_arr, 95))
    null_p99 = float(np.percentile(null_arr, 99))

    auc_val = float(
        np.mean([np.mean((pos_arr > y) + 0.5 * (pos_arr == y)) for y in null_arr])
    )

    detail = (
        f"pos p5={pos_p5:.3f} / med={pos_med:.3f}, "
        f"null med={null_med:.3f} / p95={null_p95:.3f} / p99={null_p99:.3f}, "
        f"AUC={auc_val:.3f}"
    )

    if auc_val >= 0.92 and (pos_med - null_p95) >= 0.15:
        results.append(("distributions", "PASS", detail))
    else:
        results.append(
            ("distributions", "FAIL", f"{detail} (failed threshold: AUC>=0.92, margin>=0.15)")
        )

    return results
