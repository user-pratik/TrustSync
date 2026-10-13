"""Unit tests for lip-audio synchrony scoring and attribution."""
from __future__ import annotations

import dataclasses
import math
import time
import numpy as np
import pytest

from trustsync.config import Config
from trustsync.synth import make_env, make_lip, make_static_lip
from trustsync.sync.scorer import sync_score, attribute


def make_pair(
    rng: np.random.Generator,
    lag_s: float,
    noise: float,
    dur_s: float = 2.0,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic (lip, env) signal pair aligned on an fps grid.

    Args:
        rng: Seeded numpy RandomGenerator.
        lag_s: Audio lag in seconds relative to lip movement.
        noise: Standard deviation of noise added to lip signal.
        dur_s: Duration in seconds.
        fps: Sampling rate in Hz.

    Returns:
        tuple[np.ndarray, np.ndarray]: Sliced (lip_fps, env_fps) arrays of shape (round(dur_s * fps),).
    """
    raw_env = make_env(dur_s + 1.0, 100.0, [(0.3, dur_s + 0.7)], rng)
    env_times = np.arange(len(raw_env), dtype=np.float64) / 100.0
    n = round(dur_s * fps)
    t_grid = 0.5 + np.arange(n, dtype=np.float64) / fps
    env_fps = np.interp(t_grid, env_times, raw_env)

    lip_full = make_lip(raw_env, 100.0, fps, lag_s, 0.05, noise, rng)
    start_idx = round(0.5 * fps)
    lip_fps = lip_full[start_idx : start_idx + n]

    return lip_fps, env_fps


def test_sync_positive_lag_zero(spec_cfg: Config) -> None:
    """Verify lag 0 with noise 0.003 scores > 0.6 in >= 95% of 200 seeds."""
    cfg = spec_cfg.sync
    above_count = 0
    total = 200
    for s in range(total):
        rng = np.random.default_rng(1000 + s)
        lip, env = make_pair(rng, 0.0, 0.003)
        res = sync_score(lip, env, 30.0, cfg)
        if res.score > 0.6:
            above_count += 1
    assert above_count / total >= 0.95


def test_sync_lag_recovery(spec_cfg: Config) -> None:
    """Verify lag recovery within 1 frame (1/30 s) in >= 90% of 100 seeds for 5 lags."""
    cfg = spec_cfg.sync
    for lag in (-0.1, -0.0667, 0.0, 0.0667, 0.1):
        rec_ok = 0
        total = 100
        for s in range(total):
            rng = np.random.default_rng(2000 + s)
            lip, env = make_pair(rng, lag, 0.003)
            res = sync_score(lip, env, 30.0, cfg)
            # Within 1 frame = 1/30 s = 0.0333 s tolerance
            if abs(res.lag_s - lag) <= (1.0 / 30.0 + 1e-6):
                if lag == 0.0 or (np.sign(res.lag_s) == np.sign(lag)):
                    rec_ok += 1
        assert rec_ok / total >= 0.90, f"Lag recovery failed for lag {lag}: {rec_ok}/{total}"


def test_sync_sign_deterministic(spec_cfg: Config) -> None:
    """Verify sign unit test without randomness: lip at peak 20, env at peak 23."""
    cfg = spec_cfg.sync
    # shape: (45,)
    i = np.arange(45, dtype=np.float64)
    lip = 0.02 + 0.05 * np.exp(-((i - 20.0) ** 2) / 8.0)
    env = 0.002 + 0.1 * np.exp(-((i - 23.0) ** 2) / 8.0)
    res = sync_score(lip, env, 30.0, cfg)
    assert abs(res.lag_s - 0.1) < 1e-9  # 3 frames at 30 fps = 0.1 s
    assert res.score > 0.7


def test_sync_static_mouth(spec_cfg: Config) -> None:
    """Verify static mouth yields score 0.0 and valid True over 200 seeds."""
    cfg = spec_cfg.sync
    for s in range(200):
        rng = np.random.default_rng(3000 + s)
        lip = make_static_lip(2.0, 30.0, 0.001, rng)
        env = make_env(2.0, 30.0, [(0.3, 1.7)], rng)
        res = sync_score(lip, env, 30.0, cfg)
        assert res.score == 0.0
        assert res.valid is True


def test_sync_mismatched_auc(spec_cfg: Config) -> None:
    """Verify AUC >= 0.92 and median(pos) - p95(null) >= 0.15 on 500 pairs."""
    cfg = spec_cfg.sync
    null_scores: list[float] = []
    pos_scores: list[float] = []

    for s in range(500):
        rng1 = np.random.default_rng(4000 + s)
        rng2 = np.random.default_rng(5000 + s)
        lip_null, _ = make_pair(rng1, 0.0, 0.003)
        _, env_null = make_pair(rng2, 0.0, 0.003)
        null_scores.append(sync_score(lip_null, env_null, 30.0, cfg).score)

        lip_pos, env_pos = make_pair(rng1, 0.0, 0.003)
        pos_scores.append(sync_score(lip_pos, env_pos, 30.0, cfg).score)

    pos_arr = np.array(pos_scores)
    null_arr = np.array(null_scores)

    # Rank formula for AUC
    auc_val = float(
        np.mean([np.mean((pos_arr > y) + 0.5 * (pos_arr == y)) for y in null_arr])
    )
    med_pos = float(np.median(pos_arr))
    p95_null = float(np.percentile(null_arr, 95))

    assert auc_val >= 0.92, f"AUC {auc_val} < 0.92"
    assert (med_pos - p95_null) >= 0.15, f"Separation {med_pos - p95_null} < 0.15"


def test_sync_noise_monotonicity(spec_cfg: Config) -> None:
    """Verify median score decreases monotonically with increasing noise."""
    cfg = spec_cfg.sync
    medians: list[float] = []
    for noise in (0.003, 0.01, 0.02, 0.04):
        scs = [
            sync_score(*make_pair(np.random.default_rng(6000 + s), 0.0, noise), 30.0, cfg).score
            for s in range(100)
        ]
        medians.append(float(np.median(scs)))

    # Non-strictly decreasing: medians[i] <= medians[i - 1]
    for i in range(1, len(medians)):
        assert medians[i] <= medians[i - 1]


def test_sync_degenerate_cases(spec_cfg: Config) -> None:
    """Verify degenerate inputs: short signal, constant signal, NaN, length mismatch."""
    cfg = spec_cfg.sync
    fps = 30.0
    short_n = round(cfg.min_window_s * fps) - 1

    # Short signal
    res_short = sync_score(np.zeros(short_n), np.zeros(short_n), fps, cfg)
    assert res_short.valid is False and res_short.score == 0.0

    # Constant env
    lip = np.linspace(0.01, 0.06, 60)
    env_const = np.full(60, 0.05)
    res_const = sync_score(lip, env_const, fps, cfg)
    assert res_const.score == 0.0
    assert not math.isnan(res_const.score) and not math.isnan(res_const.lag_s)

    # All-zero env
    res_zero = sync_score(lip, np.zeros(60), fps, cfg)
    assert res_zero.score == 0.0

    # NaN in lip
    lip_nan = lip.copy()
    lip_nan[5] = np.nan
    res_nan = sync_score(lip_nan, env_const, fps, cfg)
    assert res_nan.valid is False

    # Unequal lengths
    with pytest.raises(ValueError):
        sync_score(np.zeros(60), np.zeros(59), fps, cfg)


def test_sync_invariance(spec_cfg: Config) -> None:
    """Verify score invariance under positive scaling of env and affine transform of lip."""
    cfg = dataclasses.replace(spec_cfg.sync, env_eps=0.0)
    rng = np.random.default_rng(7000)
    lip, env = make_pair(rng, 0.0, 0.003)
    base_res = sync_score(lip, env, 30.0, cfg)

    # Multiplied env by 7
    res_env_scaled = sync_score(lip, env * 7.0, 30.0, cfg)
    assert abs(res_env_scaled.score - base_res.score) < 1e-9

    # Multiplied lip by 3 and added 5
    res_lip_scaled = sync_score(lip * 3.0 + 5.0, env, 30.0, cfg)
    assert abs(res_lip_scaled.score - base_res.score) < 1e-9


def test_attribute_cases() -> None:
    """Verify attribute ties, highest track, and empty dict handling."""
    assert attribute({1: 0.7, 2: 0.2}) == (1, 0.5)
    assert attribute({3: 0.6}) == (3, 0.6)
    assert attribute({}) == (None, 0.0)
    assert attribute({4: 0.5, 2: 0.5}) == (2, 0.0)


def test_multi_face_attribution(spec_cfg: Config) -> None:
    """Verify multi-face attribution accuracy with moving and static other tracks."""
    cfg = spec_cfg.sync
    picks_a = 0
    picks_a_static = 0
    total = 200

    for s in range(total):
        rng_a = np.random.default_rng(8000 + s)
        rng_b = np.random.default_rng(9000 + s)

        lip_a, env = make_pair(rng_a, 0.033, 0.003)
        lip_b, _ = make_pair(rng_b, 0.0, 0.003)
        sc_a = sync_score(lip_a, env, 30.0, cfg).score
        sc_b = sync_score(lip_b, env, 30.0, cfg).score

        best_id, _ = attribute({1: sc_a, 2: sc_b})
        if best_id == 1:
            picks_a += 1

        lip_b_static = make_static_lip(2.0, 30.0, 0.001, rng_b)
        sc_b_static = sync_score(lip_b_static, env, 30.0, cfg).score
        best_id_stat, _ = attribute({1: sc_a, 2: sc_b_static})
        if best_id_stat == 1:
            picks_a_static += 1

    assert picks_a / total >= 0.90
    assert picks_a_static / total >= 0.99


def test_sync_score_speed(spec_cfg: Config) -> None:
    """Verify 1000 calls of sync_score with N=60 finish in under 1.5 s."""
    cfg = spec_cfg.sync
    lip = np.random.default_rng(1234).standard_normal(60) * 0.02 + 0.05
    env = np.random.default_rng(5678).standard_normal(60) * 0.02 + 0.05
    t0 = time.perf_counter()
    for _ in range(1000):
        sync_score(lip, env, 30.0, cfg)
    dt = time.perf_counter() - t0
    assert dt < 1.5, f"1000 calls took {dt:.3f} s (expected < 1.5 s)"
