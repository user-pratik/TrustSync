"""Lip-audio synchrony scoring and speaker attribution."""
from __future__ import annotations

import numpy as np
import scipy.ndimage

from trustsync.config import SyncCfg
from trustsync.types import SyncResult


def sync_score(
    lip: np.ndarray, env: np.ndarray, fps: float, cfg: SyncCfg
) -> SyncResult:
    """Compute lip-audio synchrony score across a lag search window.

    Args:
        lip: Lip aperture signal array shape (N,).
        env: Audio envelope signal array shape (N,).
        fps: Sampling rate / video frame rate in Hz.
        cfg: SyncCfg configuration.

    Returns:
        SyncResult: Synchrony score, lag in seconds, lip std, valid flag, and per-feature correlations.

    Raises:
        ValueError: If lip and env have different lengths.
    """
    if len(lip) != len(env):
        raise ValueError(
            f"lip and env lengths must match, got {len(lip)} and {len(env)}"
        )

    if np.isnan(lip).any() or np.isnan(env).any():
        return SyncResult(
            score=0.0, lag_s=0.0, lip_std=0.0, valid=False, r_open=0.0, r_vel=0.0
        )

    n = len(lip)
    min_len = round(cfg.min_window_s * fps)
    if n < min_len:
        return SyncResult(
            score=0.0, lag_s=0.0, lip_std=0.0, valid=False, r_open=0.0, r_vel=0.0
        )

    lip_64 = np.asarray(lip, dtype=np.float64)
    env_64 = np.asarray(env, dtype=np.float64)

    # 2. Detrend lip signal
    k = max(1, round(fps))
    baseline = scipy.ndimage.uniform_filter1d(lip_64, k, mode="nearest")
    lip_d = lip_64 - baseline
    lip_std = float(np.std(lip_d))

    if lip_std < cfg.min_lip_std:
        return SyncResult(
            score=0.0,
            lag_s=0.0,
            lip_std=lip_std,
            valid=True,
            r_open=0.0,
            r_vel=0.0,
        )

    # 3. Feature extraction
    f_open = lip_d
    lip_grad = np.gradient(lip_d) * fps
    f_vel = scipy.ndimage.uniform_filter1d(np.abs(lip_grad), 3, mode="nearest")
    e = np.log(np.maximum(env_64, 0.0) + cfg.env_eps)

    # Precompute cumulative sums for fast slice statistics
    c_e = np.empty(n + 1, dtype=np.float64)
    c_e[0] = 0.0
    np.cumsum(e, out=c_e[1:])

    c_e2 = np.empty(n + 1, dtype=np.float64)
    c_e2[0] = 0.0
    np.cumsum(e * e, out=c_e2[1:])

    c_fo = np.empty(n + 1, dtype=np.float64)
    c_fo[0] = 0.0
    np.cumsum(f_open, out=c_fo[1:])

    c_fo2 = np.empty(n + 1, dtype=np.float64)
    c_fo2[0] = 0.0
    np.cumsum(f_open * f_open, out=c_fo2[1:])

    c_fv = np.empty(n + 1, dtype=np.float64)
    c_fv[0] = 0.0
    np.cumsum(f_vel, out=c_fv[1:])

    c_fv2 = np.empty(n + 1, dtype=np.float64)
    c_fv2[0] = 0.0
    np.cumsum(f_vel * f_vel, out=c_fv2[1:])

    # 4. Lag search in -K..K
    max_k = round(cfg.max_lag_s * fps)
    r_open_best = -1.0
    lag_open_best = 0
    r_vel_best = -1.0
    lag_vel_best = 0

    for j in range(-max_k, max_k + 1):
        if j >= 0:
            ao = f_open[: n - j]
            av = f_vel[: n - j]
            b = e[j:]
            sb, eb = j, n
            sa, ea = 0, n - j
        else:
            ao = f_open[-j:]
            av = f_vel[-j:]
            b = e[: n + j]
            sb, eb = 0, n + j
            sa, ea = -j, n

        overlap_len = n - abs(j)
        sum_b = c_e[eb] - c_e[sb]
        mb = sum_b / overlap_len
        vb = (c_e2[eb] - c_e2[sb]) - overlap_len * mb * mb
        if vb < 1e-16:
            continue
        norm_b = np.sqrt(vb)

        sum_ao = c_fo[ea] - c_fo[sa]
        mao = sum_ao / overlap_len
        vao = (c_fo2[ea] - c_fo2[sa]) - overlap_len * mao * mao
        if vao >= 1e-16:
            cov_o = np.dot(ao, b) - overlap_len * mao * mb
            ro = float(cov_o / (np.sqrt(vao) * norm_b))
            if ro > r_open_best:
                r_open_best = ro
                lag_open_best = j

        sum_av = c_fv[ea] - c_fv[sa]
        mav = sum_av / overlap_len
        vav = (c_fv2[ea] - c_fv2[sa]) - overlap_len * mav * mb
        if vav >= 1e-16:
            cov_v = np.dot(av, b) - overlap_len * mav * mb
            rv = float(cov_v / (np.sqrt(vav) * norm_b))
            if rv > r_vel_best:
                r_vel_best = rv
                lag_vel_best = j

    # 5. Winning feature and result aggregation
    if r_open_best >= r_vel_best:
        winning_lag = lag_open_best
    else:
        winning_lag = lag_vel_best

    score = float(max(0.0, r_open_best, r_vel_best))
    lag_s = float(winning_lag / fps)

    return SyncResult(
        score=score,
        lag_s=lag_s,
        lip_std=lip_std,
        valid=True,
        r_open=float(r_open_best),
        r_vel=float(r_vel_best),
    )


def attribute(scores: dict[int, float]) -> tuple[int | None, float]:
    """Attribute speech to the track with the highest synchrony score.

    Args:
        scores: Dict mapping track_id to synchrony score.

    Returns:
        tuple[int | None, float]: (best_speaker_id, margin), where margin is highest minus
        second highest score (or highest if only one track). Ties are broken by lowest track_id.
    """
    if not scores:
        return (None, 0.0)

    # Sort descending by score, ascending by track_id for tie-breaking
    sorted_tracks = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_id, best_score = sorted_tracks[0]

    if len(sorted_tracks) == 1:
        return (best_id, float(best_score))

    second_score = sorted_tracks[1][1]
    margin = round(float(best_score - second_score), 4)
    return (best_id, margin)
