"""Unit tests for addressee classification."""
from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from trustsync.config import Config
from trustsync.addressee.classifier import directed_score


def test_directed_score_default_config(spec_cfg: Config) -> None:
    """Verify directed_score with default config matches expected values."""
    cfg = spec_cfg.addressee

    # (0, 0, 0) gives 1.0
    assert directed_score(0.0, 0.0, 0.0, cfg) == pytest.approx(1.0)

    # (25, 0, 0) gives 1.0
    assert directed_score(25.0, 0.0, 0.0, cfg) == pytest.approx(1.0)

    # (37.5, 0, 0) gives 0.5
    assert directed_score(37.5, 0.0, 0.0, cfg) == pytest.approx(0.5)

    # (50, 0, 0) gives 0.0
    assert directed_score(50.0, 0.0, 0.0, cfg) == pytest.approx(0.0)

    # (0, 30, 0) gives 0.5
    assert directed_score(0.0, 30.0, 0.0, cfg) == pytest.approx(0.5)

    # (0, 0, 0.6) gives 0.7
    assert directed_score(0.0, 0.0, 0.6, cfg) == pytest.approx(0.7)

    # gaze 1.2 gives 0.7
    assert directed_score(0.0, 0.0, 1.2, cfg) == pytest.approx(0.7)


def test_directed_score_use_gaze_false(spec_cfg: Config) -> None:
    """Verify use_gaze=False ignores gaze offset."""
    cfg = dataclasses.replace(spec_cfg.addressee, use_gaze=False)
    assert directed_score(0.0, 0.0, 0.6, cfg) == pytest.approx(1.0)
    assert directed_score(0.0, 0.0, 1.2, cfg) == pytest.approx(1.0)


def test_directed_score_nan_inputs(spec_cfg: Config) -> None:
    """Verify any NaN input returns 0.0."""
    cfg = spec_cfg.addressee
    assert directed_score(float("nan"), 0.0, 0.0, cfg) == 0.0
    assert directed_score(0.0, float("nan"), 0.0, cfg) == 0.0
    assert directed_score(0.0, 0.0, float("nan"), cfg) == 0.0


def test_directed_score_yaw_offset(spec_cfg: Config) -> None:
    """Verify yaw_offset_deg=10 shifts center so (10, 0, 0) gives 1.0."""
    cfg = dataclasses.replace(spec_cfg.addressee, yaw_offset_deg=10.0)
    assert directed_score(10.0, 0.0, 0.0, cfg) == pytest.approx(1.0)


def test_directed_score_monotonicity(spec_cfg: Config) -> None:
    """Verify directed_score is monotonic non-increasing with abs(yaw) from 0 to 90."""
    cfg = spec_cfg.addressee
    # yaws shape: (91,)
    yaws = np.linspace(0.0, 90.0, 91)

    pos_scores = [directed_score(float(y), 0.0, 0.0, cfg) for y in yaws]
    for i in range(1, len(pos_scores)):
        assert pos_scores[i] <= pos_scores[i - 1] + 1e-9

    neg_scores = [directed_score(float(-y), 0.0, 0.0, cfg) for y in yaws]
    for i in range(1, len(neg_scores)):
        assert neg_scores[i] <= neg_scores[i - 1] + 1e-9


def test_directed_score_random_bounded(spec_cfg: Config) -> None:
    """Verify 10,000 random inputs produce outputs strictly clipped in [0.0, 1.0]."""
    cfg = spec_cfg.addressee
    rng = np.random.default_rng(1234)
    # shapes: (10000,)
    yaws = rng.uniform(-180.0, 180.0, 10000)
    pitches = rng.uniform(-90.0, 90.0, 10000)
    gazes = rng.uniform(-2.0, 3.0, 10000)

    for y, p, g in zip(yaws, pitches, gazes):
        s = directed_score(float(y), float(p), float(g), cfg)
        assert 0.0 <= s <= 1.0
