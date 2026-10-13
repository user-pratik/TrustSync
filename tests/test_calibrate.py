"""Tests for calibration sweeps, LOSO validation, and latency estimation."""
from __future__ import annotations

import numpy as np
import pytest

from tests.helpers_eval import fake_clip
from trustsync.config import Config
from trustsync.eval.calibrate import estimate_av_lag, loso, sweep_rule_thresholds
from trustsync.eval.metrics import compute_metrics, decide_clip
from trustsync.fusion.gate import RuleGate


def test_calibrate_perfectly_separable(spec_cfg: Config) -> None:
    live = fake_clip("live_single", "ACCEPT", 0.8, 0.02, 0.9, subject="s1")
    replay = fake_clip("replay_spoof", "REJECT_SPOOF", 0.05, 0.001, 0.9, subject="s2")
    not_addr = fake_clip("not_addressed", "REJECT_NOT_ADDRESSED", 0.8, 0.02, 0.1, subject="s3", addressed=False)

    records = [live, replay, not_addr]
    new_cfg, table = sweep_rule_thresholds(records, spec_cfg, max_far=0.02)

    outcomes = [decide_clip(r, RuleGate(new_cfg)) for r in records]
    metrics = compute_metrics(records, outcomes, new_cfg)

    assert metrics["decision_accuracy"] == 1.0
    assert metrics["far_all_reject"] == 0.0

    # Thresholds strictly between class values
    assert 0.05 < new_cfg.gate.min_sync < 0.8
    assert 0.1 < new_cfg.gate.min_directed < 0.9
    assert 0.001 < new_cfg.sync.min_lip_std < 0.02


def test_calibrate_overlapping_far_bound(spec_cfg: Config) -> None:
    rng = np.random.default_rng(42)
    sync_live = np.clip(rng.normal(0.55, 0.15, 60), 0.01, 0.99)
    sync_replay = np.clip(rng.normal(0.30, 0.15, 60), 0.01, 0.99)

    live_clips = [
        fake_clip("live_single", "ACCEPT", float(s), 0.02, 0.9, subject=f"sub_{i % 3}")
        for i, s in enumerate(sync_live)
    ]
    replay_clips = [
        fake_clip("replay_spoof", "REJECT_SPOOF", float(s), 0.02, 0.9, subject=f"sub_{i % 3}")
        for i, s in enumerate(sync_replay)
    ]
    records = live_clips + replay_clips

    new_cfg, table = sweep_rule_thresholds(records, spec_cfg, max_far=0.02)
    outcomes = [decide_clip(r, RuleGate(new_cfg)) for r in records]
    metrics = compute_metrics(records, outcomes, new_cfg)

    assert metrics["far_all_reject"] <= 0.02


def test_loso_three_subjects_spy(spec_cfg: Config) -> None:
    c1 = fake_clip("live_single", "ACCEPT", 0.8, 0.02, 0.9, subject="sub1")
    c2 = fake_clip("replay_spoof", "REJECT_SPOOF", 0.1, 0.001, 0.9, subject="sub2")
    c3 = fake_clip("live_single", "ACCEPT", 0.7, 0.02, 0.9, subject="sub3")
    records = [c1, c2, c3]

    spy_train_subjects: list[set[str]] = []

    def fit_fn(train_records, cfg):
        subjs = {r.subject for r in train_records}
        spy_train_subjects.append(subjs)
        return RuleGate(cfg)

    def predict_fn(model, test_records, cfg):
        return [decide_clip(r, model) for r in test_records]

    res = loso(records, fit_fn, predict_fn, spec_cfg)

    assert len(spy_train_subjects) == 3
    # Verify held-out subject was never in training fold
    assert "sub1" not in spy_train_subjects[0]
    assert "sub2" not in spy_train_subjects[1]
    assert "sub3" not in spy_train_subjects[2]

    assert res["pooled"]["n_clips"] == len(records)
    assert set(res["folds"].keys()) == {"sub1", "sub2", "sub3"}


def test_estimate_av_lag(spec_cfg: Config) -> None:
    # Empty input handled
    empty_res = estimate_av_lag([], spec_cfg)
    assert empty_res["median"] is None
    assert empty_res["saturated_fraction"] == 0.0

    # Known median and saturation
    sat_lag = spec_cfg.sync.max_lag_s  # 0.20
    c1 = fake_clip("live_single", "ACCEPT", 0.7, 0.02, 0.9, lag_samples=[0.05, 0.05, 0.05, sat_lag])
    res = estimate_av_lag([c1], spec_cfg)

    assert abs(res["median"] - 0.05) <= 0.01
    assert res["saturated_fraction"] == 0.25
