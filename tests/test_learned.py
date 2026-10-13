"""Tests for learned trust gate, feature extraction, training, and model serialization."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from tests.helpers_eval import fake_clip
from trustsync.config import Config
from trustsync.fusion.gate import RuleGate, make_gate
from trustsync.fusion.learned import (
    FEATURE_NAMES,
    LearnedGate,
    load_gate,
    save_gate,
    summary_features,
    train_learned_gate,
)
from trustsync.types import TrackSummary, UtteranceSummary


def test_feature_vector_extraction() -> None:
    t0 = TrackSummary(
        track_id=0,
        sync=0.7,
        lip_std=0.03,
        directed=0.85,
        quality=0.9,
        yaw=10.0,
        pitch=-5.0,
    )
    t1 = TrackSummary(
        track_id=1,
        sync=0.3,
        lip_std=0.01,
        directed=0.4,
        quality=0.6,
        yaw=20.0,
        pitch=0.0,
    )
    summary = UtteranceSummary(
        t_start=1.0,
        t_end=2.5,
        duration_s=1.5,
        tracks=[t0, t1],
    )

    feats = summary_features(summary)
    assert len(feats) == 10
    assert len(FEATURE_NAMES) == 10

    assert feats[0] == 0.7  # sync_best
    assert feats[1] == 0.3  # sync_second
    assert abs(feats[2] - 0.4) < 1e-6  # margin
    assert feats[3] == 0.03  # lip_std_best
    assert feats[4] == 0.85  # directed_best
    assert feats[5] == 10.0  # abs_yaw_best
    assert feats[6] == 5.0  # abs_pitch_best
    assert feats[7] == 0.9  # quality_max
    assert feats[8] == 2.0  # n_tracks
    assert feats[9] == 1.5  # duration_s

    # Test NaN yaw / pitch becomes 90
    t_nan = TrackSummary(
        track_id=0,
        sync=0.5,
        lip_std=0.02,
        directed=0.5,
        quality=0.8,
        yaw=float("nan"),
        pitch=float("nan"),
    )
    s_nan = UtteranceSummary(1.0, 2.0, 1.0, [t_nan])
    feats_nan = summary_features(s_nan)
    assert feats_nan[5] == 90.0
    assert feats_nan[6] == 90.0


def test_learned_gate_lifecycle(tmp_path: Path, spec_cfg: Config) -> None:
    # 1. Separable dataset
    live_clips = [
        fake_clip("live_single", "ACCEPT", 0.8, 0.03, 0.9, subject=f"sub_{i}")
        for i in range(15)
    ]
    spoof_clips = [
        fake_clip("replay_spoof", "REJECT_SPOOF", 0.05, 0.001, 0.9, subject=f"sub_{i}")
        for i in range(15)
    ]
    records = live_clips + spoof_clips

    bundle = train_learned_gate(records, spec_cfg, kind="logreg")
    assert bundle["feature_names"] == FEATURE_NAMES

    # Evaluate training accuracy
    X = np.vstack([
        summary_features(max(r.utts, key=lambda u: u.t_end - u.t_start).summary)
        for r in records
    ])
    y = np.array([1 if r.expected == "ACCEPT" else 0 for r in records])
    probs = bundle["model"].predict_proba(X)[:, 1]
    preds = (probs >= bundle["threshold"]).astype(int)
    acc = float(np.mean(preds == y))
    assert acc >= 0.95

    # 2. Save then load gives identical probabilities
    model_path = tmp_path / "gate.joblib"
    save_gate(bundle, model_path)
    loaded = load_gate(model_path)
    loaded_probs = loaded["model"].predict_proba(X)[:, 1]
    assert np.allclose(probs, loaded_probs)

    # 3. LearnedGate behavior
    cfg_learned = dataclasses.replace(
        spec_cfg,
        gate=dataclasses.replace(
            spec_cfg.gate,
            mode="learned",
            model_path=str(model_path),
        ),
    )
    gate = LearnedGate(cfg_learned)

    # Pre-model rejections
    no_tracks = UtteranceSummary(1.0, 2.0, 1.0, [])
    assert gate.decide(no_tracks).decision == "REJECT_NO_FACE"

    low_qual = UtteranceSummary(
        1.0,
        2.0,
        1.0,
        [TrackSummary(0, 0.9, 0.05, 0.9, quality=0.1, yaw=0.0, pitch=0.0)],
    )
    assert gate.decide(low_qual).decision == "REJECT_LOW_QUALITY"

    # Live accepted, spoof rejected
    sum_live = UtteranceSummary(
        1.0,
        2.0,
        1.0,
        [TrackSummary(0, 0.85, 0.04, 0.95, quality=0.9, yaw=0.0, pitch=0.0)],
    )
    dec_live = gate.decide(sum_live)
    assert dec_live.decision == "ACCEPT"
    assert dec_live.speaker_id == 0

    sum_spoof = UtteranceSummary(
        1.0,
        2.0,
        1.0,
        [TrackSummary(0, 0.05, 0.001, 0.95, quality=0.9, yaw=0.0, pitch=0.0)],
    )
    dec_spoof = gate.decide(sum_spoof)
    assert dec_spoof.decision == "REJECT_SPOOF"

    # 4. make_gate factory
    cfg_rule = dataclasses.replace(
        spec_cfg, gate=dataclasses.replace(spec_cfg.gate, mode="rule")
    )
    assert isinstance(make_gate(cfg_rule), RuleGate)
    assert isinstance(make_gate(cfg_learned), LearnedGate)
