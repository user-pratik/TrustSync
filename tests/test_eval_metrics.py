"""Tests for evaluation metrics and serialization."""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from trustsync.config import Config
from trustsync.eval.metrics import (
    ClipOutcome,
    ClipRecord,
    UttRecord,
    auc,
    compute_metrics,
    decide_clip,
    records_from_json,
    records_to_json,
    side_of,
)
from trustsync.fusion.gate import RuleGate
from trustsync.types import TrackSummary, UtteranceSummary

from tests.helpers_eval import fake_clip


def test_eval_metrics_hand_computed(spec_cfg: Config) -> None:
    gate = RuleGate(spec_cfg)

    # Build 12 clips
    c1 = fake_clip("replay_spoof", "REJECT_SPOOF", 0.6, 0.01, 0.9, 1.0)
    c2 = fake_clip("replay_spoof", "REJECT_SPOOF", 0.1, 0.001, 0.9, 1.0)
    c3 = fake_clip(
        "multi_one_speaks", "ACCEPT", 0.7, 0.02, 0.8, 1.0, n_tracks=2, speaker_side="left", speaker_cx=0.2
    )
    c4 = fake_clip(
        "multi_one_speaks", "ACCEPT", 0.7, 0.02, 0.8, 1.0, n_tracks=2, speaker_side="right", speaker_cx=0.8
    )
    c5 = fake_clip(
        "multi_one_speaks", "ACCEPT", 0.7, 0.02, 0.8, 1.0, n_tracks=2, speaker_side="right", speaker_cx=0.2
    )
    c6 = fake_clip("live_single", "ACCEPT", 0.7, 0.02, 0.9, 1.0)
    c7 = fake_clip("live_single", "ACCEPT", 0.7, 0.02, 0.9, 1.0)
    c8 = fake_clip("not_addressed", "REJECT_NOT_ADDRESSED", 0.7, 0.02, 0.1, 1.0, addressed=False)
    c9 = fake_clip("occlusion_lowlight", "REJECT_ANY", 0.7, 0.02, 0.9, quality=0.2)
    c10 = fake_clip("silence", "NONE", None, 0.0, 0.0, duration=0.0)
    c11 = fake_clip("silence", "NONE", 0.6, 0.01, 0.8, 1.0, duration=1.0)
    c12 = fake_clip("live_single", "ACCEPT", None, 0.0, 0.0, duration=0.0)

    clips = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    outcomes = [decide_clip(c, gate) for c in clips]

    m = compute_metrics(clips, outcomes, spec_cfg)

    assert m["n_clips"] == 12
    assert m["decision_accuracy"] == pytest.approx(9 / 12)
    assert m["far_spoof"] == pytest.approx(0.5)
    assert m["far_all_reject"] == pytest.approx(0.25)
    assert m["frr"] == pytest.approx(1 / 6)
    assert m["liveness"]["precision"] == pytest.approx(6 / 7)
    assert m["liveness"]["recall"] == pytest.approx(6 / 7)
    assert m["attribution_accuracy"] == pytest.approx(2 / 3)
    assert m["addressee_accuracy"] == pytest.approx(1.0)
    assert m["reason_accuracy"] == pytest.approx(2 / 3)
    assert m["false_trigger_rate"] == pytest.approx(0.5)
    assert m["missed_utterance_rate"] == pytest.approx(0.1)

    assert m["per_scenario"]["live_single"]["n"] == 3
    assert m["per_scenario"]["replay_spoof"]["n"] == 2
    assert m["per_scenario"]["multi_one_speaks"]["n"] == 3
    assert m["per_scenario"]["not_addressed"]["n"] == 1
    assert m["per_scenario"]["occlusion_lowlight"]["n"] == 1
    assert m["per_scenario"]["silence"]["n"] == 2

    assert m["confusion"]["ACCEPT"]["ACCEPT"] == 5
    assert m["confusion"]["ACCEPT"]["NONE"] == 1
    assert m["confusion"]["REJECT_SPOOF"]["ACCEPT"] == 1
    assert m["confusion"]["REJECT_SPOOF"]["REJECT_SPOOF"] == 1
    assert m["confusion"]["NONE"]["NONE"] == 1
    assert m["confusion"]["NONE"]["ACCEPT"] == 1


def test_eval_metrics_edge_cases(spec_cfg: Config) -> None:
    # Empty list gives zeros
    m = compute_metrics([], [], spec_cfg)
    assert m["n_clips"] == 0
    assert m["decision_accuracy"] == 0.0
    assert m["far_spoof"] == 0.0
    assert m["far_all_reject"] == 0.0
    assert m["frr"] == 0.0
    assert m["liveness"]["precision"] == 0.0
    assert m["attribution_accuracy"] == 0.0
    assert m["addressee_accuracy"] == 0.0

    # AUC tests
    assert auc([1, 2], [0, 0]) == 1.0
    assert auc([1], [1]) == 0.5

    # side_of boundaries
    assert side_of(0.4) == "center"
    assert side_of(0.6) == "center"
    assert side_of(0.39) == "left"
    assert side_of(0.61) == "right"


def test_json_roundtrip_lossless_with_nan() -> None:
    c = fake_clip("live_single", "ACCEPT", 0.7, 0.02, 0.9, 1.0)
    # inject NaN
    c.utts[0].summary.tracks[0].yaw = float("nan")
    c.lag_samples = [float("nan"), 0.12]

    serialized = records_to_json([c])
    assert "null" in serialized

    restored = records_from_json(serialized)
    assert len(restored) == 1
    r = restored[0]
    assert r.id == c.id
    assert r.scenario == c.scenario
    assert r.expected == c.expected
    assert math.isnan(r.utts[0].summary.tracks[0].yaw)
    assert r.utts[0].summary.tracks[0].pitch == 0.0
    assert math.isnan(r.lag_samples[0])
    assert abs(r.lag_samples[1] - 0.12) < 1e-4
