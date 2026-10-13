"""Unit tests for RuleGate decision logic."""
from __future__ import annotations

import dataclasses
import pytest

from trustsync.config import Config
from trustsync.fusion.gate import RuleGate
from trustsync.types import TrackSummary, UtteranceSummary


def _make_summary(tracks: list[TrackSummary]) -> UtteranceSummary:
    return UtteranceSummary(t_start=1.0, t_end=3.0, duration_s=2.0, tracks=tracks)


def test_gate_rule_1_no_face(spec_cfg: Config) -> None:
    """Rule 1: no tracks -> REJECT_NO_FACE."""
    gate = RuleGate(spec_cfg)
    dec = gate.decide(_make_summary([]))
    assert dec.decision == "REJECT_NO_FACE"
    assert dec.speaker_id is None


def test_gate_rule_2_low_quality_all(spec_cfg: Config) -> None:
    """Rule 2: max quality over tracks < min_quality -> REJECT_LOW_QUALITY."""
    gate = RuleGate(spec_cfg)
    tr = TrackSummary(
        track_id=0, sync=0.8, lip_std=0.01, directed=0.9, quality=0.3, yaw=0.0, pitch=0.0
    )
    dec = gate.decide(_make_summary([tr]))
    assert dec.decision == "REJECT_LOW_QUALITY"


def test_gate_rule_3_spoof_sync_or_lip_std(spec_cfg: Config) -> None:
    """Rule 3: best.sync < min_sync or best.lip_std < min_lip_std -> REJECT_SPOOF."""
    gate = RuleGate(spec_cfg)
    # Low sync
    tr_low_sync = TrackSummary(
        track_id=0, sync=0.1, lip_std=0.01, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    dec1 = gate.decide(_make_summary([tr_low_sync]))
    assert dec1.decision == "REJECT_SPOOF"

    # Low lip_std (static mouth spoof)
    tr_low_lip = TrackSummary(
        track_id=0, sync=0.8, lip_std=0.001, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    dec2 = gate.decide(_make_summary([tr_low_lip]))
    assert dec2.decision == "REJECT_SPOOF"


def test_gate_rule_4_ambiguous_speaker(spec_cfg: Config) -> None:
    """Rule 4: two or more tracks with margin < min_margin -> REJECT_AMBIGUOUS."""
    gate = RuleGate(spec_cfg)
    tr0 = TrackSummary(
        track_id=0, sync=0.75, lip_std=0.01, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    tr1 = TrackSummary(
        track_id=1, sync=0.70, lip_std=0.01, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    # min_margin is 0.10, margin is 0.05
    dec = gate.decide(_make_summary([tr0, tr1]))
    assert dec.decision == "REJECT_AMBIGUOUS"


def test_gate_rule_5_best_quality(spec_cfg: Config) -> None:
    """Rule 5: best track quality < min_quality -> REJECT_LOW_QUALITY."""
    gate = RuleGate(spec_cfg)
    # tr1 has high quality so rule 2 passes, but best sync is tr0 with quality < 0.5
    tr0 = TrackSummary(
        track_id=0, sync=0.8, lip_std=0.01, directed=0.9, quality=0.3, yaw=0.0, pitch=0.0
    )
    tr1 = TrackSummary(
        track_id=1, sync=0.4, lip_std=0.01, directed=0.9, quality=0.8, yaw=0.0, pitch=0.0
    )
    dec = gate.decide(_make_summary([tr0, tr1]))
    assert dec.decision == "REJECT_LOW_QUALITY"


def test_gate_rule_6_not_addressed(spec_cfg: Config) -> None:
    """Rule 6: best.directed < min_directed -> REJECT_NOT_ADDRESSED."""
    gate = RuleGate(spec_cfg)
    tr = TrackSummary(
        track_id=0, sync=0.8, lip_std=0.01, directed=0.3, quality=0.9, yaw=40.0, pitch=0.0
    )
    dec = gate.decide(_make_summary([tr]))
    assert dec.decision == "REJECT_NOT_ADDRESSED"


def test_gate_rule_7_accept(spec_cfg: Config) -> None:
    """Rule 7: all checks pass -> ACCEPT."""
    gate = RuleGate(spec_cfg)
    tr = TrackSummary(
        track_id=0, sync=0.8, lip_std=0.01, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    dec = gate.decide(_make_summary([tr]))
    assert dec.decision == "ACCEPT"
    assert dec.speaker_id == 0


def test_gate_precedence_rules_2_and_3(spec_cfg: Config) -> None:
    """Precedence: violating rule 2 and 3 gives REJECT_LOW_QUALITY (rule 2 wins)."""
    gate = RuleGate(spec_cfg)
    tr = TrackSummary(
        track_id=0, sync=0.1, lip_std=0.001, directed=0.9, quality=0.2, yaw=0.0, pitch=0.0
    )
    dec = gate.decide(_make_summary([tr]))
    assert dec.decision == "REJECT_LOW_QUALITY"


def test_gate_config_threshold_effect(spec_cfg: Config) -> None:
    """Changing cfg.gate.min_sync flips the decision between ACCEPT and REJECT_SPOOF."""
    tr = TrackSummary(
        track_id=0, sync=0.50, lip_std=0.01, directed=0.9, quality=0.9, yaw=0.0, pitch=0.0
    )
    gate_default = RuleGate(spec_cfg)  # min_sync = 0.35
    assert gate_default.decide(_make_summary([tr])).decision == "ACCEPT"

    cfg_high = dataclasses.replace(
        spec_cfg, gate=dataclasses.replace(spec_cfg.gate, min_sync=0.60)
    )
    gate_high = RuleGate(cfg_high)
    assert gate_high.decide(_make_summary([tr])).decision == "REJECT_SPOOF"
