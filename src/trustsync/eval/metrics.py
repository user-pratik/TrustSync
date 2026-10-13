"""Evaluation metrics, data structures, and serialization for TrustSync."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from trustsync.config import Config
from trustsync.fusion.gate import Gate
from trustsync.types import GateDecision, TrackSummary, UtteranceSummary


@dataclass
class UttRecord:
    """Per-utterance record with track summaries and face centroids at utterance_final."""

    t_start: float
    t_end: float
    summary: UtteranceSummary
    face_cx: dict[int, float]  # bbox centre x per track at the utterance_final frame


@dataclass
class ClipRecord:
    """Evaluation clip record containing metadata, utterances, and AV lag samples."""

    id: str
    scenario: str
    subject: str
    subjects: list[str]
    expected: str
    speaker_side: str
    addressed: bool
    utts: list[UttRecord]
    lag_samples: list[float]


@dataclass
class ClipOutcome:
    """Outcome of evaluating a clip through a trust gate."""

    decision: str
    speaker_id: int | None
    speaker_side: str | None
    summary: UtteranceSummary | None
    gd: GateDecision | None


def side_of(cx: float) -> str:
    """Determine spatial side relative to robot camera.

    Args:
        cx: Normalized horizontal center coordinate in [0, 1].

    Returns:
        'left' if cx < 0.4, 'right' if cx > 0.6, else 'center'.
    """
    if cx < 0.4:
        return "left"
    if cx > 0.6:
        return "right"
    return "center"


def _sanitize_for_json(obj: Any) -> Any:
    """Convert floats (including NaN) to JSON-serializable values."""
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def records_to_json(records: list[ClipRecord]) -> str:
    """Serialize list of ClipRecord losslessly to JSON, storing NaN as null."""
    data = []
    for r in records:
        rec_dict = {
            "id": r.id,
            "scenario": r.scenario,
            "subject": r.subject,
            "subjects": r.subjects,
            "expected": r.expected,
            "speaker_side": r.speaker_side,
            "addressed": r.addressed,
            "lag_samples": r.lag_samples,
            "utts": [
                {
                    "t_start": u.t_start,
                    "t_end": u.t_end,
                    "face_cx": {str(k): v for k, v in u.face_cx.items()},
                    "summary": {
                        "t_start": u.summary.t_start,
                        "t_end": u.summary.t_end,
                        "duration_s": u.summary.duration_s,
                        "tracks": [asdict(t) for t in u.summary.tracks],
                    },
                }
                for u in r.utts
            ],
        }
        data.append(rec_dict)
    sanitized = _sanitize_for_json(data)
    return json.dumps(sanitized, indent=2)


def records_from_json(text: str) -> list[ClipRecord]:
    """Deserialize JSON text to list of ClipRecord, restoring null to NaN in float fields."""
    raw_list = json.loads(text)
    records: list[ClipRecord] = []

    def _to_float(v: Any) -> float:
        return float("nan") if v is None else float(v)

    for item in raw_list:
        utts: list[UttRecord] = []
        for u in item.get("utts", []):
            tracks: list[TrackSummary] = []
            for t in u.get("summary", {}).get("tracks", []):
                tracks.append(
                    TrackSummary(
                        track_id=int(t["track_id"]),
                        sync=_to_float(t.get("sync")),
                        lip_std=_to_float(t.get("lip_std")),
                        directed=_to_float(t.get("directed")),
                        quality=_to_float(t.get("quality")),
                        yaw=_to_float(t.get("yaw")),
                        pitch=_to_float(t.get("pitch")),
                    )
                )
            summary_dict = u.get("summary", {})
            summary = UtteranceSummary(
                t_start=float(summary_dict.get("t_start", 0.0)),
                t_end=float(summary_dict.get("t_end", 0.0)),
                duration_s=float(summary_dict.get("duration_s", 0.0)),
                tracks=tracks,
            )
            face_cx = {
                int(k): _to_float(v) for k, v in u.get("face_cx", {}).items()
            }
            utts.append(
                UttRecord(
                    t_start=float(u["t_start"]),
                    t_end=float(u["t_end"]),
                    summary=summary,
                    face_cx=face_cx,
                )
            )

        lag_samples = [_to_float(x) for x in item.get("lag_samples", [])]
        subjects = list(item.get("subjects", []))
        subject = item.get("subject", subjects[0] if subjects else "")
        records.append(
            ClipRecord(
                id=str(item["id"]),
                scenario=str(item["scenario"]),
                subject=subject,
                subjects=subjects,
                expected=str(item["expected"]),
                speaker_side=str(item.get("speaker_side", "center")),
                addressed=bool(item.get("addressed", True)),
                utts=utts,
                lag_samples=lag_samples,
            )
        )
    return records


def decide_clip(rec: ClipRecord, gate: Gate) -> ClipOutcome:
    """Decide clip outcome using longest utterance and specified gate."""
    if not rec.utts:
        return ClipOutcome("NONE", None, None, None, None)

    # Pick utterance with longest duration
    longest_utt = max(rec.utts, key=lambda u: u.t_end - u.t_start)
    gd = gate.decide(longest_utt.summary)

    speaker_side: str | None = None
    if gd.speaker_id is not None and gd.speaker_id in longest_utt.face_cx:
        speaker_side = side_of(longest_utt.face_cx[gd.speaker_id])

    return ClipOutcome(
        decision=gd.decision,
        speaker_id=gd.speaker_id,
        speaker_side=speaker_side,
        summary=longest_utt.summary,
        gd=gd,
    )


def auc(pos: list[float], neg: list[float]) -> float:
    """Compute Area Under ROC Curve via rank formula with ties counting half."""
    if not pos or not neg:
        return 0.0

    n1 = len(pos)
    n2 = len(neg)
    s = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                s += 1.0
            elif p == n:
                s += 0.5
    return s / (n1 * n2)


def compute_metrics(
    records: list[ClipRecord], outcomes: list[ClipOutcome], cfg: Config
) -> dict:
    """Compute comprehensive evaluation metrics for recorded sessions and gate outcomes.

    Definitions:
        - A clip is 'should accept' if expected == ACCEPT.
        - A clip is 'should reject' if expected in (REJECT_SPOOF, REJECT_NOT_ADDRESSED, REJECT_ANY).
        - Silence (expected NONE) must produce no utterance.
        - decision_accuracy: share of clips where accept/reject agrees (for NONE: outcome decision is NONE).
        - far_spoof: share of replay_spoof clips whose decision is ACCEPT.
        - far_all_reject: share of should-reject clips accepted.
        - frr: share of should-accept clips not accepted.
        - false_trigger_rate: share of silence clips with at least one utterance.
        - missed_utterance_rate: share of non-silence clips with no utterance.
        - liveness: positive = live; truth live for scenarios live_single, multi_one_speaks,
          not_addressed and not live for replay_spoof (other scenarios excluded);
          predicted live = best-sync track has sync >= cfg.gate.min_sync and
          lip_std >= cfg.sync.min_lip_std (no utterance = not live).
        - attribution_accuracy: over multi_one_speaks clips with an utterance, the share
          where outcome speaker_side equals clip speaker_side (independent of accept).
        - addressee_accuracy: over live_single and not_addressed clips with an utterance,
          share where (best-sync track directed >= cfg.gate.min_directed) equals clip.addressed.
        - reason_accuracy: over replay_spoof and not_addressed clips, share where decision string
          equals expected.
    """
    n_clips = len(records)
    if n_clips == 0 or len(outcomes) != n_clips:
        return {
            "n_clips": n_clips,
            "decision_accuracy": 0.0,
            "far_spoof": 0.0,
            "far_all_reject": 0.0,
            "frr": 0.0,
            "false_trigger_rate": 0.0,
            "missed_utterance_rate": 0.0,
            "liveness": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "attribution_accuracy": 0.0,
            "addressee_accuracy": 0.0,
            "reason_accuracy": 0.0,
            "per_scenario": {},
            "confusion": {},
        }

    # 1. Decision accuracy
    correct_decision = 0
    for r, o in zip(records, outcomes):
        if r.expected == "ACCEPT":
            if o.decision == "ACCEPT":
                correct_decision += 1
        elif r.expected in ("REJECT_SPOOF", "REJECT_NOT_ADDRESSED", "REJECT_ANY"):
            if o.decision.startswith("REJECT_"):
                correct_decision += 1
        elif r.expected == "NONE":
            if o.decision == "NONE":
                correct_decision += 1
    decision_accuracy = correct_decision / n_clips if n_clips > 0 else 0.0

    # 2. FAR spoof
    spoof_clips = [
        (r, o) for r, o in zip(records, outcomes) if r.scenario == "replay_spoof"
    ]
    far_spoof = (
        sum(1 for _, o in spoof_clips if o.decision == "ACCEPT") / len(spoof_clips)
        if spoof_clips
        else 0.0
    )

    # 3. FAR all reject
    should_reject = [
        (r, o)
        for r, o in zip(records, outcomes)
        if r.expected in ("REJECT_SPOOF", "REJECT_NOT_ADDRESSED", "REJECT_ANY")
    ]
    far_all_reject = (
        sum(1 for _, o in should_reject if o.decision == "ACCEPT") / len(should_reject)
        if should_reject
        else 0.0
    )

    # 4. FRR
    should_accept = [(r, o) for r, o in zip(records, outcomes) if r.expected == "ACCEPT"]
    frr = (
        sum(1 for _, o in should_accept if o.decision != "ACCEPT") / len(should_accept)
        if should_accept
        else 0.0
    )

    # 5. False trigger rate (silence clips with >= 1 utterance)
    silence_clips = [(r, o) for r, o in zip(records, outcomes) if r.scenario == "silence"]
    false_trigger_rate = (
        sum(1 for r, _ in silence_clips if len(r.utts) > 0) / len(silence_clips)
        if silence_clips
        else 0.0
    )

    # 6. Missed utterance rate (non-silence clips with no utterance)
    non_silence_clips = [(r, o) for r, o in zip(records, outcomes) if r.scenario != "silence"]
    missed_utterance_rate = (
        sum(1 for r, _ in non_silence_clips if len(r.utts) == 0) / len(non_silence_clips)
        if non_silence_clips
        else 0.0
    )

    # 7. Liveness precision, recall, f1
    tp = fp = fn = 0
    for r, o in zip(records, outcomes):
        if r.scenario not in ("live_single", "multi_one_speaks", "not_addressed", "replay_spoof"):
            continue
        truth_live = r.scenario != "replay_spoof"
        pred_live = False
        if o.summary is not None and o.summary.tracks:
            best_t = max(o.summary.tracks, key=lambda t: t.sync)
            if best_t.sync >= cfg.gate.min_sync and best_t.lip_std >= cfg.sync.min_lip_std:
                pred_live = True

        if truth_live and pred_live:
            tp += 1
        elif not truth_live and pred_live:
            fp += 1
        elif truth_live and not pred_live:
            fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    liveness = {"precision": prec, "recall": rec, "f1": f1}

    # 8. Attribution accuracy (multi_one_speaks clips with an utterance)
    multi_utts = [
        (r, o)
        for r, o in zip(records, outcomes)
        if r.scenario == "multi_one_speaks" and len(r.utts) > 0
    ]
    attribution_accuracy = (
        sum(1 for r, o in multi_utts if o.speaker_side == r.speaker_side) / len(multi_utts)
        if multi_utts
        else 0.0
    )

    # 9. Addressee accuracy (live_single and not_addressed clips with an utterance)
    addr_clips = [
        (r, o)
        for r, o in zip(records, outcomes)
        if r.scenario in ("live_single", "not_addressed") and len(r.utts) > 0
    ]
    addr_correct = 0
    for r, o in addr_clips:
        if o.summary is not None and o.summary.tracks:
            best_t = max(o.summary.tracks, key=lambda t: t.sync)
            pred_addr = best_t.directed >= cfg.gate.min_directed
            if pred_addr == r.addressed:
                addr_correct += 1
    addressee_accuracy = addr_correct / len(addr_clips) if addr_clips else 0.0

    # 10. Reason accuracy (replay_spoof and not_addressed clips)
    reason_clips = [
        (r, o)
        for r, o in zip(records, outcomes)
        if r.scenario in ("replay_spoof", "not_addressed")
    ]
    reason_accuracy = (
        sum(1 for r, o in reason_clips if o.decision == r.expected) / len(reason_clips)
        if reason_clips
        else 0.0
    )

    # 11. Per-scenario statistics
    scenarios = {r.scenario for r in records}
    per_scenario: dict[str, dict[str, Any]] = {}
    for sc in sorted(scenarios):
        sc_pairs = [(r, o) for r, o in zip(records, outcomes) if r.scenario == sc]
        n_sc = len(sc_pairs)
        n_acc = sum(1 for _, o in sc_pairs if o.decision == "ACCEPT")
        n_correct = 0
        for r, o in sc_pairs:
            if r.expected == "ACCEPT":
                if o.decision == "ACCEPT":
                    n_correct += 1
            elif r.expected in ("REJECT_SPOOF", "REJECT_NOT_ADDRESSED", "REJECT_ANY"):
                if o.decision.startswith("REJECT_"):
                    n_correct += 1
            elif r.expected == "NONE":
                if o.decision == "NONE":
                    n_correct += 1
        per_scenario[sc] = {
            "n": n_sc,
            "accuracy": n_correct / n_sc if n_sc > 0 else 0.0,
            "accept_rate": n_acc / n_sc if n_sc > 0 else 0.0,
        }

    # 12. Confusion matrix
    confusion: dict[str, dict[str, int]] = {}
    for r, o in zip(records, outcomes):
        exp = r.expected
        dec = o.decision
        if exp not in confusion:
            confusion[exp] = {}
        confusion[exp][dec] = confusion[exp].get(dec, 0) + 1

    return {
        "n_clips": n_clips,
        "decision_accuracy": decision_accuracy,
        "far_spoof": far_spoof,
        "far_all_reject": far_all_reject,
        "frr": frr,
        "false_trigger_rate": false_trigger_rate,
        "missed_utterance_rate": missed_utterance_rate,
        "liveness": liveness,
        "attribution_accuracy": attribution_accuracy,
        "addressee_accuracy": addressee_accuracy,
        "reason_accuracy": reason_accuracy,
        "per_scenario": per_scenario,
        "confusion": confusion,
    }
