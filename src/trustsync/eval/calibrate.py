"""Calibration routines for rule thresholds and leave-one-subject-out validation."""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable
import numpy as np

from trustsync.config import Config
from trustsync.eval.metrics import ClipOutcome, ClipRecord, compute_metrics, decide_clip
from trustsync.fusion.gate import RuleGate


def sweep_rule_thresholds(
    records: list[ClipRecord], cfg: Config, max_far: float = 0.02
) -> tuple[Config, list[dict[str, Any]]]:
    """Perform grid search over RuleGate thresholds to optimize accuracy subject to FAR constraint.

    Grid:
        min_sync: 0.05..0.60 step 0.05 (12 values)
        min_margin: (0.0, 0.05, 0.10, 0.15, 0.20)
        min_directed: (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
        sync.min_lip_std: (0.002, 0.003, 0.004, 0.006, 0.008, 0.012)
    """
    sync_vals = [round(0.05 * i, 2) for i in range(1, 13)]
    margin_vals = (0.0, 0.05, 0.10, 0.15, 0.20)
    directed_vals = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    lip_std_vals = (0.002, 0.003, 0.004, 0.006, 0.008, 0.012)

    table: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for s in sync_vals:
        for m in margin_vals:
            for d in directed_vals:
                for ls in lip_std_vals:
                    cand_cfg = dataclasses.replace(
                        cfg,
                        gate=dataclasses.replace(
                            cfg.gate,
                            min_sync=float(s),
                            min_margin=float(m),
                            min_directed=float(d),
                        ),
                        sync=dataclasses.replace(
                            cfg.sync,
                            min_lip_std=float(ls),
                        ),
                    )
                    gate = RuleGate(cand_cfg)
                    outcomes = [decide_clip(rec, gate) for rec in records]
                    metrics = compute_metrics(records, outcomes, cand_cfg)

                    entry = {
                        "min_sync": float(s),
                        "min_margin": float(m),
                        "min_directed": float(d),
                        "min_lip_std": float(ls),
                        "decision_accuracy": float(metrics["decision_accuracy"]),
                        "far_all_reject": float(metrics["far_all_reject"]),
                        "far_spoof": float(metrics["far_spoof"]),
                        "frr": float(metrics["frr"]),
                    }
                    table.append(entry)
                    candidates.append(entry)

    # Selection logic:
    # 1. Among combinations with far_all_reject <= max_far choose highest decision_accuracy.
    #    If none satisfies constraint, choose lowest far_all_reject then highest accuracy.
    matching_far = [c for c in candidates if c["far_all_reject"] <= max_far]
    if matching_far:
        best_acc = max(c["decision_accuracy"] for c in matching_far)
        # Take combinations with FAR rule and within 0.01 of best accuracy
        plateau_set = [
            c
            for c in matching_far
            if c["decision_accuracy"] >= best_acc - 0.01
        ]
    else:
        min_far = min(c["far_all_reject"] for c in candidates)
        far_matches = [c for c in candidates if c["far_all_reject"] == min_far]
        best_acc = max(c["decision_accuracy"] for c in far_matches)
        plateau_set = [
            c
            for c in far_matches
            if c["decision_accuracy"] >= best_acc - 0.01
        ]

    # Find plateau center: nearest to per-parameter median of that set
    med_sync = float(np.median([c["min_sync"] for c in plateau_set]))
    med_margin = float(np.median([c["min_margin"] for c in plateau_set]))
    med_directed = float(np.median([c["min_directed"] for c in plateau_set]))
    med_lip_std = float(np.median([c["min_lip_std"] for c in plateau_set]))

    range_sync = max(sync_vals) - min(sync_vals) or 1.0
    range_margin = max(margin_vals) - min(margin_vals) or 1.0
    range_directed = max(directed_vals) - min(directed_vals) or 1.0
    range_lip_std = max(lip_std_vals) - min(lip_std_vals) or 1.0

    best_cand = min(
        plateau_set,
        key=lambda c: (
            ((c["min_sync"] - med_sync) / range_sync) ** 2
            + ((c["min_margin"] - med_margin) / range_margin) ** 2
            + ((c["min_directed"] - med_directed) / range_directed) ** 2
            + ((c["min_lip_std"] - med_lip_std) / range_lip_std) ** 2
        ),
    )

    new_cfg = dataclasses.replace(
        cfg,
        gate=dataclasses.replace(
            cfg.gate,
            min_sync=float(best_cand["min_sync"]),
            min_margin=float(best_cand["min_margin"]),
            min_directed=float(best_cand["min_directed"]),
        ),
        sync=dataclasses.replace(
            cfg.sync,
            min_lip_std=float(best_cand["min_lip_std"]),
        ),
    )

    return new_cfg, table


def loso(
    records: list[ClipRecord],
    fit_fn: Callable[[list[ClipRecord], Config], Any],
    predict_fn: Callable[[Any, list[ClipRecord], Config], list[ClipOutcome]],
    cfg: Config,
) -> dict[str, Any]:
    """Leave-one-subject-out cross-validation by ClipRecord.subject.

    Returns:
        dict: {'folds': {subject: metrics}, 'pooled': compute_metrics on all held-out outcomes}
    """
    subjects = sorted(list({r.subject for r in records}))
    folds: dict[str, dict[str, Any]] = {}

    all_held_out_records: list[ClipRecord] = []
    all_held_out_outcomes: list[ClipOutcome] = []

    for subj in subjects:
        test_records = [r for r in records if r.subject == subj]
        train_records = [r for r in records if r.subject != subj]

        # Enforce no leakage: a subject's records must never be used in its own fit
        assert all(r.subject != subj for r in train_records), (
            f"Data leakage detected: {subj} found in training records"
        )

        model = fit_fn(train_records, cfg)
        outcomes = predict_fn(model, test_records, cfg)
        fold_metrics = compute_metrics(test_records, outcomes, cfg)
        folds[subj] = fold_metrics

        all_held_out_records.extend(test_records)
        all_held_out_outcomes.extend(outcomes)

    pooled = compute_metrics(all_held_out_records, all_held_out_outcomes, cfg)
    return {"folds": folds, "pooled": pooled}


def estimate_av_lag(records: list[ClipRecord], cfg: Config) -> dict[str, Any]:
    """Estimate audio-visual latency distribution from lag samples.

    Returns:
        dict(median, p10, p90, n, saturated_fraction)
    """
    all_lags: list[float] = []
    for r in records:
        for lag in r.lag_samples:
            if not math.isnan(lag):
                all_lags.append(float(lag))

    n = len(all_lags)
    if n == 0:
        return {
            "median": None,
            "p10": None,
            "p90": None,
            "n": 0,
            "saturated_fraction": 0.0,
        }

    sat_limit = cfg.sync.max_lag_s - (1.0 / cfg.video.fps)
    sat_count = sum(1 for lag in all_lags if abs(lag) >= sat_limit)
    saturated_fraction = sat_count / n

    arr = np.array(all_lags, dtype=np.float64)
    med = float(np.percentile(arr, 50))
    p10 = float(np.percentile(arr, 10))
    p90 = float(np.percentile(arr, 90))

    return {
        "median": med,
        "p10": p10,
        "p90": p90,
        "n": n,
        "saturated_fraction": saturated_fraction,
    }
