"""Learned trust gate classifier with feature extraction and probability calibration."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trustsync.config import Config
from trustsync.eval.metrics import ClipRecord
from trustsync.fusion.gate import Gate
from trustsync.types import GateDecision, UtteranceSummary

FEATURE_NAMES = [
    "sync_best",
    "sync_second",
    "margin",
    "lip_std_best",
    "directed_best",
    "abs_yaw_best",
    "abs_pitch_best",
    "quality_max",
    "n_tracks",
    "duration_s",
]


def summary_features(summary: UtteranceSummary) -> np.ndarray:
    """Extract a 10-dimensional feature vector from an UtteranceSummary.

    Features:
        [sync_best, sync_second, margin, lip_std_best, directed_best,
         abs_yaw_best, abs_pitch_best, quality_max, n_tracks, duration_s]
    """
    if not summary.tracks:
        return np.zeros(10, dtype=np.float64)

    sorted_tracks = sorted(summary.tracks, key=lambda t: (-t.sync, t.track_id))
    best = sorted_tracks[0]
    second_sync = sorted_tracks[1].sync if len(sorted_tracks) > 1 else 0.0

    sync_best = float(best.sync)
    margin = float(sync_best - second_sync)
    lip_std_best = float(best.lip_std)
    directed_best = float(best.directed)

    yaw_val = 90.0 if math.isnan(best.yaw) else float(best.yaw)
    pitch_val = 90.0 if math.isnan(best.pitch) else float(best.pitch)
    abs_yaw_best = abs(yaw_val)
    abs_pitch_best = abs(pitch_val)

    quality_max = float(max(t.quality for t in summary.tracks))
    n_tracks = float(len(summary.tracks))
    duration_s = float(summary.duration_s)

    return np.array(
        [
            sync_best,
            second_sync,
            margin,
            lip_std_best,
            directed_best,
            abs_yaw_best,
            abs_pitch_best,
            quality_max,
            n_tracks,
            duration_s,
        ],
        dtype=np.float64,
    )


def train_learned_gate(
    records: list[ClipRecord], cfg: Config, kind: str = "logreg", max_far: float = 0.02
) -> dict[str, Any]:
    """Train a supervised classifier on extracted utterance features and calibrate cutoff threshold."""
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for r in records:
        if r.expected == "NONE" or not r.utts:
            continue
        longest_utt = max(r.utts, key=lambda u: u.t_end - u.t_start)
        if not longest_utt.summary.tracks:
            continue

        feat = summary_features(longest_utt.summary)
        label = 1 if r.expected == "ACCEPT" else 0
        X_list.append(feat)
        y_list.append(label)

    if not X_list:
        raise ValueError("No training samples found in provided records")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)

    if kind == "logreg":
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=1000
                    ),
                ),
            ]
        )
    elif kind == "gboost":
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, random_state=1234
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    model.fit(X, y)

    # Threshold scan over 0.05..0.95 step 0.01 (91 values)
    probs = model.predict_proba(X)[:, 1]
    thresholds = [round(0.05 + 0.01 * i, 2) for i in range(91)]

    candidates: list[tuple[float, float, float]] = []  # (thresh, far, acc)
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        neg_mask = y == 0
        far = (
            float(np.mean(preds[neg_mask] == 1))
            if np.sum(neg_mask) > 0
            else 0.0
        )
        acc = float(np.mean(preds == y))
        candidates.append((thresh, far, acc))

    matching_far = [c for c in candidates if c[1] <= max_far]
    if matching_far:
        best_cand = max(matching_far, key=lambda c: (c[2], -c[1]))
    else:
        min_far = min(c[1] for c in candidates)
        far_matches = [c for c in candidates if c[1] == min_far]
        best_cand = max(far_matches, key=lambda c: c[2])

    chosen_thresh = float(best_cand[0])

    return {
        "model": model,
        "threshold": chosen_thresh,
        "feature_names": FEATURE_NAMES,
        "kind": kind,
    }


def save_gate(bundle: dict[str, Any], path: Path | str) -> None:
    """Save trained model bundle to disk using joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(p))


def load_gate(path: Path | str) -> dict[str, Any]:
    """Load model bundle from disk using joblib."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(str(p))


class LearnedGate(Gate):
    """Supervised decision gate utilizing trained classifier with rule-based fallbacks."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        model_path = Path(cfg.gate.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Gate model file not found: {model_path}")
        bundle = load_gate(model_path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.kind = bundle.get("kind", "learned")

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        """Evaluate utterance summary through learned classifier."""
        # 1. No tracks
        if not summary.tracks:
            return GateDecision(
                decision="REJECT_NO_FACE",
                reason="no tracks detected during utterance",
                speaker_id=None,
                liveness=0.0,
                attribution_conf=0.0,
                addressee=0.0,
            )

        # 2. Max quality < min_quality
        max_qual = max(tr.quality for tr in summary.tracks)
        if max_qual < self.cfg.gate.min_quality:
            return GateDecision(
                decision="REJECT_LOW_QUALITY",
                reason=(
                    f"max quality {max_qual:.2f} below min "
                    f"{self.cfg.gate.min_quality:.2f}"
                ),
                speaker_id=None,
                liveness=0.0,
                attribution_conf=0.0,
                addressee=0.0,
            )

        sorted_tracks = sorted(
            summary.tracks, key=lambda tr: (-tr.sync, tr.track_id)
        )
        best = sorted_tracks[0]
        second = sorted_tracks[1] if len(sorted_tracks) > 1 else None
        second_sync = second.sync if second is not None else 0.0

        liveness = float(best.sync)
        attribution_conf = float(best.sync - second_sync)
        addressee = float(best.directed)

        feats = summary_features(summary).reshape(1, -1)
        p = float(self.model.predict_proba(feats)[0, 1])

        if p >= self.threshold:
            return GateDecision(
                decision="ACCEPT",
                reason=f"learned score {p:.2f} >= threshold {self.threshold:.2f}",
                speaker_id=best.track_id,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # Reject: evaluate rules 3, 4, 6 in RuleGate order with cfg thresholds
        # Rule 3: sync or lip_std
        if (
            best.sync < self.cfg.gate.min_sync
            or best.lip_std < self.cfg.sync.min_lip_std
        ):
            return GateDecision(
                decision="REJECT_SPOOF",
                reason=(
                    f"learned score {p:.2f} rejected; best sync {best.sync:.2f} < "
                    f"{self.cfg.gate.min_sync:.2f} or lip motion {best.lip_std:.4f} < "
                    f"{self.cfg.sync.min_lip_std:.4f}"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # Rule 4: margin
        if (
            len(sorted_tracks) >= 2
            and (best.sync - second_sync) < self.cfg.gate.min_margin
        ):
            return GateDecision(
                decision="REJECT_AMBIGUOUS",
                reason=(
                    f"learned score {p:.2f} rejected; sync margin "
                    f"{best.sync - second_sync:.2f} < {self.cfg.gate.min_margin:.2f}"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # Rule 6: directed
        if best.directed < self.cfg.gate.min_directed:
            return GateDecision(
                decision="REJECT_NOT_ADDRESSED",
                reason=(
                    f"learned score {p:.2f} rejected; directed score "
                    f"{best.directed:.2f} < {self.cfg.gate.min_directed:.2f}"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # If none fails
        return GateDecision(
            decision="REJECT_AMBIGUOUS",
            reason=f"learned score {p:.2f} < threshold {self.threshold:.2f}",
            speaker_id=None,
            liveness=liveness,
            attribution_conf=attribution_conf,
            addressee=addressee,
        )
