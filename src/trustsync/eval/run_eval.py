"""Session processing, caching, evaluation, ablation, and full CLI runner."""
from __future__ import annotations

import argparse
import dataclasses
import difflib
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from trustsync.audio.sources import FileAudioSource
from trustsync.config import Config, load_config, save_config
from trustsync.eval.calibrate import estimate_av_lag, loso, sweep_rule_thresholds
from trustsync.eval.metrics import (
    ClipOutcome,
    ClipRecord,
    UttRecord,
    compute_metrics,
    decide_clip,
    records_from_json,
    records_to_json,
)
from trustsync.eval.plots import (
    plot_ablation,
    plot_box_by_scenario,
    plot_confusion,
    plot_lag_hist,
    plot_roc,
    plot_timeline,
)
from trustsync.fusion.gate import Gate, RuleGate
from trustsync.fusion.learned import (
    LearnedGate,
    save_gate,
    summary_features,
    train_learned_gate,
)
from trustsync.io.video_source import FileVideoSource
from trustsync.pipeline import TrustSyncPipeline
from trustsync.types import GateDecision, UtteranceSummary


class RecordingGate(Gate):
    """Gate wrapper that records all evaluated utterance summaries."""

    def __init__(self, inner: Gate) -> None:
        self.inner = inner
        self.summaries: list[UtteranceSummary] = []

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        self.summaries.append(summary)
        return self.inner.decide(summary)


# ---------------------------------------------------------
# Ablation Gates
# ---------------------------------------------------------

class AcceptAll(Gate):
    """B0: Naive voice interface that accepts if any utterance occurs."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        if not summary.tracks:
            return GateDecision(
                decision="ACCEPT",
                reason="naive voice interface: accept any utterance",
                speaker_id=None,
                liveness=0.0,
                attribution_conf=0.0,
                addressee=0.0,
            )
        best = max(summary.tracks, key=lambda t: t.sync)
        return GateDecision(
            decision="ACCEPT",
            reason="naive voice interface: accept any utterance",
            speaker_id=best.track_id,
            liveness=best.sync,
            attribution_conf=0.0,
            addressee=best.directed,
        )


class SyncOnly(Gate):
    """A1: Evaluates only lip-audio sync score against threshold."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        if not summary.tracks:
            return GateDecision("REJECT_NO_FACE", "no tracks", None, 0.0, 0.0, 0.0)
        best = max(summary.tracks, key=lambda t: t.sync)
        if best.sync < self.cfg.gate.min_sync:
            return GateDecision(
                decision="REJECT_SPOOF",
                reason=f"sync {best.sync:.2f} < {self.cfg.gate.min_sync:.2f}",
                speaker_id=None,
                liveness=best.sync,
                attribution_conf=0.0,
                addressee=best.directed,
            )
        return GateDecision(
            decision="ACCEPT",
            reason=f"sync {best.sync:.2f} >= min_sync",
            speaker_id=best.track_id,
            liveness=best.sync,
            attribution_conf=0.0,
            addressee=best.directed,
        )


class SyncMargin(Gate):
    """A2: Evaluates sync score and speaker attribution margin."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        if not summary.tracks:
            return GateDecision("REJECT_NO_FACE", "no tracks", None, 0.0, 0.0, 0.0)
        sorted_tracks = sorted(summary.tracks, key=lambda t: (-t.sync, t.track_id))
        best = sorted_tracks[0]
        second = sorted_tracks[1] if len(sorted_tracks) > 1 else None
        second_sync = second.sync if second is not None else 0.0
        margin = best.sync - second_sync

        if best.sync < self.cfg.gate.min_sync:
            return GateDecision(
                "REJECT_SPOOF",
                f"sync {best.sync:.2f} < {self.cfg.gate.min_sync:.2f}",
                None,
                best.sync,
                margin,
                best.directed,
            )
        if len(sorted_tracks) >= 2 and margin < self.cfg.gate.min_margin:
            return GateDecision(
                "REJECT_AMBIGUOUS",
                f"margin {margin:.2f} < {self.cfg.gate.min_margin:.2f}",
                None,
                best.sync,
                margin,
                best.directed,
            )
        return GateDecision(
            "ACCEPT",
            f"sync and margin pass",
            best.track_id,
            best.sync,
            margin,
            best.directed,
        )


class SyncMarginAddressee(Gate):
    """A3: Evaluates sync score, attribution margin, and addressee score."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        if not summary.tracks:
            return GateDecision("REJECT_NO_FACE", "no tracks", None, 0.0, 0.0, 0.0)
        sorted_tracks = sorted(summary.tracks, key=lambda t: (-t.sync, t.track_id))
        best = sorted_tracks[0]
        second = sorted_tracks[1] if len(sorted_tracks) > 1 else None
        second_sync = second.sync if second is not None else 0.0
        margin = best.sync - second_sync

        if best.sync < self.cfg.gate.min_sync:
            return GateDecision("REJECT_SPOOF", "sync reject", None, best.sync, margin, best.directed)
        if len(sorted_tracks) >= 2 and margin < self.cfg.gate.min_margin:
            return GateDecision("REJECT_AMBIGUOUS", "margin reject", None, best.sync, margin, best.directed)
        if best.directed < self.cfg.gate.min_directed:
            return GateDecision("REJECT_NOT_ADDRESSED", "addressee reject", None, best.sync, margin, best.directed)

        return GateDecision("ACCEPT", "accepted", best.track_id, best.sync, margin, best.directed)


class BundleGate(Gate):
    """Temporary evaluator for a trained model bundle."""

    def __init__(self, cfg: Config, bundle: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        if not summary.tracks:
            return GateDecision("REJECT_NO_FACE", "no tracks", None, 0.0, 0.0, 0.0)
        max_qual = max(tr.quality for tr in summary.tracks)
        if max_qual < self.cfg.gate.min_quality:
            return GateDecision("REJECT_LOW_QUALITY", "low quality", None, 0.0, 0.0, 0.0)

        sorted_tracks = sorted(summary.tracks, key=lambda tr: (-tr.sync, tr.track_id))
        best = sorted_tracks[0]
        second = sorted_tracks[1] if len(sorted_tracks) > 1 else None
        second_sync = second.sync if second is not None else 0.0
        margin = float(best.sync - second_sync)

        feats = summary_features(summary).reshape(1, -1)
        p = float(self.model.predict_proba(feats)[0, 1])

        if p >= self.threshold:
            return GateDecision("ACCEPT", f"score {p:.2f}", best.track_id, best.sync, margin, best.directed)

        if best.sync < self.cfg.gate.min_sync or best.lip_std < self.cfg.sync.min_lip_std:
            return GateDecision("REJECT_SPOOF", f"score {p:.2f} spoof", None, best.sync, margin, best.directed)
        if len(sorted_tracks) >= 2 and margin < self.cfg.gate.min_margin:
            return GateDecision("REJECT_AMBIGUOUS", f"score {p:.2f} margin", None, best.sync, margin, best.directed)
        if best.directed < self.cfg.gate.min_directed:
            return GateDecision("REJECT_NOT_ADDRESSED", f"score {p:.2f} addressee", None, best.sync, margin, best.directed)

        return GateDecision("REJECT_AMBIGUOUS", f"score {p:.2f} rejected", None, best.sync, margin, best.directed)


def build_ablation(
    records: list[ClipRecord], cfg: Config, learned_bundle: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Build ablation study table comparing B0, A1, A2, A3, A4, A5."""
    learned_gate = BundleGate(cfg, learned_bundle) if learned_bundle else RuleGate(cfg)

    variants: list[tuple[str, Gate]] = [
        ("B0", AcceptAll(cfg)),
        ("A1", SyncOnly(cfg)),
        ("A2", SyncMargin(cfg)),
        ("A3", SyncMarginAddressee(cfg)),
        ("A4", RuleGate(cfg)),
        ("A5", learned_gate),
    ]

    rows: list[dict[str, Any]] = []
    for name, gate in variants:
        outcomes = [decide_clip(r, gate) for r in records]
        m = compute_metrics(records, outcomes, cfg)
        rows.append({
            "name": name,
            "accuracy": float(m["decision_accuracy"]),
            "far_all_reject": float(m["far_all_reject"]),
            "frr": float(m["frr"]),
            "liveness_f1": float(m["liveness"]["f1"]),
        })

    return rows


# ---------------------------------------------------------
# Processing and Cache
# ---------------------------------------------------------

def compute_cache_hash(cfg: Config) -> str:
    """Compute sha256 hash of configuration sections affecting perception features."""
    cfg_dict = {
        "vad": dataclasses.asdict(cfg.vad),
        "utterance": dataclasses.asdict(cfg.utterance),
        "tracker": dataclasses.asdict(cfg.tracker),
        "lip": dataclasses.asdict(cfg.lip),
        "sync": dataclasses.asdict(cfg.sync),
        "quality": dataclasses.asdict(cfg.quality),
        "addressee": dataclasses.asdict(cfg.addressee),
        "video": dataclasses.asdict(cfg.video),
    }
    raw = json.dumps(cfg_dict, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def process_session(session_dir: Path, cfg: Config) -> ClipRecord:
    """Process a session directory through TrustSync pipeline to produce a ClipRecord."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    video_source = FileVideoSource(session_dir / "video.mp4", session_dir / "frame_times.csv")
    audio_source = FileAudioSource(
        session_dir / "audio.wav", t0=float(meta["audio_t0"]), cfg=cfg.audio
    )

    rec_gate = RecordingGate(RuleGate(cfg))
    pipeline = TrustSyncPipeline(cfg, audio=audio_source, gate=rec_gate)

    utts: list[UttRecord] = []
    lag_samples: list[float] = []

    while True:
        ok, frame, t = video_source.read()
        if not ok or frame is None:
            break

        res = pipeline.process(frame, t)

        if res.decision == "LISTENING" and res.faces:
            best_face = max(res.faces, key=lambda f: f.sync)
            if best_face.sync > 0.5:
                lag_samples.append(float(best_face.best_lag_s))

        if res.utterance_final and rec_gate.summaries:
            summary = rec_gate.summaries[-1]
            face_cx: dict[int, float] = {}
            for f_score in res.faces:
                cx = (f_score.bbox[0] + f_score.bbox[2]) / 2.0
                face_cx[f_score.track_id] = float(cx)

            utts.append(
                UttRecord(
                    t_start=float(summary.t_start),
                    t_end=float(summary.t_end),
                    summary=summary,
                    face_cx=face_cx,
                )
            )

    subjects = list(meta.get("subjects", []))
    subject = subjects[0] if subjects else "unknown"

    return ClipRecord(
        id=str(meta["id"]),
        scenario=str(meta["scenario"]),
        subject=subject,
        subjects=subjects,
        expected=str(meta["expected"]),
        speaker_side=str(meta.get("speaker_side", "center")),
        addressed=bool(meta.get("addressed", True)),
        utts=utts,
        lag_samples=lag_samples,
    )


def load_or_process(session_dir: Path, cfg: Config, cache_dir: Path) -> ClipRecord:
    """Load cached ClipRecord or process session and save to cache."""
    session_dir = Path(session_dir)
    cache_dir = Path(cache_dir)
    meta_path = session_dir / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = meta["id"]
    cache_file = cache_dir / f"{session_id}.json"
    current_hash = compute_cache_hash(cfg)

    if cache_file.is_file():
        cache_mtime = cache_file.stat().st_mtime
        session_files = [p for p in session_dir.iterdir() if p.is_file()]
        max_session_mtime = max((p.stat().st_mtime for p in session_files), default=0.0)

        if cache_mtime > max_session_mtime:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("config_hash") == current_hash:
                    records = records_from_json(json.dumps([cached_data["record"]]))
                    if records:
                        return records[0]
            except Exception:
                pass

    rec = process_session(session_dir, cfg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.loads(records_to_json([rec]))[0]
    payload = {
        "config_hash": current_hash,
        "record": serialized,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return rec


# ---------------------------------------------------------
# CLI Main
# ---------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TrustSync offline evaluation and calibration.")
    parser.add_argument("--sessions", default="data/sessions", help="Sessions directory")
    parser.add_argument("--records-json", default=None, help="Pre-computed records JSON file")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--cache", default="out/cache", help="Cache directory")
    parser.add_argument("--calibrate", action="store_true", help="Perform threshold calibration sweep")
    parser.add_argument("--learned", action="store_true", help="Train and validate learned gate")
    parser.add_argument("--write-config", action="store_true", help="Update config.yaml with calibrated values")
    args = parser.parse_args(argv)

    root = Path.cwd()
    cfg = load_config()

    # 1. Load or process records
    if args.records_json:
        records_path = Path(args.records_json)
        records = records_from_json(records_path.read_text(encoding="utf-8"))
    else:
        sessions_path = Path(args.sessions)
        if not sessions_path.is_dir():
            print(f"Sessions directory {sessions_path} does not exist.")
            records = []
        else:
            session_dirs = [
                d for d in sessions_path.iterdir()
                if d.is_dir() and (d / "meta.json").is_file()
            ]
            records = [load_or_process(d, cfg, Path(args.cache)) for d in session_dirs]

    # 2. Current metrics
    current_outcomes = [decide_clip(r, RuleGate(cfg)) for r in records]
    current_metrics = compute_metrics(records, current_outcomes, cfg)

    # 3. Calibrate
    calibrated_cfg = cfg
    calibrated_metrics: dict[str, Any] | None = None
    loso_rule: dict[str, Any] | None = None

    if args.calibrate and records:
        calibrated_cfg, sweep_table = sweep_rule_thresholds(records, cfg)
        cal_outcomes = [decide_clip(r, RuleGate(calibrated_cfg)) for r in records]
        calibrated_metrics = compute_metrics(records, cal_outcomes, calibrated_cfg)
        calibrated_metrics["thresholds"] = {
            "min_sync": calibrated_cfg.gate.min_sync,
            "min_margin": calibrated_cfg.gate.min_margin,
            "min_directed": calibrated_cfg.gate.min_directed,
            "min_lip_std": calibrated_cfg.sync.min_lip_std,
        }

        # LOSO with sweep
        def fit_sweep(train_recs: list[ClipRecord], base_cfg: Config) -> Gate:
            c, _ = sweep_rule_thresholds(train_recs, base_cfg)
            return RuleGate(c)

        def pred_sweep(model: Gate, test_recs: list[ClipRecord], base_cfg: Config) -> list[ClipOutcome]:
            return [decide_clip(r, model) for r in test_recs]

        loso_rule = loso(records, fit_sweep, pred_sweep, cfg)

    # 4. Learned gate
    loso_learned: dict[str, Any] | None = None
    learned_kind: str | None = None
    learned_bundle: dict[str, Any] | None = None

    if args.learned and records:
        best_acc = -1.0
        best_far = 999.0

        for kind in ("logreg", "gboost"):
            try:
                bundle = train_learned_gate(records, calibrated_cfg, kind=kind)

                def fit_k(train_recs: list[ClipRecord], c: Config) -> dict[str, Any]:
                    return train_learned_gate(train_recs, c, kind=kind)

                def pred_k(m_bundle: dict[str, Any], test_recs: list[ClipRecord], c: Config) -> list[ClipOutcome]:
                    g = BundleGate(c, m_bundle)
                    return [decide_clip(r, g) for r in test_recs]

                loso_res = loso(records, fit_k, pred_k, calibrated_cfg)
                pooled_acc = loso_res["pooled"]["decision_accuracy"]
                pooled_far = loso_res["pooled"]["far_all_reject"]

                # Selection: higher pooled accuracy subject to pooled FAR <= 0.02 (else lower FAR)
                is_better = False
                if pooled_far <= 0.02:
                    if best_far > 0.02 or pooled_acc > best_acc:
                        is_better = True
                else:
                    if best_far > 0.02 and pooled_far < best_far:
                        is_better = True

                if is_better:
                    best_acc = pooled_acc
                    best_far = pooled_far
                    loso_learned = loso_res
                    learned_kind = kind
                    learned_bundle = bundle
            except Exception as e:
                print(f"Warning: training {kind} failed: {e}")

        if learned_bundle is not None:
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)
            save_gate(learned_bundle, models_dir / "gate.joblib")

    # 5. Ablation study
    ablation_rows = build_ablation(records, calibrated_cfg, learned_bundle)

    # 6. AV Latency estimation
    av_lag_res = estimate_av_lag(records, calibrated_cfg)

    # 7. Generate report figures
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ROC sync
    pos_sync = [
        max(u.summary.tracks, key=lambda t: t.sync).sync
        for r in records
        if r.scenario in ("live_single", "multi_one_speaks", "not_addressed") and r.utts
        for u in [max(r.utts, key=lambda ut: ut.t_end - ut.t_start)]
        if u.summary.tracks
    ]
    neg_sync = [
        max(u.summary.tracks, key=lambda t: t.sync).sync
        for r in records
        if r.scenario == "replay_spoof" and r.utts
        for u in [max(r.utts, key=lambda ut: ut.t_end - ut.t_start)]
        if u.summary.tracks
    ]
    p_roc_sync = plot_roc(pos_sync, neg_sync, "ROC: Lip-Audio Sync", fig_dir / "roc_sync.png")

    # ROC directed
    pos_dir = [
        max(u.summary.tracks, key=lambda t: t.sync).directed
        for r in records
        if r.addressed and r.utts
        for u in [max(r.utts, key=lambda ut: ut.t_end - ut.t_start)]
        if u.summary.tracks
    ]
    neg_dir = [
        max(u.summary.tracks, key=lambda t: t.sync).directed
        for r in records
        if not r.addressed and r.utts
        for u in [max(r.utts, key=lambda ut: ut.t_end - ut.t_start)]
        if u.summary.tracks
    ]
    p_roc_dir = plot_roc(pos_dir, neg_dir, "ROC: Addressee Directed Score", fig_dir / "roc_directed.png")

    # Box by scenario
    vals_by_sc: dict[str, list[float]] = {}
    for r in records:
        if r.utts:
            u = max(r.utts, key=lambda ut: ut.t_end - ut.t_start)
            if u.summary.tracks:
                sync_val = max(u.summary.tracks, key=lambda t: t.sync).sync
                vals_by_sc.setdefault(r.scenario, []).append(sync_val)
    p_box = plot_box_by_scenario(
        vals_by_sc, "Lip-Audio Sync Score", calibrated_cfg.gate.min_sync, fig_dir / "sync_by_scenario.png"
    )

    p_conf = plot_confusion(current_metrics["confusion"], fig_dir / "confusion.png")
    p_abl = plot_ablation(ablation_rows, fig_dir / "ablation.png")

    all_lags = [lag for r in records for lag in r.lag_samples if not math.isnan(lag)]
    p_lag = plot_lag_hist(all_lags, calibrated_cfg.sync.max_lag_s, fig_dir / "lag_hist.png")

    figures = [
        {"name": "roc_sync", "path": str(p_roc_sync), "caption": "ROC curve for lip-audio synchrony."},
        {"name": "roc_directed", "path": str(p_roc_dir), "caption": "ROC curve for addressee classification."},
        {"name": "sync_by_scenario", "path": str(p_box), "caption": "Sync score distribution across scenarios."},
        {"name": "confusion", "path": str(p_conf), "caption": "Decision confusion matrix."},
        {"name": "ablation", "path": str(p_abl), "caption": "Ablation study of decision gate variants."},
        {"name": "lag_hist", "path": str(p_lag), "caption": "Observed audio-visual latency histogram."},
    ]

    # Scenario timelines
    scenarios_seen = set()
    for r in records:
        if r.scenario not in scenarios_seen:
            scenarios_seen.add(r.scenario)
            timeline_file = Path("out") / r.id / "frames.csv"
            p_time = plot_timeline(timeline_file, f"Timeline: {r.scenario}", fig_dir / f"timeline_{r.scenario}.png")
            figures.append({
                "name": f"timeline_{r.scenario}",
                "path": str(p_time),
                "caption": f"Per-frame temporal signal timeline for {r.scenario}.",
            })

    # 8. Write results.json
    per_sc_counts: dict[str, int] = {}
    for r in records:
        per_sc_counts[r.scenario] = per_sc_counts.get(r.scenario, 0) + 1

    subjects_list = sorted(list({r.subject for r in records}))

    results_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_clips": len(records),
            "per_scenario_counts": per_sc_counts,
            "subjects": subjects_list,
        },
        "config": {
            "gate": dataclasses.asdict(calibrated_cfg.gate),
            "min_lip_std": calibrated_cfg.sync.min_lip_std,
            "addressee": dataclasses.asdict(calibrated_cfg.addressee),
        },
        "current": current_metrics,
        "calibrated": calibrated_metrics,
        "loso_rule": loso_rule,
        "loso_learned": loso_learned,
        "learned_kind": learned_kind,
        "ablation": ablation_rows,
        "av_lag": av_lag_res,
        "fps": None,
        "figures": figures,
        "notes": ["Phase 8 evaluation run"],
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"Saved evaluation results to {results_path}")

    # 9. Handle --write-config
    if args.write_config:
        cfg_file = Path("config.yaml")
        if cfg_file.is_file():
            bak_file = Path("config.yaml.bak")
            shutil.copyfile(cfg_file, bak_file)

            old_text = cfg_file.read_text(encoding="utf-8")
            mode_to_set = "rule"
            if args.learned and loso_learned and loso_learned["pooled"]["far_all_reject"] <= 0.02:
                mode_to_set = "learned"

            new_gate_cfg = dataclasses.replace(
                cfg.gate,
                min_sync=calibrated_cfg.gate.min_sync,
                min_margin=calibrated_cfg.gate.min_margin,
                min_directed=calibrated_cfg.gate.min_directed,
                mode=mode_to_set,
            )
            updated_cfg = dataclasses.replace(
                cfg,
                gate=new_gate_cfg,
                sync=dataclasses.replace(cfg.sync, min_lip_std=calibrated_cfg.sync.min_lip_std),
            )
            save_config(updated_cfg, cfg_file)
            new_text = cfg_file.read_text(encoding="utf-8")

            diff = list(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile="config.yaml.bak",
                tofile="config.yaml",
            ))
            print("--- Configuration Diff ---")
            print("".join(diff))


if __name__ == "__main__":
    main()
