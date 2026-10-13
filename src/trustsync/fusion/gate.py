"""Decision gate implementing rule-based multi-modal fusion."""
from __future__ import annotations

from trustsync.config import Config
from trustsync.types import GateDecision, UtteranceSummary


class Gate:
    """Base gate interface."""

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        """Make decision on an utterance summary.

        Args:
            summary: Aggregated UtteranceSummary across tracks.

        Returns:
            GateDecision with decision, reason, speaker_id, and scores.
        """
        raise NotImplementedError


class RuleGate(Gate):
    """Rule-based decision gate following SPEC section 6."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def decide(self, summary: UtteranceSummary) -> GateDecision:
        """Evaluate utterance summary against rules in section 6 order.

        Args:
            summary: Aggregated UtteranceSummary across tracks.

        Returns:
            GateDecision: Result of rule evaluation.
        """
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

        # 2. Max quality across tracks < min_quality
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

        # 3. Best track by highest sync (ties broken by lowest track_id)
        sorted_tracks = sorted(
            summary.tracks, key=lambda tr: (-tr.sync, tr.track_id)
        )
        best = sorted_tracks[0]
        second = sorted_tracks[1] if len(sorted_tracks) > 1 else None
        second_sync = second.sync if second is not None else 0.0

        liveness = float(best.sync)
        attribution_conf = float(best.sync - second_sync)
        addressee = float(best.directed)

        if (
            best.sync < self.cfg.gate.min_sync
            or best.lip_std < self.cfg.sync.min_lip_std
        ):
            return GateDecision(
                decision="REJECT_SPOOF",
                reason=(
                    f"best sync {best.sync:.2f} below min {self.cfg.gate.min_sync:.2f} "
                    f"or lip motion {best.lip_std:.4f} below min {self.cfg.sync.min_lip_std:.4f}: "
                    "audio not explained by any visible mouth"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # 4. Ambiguous speaker attribution if two or more tracks
        if (
            len(sorted_tracks) >= 2
            and (best.sync - second_sync) < self.cfg.gate.min_margin
        ):
            return GateDecision(
                decision="REJECT_AMBIGUOUS",
                reason=(
                    f"margin {best.sync - second_sync:.2f} below min "
                    f"{self.cfg.gate.min_margin:.2f} between top tracks"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # 5. Best track quality < min_quality
        if best.quality < self.cfg.gate.min_quality:
            return GateDecision(
                decision="REJECT_LOW_QUALITY",
                reason=(
                    f"best track quality {best.quality:.2f} below min "
                    f"{self.cfg.gate.min_quality:.2f}"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # 6. Best track directed < min_directed
        if best.directed < self.cfg.gate.min_directed:
            return GateDecision(
                decision="REJECT_NOT_ADDRESSED",
                reason=(
                    f"best track directed {best.directed:.2f} below min "
                    f"{self.cfg.gate.min_directed:.2f}"
                ),
                speaker_id=None,
                liveness=liveness,
                attribution_conf=attribution_conf,
                addressee=addressee,
            )

        # 7. ACCEPT
        return GateDecision(
            decision="ACCEPT",
            reason=(
                f"speaker {best.track_id} accepted with sync {best.sync:.2f} "
                f"and directed {best.directed:.2f}"
            ),
            speaker_id=best.track_id,
            liveness=liveness,
            attribution_conf=attribution_conf,
            addressee=addressee,
        )


def make_gate(cfg: Config) -> Gate:
    """Factory returning RuleGate when cfg.gate.mode == 'rule' and LearnedGate when 'learned'."""
    if cfg.gate.mode == "learned":
        from trustsync.fusion.learned import LearnedGate
        return LearnedGate(cfg)
    return RuleGate(cfg)
