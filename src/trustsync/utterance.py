"""Utterance state machine and event detection."""
from __future__ import annotations

from dataclasses import dataclass, field
from trustsync.config import UtteranceCfg


@dataclass
class UtteranceEvent:
    """Completed utterance event payload."""

    t_start: float
    t_end: float
    kept: bool
    per_track: dict[int, dict[str, list[float]]] = field(default_factory=dict)


class UtteranceTracker:
    """State machine tracking speech activity and aggregating utterance observations."""

    def __init__(self, ucfg: UtteranceCfg, trim_s: float) -> None:
        self.ucfg = ucfg
        self.trim_s = trim_s
        self._active = False
        self._forced_suppress = False
        self._t_start = 0.0
        self._last_speech_t = 0.0
        self._per_track: dict[int, dict[str, list[float]]] = {}

    @property
    def active(self) -> bool:
        """Return True if an utterance is currently being recorded."""
        return self._active

    def update(
        self, t: float, speech: bool, obs: dict[int, dict[str, float]]
    ) -> UtteranceEvent | None:
        """Update utterance state machine with current frame observations.

        Args:
            t: Monotonic timestamp of the frame.
            speech: Voice activity detection flag for current frame.
            obs: Per-track observations mapping track_id to {'directed', 'quality', 'yaw', 'pitch'}.

        Returns:
            UtteranceEvent if an utterance concluded on this frame, else None.
        """
        if self._forced_suppress:
            if not speech:
                self._forced_suppress = False
            return None

        if not self._active:
            if speech:
                # Start new utterance
                self._active = True
                self._t_start = t
                self._last_speech_t = t
                self._per_track = {}
                self._record_obs(obs)
            return None

        # Currently active
        if speech:
            self._last_speech_t = t
            self._record_obs(obs)
            # Check maximum duration exceeded
            if (t - self._t_start) >= self.ucfg.max_s:
                self._forced_suppress = True
                return self._finalize_event()
            return None
        else:
            # Speech transitioned to False -> conclude utterance
            return self._finalize_event()

    def _record_obs(self, obs: dict[int, dict[str, float]]) -> None:
        for tid, values in obs.items():
            if tid not in self._per_track:
                self._per_track[tid] = {
                    "directed": [],
                    "quality": [],
                    "yaw": [],
                    "pitch": [],
                }
            for key in ("directed", "quality", "yaw", "pitch"):
                if key in values:
                    self._per_track[tid][key].append(float(values[key]))

    def _finalize_event(self) -> UtteranceEvent:
        self._active = False
        t_end = self._last_speech_t - self.trim_s
        kept = (t_end - self._t_start) >= self.ucfg.min_s
        return UtteranceEvent(
            t_start=self._t_start,
            t_end=t_end,
            kept=kept,
            per_track=self._per_track,
        )
