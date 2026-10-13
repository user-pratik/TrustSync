"""TrustSync core data structures and types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any
import numpy as np

DECISIONS: tuple[str, ...] = (
    "IDLE",
    "LISTENING",
    "ACCEPT",
    "REJECT_SPOOF",
    "REJECT_AMBIGUOUS",
    "REJECT_NOT_ADDRESSED",
    "REJECT_LOW_QUALITY",
    "REJECT_NO_FACE",
)


@dataclass(frozen=True)
class FaceObs:
    """Observation of a single detected face in one video frame.

    pts shape: (478, 3) normalized x, y, and relative z.
    bbox: (x1, y1, x2, y2) normalized in [0, 1].
    """
    track_id: int
    bbox: tuple[float, float, float, float]
    pts: np.ndarray
    mouth_open: float
    quality: float


@dataclass
class FaceScores:
    """Per-face score metrics computed during frame processing."""
    track_id: int
    bbox: tuple[float, float, float, float]
    sync: float
    lip_std: float
    best_lag_s: float
    yaw: float
    pitch: float
    roll: float
    gaze_off: float
    directed: float
    quality: float


@dataclass
class SyncResult:
    """Audiovisual synchrony analysis result."""
    score: float
    lag_s: float
    lip_std: float
    valid: bool
    r_open: float
    r_vel: float


@dataclass
class TrackSummary:
    """Summary of scores for a face track aggregated over an utterance."""
    track_id: int
    sync: float
    lip_std: float
    directed: float
    quality: float
    yaw: float
    pitch: float


@dataclass
class UtteranceSummary:
    """Summary of all visible tracks across an active speech utterance."""
    t_start: float
    t_end: float
    duration_s: float
    tracks: list[TrackSummary]


@dataclass
class GateDecision:
    """Output of the fusion gate for an utterance."""
    decision: str
    reason: str
    speaker_id: int | None
    liveness: float
    attribution_conf: float
    addressee: float


@dataclass
class FrameResult:
    """Complete per-frame pipeline output."""
    t: float
    frame_idx: int
    fps: float
    faces: list[FaceScores]
    speech_active: bool
    env: float
    noise_floor: float
    speaker_id: int | None
    attribution_conf: float
    liveness: float
    addressee: float
    decision: str
    reason: str
    utterance_final: bool


def _clean_for_json(val: Any) -> Any:
    if isinstance(val, (float, np.floating)):
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return round(f_val, 4)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (tuple, list)):
        return [_clean_for_json(v) for v in val]
    if is_dataclass(val) and not isinstance(val, type):
        return {k: _clean_for_json(v) for k, v in asdict(val).items()}
    if isinstance(val, dict):
        return {k: _clean_for_json(v) for k, v in val.items()}
    if isinstance(val, np.ndarray):
        return _clean_for_json(val.tolist())
    return val


def frame_result_to_dict(r: FrameResult) -> dict[str, Any]:
    """Convert FrameResult to a JSON-safe dictionary.

    Python floats are rounded to 4 decimals, NaN values become None,
    and tuples become lists.
    """
    raw = asdict(r)
    return _clean_for_json(raw)
