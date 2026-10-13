"""Addressee classification evaluating whether a face is looking towards the robot."""
from __future__ import annotations

import math
import numpy as np

from trustsync.config import AddresseeCfg


def directed_score(
    yaw: float, pitch: float, gaze_off: float, cfg: AddresseeCfg
) -> float:
    """Compute addressee directedness score in [0.0, 1.0].

    Args:
        yaw: Estimated head yaw in degrees.
        pitch: Estimated head pitch in degrees.
        gaze_off: Normalized gaze offset in [0.0, 1.0].
        cfg: Addressee configuration containing tolerance thresholds and weights.

    Returns:
        float: Directed score in [0.0, 1.0], or 0.0 if any input is NaN.
    """
    if (
        math.isnan(yaw)
        or math.isnan(pitch)
        or math.isnan(gaze_off)
        or math.isinf(yaw)
        or math.isinf(pitch)
        or math.isinf(gaze_off)
    ):
        return 0.0

    yc = yaw - cfg.yaw_offset_deg
    pc = pitch - cfg.pitch_offset_deg

    d = max(abs(yc) / cfg.max_yaw_deg, abs(pc) / cfg.max_pitch_deg)

    if d <= 1.0:
        pose = 1.0
    else:
        pose = max(0.0, 2.0 - d)

    if cfg.use_gaze:
        gaze_factor = min(1.0, max(0.0, gaze_off / cfg.gaze_max_off))
        pose *= 1.0 - cfg.gaze_weight * gaze_factor

    return float(np.clip(pose, 0.0, 1.0))
