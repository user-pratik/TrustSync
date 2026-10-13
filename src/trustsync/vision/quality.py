"""Face image quality assessment based on size, brightness, and edge margin."""
from __future__ import annotations

import cv2
import numpy as np

from trustsync.config import QualityCfg


def face_quality(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    pts: np.ndarray,
    cfg: QualityCfg,
) -> float:
    """Assess overall image quality of a detected face.

    Args:
        frame_bgr: Input BGR image array shape (H, W, 3).
        bbox: Normalized bounding box (x1, y1, x2, y2).
        pts: Normalized landmark coordinates shape (478, 3).
        cfg: Quality configuration thresholds.

    Returns:
        float: Composite quality score in [0.0, 1.0].
    """
    h, w = frame_bgr.shape[:2]

    # 1. Size score
    face_height_px = (bbox[3] - bbox[1]) * h
    s_size = float(np.clip(face_height_px / cfg.min_face_px, 0.0, 1.0))

    # 2. Brightness score
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    x1_px = int(round(np.clip(bbox[0], 0.0, 1.0) * w))
    y1_px = int(round(np.clip(bbox[1], 0.0, 1.0) * h))
    x2_px = int(round(np.clip(bbox[2], 0.0, 1.0) * w))
    y2_px = int(round(np.clip(bbox[3], 0.0, 1.0) * h))

    if x2_px <= x1_px or y2_px <= y1_px:
        mean_brightness = 0.0
    else:
        crop = gray[y1_px:y2_px, x1_px:x2_px]
        mean_brightness = float(np.mean(crop))

    s_bright = float(np.clip(mean_brightness / cfg.min_brightness, 0.0, 1.0))

    # 3. Edge margin score
    xs = pts[:, 0]
    ys = pts[:, 1]
    outside = (xs < 0.02) | (xs > 0.98) | (ys < 0.02) | (ys > 0.98)
    frac_outside = float(np.mean(outside))
    s_edge = 1.0 if frac_outside <= cfg.max_edge_frac else 0.0

    return float(s_size * s_bright * s_edge)
