"""HUD overlay renderer for live visual monitoring."""
from __future__ import annotations

import cv2
import numpy as np

from trustsync.types import FrameResult


COLOR_ACCEPT = (191, 212, 45)      # BGR
COLOR_REJECT = (113, 113, 248)     # BGR
COLOR_LISTEN = (36, 191, 251)      # BGR
COLOR_IDLE = (139, 116, 100)       # BGR
COLOR_TEXT = (243, 237, 230)       # BGR


def _get_decision_color(decision: str) -> tuple[int, int, int]:
    if decision == "ACCEPT":
        return COLOR_ACCEPT
    elif decision.startswith("REJECT"):
        return COLOR_REJECT
    elif decision == "LISTENING":
        return COLOR_LISTEN
    return COLOR_IDLE


def draw_hud(
    frame_bgr: np.ndarray, result: FrameResult, env_hist: list[float]
) -> np.ndarray:
    """Render heads-up display overlay on video frame.

    Args:
        frame_bgr: Input BGR video frame image array.
        result: FrameResult containing perception metrics and decisions.
        env_hist: Rolling history of audio envelope values.

    Returns:
        np.ndarray: Fresh copy of frame with rendered overlay annotations.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if h == 0 or w == 0:
        return out

    scale = max(0.5, float(h) / 480.0)
    line_thick = max(1, round(2 * scale))

    # (a) Per-face annotations
    best_sync_tid: int | None = None
    if result.faces:
        best_fs = max(result.faces, key=lambda f: f.sync)
        best_sync_tid = best_fs.track_id

    for face in result.faces:
        x1_n, y1_n, x2_n, y2_n = face.bbox
        x1 = int(np.clip(round(x1_n * w), 0, w - 1))
        y1 = int(np.clip(round(y1_n * h), 0, h - 1))
        x2 = int(np.clip(round(x2_n * w), 0, w - 1))
        y2 = int(np.clip(round(y2_n * h), 0, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

        # Determine face color
        if result.speaker_id is not None and face.track_id == result.speaker_id:
            box_col = COLOR_ACCEPT
        elif result.decision.startswith("REJECT") and face.track_id == best_sync_tid:
            box_col = COLOR_REJECT
        else:
            box_col = COLOR_IDLE

        # Corner brackets (arm length 15% of side)
        arm_w = max(4, round(0.15 * box_w))
        arm_h = max(4, round(0.15 * box_h))

        # Top-left
        cv2.line(out, (x1, y1), (x1 + arm_w, y1), box_col, line_thick, cv2.LINE_AA)
        cv2.line(out, (x1, y1), (x1, y1 + arm_h), box_col, line_thick, cv2.LINE_AA)
        # Top-right
        cv2.line(out, (x2, y1), (x2 - arm_w, y1), box_col, line_thick, cv2.LINE_AA)
        cv2.line(out, (x2, y1), (x2, y1 + arm_h), box_col, line_thick, cv2.LINE_AA)
        # Bottom-left
        cv2.line(out, (x1, y2), (x1 + arm_w, y2), box_col, line_thick, cv2.LINE_AA)
        cv2.line(out, (x1, y2), (x1, y2 - arm_h), box_col, line_thick, cv2.LINE_AA)
        # Bottom-right
        cv2.line(out, (x2, y2), (x2 - arm_w, y2), box_col, line_thick, cv2.LINE_AA)
        cv2.line(out, (x2, y2), (x2, y2 - arm_h), box_col, line_thick, cv2.LINE_AA)

        # Label above box on dark translucent chip
        sync_val = 0.0 if np.isnan(face.sync) else face.sync
        label_text = f"id {face.track_id} sync {sync_val:.2f}"
        font_scale = 0.45 * scale
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        chip_y2 = max(0, y1 - 4)
        chip_y1 = max(0, chip_y2 - th - 6)
        chip_x1 = x1
        chip_x2 = min(w, x1 + tw + 8)

        if chip_y2 > chip_y1 and chip_x2 > chip_x1:
            sub = out[chip_y1:chip_y2, chip_x1:chip_x2]
            dark = np.zeros_like(sub)
            cv2.addWeighted(dark, 0.6, sub, 0.4, 0, sub)
            cv2.putText(
                out,
                label_text,
                (chip_x1 + 4, chip_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_TEXT,
                1,
                cv2.LINE_AA,
            )

        # Below box: horizontal sync fill bar
        bar_h = max(3, round(4 * scale))
        bar_y1 = min(h - 1, y2 + 4)
        bar_y2 = min(h - 1, bar_y1 + bar_h)
        if bar_y2 > bar_y1:
            cv2.rectangle(out, (x1, bar_y1), (x2, bar_y2), (50, 50, 50), -1)
            fill_w = int(np.clip(sync_val, 0.0, 1.0) * box_w)
            if fill_w > 0:
                cv2.rectangle(
                    out, (x1, bar_y1), (x1 + fill_w, bar_y2), box_col, -1
                )

        # Gaze / head arrow from box centre
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        yaw = 0.0 if np.isnan(face.yaw) else face.yaw
        pitch = 0.0 if np.isnan(face.pitch) else face.pitch
        max_arrow = round(100 * scale)
        dx = int(np.clip(3.0 * scale * yaw, -max_arrow, max_arrow))
        dy = int(np.clip(3.0 * scale * pitch, -max_arrow, max_arrow))
        if abs(dx) > 2 or abs(dy) > 2:
            cv2.arrowedLine(
                out,
                (cx, cy),
                (cx + dx, cy + dy),
                box_col,
                line_thick,
                cv2.LINE_AA,
                tipLength=0.3,
            )

    # (b) Translucent dark panel top-left
    panel_w = min(w, round(280 * scale))
    panel_h = min(h, round(105 * scale))
    sub_p = out[0:panel_h, 0:panel_w]
    dark_p = np.zeros_like(sub_p)
    cv2.addWeighted(dark_p, 0.55, sub_p, 0.45, 0, sub_p)

    dec_col = _get_decision_color(result.decision)
    cv2.putText(
        out,
        result.decision,
        (round(12 * scale), round(30 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9 * scale,
        dec_col,
        line_thick,
        cv2.LINE_AA,
    )

    reason_trunc = result.reason[:70] if result.reason else ""
    cv2.putText(
        out,
        reason_trunc,
        (round(12 * scale), round(52 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4 * scale,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    spk_str = f"speaker: {result.speaker_id}" if result.speaker_id is not None else "speaker: -"
    fps_val = int(round(result.fps)) if not np.isnan(result.fps) else 0
    fps_str = f"FPS {fps_val:02d}"
    cv2.putText(
        out,
        f"{spk_str}   {fps_str}",
        (round(12 * scale), round(80 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48 * scale,
        COLOR_TEXT,
        1,
        cv2.LINE_AA,
    )

    # (c) Bottom audio strip (40 px high)
    strip_h = round(40 * scale)
    strip_y1 = max(0, h - strip_h)
    strip_sub = out[strip_y1:h, 0:w]
    dark_s = np.zeros_like(strip_sub)
    cv2.addWeighted(dark_s, 0.65, strip_sub, 0.35, 0, strip_sub)

    strip_col = COLOR_LISTEN if result.speech_active else COLOR_IDLE

    if env_hist:
        pts_list: list[tuple[int, int]] = []
        recent = env_hist[-200:]
        n_pts = len(recent)
        step_x = float(w) / max(1, n_pts - 1) if n_pts > 1 else float(w)
        max_val = max(0.1, max(recent))

        for idx, val in enumerate(recent):
            px = int(round(idx * step_x))
            norm_v = np.clip(val / max_val, 0.0, 1.0)
            py = int(round(h - 4 - norm_v * (strip_h - 8)))
            pts_list.append((px, py))

        if len(pts_list) >= 2:
            pts_arr = np.array(pts_list, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts_arr], isClosed=False, color=strip_col, thickness=line_thick, lineType=cv2.LINE_AA)

    return out
