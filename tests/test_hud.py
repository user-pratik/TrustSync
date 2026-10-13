"""Unit tests for HUD overlay renderer."""
from __future__ import annotations

import numpy as np

from trustsync.overlay.hud import draw_hud
from trustsync.types import DECISIONS, FaceScores, FrameResult


def _make_dummy_result(decision: str = "IDLE") -> FrameResult:
    return FrameResult(
        t=10.0,
        frame_idx=0,
        fps=30.0,
        faces=[],
        speech_active=False,
        env=0.01,
        noise_floor=0.005,
        speaker_id=None,
        attribution_conf=0.0,
        liveness=0.0,
        addressee=0.0,
        decision=decision,
        reason="test reason",
        utterance_final=False,
    )


def test_hud_rendering_properties() -> None:
    """Verify output shape equals input, input unchanged, and output differs."""
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)
    frame_orig = frame.copy()
    res = _make_dummy_result("ACCEPT")

    out = draw_hud(frame, res, [0.01, 0.02, 0.03])
    assert out.shape == frame.shape
    assert np.array_equal(frame, frame_orig)  # input unchanged
    assert not np.array_equal(out, frame)      # output differs


def test_hud_every_decision_renders() -> None:
    """Verify every decision in DECISIONS renders cleanly without exception."""
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    for dec in DECISIONS:
        res = _make_dummy_result(dec)
        out = draw_hud(frame, res, [0.01, 0.05])
        assert out.shape == frame.shape


def test_hud_robustness_edge_cases() -> None:
    """Verify zero faces, empty env_hist, out-of-bounds bbox, and NaN yaw do not crash."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Zero faces and empty env_hist
    res_empty = _make_dummy_result("IDLE")
    out1 = draw_hud(frame, res_empty, [])
    assert out1.shape == frame.shape

    # 2. Face bbox partly outside [0, 1]
    face_oob = FaceScores(
        track_id=0,
        bbox=(-0.2, -0.1, 1.2, 1.1),
        sync=0.5,
        lip_std=0.01,
        best_lag_s=0.0,
        yaw=float("nan"),  # 3. NaN yaw
        pitch=float("nan"),
        roll=float("nan"),
        gaze_off=0.1,
        directed=0.8,
        quality=0.9,
    )
    res_oob = FrameResult(
        t=10.0,
        frame_idx=1,
        fps=30.0,
        faces=[face_oob],
        speech_active=True,
        env=0.05,
        noise_floor=0.005,
        speaker_id=0,
        attribution_conf=0.5,
        liveness=0.5,
        addressee=0.8,
        decision="ACCEPT",
        reason="robustness check",
        utterance_final=True,
    )
    out2 = draw_hud(frame, res_oob, [0.02] * 250)
    assert out2.shape == frame.shape
