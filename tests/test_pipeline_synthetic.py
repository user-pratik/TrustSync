"""Synthetic end-to-end pipeline verification tests."""
from __future__ import annotations

import tracemalloc
import numpy as np

from tests.helpers_world import build_world, run_world


def test_pipeline_live_single() -> None:
    """Verify live_single: exactly 1 utterance_final frame, ACCEPT with speaker 0, LISTENING before it."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng,
        [(3.0, 4.5)],
        [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}],
    )
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    final_res = finals[0]
    assert final_res.decision == "ACCEPT"
    assert final_res.speaker_id == 0

    # Frames between speech start and final frame are LISTENING
    final_idx = final_res.frame_idx
    speech_indices = [r.frame_idx for r in res if r.speech_active and r.frame_idx < final_idx]
    assert len(speech_indices) > 0
    for idx in speech_indices:
        assert res[idx].decision == "LISTENING"


def test_pipeline_replay_spoof() -> None:
    """Verify replay_spoof (static lip): final decision is REJECT_SPOOF."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng,
        [(3.0, 4.5)],
        [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "static"}],
    )
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "REJECT_SPOOF"


def test_pipeline_multi_static_other() -> None:
    """Verify multi_static_other: face 0 static, face 1 matched -> ACCEPT speaker 1."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng,
        [(3.0, 4.5)],
        [
            {"track_id": 0, "bbox": (0.1, 0.2, 0.4, 0.8), "mode": "static"},
            {"track_id": 1, "bbox": (0.6, 0.2, 0.9, 0.8), "mode": "matched"},
        ],
    )
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "ACCEPT"
    assert finals[0].speaker_id == 1


def test_pipeline_multi_independent_other() -> None:
    """Verify multi_independent_other over 20 seeds: never ACCEPT speaker 0, >= 60% ACCEPT speaker 1, rest AMBIGUOUS."""
    accept_1_count = 0
    for s in range(20):
        rng = np.random.default_rng(7000 + s)
        p, f, t = build_world(
            rng,
            [(3.0, 4.5)],
            [
                {"track_id": 0, "bbox": (0.1, 0.2, 0.4, 0.8), "mode": "independent"},
                {"track_id": 1, "bbox": (0.6, 0.2, 0.9, 0.8), "mode": "matched"},
            ],
        )
        res = run_world(p, f, t)
        finals = [r for r in res if r.utterance_final]
        assert len(finals) == 1
        d = finals[0].decision
        spk = finals[0].speaker_id
        assert not (d == "ACCEPT" and spk == 0)
        if d == "ACCEPT" and spk == 1:
            accept_1_count += 1
        else:
            assert d == "REJECT_AMBIGUOUS"

    assert accept_1_count / 20.0 >= 0.60


def test_pipeline_not_addressed() -> None:
    """Verify not_addressed (yaw 50): final decision is REJECT_NOT_ADDRESSED."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng,
        [(3.0, 4.5)],
        [{"track_id": 0, "mode": "matched", "yaw": 50.0}],
    )
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "REJECT_NOT_ADDRESSED"


def test_pipeline_low_quality() -> None:
    """Verify low_quality (quality 0.2): final decision is REJECT_LOW_QUALITY."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng,
        [(3.0, 4.5)],
        [{"track_id": 0, "mode": "matched", "quality": 0.2}],
    )
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "REJECT_LOW_QUALITY"


def test_pipeline_no_face() -> None:
    """Verify no_face (script all empty): final decision is REJECT_NO_FACE."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(rng, [(3.0, 4.5)], [])
    res = run_world(p, f, t)
    finals = [r for r in res if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "REJECT_NO_FACE"


def test_pipeline_silence() -> None:
    """Verify silence (bursts []): no utterance_final frame, every decision IDLE."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(rng, [], [{"track_id": 0, "mode": "matched"}])
    res = run_world(p, f, t)
    assert not any(r.utterance_final for r in res)
    assert all(r.decision == "IDLE" for r in res)


def test_pipeline_short_click() -> None:
    """Verify short_click (0.2 s burst): no utterance_final frame."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(rng, [(3.0, 3.2)], [{"track_id": 0, "mode": "matched"}])
    res = run_world(p, f, t)
    assert not any(r.utterance_final for r in res)


def test_pipeline_calibration() -> None:
    """Verify calibration: no LISTENING frame earlier than 100.0 + vad.calib_s."""
    rng = np.random.default_rng(1234)
    # Speech burst starts early at 0.2 s
    p, f, t = build_world(rng, [(0.2, 2.0)], [{"track_id": 0, "mode": "matched"}])
    res = run_world(p, f, t)
    listening_times = [r.t for r in res if r.decision == "LISTENING"]
    for l_t in listening_times:
        assert l_t >= (100.0 + p.cfg.vad.calib_s)


def test_pipeline_robustness_nan_pts() -> None:
    """Verify robustness: a face whose pts are all NaN for 10 frames does not raise and has directed 0."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(rng, [(3.0, 4.5)], [{"track_id": 0, "mode": "matched"}])
    for k in range(10):
        p.tracker.script[k][0].pts[:] = np.nan
    res = run_world(p, f[:20], t[:20])
    for r in res[:10]:
        assert r.faces[0].directed == 0.0


def test_pipeline_memory_growth() -> None:
    """Verify memory growth for 3000 frames is below 50 MB."""
    rng = np.random.default_rng(1234)
    p, f, t = build_world(
        rng, [(1.0, 3.0), (4.0, 6.0)], [{"track_id": 0, "mode": "matched"}], duration_s=100.0
    )
    tracemalloc.start()
    run_world(p, f, t)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    growth_mb = peak / (1024.0 * 1024.0)
    assert growth_mb < 50.0, f"Memory growth was {growth_mb:.2f} MB (expected < 50 MB)"
