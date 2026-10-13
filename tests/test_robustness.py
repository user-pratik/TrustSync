"""Robustness, fuzzing, degradation, and long-run tests for TrustSync pipeline."""
from __future__ import annotations

import time
import tracemalloc
import cv2
import numpy as np
import pytest

from tests.helpers_world import build_world, run_world
from trustsync.audio.sources import NumpyAudioSource
from trustsync.config import Config
from trustsync.pipeline import TrustSyncPipeline
from trustsync.synth import FakeFaceTracker, make_env, make_lip, make_pts, make_static_lip
from trustsync.types import FaceObs, FrameResult
from trustsync.vision.face_tracker import FaceTracker
from trustsync.vision.quality import face_quality


def test_robustness_frame_fuzz(spec_cfg: Config) -> None:
    """a. Frame fuzz through real FaceTracker: black, white, noise, tiny frames; shape/type checks."""
    tracker = FaceTracker(spec_cfg)
    try:
        # Dummy audio source returning quiet baseline
        env_quiet = np.full(100, 0.002, dtype=np.float32)
        audio = NumpyAudioSource(env_quiet, env_hz=100.0, t0=100.0)
        pipeline = TrustSyncPipeline(spec_cfg, audio=audio, tracker=tracker)

        fuzz_frames = [
            ("black", np.zeros((480, 640, 3), dtype=np.uint8)),
            ("white", np.full((480, 640, 3), 255, dtype=np.uint8)),
            ("noise", np.random.default_rng(1234).integers(0, 256, (480, 640, 3), dtype=np.uint8)),
            ("8x8", np.zeros((8, 8, 3), dtype=np.uint8)),
            ("1x1", np.zeros((1, 1, 3), dtype=np.uint8)),
        ]

        for name, frame in fuzz_frames:
            res = pipeline.process(frame, 100.0)
            assert res.decision != "ACCEPT", f"Frame {name} falsely accepted"
            # No face detected or any detected face has quality below 0.5
            assert len(res.faces) == 0 or all(f.quality < 0.5 for f in res.faces), (
                f"Frame {name} produced unexpected high-quality face"
            )

        # Grayscale (2-D) raises ValueError
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        with pytest.raises(ValueError, match="frame must be HxWx3 uint8"):
            pipeline.process(gray_frame, 100.0)

        # uint16 raises ValueError
        u16_frame = np.zeros((480, 640, 3), dtype=np.uint16)
        with pytest.raises(ValueError, match="frame must be HxWx3 uint8"):
            pipeline.process(u16_frame, 100.0)

        # 4-channel raises ValueError
        rgba_frame = np.zeros((480, 640, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="frame must be HxWx3 uint8"):
            pipeline.process(rgba_frame, 100.0)

        # NaN t raises ValueError
        good_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="t must be finite"):
            pipeline.process(good_frame, float("nan"))

    finally:
        tracker.close()


def test_robustness_face_flicker(spec_cfg: Config) -> None:
    """b. Face flicker: 20% random dropped face frames across 30 seeds; >=90% end ACCEPT."""
    bursts = [(3.0, 4.5)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}]

    accept_count = 0
    total_seeds = 30

    for seed in range(total_seeds):
        rng = np.random.default_rng(3000 + seed)
        pipeline, frames, times = build_world(
            rng=rng,
            bursts=bursts,
            faces=faces,
            cfg=spec_cfg,
            flicker_prob=0.20,
        )
        results = run_world(pipeline, frames, times)
        finals = [r for r in results if r.utterance_final]
        if finals:
            final_res = finals[0]
            if final_res.decision == "ACCEPT":
                assert final_res.speaker_id == 0, "Speaker was not track 0"
                accept_count += 1
            else:
                assert final_res.speaker_id != 1, "Wrong speaker accepted"

    share = accept_count / total_seeds
    assert share >= 0.90, f"Flicker accept share {share:.2f} < 0.90"


def test_robustness_track_id_switch(spec_cfg: Config, capsys: pytest.CaptureFixture[str]) -> None:
    """c. Track id switch mid-utterance: never ACCEPT with wrong speaker, is REJECT_*."""
    rng = np.random.default_rng(1234)
    bursts = [(3.0, 4.5)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}]

    # Switch track ID from 0 to 1 at t=103.75 s (middle of utterance, start t0=100.0)
    pipeline, frames, times = build_world(
        rng=rng,
        bursts=bursts,
        faces=faces,
        cfg=spec_cfg,
        id_switch=(0, 1, 103.75),
    )
    results = run_world(pipeline, frames, times)
    finals = [r for r in results if r.utterance_final]
    assert len(finals) == 1, "Expected one final decision"
    decision = finals[0].decision
    print(f"Track id switch decision: {decision}")
    assert decision != "ACCEPT", "Switched track must not ACCEPT"
    assert decision.startswith("REJECT_"), f"Decision {decision} must be a REJECT_*"


def test_robustness_two_faces_same_speech(spec_cfg: Config) -> None:
    """d. Two faces both driven by same speech: REJECT_AMBIGUOUS in >=90% of 20 seeds, never ACCEPT."""
    bursts = [(3.0, 4.5)]
    faces = [
        {"track_id": 0, "bbox": (0.1, 0.2, 0.4, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"},
        {"track_id": 1, "bbox": (0.6, 0.2, 0.9, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"},
    ]

    ambiguous_count = 0
    total_seeds = 20

    for seed in range(total_seeds):
        rng = np.random.default_rng(4000 + seed)
        pipeline, frames, times = build_world(rng=rng, bursts=bursts, faces=faces, cfg=spec_cfg)
        results = run_world(pipeline, frames, times)
        finals = [r for r in results if r.utterance_final]
        assert len(finals) == 1
        decision = finals[0].decision
        assert decision != "ACCEPT", f"Seed {seed} falsely accepted"
        if decision == "REJECT_AMBIGUOUS":
            ambiguous_count += 1

    share = ambiguous_count / total_seeds
    assert share >= 0.90, f"Ambiguous share {share:.2f} < 0.90"


def test_robustness_loud_audio(spec_cfg: Config) -> None:
    """e. Loud audio: env * 20 gives same final decision as normal in 20 seeds."""
    bursts = [(3.0, 4.5)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}]

    total_seeds = 20
    for seed in range(total_seeds):
        rng1 = np.random.default_rng(5000 + seed)
        p1, f1, t1 = build_world(rng=rng1, bursts=bursts, faces=faces, cfg=spec_cfg, env_multiplier=1.0)
        r1 = run_world(p1, f1, t1)
        finals1 = [r for r in r1 if r.utterance_final]

        rng2 = np.random.default_rng(5000 + seed)
        p2, f2, t2 = build_world(rng=rng2, bursts=bursts, faces=faces, cfg=spec_cfg, env_multiplier=20.0)
        r2 = run_world(p2, f2, t2)
        finals2 = [r for r in r2 if r.utterance_final]

        assert len(finals1) == len(finals2) == 1
        assert finals1[0].decision == finals2[0].decision, (
            f"Seed {seed} mismatch: normal={finals1[0].decision}, loud={finals2[0].decision}"
        )


def test_robustness_noisy_audio(spec_cfg: Config) -> None:
    """f. Noisy audio: baseline noise 0.02: >=50% end ACCEPT across 30 seeds."""
    bursts = [(3.0, 4.5)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}]

    accept_count = 0
    total_seeds = 30

    for seed in range(total_seeds):
        rng = np.random.default_rng(6000 + seed)
        pipeline, frames, times = build_world(
            rng=rng,
            bursts=bursts,
            faces=faces,
            cfg=spec_cfg,
            baseline_noise=0.02,
        )
        results = run_world(pipeline, frames, times)
        finals = [r for r in results if r.utterance_final]
        if finals and finals[0].decision == "ACCEPT":
            accept_count += 1

    share = accept_count / total_seeds
    print(f"Noisy audio accept share: {share:.2f} ({accept_count}/{total_seeds})")
    assert share >= 0.50, f"Noisy audio accept share {share:.2f} < 0.50"


def test_robustness_clock_jump(spec_cfg: Config) -> None:
    """g. Clock jump: at frame 120 shift later t by -5.0 s: no exception, second utterance decided."""
    rng = np.random.default_rng(7000)
    fps = 30.0
    duration_s = 14.0
    n_frames = round(duration_s * fps)

    # Audio timeline covers t from 90.0 to 120.0
    t_audio_base = 90.0
    audio_dur = 30.0
    # Burst 1 at 102.0..103.5 (before jump). Frame 120 happens at 100.0 + 120/30 = 104.0.
    # At frame 120, t shifts to 104.0 - 5.0 = 99.0.
    # Burst 2 at 101.5..103.0 in the post-jump timeline.
    bursts = [(12.0, 13.5), (11.5, 13.0)]  # relative to respective base
    # We will build envelope explicitly to match the timeline
    env_hz = 100.0
    n_env = round(audio_dur * env_hz)
    env = make_env(audio_dur, env_hz, [(12.0, 13.5)], rng)

    # Second burst placed at t=101.5..103.0 (which is offset 11.5..13.0 from 90.0)
    # The first burst is at 102.0..103.5 (offset 12.0..13.5 from 90.0)
    audio = NumpyAudioSource(env, env_hz=env_hz, t0=t_audio_base)

    # Build lip matching speech burst
    lip = make_lip(env, env_hz, fps, lag_s=0.033, gain=0.05, noise=0.003, rng=rng)

    # Frame times
    times: list[float] = []
    for k in range(n_frames):
        t_k = 100.0 + k / fps
        if k >= 120:
            t_k -= 5.0
        times.append(t_k)

    # Script for fake face tracker: face present for entire duration
    script: list[list[FaceObs]] = []
    for k in range(n_frames):
        # Sample lip using the frame's audio-timeline index
        audio_t = times[k] - t_audio_base
        lip_idx = int(round(np.clip(audio_t * fps, 0, len(lip) - 1)))
        lip_val = float(lip[lip_idx])
        pts = make_pts(mouth_gap_ratio=lip_val)
        script.append([
            FaceObs(
                track_id=0,
                bbox=(0.3, 0.2, 0.7, 0.8),
                pts=pts,
                mouth_open=lip_val,
                quality=1.0,
            )
        ])

    pipeline = TrustSyncPipeline(spec_cfg, audio=audio, tracker=FakeFaceTracker(script))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frames = [dummy_frame for _ in range(n_frames)]

    results = run_world(pipeline, frames, times)
    finals = [r for r in results if r.utterance_final]
    # Should complete without exception and decide both utterances
    assert len(finals) >= 2, f"Expected at least 2 utterances decided across clock jump, got {len(finals)}"


def test_robustness_mouth_covered_simulated(spec_cfg: Config) -> None:
    """h. Mouth covered simulated: quality 1, static lip signal: REJECT_SPOOF (never ACCEPT)."""
    rng = np.random.default_rng(8000)
    bursts = [(3.0, 4.5)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "static"}]

    pipeline, frames, times = build_world(rng=rng, bursts=bursts, faces=faces, cfg=spec_cfg)
    results = run_world(pipeline, frames, times)
    finals = [r for r in results if r.utterance_final]
    assert len(finals) == 1
    assert finals[0].decision == "REJECT_SPOOF"


def test_robustness_darkness_curve(spec_cfg: Config) -> None:
    """i. Darkness curve: brightness 5, 15, 30, 45, 60, 90: face_quality non-decreasing."""
    brightnesses = [5, 15, 30, 45, 60, 90]
    qualities: list[float] = []

    bbox = (0.2, 0.2, 0.8, 0.8)
    pts = make_pts(w=640, h=480)

    for b in brightnesses:
        frame = np.full((480, 640, 3), b, dtype=np.uint8)
        # Draw a bright ellipse at center
        cv2.ellipse(frame, (320, 240), (100, 150), 0, 0, 360, (b + 10, b + 10, b + 10), -1)
        q = face_quality(frame, bbox, pts, spec_cfg.quality)
        qualities.append(q)

    for i in range(len(qualities) - 1):
        assert qualities[i] <= qualities[i + 1] + 1e-6, (
            f"Quality decreased from {qualities[i]} at brightness {brightnesses[i]} "
            f"to {qualities[i+1]} at brightness {brightnesses[i+1]}"
        )


@pytest.mark.slow
def test_robustness_long_run(spec_cfg: Config) -> None:
    """j. Long run: 18000 frames (10 min at 30 fps), memory < 100 MB, p99 < 5*p50, 1 final per burst."""
    rng = np.random.default_rng(1234)
    fps = 30.0
    n_frames = 18000  # 600 seconds
    burst_period_s = 6.0
    n_bursts = int(600 // burst_period_s)  # 100 bursts

    # Bursts every 6 s: [1.5 + 6k, 3.0 + 6k]
    bursts = [(1.5 + burst_period_s * i, 3.0 + burst_period_s * i) for i in range(n_bursts)]
    faces = [{"track_id": 0, "bbox": (0.3, 0.2, 0.7, 0.8), "yaw": 0.0, "quality": 1.0, "mode": "matched"}]

    pipeline, frames, times = build_world(
        rng=rng,
        bursts=bursts,
        faces=faces,
        cfg=spec_cfg,
        duration_s=600.0,
        fps=fps,
        t0=100.0,
    )

    # 1. Check memory growth over 3000 frames with tracemalloc
    tracemalloc.start()
    for k in range(3000):
        pipeline.process(frames[k], times[k])
    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_growth_mb = mem_peak / (1024 * 1024)
    assert mem_growth_mb < 100.0, f"Memory growth {mem_growth_mb:.2f} MB exceeds 100 MB"

    # 2. Run full 18000 frames measuring per-frame latency without tracemalloc overhead
    rng2 = np.random.default_rng(1234)
    pipeline2, frames2, times2 = build_world(
        rng=rng2,
        bursts=bursts,
        faces=faces,
        cfg=spec_cfg,
        duration_s=600.0,
        fps=fps,
        t0=100.0,
    )
    frame_durations: list[float] = []
    final_count = 0

    for k in range(n_frames):
        t0 = time.perf_counter()
        res = pipeline2.process(frames2[k], times2[k])
        dt = time.perf_counter() - t0
        frame_durations.append(dt)
        if res.utterance_final:
            final_count += 1

    dts = np.array(frame_durations[30:])
    p50 = float(np.percentile(dts, 50))
    p99 = float(np.percentile(dts, 99))
    assert p99 < 5.0 * p50, f"p99 latency ({p99*1000:.2f} ms) >= 5 * p50 ({p50*1000:.2f} ms)"

    # Exactly one utterance_final per burst
    assert final_count == n_bursts, f"Expected {n_bursts} final utterances, got {final_count}"
