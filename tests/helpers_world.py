"""Synthetic world fixture helpers for pipeline testing."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf

from tools.validate_session import EXPECTED_PER_SCENARIO
from trustsync.audio.sources import NumpyAudioSource
from trustsync.config import Config
from trustsync.pipeline import TrustSyncPipeline
from trustsync.synth import FakeFaceTracker, make_env, make_lip, make_pts, make_static_lip
from trustsync.types import FaceObs, FrameResult


def build_world(
    rng: np.random.Generator,
    bursts: list[tuple[float, float]],
    faces: list[dict],
    cfg: Config | None = None,
    duration_s: float = 8.0,
    fps: float = 30.0,
    t0: float = 100.0,
    flicker_prob: float = 0.0,
    id_switch: tuple[int, int, float] | None = None,
    env_multiplier: float = 1.0,
    baseline_noise: float | None = None,
) -> tuple[TrustSyncPipeline, list[np.ndarray], list[float]]:
    """Build a synthetic test world with scripted FaceObs and audio envelope.

    Args:
        rng: Seeded numpy RandomGenerator.
        bursts: Speech burst intervals in seconds.
        faces: List of dicts with keys:
            track_id (int), bbox (tuple), yaw (float), quality (float), mode ('matched'|'static'|'independent').
        cfg: Config instance (uses default if None).
        duration_s: World duration in seconds.
        fps: Video frame rate.
        t0: Start timestamp in seconds.
        flicker_prob: Probability of dropping face detections per frame.
        id_switch: Optional tuple (old_id, new_id, switch_t) switching track ID mid-world.
        env_multiplier: Multiplier for audio envelope.
        baseline_noise: Optional elevated baseline noise level.

    Returns:
        tuple: (pipeline, dummy_frames, timestamps).
    """
    if cfg is None:
        from trustsync.config import load_config
        cfg_path = Path(__file__).resolve().parent / "fixtures" / "config_spec_defaults.yaml"
        cfg = load_config(cfg_path)

    n_frames = round(duration_s * fps)
    times = [t0 + k / fps for k in range(n_frames)]

    # Generate 100 Hz envelope
    env = make_env(duration_s, 100.0, bursts, rng)
    if baseline_noise is not None:
        t_env = np.arange(len(env), dtype=np.float64) / 100.0
        outside_bursts = np.ones(len(env), dtype=bool)
        for start_s, end_s in bursts:
            outside_bursts &= ~((t_env >= start_s) & (t_env <= end_s))
        env[outside_bursts] += np.float32(baseline_noise - 0.002)

    if env_multiplier != 1.0:
        env = (env * env_multiplier).astype(np.float32)

    audio_source = NumpyAudioSource(env, env_hz=100.0, t0=t0)

    # Generate per-face lip signals
    lip_signals: dict[int, np.ndarray] = {}
    matched_lip_sig: np.ndarray | None = None
    for face_info in faces:
        tid = face_info["track_id"]
        mode = face_info.get("mode", "matched")
        if mode == "matched":
            if matched_lip_sig is None:
                matched_lip_sig = make_lip(env, 100.0, fps, lag_s=0.033, gain=0.05, noise=0.003, rng=rng)
            lip_sig = matched_lip_sig
        elif mode == "static":
            lip_sig = make_static_lip(duration_s, fps, noise=0.001, rng=rng)
        elif mode == "independent":
            rng_other = np.random.default_rng(int(rng.integers(1, 100000)))
            env_other = make_env(duration_s, 100.0, [(1.0, 6.0)], rng_other)
            lip_sig = make_lip(env_other, 100.0, fps, lag_s=0.0, gain=0.05, noise=0.003, rng=rng_other)
        else:
            lip_sig = make_static_lip(duration_s, fps, noise=0.001, rng=rng)
        lip_signals[tid] = lip_sig

    # Build script for FakeFaceTracker
    script: list[list[FaceObs]] = []
    for k in range(n_frames):
        t_curr = times[k]
        if flicker_prob > 0.0 and float(rng.uniform(0.0, 1.0)) < flicker_prob:
            script.append([])
            continue

        frame_faces: list[FaceObs] = []
        for face_info in faces:
            orig_tid = face_info["track_id"]
            tid = orig_tid
            if id_switch is not None:
                old_id, new_id, switch_t = id_switch
                if orig_tid == old_id and t_curr >= switch_t:
                    tid = new_id

            yaw = float(face_info.get("yaw", 0.0))
            quality = float(face_info.get("quality", 1.0))
            bbox = face_info.get("bbox", (0.3, 0.2, 0.7, 0.8))

            lip_val = float(lip_signals[orig_tid][k])
            pts = make_pts(
                yaw=yaw,
                pitch=0.0,
                roll=0.0,
                mouth_gap_ratio=lip_val,
                center=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
            )
            frame_faces.append(
                FaceObs(
                    track_id=tid,
                    bbox=bbox,
                    pts=pts,
                    mouth_open=lip_val,
                    quality=quality,
                )
            )
        script.append(frame_faces)

    fake_tracker = FakeFaceTracker(script)
    pipeline = TrustSyncPipeline(cfg=cfg, audio=audio_source, tracker=fake_tracker)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frames = [dummy_frame for _ in range(n_frames)]

    return pipeline, frames, times


def run_world(
    pipeline: TrustSyncPipeline, frames: list[np.ndarray], times: list[float]
) -> list[FrameResult]:
    """Feed frames sequentially through pipeline and return all FrameResult objects."""
    results: list[FrameResult] = []
    for frame, t in zip(frames, times):
        res = pipeline.process(frame, t)
        results.append(res)
    return results


def make_pseudo_session(
    tmp_path: Path, rng: np.random.Generator, scenario: str = "replay_spoof"
) -> Path:
    """Build a pseudo session directory with video.mp4, audio.wav, frame_times.csv, and meta.json."""
    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "tests" / "fixtures"
    frontal_path = fixtures_dir / "face_frontal.jpg"

    session_dir = tmp_path / f"pseudo_{scenario}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if frontal_path.is_file():
        img = cv2.imread(str(frontal_path))
    else:
        img = np.full((480, 640, 3), 120, dtype=np.uint8)

    h, w = img.shape[:2]
    video_path = session_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(240):
        vw.write(img)
    vw.release()

    csv_path = session_dir / "frame_times.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t"])
        for k in range(240):
            writer.writerow([k, f"{100.0 + k / 30.0:.4f}"])

    env_100hz = make_env(8.0, 100.0, [(3.0, 4.5)], rng)
    t_env = np.arange(len(env_100hz), dtype=np.float64) / 100.0
    t_audio = np.arange(128000, dtype=np.float64) / 16000.0
    env_16k = np.interp(t_audio, t_env, env_100hz)

    noise = rng.standard_normal(128000).astype(np.float32)
    audio_data = (noise * env_16k).astype(np.float32)
    max_val = max(1e-4, float(np.max(np.abs(audio_data))))
    audio_data = (audio_data / max_val * 0.5).astype(np.float32)

    audio_path = session_dir / "audio.wav"
    sf.write(str(audio_path), audio_data, 16000)

    expected = EXPECTED_PER_SCENARIO.get(scenario, "REJECT_SPOOF")
    meta = {
        "id": f"pseudo_{scenario}_001",
        "scenario": scenario,
        "subjects": ["test_subject"],
        "speaker_side": "center",
        "addressed": True,
        "expected": expected,
        "lighting": "normal",
        "occlusion": False,
        "spoof_device": "phone" if scenario == "replay_spoof" else None,
        "audio_t0": 100.0,
        "video_fps": 30.0,
        "notes": "pseudo session test",
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return session_dir
