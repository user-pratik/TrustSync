"""TrustSync latency, throughput, and sub-system benchmark tool."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mediapipe
import numpy as np

from trustsync.addressee.classifier import directed_score
from trustsync.audio.sources import FileAudioSource, LiveAudioSource, NumpyAudioSource
from trustsync.config import Config, load_config
from trustsync.io.video_source import FileVideoSource, WebcamSource
from trustsync.overlay.hud import draw_hud
from trustsync.pipeline import TrustSyncPipeline, _window_sync
from trustsync.synth import make_env
from trustsync.vision.face_tracker import FaceTracker
from trustsync.vision.gaze import gaze_off
from trustsync.vision.head_pose import estimate_pose


def compute_percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    """Compute mean, p50, p95, and p99 percentiles in milliseconds.

    Args:
        values: Sequence of durations in milliseconds.

    Returns:
        dict: Keys mean_ms, p50_ms, p95_ms, p99_ms rounded to 4 decimals.
    """
    if len(values) == 0:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def compute_fps_from_ms(mean_ms: float, p95_ms: float) -> dict[str, float]:
    """Compute throughput mean and p5 FPS from millisecond latencies.

    Args:
        mean_ms: Average pipeline execution time in ms.
        p95_ms: 95th percentile pipeline execution time in ms.

    Returns:
        dict: Keys mean, p5.
    """
    mean_fps = round(1000.0 / mean_ms, 2) if mean_ms > 0.0 else 0.0
    p5_fps = round(1000.0 / p95_ms, 2) if p95_ms > 0.0 else 0.0
    return {"mean": mean_fps, "p5": p5_fps}


def format_table(stages: dict[str, dict[str, float]]) -> str:
    """Format stages timing dictionary into an aligned text table.

    Args:
        stages: Dictionary mapping stage names to timing stats dicts.

    Returns:
        str: Aligned table formatted as text.
    """
    headers = ("stage", "mean ms", "p50", "p95", "p99")
    rows: list[tuple[str, str, str, str, str]] = []
    for name, stats in stages.items():
        rows.append((
            name,
            f"{stats['mean_ms']:.2f}",
            f"{stats['p50_ms']:.2f}",
            f"{stats['p95_ms']:.2f}",
            f"{stats['p99_ms']:.2f}",
        ))
    all_rows = [headers] + rows
    col_widths = [max(len(r[i]) for r in all_rows) for i in range(5)]
    lines = [
        " | ".join(headers[i].ljust(col_widths[i]) for i in range(5)),
        "-+-".join("-" * col_widths[i] for i in range(5)),
    ]
    for r in rows:
        lines.append(" | ".join(r[i].ljust(col_widths[i]) for i in range(5)))
    return "\n".join(lines)


def update_results_fps(results_path: Path, fps_data: dict[str, Any]) -> None:
    """Atomically update only the 'fps' key of results.json using a temp file.

    Args:
        results_path: Path to reports/results.json.
        fps_data: FPS metrics dictionary {mean, p5, source, machine: processor}.
    """
    if not results_path.is_file():
        return
    data = json.loads(results_path.read_text(encoding="utf-8"))
    data["fps"] = fps_data
    temp_path = results_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp_path.replace(results_path)


def run_benchmark(
    source_type: str = "photo",
    session_path: str | Path | None = None,
    n_frames: int = 300,
    out_path: str | Path = "reports/benchmark.json",
    cfg: Config | None = None,
) -> dict[str, Any]:
    """Execute end-to-end performance benchmark.

    Args:
        source_type: One of 'photo', 'session', 'webcam'.
        session_path: Path to recorded session directory if source_type == 'session'.
        n_frames: Number of frames to evaluate.
        out_path: Destination path for benchmark.json.
        cfg: Optional configuration instance.

    Returns:
        dict: Complete benchmark results dictionary.
    """
    if cfg is None:
        cfg = load_config()

    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "tests" / "fixtures"

    burst_ends: list[float] = []
    frames_bgr: list[np.ndarray] = []
    timestamps: list[float] = []
    video_source: Any = None
    audio_source: Any = None

    if source_type == "photo":
        photo_file = fixtures_dir / "face_frontal.jpg"
        if not photo_file.is_file():
            raise FileNotFoundError(f"Fixture {photo_file} not found")
        img = cv2.imread(str(photo_file))
        if img is None:
            raise RuntimeError(f"Failed to read image {photo_file}")

        # Synthetic audio: 100 Hz envelope with bursts every 3 s
        duration_s = (n_frames / 30.0) + 5.0
        burst_period = 3.0
        num_bursts = int((n_frames / 30.0) // burst_period) + 1
        bursts = [(1.5 + burst_period * i, 2.7 + burst_period * i) for i in range(num_bursts)]
        burst_ends = [b[1] for b in bursts]

        rng = np.random.default_rng(1234)
        env = make_env(duration_s, 100.0, bursts, rng)
        audio_source = NumpyAudioSource(env, env_hz=100.0, t0=0.0)

        for k in range(n_frames):
            frames_bgr.append(img.copy())
            timestamps.append(k / 30.0)

    elif source_type == "session":
        if session_path is None:
            raise ValueError("--session PATH is required when source is session")
        s_path = Path(session_path)
        meta = json.loads((s_path / "meta.json").read_text(encoding="utf-8"))
        video_source = FileVideoSource(s_path / "video.mp4", s_path / "frame_times.csv")
        audio_source = FileAudioSource(s_path / "audio.wav", t0=float(meta["audio_t0"]), cfg=cfg.audio)

    elif source_type == "webcam":
        video_source = WebcamSource(cfg)
        audio_source = LiveAudioSource(cfg.audio)
        audio_source.start()

    else:
        raise ValueError(f"Unknown source: {source_type}")

    # Build standalone tracker and pipeline
    face_tracker = FaceTracker(cfg)
    pipeline = TrustSyncPipeline(cfg, audio=audio_source, tracker=face_tracker)

    # Stage timing collections (in ms)
    times_tracker: list[float] = []
    times_pose: list[float] = []
    times_sync: list[float] = []
    times_pipeline: list[float] = []
    times_jpeg: list[float] = []
    times_hud: list[float] = []
    decision_latencies: list[float] = []

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), cfg.server.jpeg_quality]
    env_hist: list[float] = []
    completed_burst_idx = 0

    warmup_count = 30
    frame_idx = 0

    try:
        while frame_idx < n_frames:
            if source_type == "photo":
                frame = frames_bgr[frame_idx]
                t = timestamps[frame_idx]
            else:
                ok, read_frame, t = video_source.read()
                if not ok or read_frame is None:
                    break
                frame = read_frame

            env_now = float(audio_source.envelope_at(np.array([t], dtype=np.float64))[0])
            env_hist.append(env_now)
            if len(env_hist) > 200:
                env_hist.pop(0)

            h, w = frame.shape[:2]

            # 1. Measure standalone face_tracker.process
            t0 = time.perf_counter()
            faces = face_tracker.process(frame)
            dt_tracker = (time.perf_counter() - t0) * 1000.0

            # 2. Measure pose + gaze + directed for all faces
            t0 = time.perf_counter()
            for face in faces:
                yaw, pitch, roll = estimate_pose(face.pts, w, h)
                goff = gaze_off(face.pts, w, h) if cfg.addressee.use_gaze else 0.0
                _ = directed_score(yaw, pitch, goff, cfg.addressee)
            dt_pose = (time.perf_counter() - t0) * 1000.0

            # 3. Measure running sync for all faces
            t0 = time.perf_counter()
            for face in faces:
                buf = pipeline._buffers.get(face.track_id, deque())
                _ = _window_sync(buf, t, audio_source, cfg)
            dt_sync = (time.perf_counter() - t0) * 1000.0

            # 4. Measure full pipeline.process
            t0 = time.perf_counter()
            result = pipeline.process(frame, t)
            dt_pipeline = (time.perf_counter() - t0) * 1000.0

            # Check decision latency if utterance completed
            if result.utterance_final and burst_ends:
                # Find matching burst end
                if completed_burst_idx < len(burst_ends):
                    true_burst_end = burst_ends[completed_burst_idx]
                    lat = max(0.0, t - true_burst_end)
                    decision_latencies.append(lat)
                    completed_burst_idx += 1

            # 5. Measure JPEG encode
            t0 = time.perf_counter()
            _, _ = cv2.imencode(".jpg", frame, encode_params)
            dt_jpeg = (time.perf_counter() - t0) * 1000.0

            # 6. Measure draw_hud
            t0 = time.perf_counter()
            _ = draw_hud(frame, result, env_hist)
            dt_hud = (time.perf_counter() - t0) * 1000.0

            if frame_idx >= warmup_count:
                times_tracker.append(dt_tracker)
                times_pose.append(dt_pose)
                times_sync.append(dt_sync)
                times_pipeline.append(dt_pipeline)
                times_jpeg.append(dt_jpeg)
                times_hud.append(dt_hud)

            frame_idx += 1

    finally:
        face_tracker.close()
        if hasattr(audio_source, "close"):
            audio_source.close()
        if video_source is not None and hasattr(video_source, "close"):
            video_source.close()

    stages_stats = {
        "face_tracker.process": compute_percentiles(times_tracker),
        "pose+gaze+directed": compute_percentiles(times_pose),
        "running sync": compute_percentiles(times_sync),
        "full pipeline.process": compute_percentiles(times_pipeline),
        "JPEG encode": compute_percentiles(times_jpeg),
        "draw_hud": compute_percentiles(times_hud),
    }

    fps_stats = compute_fps_from_ms(
        stages_stats["full pipeline.process"]["mean_ms"],
        stages_stats["full pipeline.process"]["p95_ms"],
    )

    if decision_latencies:
        dec_lat = {
            "mean": round(float(np.mean(decision_latencies)), 4),
            "max": round(float(np.max(decision_latencies)), 4),
        }
    else:
        dec_lat = {"mean": 0.0, "max": 0.0}

    processor_name = platform.processor() or platform.machine() or "Unknown"

    benchmark_data = {
        "machine": {
            "platform": platform.platform(),
            "processor": processor_name,
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count() or 1,
            "opencv": cv2.__version__,
            "mediapipe": mediapipe.__version__,
        },
        "source": source_type,
        "n_frames": frame_idx,
        "stages": stages_stats,
        "fps": fps_stats,
        "decision_latency_s": dec_lat,
    }

    # Print summary table
    print(format_table(stages_stats))
    print(f"\nThroughput FPS: mean={fps_stats['mean']}, p5={fps_stats['p5']}")
    print(f"Decision latency: mean={dec_lat['mean']} s, max={dec_lat['max']} s\n")

    # Write output JSON
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")
    print(f"Benchmark written to {out_file}")

    # Update results.json if it exists
    results_json_path = root / "reports" / "results.json"
    if results_json_path.is_file():
        update_results_fps(
            results_json_path,
            {
                "mean": fps_stats["mean"],
                "p5": fps_stats["p5"],
                "source": source_type,
                "machine": processor_name,
            },
        )
        print("Updated reports/results.json 'fps' key.")

    return benchmark_data


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for benchmark tool."""
    parser = argparse.ArgumentParser(description="TrustSync latency and throughput benchmark.")
    parser.add_argument(
        "--source",
        choices=["webcam", "photo", "session"],
        default="photo",
        help="Benchmark input source (default photo)",
    )
    parser.add_argument("--session", type=str, default=None, help="Session directory path")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to benchmark")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/benchmark.json",
        help="Output benchmark JSON path",
    )
    args = parser.parse_args(argv)

    run_benchmark(
        source_type=args.source,
        session_path=args.session,
        n_frames=args.frames,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
