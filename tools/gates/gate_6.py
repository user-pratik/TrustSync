"""TrustSync Gate 6: Pipeline, fusion gating, HUD, offline runner, and hardware verification."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 6.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    test_files = [
        "tests/test_gate.py",
        "tests/test_utterance.py",
        "tests/test_video_source.py",
        "tests/test_hud.py",
        "tests/test_pipeline_synthetic.py",
        "tests/test_e2e_pseudo_session.py",
    ]

    for rel_path in test_files:
        test_path = root / rel_path
        if not test_path.is_file():
            results.append((f"test_{test_path.stem}", "FAIL", f"File not found: {rel_path}"))
            continue

        cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", str(test_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "passed"
            results.append((rel_path, "PASS", summary))
        else:
            err = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-1]
            results.append((rel_path, "FAIL", f"Exit {proc.returncode}: {err}"))

    # Hardware check
    if not hardware:
        results.append(
            ("pipeline_live_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
    else:
        from trustsync.config import load_config
        from trustsync.io.video_source import WebcamSource
        from trustsync.audio.sources import LiveAudioSource
        from trustsync.pipeline import TrustSyncPipeline

        cfg = load_config()
        video_src = WebcamSource(cfg)
        audio_src = LiveAudioSource(cfg.audio)
        audio_src.start()
        pipeline = TrustSyncPipeline(cfg, audio_src)

        print("\nFacing camera, speak one sentence during the 20 s window...")
        t_end = time.monotonic() + 20.0
        fps_list: list[float] = []
        utterance_decisions: list[str] = []

        try:
            while time.monotonic() < t_end:
                ok, frame, t = video_src.read()
                if not ok or frame is None:
                    continue

                res = pipeline.process(frame, t)
                fps_list.append(res.fps)
                if res.utterance_final:
                    utterance_decisions.append(res.decision)

            mean_fps = float(np.mean(fps_list)) if fps_list else 0.0
            if mean_fps >= 15.0 and len(utterance_decisions) >= 1:
                results.append(
                    (
                        "pipeline_live_hardware",
                        "PASS",
                        f"mean FPS={mean_fps:.1f} (>= 15), decisions seen={utterance_decisions}",
                    )
                )
            else:
                results.append(
                    (
                        "pipeline_live_hardware",
                        "FAIL",
                        f"mean FPS={mean_fps:.1f} (want >= 15), decisions seen={utterance_decisions} (want >= 1)",
                    )
                )
        except Exception as e:
            results.append(("pipeline_live_hardware", "FAIL", f"Exception during live run: {e}"))
        finally:
            video_src.close()
            audio_src.close()

    return results
