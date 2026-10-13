"""TrustSync Gate 2: Audio envelope, VAD, sources and tests."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 2.

    Returns:
        List of (check_name, result, detail).
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    test_files = [
        "tests/test_envelope.py",
        "tests/test_vad.py",
        "tests/test_sources.py",
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

    # Hardware check: live recording and VAD segmentation
    if not hardware:
        results.append(
            (
                "live_speech_hardware",
                "SKIP",
                "Hardware check skipped (run with --hardware)",
            )
        )
    else:
        try:
            from trustsync.audio.envelope import segment_speech
            from trustsync.audio.sources import LiveAudioSource
            from trustsync.config import load_config

            cfg = load_config()
            print("\nSPEAK: stay silent 2 s, then say one-two-three twice")
            src = LiveAudioSource(cfg.audio)
            src.start()
            time.sleep(8.0)
            src.close()

            with src._lock:
                e_arr = np.array([pt[1] for pt in src._buffer], dtype=np.float32)

            segments = segment_speech(e_arr, float(cfg.audio.env_hz), cfg.vad)
            num_segs = len(segments)
            first_start = segments[0][0] if segments else 0.0

            if 1 <= num_segs <= 4 and first_start > 1.0:
                results.append(
                    (
                        "live_speech_hardware",
                        "PASS",
                        f"{num_segs} segments, first at {first_start:.2f} s",
                    )
                )
            else:
                results.append(
                    (
                        "live_speech_hardware",
                        "FAIL",
                        f"{num_segs} segments (expected 1..4), first at {first_start:.2f} s (expected > 1.0 s)",
                    )
                )
        except Exception as e:
            results.append(("live_speech_hardware", "FAIL", str(e)))

    return results
