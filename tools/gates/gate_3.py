"""TrustSync Gate 3: Face tracking, lip aperture, quality, and camera hardware."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 3.

    Returns:
        List of (check_name, result, detail).
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # Unit test files
    test_files = [
        "tests/test_id_tracker.py",
        "tests/test_lip.py",
        "tests/test_quality.py",
        "tests/test_face_tracker.py",
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

    # Check fixtures present
    fixtures_dir = root / "tests" / "fixtures"
    frontal_exists = (fixtures_dir / "face_frontal.jpg").is_file()
    mouth_open_exists = (fixtures_dir / "face_mouth_open.jpg").is_file()

    if frontal_exists and mouth_open_exists:
        results.append(
            ("fixtures_present", "PASS", "face_frontal.jpg and face_mouth_open.jpg present")
        )
    else:
        results.append(
            (
                "fixtures_present",
                "SKIP",
                "run tools/capture_sample.py face_frontal and face_mouth_open",
            )
        )

    # Hardware rows
    if not hardware:
        results.append(
            ("still_face_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
        results.append(
            ("mouth_motion_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
    else:
        from trustsync.config import load_config
        from trustsync.vision.face_tracker import FaceTracker

        cfg = load_config()
        cap = cv2.VideoCapture(cfg.video.camera_index)
        tracker = FaceTracker(cfg)
        try:
            # 1. Still face test: collect 60 frames
            print("\nLook at the camera and stay still for 3 s")
            frames_collected: list[np.ndarray] = []
            while len(frames_collected) < 60:
                ret, frame = cap.read()
                if ret:
                    frames_collected.append(frame)

            single_face_count = 0
            mouth_vals: list[float] = []
            track_ids: list[int] = []

            for f in frames_collected:
                faces = tracker.process(f)
                if len(faces) == 1:
                    single_face_count += 1
                    track_ids.append(faces[0].track_id)
                    mouth_vals.append(faces[0].mouth_open)

            frac_single = single_face_count / 60.0
            same_id = len(set(track_ids)) == 1 if track_ids else False
            mean_mouth = float(np.mean(mouth_vals)) if mouth_vals else 1.0

            if frac_single >= 0.90 and same_id and mean_mouth < 0.15:
                results.append(
                    (
                        "still_face_hardware",
                        "PASS",
                        f"{single_face_count}/60 single face, id={track_ids[0]}, mean mouth={mean_mouth:.3f}",
                    )
                )
            else:
                results.append(
                    (
                        "still_face_hardware",
                        "FAIL",
                        f"single={frac_single:.1%}, same_id={same_id}, mean_mouth={mean_mouth:.3f}",
                    )
                )

            # 2. Mouth motion test: 5 seconds
            print("\nOpen your mouth wide and close it 3 times in 5 s")
            t_end = time.monotonic() + 5.0
            dynamic_mouth_vals: list[float] = []
            while time.monotonic() < t_end:
                ret, frame = cap.read()
                if ret:
                    faces = tracker.process(frame)
                    if faces:
                        dynamic_mouth_vals.append(faces[0].mouth_open)

            if dynamic_mouth_vals:
                mouth_range = max(dynamic_mouth_vals) - min(dynamic_mouth_vals)
            else:
                mouth_range = 0.0

            if mouth_range > 0.15:
                results.append(
                    (
                        "mouth_motion_hardware",
                        "PASS",
                        f"mouth range={mouth_range:.3f} (> 0.15)",
                    )
                )
            else:
                results.append(
                    (
                        "mouth_motion_hardware",
                        "FAIL",
                        f"mouth range={mouth_range:.3f} (expected > 0.15)",
                    )
                )
        finally:
            tracker.close()
            cap.release()

    return results
