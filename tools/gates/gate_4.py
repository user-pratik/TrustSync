"""TrustSync Gate 4: Head pose, gaze estimation, addressee classification, and hardware verification."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 4.

    Args:
        hardware: If True, execute live camera hardware checks.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # Unit test files
    test_files = [
        "tests/test_head_pose.py",
        "tests/test_gaze.py",
        "tests/test_addressee.py",
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

    # Hardware row
    if not hardware:
        results.append(
            ("directed_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
    else:
        from trustsync.config import load_config
        from trustsync.vision.face_tracker import FaceTracker
        from trustsync.vision.head_pose import estimate_pose
        from trustsync.vision.gaze import gaze_off
        from trustsync.addressee.classifier import directed_score

        cfg = load_config()
        cap = cv2.VideoCapture(cfg.video.camera_index)
        if not cap.isOpened():
            results.append(
                ("directed_hardware", "FAIL", f"Cannot open camera index {cfg.video.camera_index}")
            )
            return results

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.height)
        tracker = FaceTracker(cfg)

        try:
            # Step 1: Look at the camera 3 s
            print("\nLook at the camera 3 s")
            t_end = time.monotonic() + 3.0
            direct_scores: list[float] = []

            while time.monotonic() < t_end:
                ret, frame = cap.read()
                if not ret:
                    continue
                faces = tracker.process(frame)
                if len(faces) == 1:
                    h, w = frame.shape[:2]
                    yaw, pitch, _ = estimate_pose(faces[0].pts, w, h)
                    gaze = gaze_off(faces[0].pts, w, h) if cfg.addressee.use_gaze else 0.0
                    score = directed_score(yaw, pitch, gaze, cfg.addressee)
                    direct_scores.append(score)

            med_direct = float(np.median(direct_scores)) if direct_scores else 0.0

            # Step 2: Turn your head about 40 degrees away for 3 s
            print("\nTurn your head about 40 degrees away for 3 s")
            t_end = time.monotonic() + 3.0
            turned_scores: list[float] = []

            while time.monotonic() < t_end:
                ret, frame = cap.read()
                if not ret:
                    continue
                faces = tracker.process(frame)
                if len(faces) == 1:
                    h, w = frame.shape[:2]
                    yaw, pitch, _ = estimate_pose(faces[0].pts, w, h)
                    gaze = gaze_off(faces[0].pts, w, h) if cfg.addressee.use_gaze else 0.0
                    score = directed_score(yaw, pitch, gaze, cfg.addressee)
                    turned_scores.append(score)

            med_turned = float(np.median(turned_scores)) if turned_scores else 1.0

            if med_direct > 0.8 and med_turned < 0.5:
                results.append(
                    (
                        "directed_hardware",
                        "PASS",
                        f"direct median={med_direct:.3f} (> 0.8), turned median={med_turned:.3f} (< 0.5)",
                    )
                )
            else:
                results.append(
                    (
                        "directed_hardware",
                        "FAIL",
                        f"direct median={med_direct:.3f} (want > 0.8), turned median={med_turned:.3f} (want < 0.5)",
                    )
                )
        finally:
            tracker.close()
            cap.release()

    return results
