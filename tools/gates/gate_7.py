"""TrustSync Gate 7: Session recorder, session validator, and recording protocol."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 7.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    test_files = [
        "tests/test_validate_session.py",
        "tests/test_recorder.py",
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
            ("recorder_live_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
    else:
        from trustsync.config import load_config
        from trustsync.io.recorder import SessionRecorder
        from tools.validate_session import validate

        print("\n[Gate 7 Hardware Test] SAY: robot, stop")
        cfg = load_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_path = Path(tmp_dir) / "gate_test_session"
            meta = {
                "id": "live_single_gatetest_001",
                "scenario": "live_single",
                "subjects": ["gatetest"],
                "speaker_side": "center",
                "addressed": True,
                "expected": "ACCEPT",
                "lighting": "normal",
                "occlusion": False,
                "spoof_device": None,
                "notes": "Gate 7 hardware verification",
                "prompt_text": "Say: robot, stop",
            }
            recorder = SessionRecorder(cfg=cfg, out_dir=temp_path, meta=meta)
            recorder.run(duration_s=10.0, preview=True)

            val_results = validate(temp_path, check_faces=True)
            failing = [f"{chk}: {detail}" for chk, res, detail in val_results if res == "FAIL"]
            if not failing:
                results.append(
                    ("recorder_live_hardware", "PASS", "10s recording passed all validation checks")
                )
            else:
                results.append(
                    ("recorder_live_hardware", "FAIL", f"Validation failures: {'; '.join(failing)}")
                )

    return results
