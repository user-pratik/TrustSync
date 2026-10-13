"""TrustSync Gate 1: Config, types, synth helpers and unit tests."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 1.

    Returns:
        List of (check_name, result, detail).
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # Test files to verify
    test_files = [
        "tests/test_config.py",
        "tests/test_types.py",
        "tests/test_synth.py",
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

    # Import speed check: import trustsync.config, trustsync.types, trustsync.synth in under 2 seconds
    t0 = time.monotonic()
    try:
        # Import dynamically or via import
        import trustsync.config  # noqa: F401
        import trustsync.synth  # noqa: F401
        import trustsync.types  # noqa: F401

        elapsed = time.monotonic() - t0
        if elapsed < 2.0:
            results.append(
                (
                    "import_config_types_synth",
                    "PASS",
                    f"Imported in {elapsed:.3f} s (< 2.0 s)",
                )
            )
        else:
            results.append(
                (
                    "import_config_types_synth",
                    "FAIL",
                    f"Import took {elapsed:.3f} s (>= 2.0 s)",
                )
            )
    except Exception as e:
        results.append(("import_config_types_synth", "FAIL", str(e)))

    return results
