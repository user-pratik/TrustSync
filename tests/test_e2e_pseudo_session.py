"""End-to-end pseudo session verification test using captured frontal photo."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

from tests.helpers_world import make_pseudo_session


def test_e2e_pseudo_session(tmp_path: Path) -> None:
    """Verify offline pipeline on a static face pseudo-session outputs REJECT_SPOOF."""
    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "tests" / "fixtures"
    frontal_path = fixtures_dir / "face_frontal.jpg"

    if not frontal_path.is_file():
        pytest.skip("tests/fixtures/face_frontal.jpg is missing")

    rng = np.random.default_rng(1234)
    session_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    out_dir = tmp_path / "out"

    # Run tools/run_offline.py through subprocess
    cmd = [
        sys.executable,
        str(root / "tools" / "run_offline.py"),
        "--session",
        str(session_dir),
        "--out",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, f"run_offline failed with code {proc.returncode}:\n{proc.stderr}"

    # Assert utterances.json has exactly 1 entry with REJECT_SPOOF
    utterances_path = out_dir / "utterances.json"
    assert utterances_path.is_file()
    with open(utterances_path, "r", encoding="utf-8") as f:
        utts = json.load(f)

    assert len(utts) == 1, f"Expected 1 utterance, got {len(utts)}"
    assert utts[0]["decision"] == "REJECT_SPOOF"
