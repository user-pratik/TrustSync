"""Integration test for evaluation CLI runner, ablation study, and report outputs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import pytest

from tests.helpers_eval import fake_clip
from trustsync.eval.metrics import records_to_json
from trustsync.eval.run_eval import main


def test_eval_run_cli_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    # Copy spec default config to tmp_path/config.yaml for testing --write-config safely
    spec_cfg_source = root / "tests" / "fixtures" / "config_spec_defaults.yaml"
    cfg_copy = tmp_path / "config.yaml"
    shutil.copyfile(spec_cfg_source, cfg_copy)

    # Build synthetic records with 3 subjects across multiple scenarios
    clips = []
    subjects = ["sub_a", "sub_b", "sub_c"]
    for i, subj in enumerate(subjects):
        clips.append(fake_clip("live_single", "ACCEPT", 0.8, 0.03, 0.9, subject=subj))
        clips.append(fake_clip("replay_spoof", "REJECT_SPOOF", 0.05, 0.001, 0.9, subject=subj))
        clips.append(fake_clip("not_addressed", "REJECT_NOT_ADDRESSED", 0.8, 0.03, 0.1, subject=subj, addressed=False))
        clips.append(fake_clip("multi_one_speaks", "ACCEPT", 0.75, 0.03, 0.85, n_tracks=2, speaker_side="left", speaker_cx=0.2, subject=subj))
    clips.append(fake_clip("silence", "NONE", None, 0.0, 0.0, duration=0.0, subject="sub_a"))

    records_file = tmp_path / "records.json"
    records_file.write_text(records_to_json(clips), encoding="utf-8")

    reports_dir = tmp_path / "reports"

    # Run in tmp_path cwd
    monkeypatch.chdir(tmp_path)
    main([
        "--records-json", str(records_file),
        "--out", str(reports_dir),
        "--calibrate",
        "--learned",
        "--write-config",
    ])

    results_file = reports_dir / "results.json"
    assert results_file.is_file(), "results.json was not created"

    with open(results_file, "r", encoding="utf-8") as f:
        res = json.load(f)

    # Verify all expected top-level keys
    expected_keys = [
        "generated_at",
        "dataset",
        "config",
        "current",
        "calibrated",
        "loso_rule",
        "loso_learned",
        "learned_kind",
        "ablation",
        "av_lag",
        "fps",
        "figures",
        "notes",
    ]
    for k in expected_keys:
        assert k in res, f"Key '{k}' missing from results.json"

    # Verify ablation rows
    ablation = res["ablation"]
    assert len(ablation) == 6
    names = [row["name"] for row in ablation]
    assert names == ["B0", "A1", "A2", "A3", "A4", "A5"]

    # B0 has far_all_reject 1.0 on a set containing should-reject clips
    assert ablation[0]["far_all_reject"] == 1.0
    # A4 (RuleGate) has lower FAR than B0
    assert ablation[4]["far_all_reject"] < ablation[0]["far_all_reject"]

    # Verify all figures exist and are > 5 KB
    figures = res["figures"]
    assert len(figures) >= 6
    for fig in figures:
        fig_path = Path(fig["path"])
        assert fig_path.is_file(), f"Figure file missing: {fig_path}"
        assert fig_path.stat().st_size > 5000, f"Figure file too small ({fig_path.stat().st_size} bytes): {fig_path}"

    # Verify --write-config created config.yaml.bak and did not touch real config.yaml
    assert (tmp_path / "config.yaml.bak").is_file()
    assert (tmp_path / "models" / "gate.joblib").is_file()
