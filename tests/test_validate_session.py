"""Tests for session validator and mutation error detection."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from tests.helpers_world import make_pseudo_session
from tools.validate_session import validate


def test_pseudo_session_validates(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    session_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")

    # 1. passes validate(check_faces=False)
    results_no_faces = validate(session_dir, check_faces=False)
    for check_name, res, detail in results_no_faces:
        assert res in ("PASS", "SKIP"), f"Check {check_name} failed unexpectedly: {detail}"

    # 2. also run once with check_faces=True and expect PASS when fixture exists
    root = Path(__file__).resolve().parents[1]
    if (root / "tests" / "fixtures" / "face_frontal.jpg").is_file():
        results_faces = validate(session_dir, check_faces=True)
        for check_name, res, detail in results_faces:
            assert res == "PASS", f"Check {check_name} failed: {detail}"


def test_mutation_delete_meta_key(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    mut_dir = tmp_path / "mut_delete_meta_key"
    shutil.copytree(base_dir, mut_dir)

    meta_file = mut_dir / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    del meta["expected"]
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("meta" in r[2].lower() or "missing" in r[2].lower() for r in fails), f"Detail didn't name problem: {fails}"


def test_mutation_shuffle_frame_times(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    mut_dir = tmp_path / "mut_shuffle_csv"
    shutil.copytree(base_dir, mut_dir)

    csv_file = mut_dir / "frame_times.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # swap row 5 and row 6
    rows[5], rows[6] = rows[6], rows[5]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("strictly increasing" in r[2].lower() or "frame_times" in r[0] for r in fails), f"Detail didn't name problem: {fails}"


def test_mutation_rewrite_audio_44100(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    mut_dir = tmp_path / "mut_audio_44100"
    shutil.copytree(base_dir, mut_dir)

    audio_file = mut_dir / "audio.wav"
    data, _ = sf.read(str(audio_file))
    sf.write(str(audio_file), data, 44100)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("44100" in r[2] or "16000" in r[2] for r in fails), f"Detail didn't name problem: {fails}"


def test_mutation_add_audio_t0(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    mut_dir = tmp_path / "mut_audio_t0"
    shutil.copytree(base_dir, mut_dir)

    meta_file = mut_dir / "meta.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["audio_t0"] += 5.0
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("audio_t0" in r[2] or "0.5" in r[2] for r in fails), f"Detail didn't name problem: {fails}"


def test_mutation_audio_zeros_live_single(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="live_single")
    mut_dir = tmp_path / "mut_audio_zeros"
    shutil.copytree(base_dir, mut_dir)

    audio_file = mut_dir / "audio.wav"
    zeros = np.zeros(128000, dtype=np.float32)
    sf.write(str(audio_file), zeros, 16000)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("segment" in r[2].lower() or "speech_vad" in r[0] for r in fails), f"Detail didn't name problem: {fails}"


def test_mutation_drop_last_frame_row(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    base_dir = make_pseudo_session(tmp_path, rng, scenario="replay_spoof")
    mut_dir = tmp_path / "mut_drop_last_row"
    shutil.copytree(base_dir, mut_dir)

    csv_file = mut_dir / "frame_times.csv"
    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows = rows[:-1]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    results = validate(mut_dir, check_faces=False)
    fails = [r for r in results if r[1] == "FAIL"]
    assert len(fails) > 0, "Expected validate to FAIL"
    assert any("csv rows" in r[2].lower() or "frame_count_match" in r[0] for r in fails), f"Detail didn't name problem: {fails}"
