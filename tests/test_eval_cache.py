"""Tests for evaluation cache invalidation and reuse."""
from __future__ import annotations

import dataclasses
from pathlib import Path
import numpy as np
import pytest

from tests.helpers_world import make_pseudo_session
from trustsync.config import Config
from trustsync.eval import run_eval
from trustsync.eval.run_eval import load_or_process


def test_eval_cache_reuse_and_invalidation(tmp_path: Path, spec_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "tests" / "fixtures" / "face_frontal.jpg").is_file():
        pytest.skip("tests/fixtures/face_frontal.jpg is missing")

    rng = np.random.default_rng(1234)
    session_dir = make_pseudo_session(tmp_path / "sessions", rng, scenario="replay_spoof")
    cache_dir = tmp_path / "cache"

    call_count = 0
    orig_process = run_eval.process_session

    def counting_process(s_dir: Path, cfg: Config):
        nonlocal call_count
        call_count += 1
        return orig_process(s_dir, cfg)

    monkeypatch.setattr(run_eval, "process_session", counting_process)

    # 1. First call: must process
    rec1 = load_or_process(session_dir, spec_cfg, cache_dir)
    assert call_count == 1
    assert rec1.id == "pseudo_replay_spoof_001"

    # 2. Second call with same cfg: must reuse cache, call_count stays 1
    rec2 = load_or_process(session_dir, spec_cfg, cache_dir)
    assert call_count == 1
    assert rec2.id == rec1.id

    # 3. Change cfg.sync.window_s: perception feature hash changes, must re-process
    cfg_new_sync = dataclasses.replace(
        spec_cfg,
        sync=dataclasses.replace(spec_cfg.sync, window_s=2.0),
    )
    rec3 = load_or_process(session_dir, cfg_new_sync, cache_dir)
    assert call_count == 2
    assert rec3.id == rec1.id

    # 4. Change cfg.gate.min_sync: gate threshold is NOT in hash, must NOT re-process
    cfg_new_gate = dataclasses.replace(
        cfg_new_sync,
        gate=dataclasses.replace(cfg_new_sync.gate, min_sync=0.9),
    )
    rec4 = load_or_process(session_dir, cfg_new_gate, cache_dir)
    assert call_count == 2
    assert rec4.id == rec1.id
