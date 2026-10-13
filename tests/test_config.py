"""Tests for TrustSync configuration management."""
from pathlib import Path
import pytest
import yaml

from trustsync.config import Config, load_config, save_config, validate


def test_fixture_matches_spec_values(spec_cfg: Config) -> None:
    """Verify that fixture loads every SPEC section 3 default value."""
    # Video
    assert spec_cfg.video.fps == 30
    assert spec_cfg.video.width == 640
    assert spec_cfg.video.height == 480
    assert spec_cfg.video.camera_index == 0
    assert spec_cfg.video.max_faces == 4

    # Audio
    assert spec_cfg.audio.sample_rate == 16000
    assert spec_cfg.audio.env_hz == 100
    assert spec_cfg.audio.device is None
    assert spec_cfg.audio.latency_s == 0.0
    assert spec_cfg.audio.ring_s == 10.0

    # VAD
    assert spec_cfg.vad.calib_s == 1.0
    assert spec_cfg.vad.on_factor == 3.0
    assert spec_cfg.vad.min_abs == 0.004
    assert spec_cfg.vad.hangover_s == 0.30
    assert spec_cfg.vad.adapt == 0.005

    # Utterance
    assert spec_cfg.utterance.min_s == 0.40
    assert spec_cfg.utterance.max_s == 4.0
    assert spec_cfg.utterance.pad_s == 0.25

    # Tracker
    assert spec_cfg.tracker.max_dist == 0.15
    assert spec_cfg.tracker.max_missed == 15

    # Lip
    assert spec_cfg.lip.buffer_s == 6.0

    # Sync
    assert spec_cfg.sync.window_s == 1.5
    assert spec_cfg.sync.min_window_s == 0.6
    assert spec_cfg.sync.max_lag_s == 0.20
    assert spec_cfg.sync.min_lip_std == 0.004
    assert spec_cfg.sync.env_eps == 0.0001

    # Quality
    assert spec_cfg.quality.min_face_px == 80
    assert spec_cfg.quality.min_brightness == 40.0
    assert spec_cfg.quality.max_edge_frac == 0.05

    # Addressee
    assert spec_cfg.addressee.max_yaw_deg == 25.0
    assert spec_cfg.addressee.max_pitch_deg == 20.0
    assert spec_cfg.addressee.yaw_offset_deg == 0.0
    assert spec_cfg.addressee.pitch_offset_deg == 0.0
    assert spec_cfg.addressee.gaze_weight == 0.3
    assert spec_cfg.addressee.gaze_max_off == 0.6
    assert spec_cfg.addressee.use_gaze is True

    # Gate
    assert spec_cfg.gate.mode == "rule"
    assert spec_cfg.gate.min_sync == 0.35
    assert spec_cfg.gate.min_margin == 0.10
    assert spec_cfg.gate.min_directed == 0.5
    assert spec_cfg.gate.min_quality == 0.5
    assert spec_cfg.gate.model_path == "models/gate.joblib"

    # Server
    assert spec_cfg.server.host == "127.0.0.1"
    assert spec_cfg.server.port == 8000
    assert spec_cfg.server.ws_hz == 20
    assert spec_cfg.server.jpeg_quality == 80


def test_live_config_equals_fixture_and_validates(spec_cfg: Config) -> None:
    """Verify live repo config.yaml has identical structure and passes validation."""
    live_cfg = load_config()
    assert live_cfg.to_dict().keys() == spec_cfg.to_dict().keys()
    for sec in live_cfg.to_dict():
        assert live_cfg.to_dict()[sec].keys() == spec_cfg.to_dict()[sec].keys()
    validate(live_cfg)


def test_save_and_load_round_trip(spec_cfg: Config, tmp_path: Path) -> None:
    """Verify save_config and load_config round-trip preserving values."""
    temp_yaml = tmp_path / "round_trip.yaml"
    save_config(spec_cfg, temp_yaml)
    loaded = load_config(temp_yaml)
    assert loaded == spec_cfg


def test_unknown_and_missing_keys_raise(spec_cfg: Config, tmp_path: Path) -> None:
    """Verify unknown and missing YAML keys raise ValueError."""
    base_dict = spec_cfg.to_dict()

    # Unknown section
    d1 = dict(base_dict)
    d1["unknown_section"] = {}
    p1 = tmp_path / "unknown_sec.yaml"
    p1.write_text(yaml.dump(d1), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown section"):
        load_config(p1)

    # Missing section
    d2 = dict(base_dict)
    del d2["video"]
    p2 = tmp_path / "missing_sec.yaml"
    p2.write_text(yaml.dump(d2), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing section"):
        load_config(p2)

    # Unknown key in section
    d3 = {k: dict(v) for k, v in base_dict.items()}
    d3["video"]["extra_key"] = 123
    p3 = tmp_path / "unknown_key.yaml"
    p3.write_text(yaml.dump(d3), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        load_config(p3)

    # Missing key in section
    d4 = {k: dict(v) for k, v in base_dict.items()}
    del d4["video"]["fps"]
    p4 = tmp_path / "missing_key.yaml"
    p4.write_text(yaml.dump(d4), encoding="utf-8")
    with pytest.raises(ValueError, match="Missing key"):
        load_config(p4)


def test_four_validate_rules_raise(spec_cfg: Config, tmp_path: Path) -> None:
    """Verify all four validate() consistency rules raise ValueError."""
    # Rule 1: audio.sample_rate % audio.env_hz != 0
    d1 = {k: dict(v) for k, v in spec_cfg.to_dict().items()}
    d1["audio"]["sample_rate"] = 16001
    p1 = tmp_path / "rule1.yaml"
    p1.write_text(yaml.dump(d1), encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        load_config(p1)

    # Rule 2: sync.max_lag_s * video.fps < 1
    d2 = {k: dict(v) for k, v in spec_cfg.to_dict().items()}
    d2["sync"]["max_lag_s"] = 0.01
    d2["video"]["fps"] = 30
    p2 = tmp_path / "rule2.yaml"
    p2.write_text(yaml.dump(d2), encoding="utf-8")
    with pytest.raises(ValueError, match="sync.max_lag_s"):
        load_config(p2)

    # Rule 3: sync.min_window_s > sync.window_s
    d3 = {k: dict(v) for k, v in spec_cfg.to_dict().items()}
    d3["sync"]["min_window_s"] = 2.0
    d3["sync"]["window_s"] = 1.0
    p3 = tmp_path / "rule3.yaml"
    p3.write_text(yaml.dump(d3), encoding="utf-8")
    with pytest.raises(ValueError, match="min_window_s"):
        load_config(p3)

    # Rule 4: utterance.min_s > utterance.max_s
    d4 = {k: dict(v) for k, v in spec_cfg.to_dict().items()}
    d4["utterance"]["min_s"] = 5.0
    d4["utterance"]["max_s"] = 4.0
    p4 = tmp_path / "rule4.yaml"
    p4.write_text(yaml.dump(d4), encoding="utf-8")
    with pytest.raises(ValueError, match="utterance.min_s"):
        load_config(p4)
