"""Unit tests for Voice Activity Detection and speech segmentation."""
import numpy as np

from trustsync.audio.envelope import VAD, segment_speech
from trustsync.config import Config
from trustsync.synth import make_env


def test_vad_speech_segmentation(spec_cfg: Config) -> None:
    """Verify segment_speech identifies 2 speech bursts with accurate boundaries."""
    rng = np.random.default_rng(1234)
    truth = [(1.5, 2.5), (4.0, 4.6)]
    env = make_env(6.0, 100.0, truth, rng)

    segments = segment_speech(env, 100.0, spec_cfg.vad)
    assert len(segments) == 2
    for (s_est, e_est), (s_true, e_true) in zip(segments, truth):
        assert abs(s_est - s_true) <= 0.06
        assert abs(e_est - e_true) <= 0.06


def test_loud_signal_during_calibration_is_false(spec_cfg: Config) -> None:
    """Verify loud signal from t=0 returns False during the calibration period."""
    dt = 1.0 / 100.0
    vad = VAD(spec_cfg.vad, dt=dt)
    calib_samples = round(spec_cfg.vad.calib_s / dt)

    for _ in range(calib_samples):
        assert vad.update(1.0) is False
    assert vad.calibrated is True


def test_burst_gap_merging_and_separation(spec_cfg: Config) -> None:
    """Verify bursts separated by 0.2 s merge, while bursts separated by 0.5 s stay two."""
    rng = np.random.default_rng(1234)

    # 0.2 s gap (< hangover_s 0.3 s) merges
    bursts_close = [(1.5, 2.0), (2.2, 2.7)]
    env_close = make_env(4.0, 100.0, bursts_close, rng)
    segs_close = segment_speech(env_close, 100.0, spec_cfg.vad)
    assert len(segs_close) == 1

    # 0.5 s gap (> hangover_s 0.3 s) stays two
    bursts_apart = [(1.5, 2.0), (2.5, 3.0)]
    env_apart = make_env(4.0, 100.0, bursts_apart, rng)
    segs_apart = segment_speech(env_apart, 100.0, spec_cfg.vad)
    assert len(segs_apart) == 2


def test_slow_noise_floor_ramp_no_detection(spec_cfg: Config) -> None:
    """Verify a slow ramp from 0.003 to 0.006 over 10 s causes no speech detection."""
    dt = 0.01
    vad = VAD(spec_cfg.vad, dt=dt)
    n = round(10.0 / dt)
    ramp = np.linspace(0.003, 0.006, n)

    for val in ramp:
        assert vad.update(float(val)) is False


def test_noise_floor_never_negative(spec_cfg: Config) -> None:
    """Verify VAD noise_floor never becomes negative."""
    dt = 0.01
    vad = VAD(spec_cfg.vad, dt=dt)

    for _ in range(200):
        vad.update(0.0)
        assert vad.noise_floor >= 0.0
