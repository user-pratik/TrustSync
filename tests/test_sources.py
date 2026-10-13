"""Unit tests for audio stream sources."""
from pathlib import Path
import time
import numpy as np
import pytest
import soundfile as sf

from trustsync.audio.sources import FileAudioSource, LiveAudioSource, NumpyAudioSource
from trustsync.config import Config


def test_numpy_audio_source() -> None:
    """Verify NumpyAudioSource exact values, linear midpoints, and boundaries."""
    env = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    env_hz = 10.0
    t0 = 1.0
    src = NumpyAudioSource(env, env_hz, t0)

    # Exact grid times: 1.0, 1.1, 1.2
    grid_times = np.array([1.0, 1.1, 1.2])
    np.testing.assert_allclose(src.envelope_at(grid_times), [0.1, 0.2, 0.3], atol=1e-6)

    # Linear midpoints: 1.05, 1.15
    midpoints = np.array([1.05, 1.15])
    np.testing.assert_allclose(src.envelope_at(midpoints), [0.15, 0.25], atol=1e-6)

    # Outside: 0.5, 1.5 -> 0.0
    outside = np.array([0.5, 1.5])
    np.testing.assert_allclose(src.envelope_at(outside), [0.0, 0.0], atol=1e-6)

    # latest() returns last (t, value)
    t_last, val_last = src.latest()
    assert abs(t_last - 1.2) < 1e-6
    assert abs(val_last - 0.3) < 1e-6


def test_file_audio_source(spec_cfg: Config, tmp_path: Path) -> None:
    """Verify FileAudioSource t0 offset, sample rate check, and stereo downmix."""
    sr = spec_cfg.audio.sample_rate
    duration_s = 3.0
    t = np.arange(int(sr * duration_s)) / sr
    audio = np.zeros(len(t), dtype=np.float32)

    # 1 kHz sine burst centered around 1.5 s with amplitude 0.5
    burst_mask = (t >= 1.2) & (t <= 1.8)
    audio[burst_mask] = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t[burst_mask])

    wav_path = tmp_path / "valid.wav"
    sf.write(str(wav_path), audio, sr)

    t0 = 100.0
    src = FileAudioSource(wav_path, t0=t0, cfg=spec_cfg.audio)
    val = src.envelope_at(np.array([t0 + 1.5]))[0]
    # Expected RMS around 0.3536
    assert abs(val - 0.3536) < 0.05

    # WAV with invalid sample rate 44100 Hz raises ValueError
    bad_wav = tmp_path / "bad_sr.wav"
    sf.write(str(bad_wav), audio[:4410], 44100)
    with pytest.raises(ValueError, match="sample_rate"):
        FileAudioSource(bad_wav, t0=0.0, cfg=spec_cfg.audio)

    # Stereo WAV is mixed down to mono
    stereo = np.column_stack([audio, audio * 0.5])
    stereo_wav = tmp_path / "stereo.wav"
    sf.write(str(stereo_wav), stereo, sr)
    stereo_src = FileAudioSource(stereo_wav, t0=t0, cfg=spec_cfg.audio)
    val_stereo = stereo_src.envelope_at(np.array([t0 + 1.5]))[0]
    assert 0.2 < val_stereo < 0.35


@pytest.mark.hardware
def test_live_audio_source_hardware(spec_cfg: Config) -> None:
    """Hardware test: LiveAudioSource produces monotonic timestamps and non-zero RMS."""
    src = LiveAudioSource(spec_cfg.audio)
    src.start()
    time.sleep(3.0)
    src.close()

    with src._lock:
        stored_times = [p[0] for p in src._buffer]
        stored_rms = [p[1] for p in src._buffer]

    assert len(stored_times) > 10
    for i in range(len(stored_times) - 1):
        assert stored_times[i] < stored_times[i + 1]
    assert any(r > 0.0 for r in stored_rms)
