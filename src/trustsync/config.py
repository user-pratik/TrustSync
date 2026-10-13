"""TrustSync configuration management."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class VideoCfg:
    fps: int
    width: int
    height: int
    camera_index: int
    max_faces: int


@dataclass(frozen=True)
class AudioCfg:
    sample_rate: int
    env_hz: int
    device: str | int | None
    latency_s: float
    ring_s: float


@dataclass(frozen=True)
class VadCfg:
    calib_s: float
    on_factor: float
    min_abs: float
    hangover_s: float
    adapt: float


@dataclass(frozen=True)
class UtteranceCfg:
    min_s: float
    max_s: float
    pad_s: float


@dataclass(frozen=True)
class TrackerCfg:
    max_dist: float
    max_missed: int


@dataclass(frozen=True)
class LipCfg:
    buffer_s: float


@dataclass(frozen=True)
class SyncCfg:
    window_s: float
    min_window_s: float
    max_lag_s: float
    min_lip_std: float
    env_eps: float


@dataclass(frozen=True)
class QualityCfg:
    min_face_px: int
    min_brightness: float
    max_edge_frac: float


@dataclass(frozen=True)
class AddresseeCfg:
    max_yaw_deg: float
    max_pitch_deg: float
    yaw_offset_deg: float
    pitch_offset_deg: float
    gaze_weight: float
    gaze_max_off: float
    use_gaze: bool


@dataclass(frozen=True)
class GateCfg:
    mode: str
    min_sync: float
    min_margin: float
    min_directed: float
    min_quality: float
    model_path: str


@dataclass(frozen=True)
class ServerCfg:
    host: str
    port: int
    ws_hz: int
    jpeg_quality: int


@dataclass(frozen=True)
class Config:
    video: VideoCfg
    audio: AudioCfg
    vad: VadCfg
    utterance: UtteranceCfg
    tracker: TrackerCfg
    lip: LipCfg
    sync: SyncCfg
    quality: QualityCfg
    addressee: AddresseeCfg
    gate: GateCfg
    server: ServerCfg

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


_SECTION_CLASSES: dict[str, type] = {
    "video": VideoCfg,
    "audio": AudioCfg,
    "vad": VadCfg,
    "utterance": UtteranceCfg,
    "tracker": TrackerCfg,
    "lip": LipCfg,
    "sync": SyncCfg,
    "quality": QualityCfg,
    "addressee": AddresseeCfg,
    "gate": GateCfg,
    "server": ServerCfg,
}


def validate(cfg: Config) -> None:
    """Validate configuration consistency constraints from SPEC section 3.

    Raises:
        ValueError: If any constraint is violated.
    """
    if cfg.audio.sample_rate % cfg.audio.env_hz != 0:
        raise ValueError(
            f"audio.sample_rate ({cfg.audio.sample_rate}) must be divisible by "
            f"audio.env_hz ({cfg.audio.env_hz})"
        )
    if cfg.sync.max_lag_s * cfg.video.fps < 1:
        raise ValueError(
            f"sync.max_lag_s * video.fps must be >= 1, "
            f"got {cfg.sync.max_lag_s} * {cfg.video.fps} = {cfg.sync.max_lag_s * cfg.video.fps}"
        )
    if cfg.sync.min_window_s > cfg.sync.window_s:
        raise ValueError(
            f"sync.min_window_s ({cfg.sync.min_window_s}) cannot be greater than "
            f"sync.window_s ({cfg.sync.window_s})"
        )
    if cfg.utterance.min_s > cfg.utterance.max_s:
        raise ValueError(
            f"utterance.min_s ({cfg.utterance.min_s}) cannot be greater than "
            f"utterance.max_s ({cfg.utterance.max_s})"
        )


def _find_default_config_path() -> Path:
    curr = Path(__file__).resolve().parent
    while True:
        candidate = curr / "config.yaml"
        if candidate.is_file():
            return candidate
        if curr.parent == curr:
            break
        curr = curr.parent
    raise FileNotFoundError("Could not find config.yaml by walking up from config.py")


def load_config(path: Path | str | None = None) -> Config:
    """Load, parse and validate TrustSync configuration.

    Args:
        path: Optional path to config YAML file. If None, walks up from this file.

    Returns:
        Validated Config dataclass instance.

    Raises:
        FileNotFoundError: If configuration file is missing.
        ValueError: If YAML syntax is invalid, keys are missing or unknown,
            or validation constraints fail.
    """
    if path is None:
        target_path = _find_default_config_path()
    else:
        target_path = Path(path)

    if not target_path.is_file():
        raise FileNotFoundError(f"Config file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file at {target_path} must be a YAML mapping")

    # Check for unknown sections
    for sec_key in raw_data:
        if sec_key not in _SECTION_CLASSES:
            raise ValueError(f"Unknown section in configuration: '{sec_key}'")

    # Check for missing sections and parse each section
    section_instances: dict[str, Any] = {}
    for sec_name, sec_cls in _SECTION_CLASSES.items():
        if sec_name not in raw_data:
            raise ValueError(f"Missing section in configuration: '{sec_name}'")

        sec_dict = raw_data[sec_name]
        if not isinstance(sec_dict, dict):
            raise ValueError(f"Section '{sec_name}' must be a mapping")

        expected_fields = {f.name: f for f in fields(sec_cls)}

        # Check for unknown keys in section
        for k in sec_dict:
            if k not in expected_fields:
                raise ValueError(f"Unknown key '{k}' in section '{sec_name}'")

        # Check for missing keys and convert types
        kwargs: dict[str, Any] = {}
        for f_name, f_info in expected_fields.items():
            if f_name not in sec_dict:
                raise ValueError(f"Missing key '{f_name}' in section '{sec_name}'")
            val = sec_dict[f_name]
            if f_info.type in (float, "float") and isinstance(val, (int, float)):
                val = float(val)
            elif (
                f_info.type in (int, "int")
                and isinstance(val, int)
                and not isinstance(val, bool)
            ):
                val = int(val)
            kwargs[f_name] = val

        section_instances[sec_name] = sec_cls(**kwargs)

    cfg = Config(**section_instances)
    validate(cfg)
    return cfg


def save_config(cfg: Config, path: Path | str) -> None:
    """Save Config instance to YAML file.

    Args:
        cfg: Config instance to save.
        path: Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict(), f, sort_keys=False)
