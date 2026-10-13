"""Generate demo replay session for TrustSync UI demonstrations."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from trustsync.synth import make_env


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    fixtures_dir = root / "tests" / "fixtures"
    frontal_path = fixtures_dir / "face_frontal.jpg"

    if not frontal_path.is_file():
        print(f"Error: Required fixture not found: {frontal_path}")
        print("Run tools/capture_sample.py face_frontal to capture it.")
        return 1

    out_dir = root / "data" / "sessions" / "demo_replay_000"
    if out_dir.exists():
        print(f"Demo session already exists at {out_dir}. Refusing to overwrite.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    img = cv2.imread(str(frontal_path))
    if img is None:
        print(f"Error: Failed to read image from {frontal_path}")
        return 1

    h, w = img.shape[:2]
    video_path = out_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    for _ in range(240):
        vw.write(img)
    vw.release()

    csv_path = out_dir / "frame_times.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "t"])
        for k in range(240):
            writer.writerow([k, f"{100.0 + k / 30.0:.4f}"])

    env_100hz = make_env(8.0, 100.0, [(3.0, 4.5)], rng)
    t_env = np.arange(len(env_100hz), dtype=np.float64) / 100.0
    t_audio = np.arange(128000, dtype=np.float64) / 16000.0
    env_16k = np.interp(t_audio, t_env, env_100hz)

    noise = rng.standard_normal(128000).astype(np.float32)
    audio_data = (noise * env_16k).astype(np.float32)
    max_val = max(1e-4, float(np.max(np.abs(audio_data))))
    audio_data = (audio_data / max_val * 0.5).astype(np.float32)

    audio_path = out_dir / "audio.wav"
    sf.write(str(audio_path), audio_data, 16000)

    meta = {
        "id": "demo_replay_000",
        "scenario": "replay_spoof",
        "subjects": ["demo"],
        "speaker_side": "center",
        "addressed": True,
        "expected": "REJECT_SPOOF",
        "lighting": "normal",
        "occlusion": False,
        "spoof_device": "phone",
        "audio_t0": 100.0,
        "video_fps": 30.0,
        "notes": "Generated pseudo session from face_frontal photo with synthetic speech envelope",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Created demo session at {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
