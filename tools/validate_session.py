"""Session recording validation tool checking all criteria of SPEC section 7."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import soundfile as sf

from trustsync.audio.envelope import compute_envelope, segment_speech
from trustsync.config import load_config
from trustsync.vision.face_tracker import FaceTracker


EXPECTED_PER_SCENARIO = {
    "live_single": "ACCEPT",
    "replay_spoof": "REJECT_SPOOF",
    "multi_one_speaks": "ACCEPT",
    "not_addressed": "REJECT_NOT_ADDRESSED",
    "occlusion_lowlight": "REJECT_ANY",
    "silence": "NONE",
}

LEGAL_SPEAKER_SIDE = {"left", "right", "center", "none"}
LEGAL_LIGHTING = {"normal", "dim"}
LEGAL_SPOOF_DEVICE = {None, "none", "phone"}


def validate(path: Path | str, check_faces: bool = True) -> list[tuple[str, str, str]]:
    """Validate a session folder against all TrustSync SPEC criteria.

    Args:
        path: Path to session folder.
        check_faces: If True, execute face tracking validation on every 5th frame.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    path = Path(path)
    results: list[tuple[str, str, str]] = []

    # 1. Check all 4 files exist
    files_needed = ["video.mp4", "audio.wav", "frame_times.csv", "meta.json"]
    missing = [f for f in files_needed if not (path / f).is_file()]
    if missing:
        results.append(("files_exist", "FAIL", f"Missing files: {missing}"))
        return results
    results.append(("files_exist", "PASS", "All 4 session files present"))

    # 2. meta.json schema and consistency
    meta_path = path / "meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        results.append(("meta_valid", "FAIL", f"Invalid json: {e}"))
        return results

    required_keys = [
        "id", "scenario", "subjects", "speaker_side", "addressed", "expected",
        "lighting", "occlusion", "spoof_device", "audio_t0", "video_fps", "notes"
    ]
    missing_keys = [k for k in required_keys if k not in meta]
    if missing_keys:
        results.append(("meta_valid", "FAIL", f"Missing meta keys: {missing_keys}"))
        return results

    scenario = meta["scenario"]
    if scenario not in EXPECTED_PER_SCENARIO:
        results.append(("meta_valid", "FAIL", f"Unknown scenario: {scenario}"))
        return results

    expected = meta["expected"]
    req_expected = EXPECTED_PER_SCENARIO[scenario]
    if expected != req_expected:
        results.append(
            ("meta_valid", "FAIL", f"Expected '{expected}' does not match scenario '{scenario}' ({req_expected})")
        )
        return results

    if meta["speaker_side"] not in LEGAL_SPEAKER_SIDE:
        results.append(("meta_valid", "FAIL", f"Invalid speaker_side: {meta['speaker_side']}"))
        return results

    if meta["lighting"] not in LEGAL_LIGHTING:
        results.append(("meta_valid", "FAIL", f"Invalid lighting: {meta['lighting']}"))
        return results

    if meta.get("spoof_device") not in LEGAL_SPOOF_DEVICE:
        results.append(("meta_valid", "FAIL", f"Invalid spoof_device: {meta.get('spoof_device')}"))
        return results

    results.append(("meta_valid", "PASS", "Metadata schema and expected label valid"))

    # 3. frame_times.csv vs video.mp4
    video_path = path / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    cap_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_times: list[float] = []
    csv_path = path / "frame_times.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_times.append(float(row["t"]))

    if len(frame_times) != cap_count:
        results.append(
            ("frame_count_match", "FAIL", f"CSV rows ({len(frame_times)}) != video frames ({cap_count})")
        )
    else:
        results.append(("frame_count_match", "PASS", f"{cap_count} frames match CSV"))

    # Strictly increasing
    is_increasing = all(
        frame_times[i] > frame_times[i - 1] for i in range(1, len(frame_times))
    )
    if not is_increasing:
        results.append(("frame_times_increasing", "FAIL", "frame_times not strictly increasing"))
    else:
        results.append(("frame_times_increasing", "PASS", "Timestamps strictly increasing"))

    # 4. Measured video FPS
    n_frames = len(frame_times)
    if n_frames >= 2 and (frame_times[-1] - frame_times[0]) > 0:
        measured_fps = (n_frames - 1) / (frame_times[-1] - frame_times[0])
    else:
        measured_fps = 0.0

    if 15.0 <= measured_fps <= 35.0:
        results.append(("measured_fps", "PASS", f"Measured FPS: {measured_fps:.2f} in [15, 35]"))
    else:
        results.append(("measured_fps", "FAIL", f"Measured FPS: {measured_fps:.2f} out of [15, 35]"))

    # 5. Audio checks (16 kHz, mono, duration, clipping, alignment)
    audio_path = path / "audio.wav"
    try:
        audio_info = sf.info(str(audio_path))
        data, sr = sf.read(str(audio_path))
    except Exception as e:
        results.append(("audio_format", "FAIL", f"Failed reading audio: {e}"))
        cap.release()
        return results

    if audio_info.samplerate != 16000 or audio_info.channels != 1:
        results.append(
            ("audio_format", "FAIL", f"Audio {audio_info.samplerate}Hz/{audio_info.channels}ch (want 16000Hz/1ch)")
        )
    else:
        results.append(("audio_format", "PASS", "16000 Hz mono"))

    audio_dur = float(audio_info.duration)
    video_dur = float(frame_times[-1] - frame_times[0]) if n_frames >= 2 else 0.0
    dur_diff = abs(audio_dur - video_dur)
    if dur_diff < 0.3:
        results.append(("duration_alignment", "PASS", f"|audio - video| = {dur_diff:.3f}s < 0.3s"))
    else:
        results.append(("duration_alignment", "FAIL", f"|audio - video| = {dur_diff:.3f}s >= 0.3s"))

    # Alignment audio_t0 vs first frame
    audio_t0 = float(meta["audio_t0"])
    first_frame_t = frame_times[0] if frame_times else 0.0
    t0_diff = abs(audio_t0 - first_frame_t)
    if t0_diff < 0.5:
        results.append(("t0_alignment", "PASS", f"|audio_t0 - first frame t| = {t0_diff:.3f}s < 0.5s"))
    else:
        results.append(("t0_alignment", "FAIL", f"|audio_t0 - first frame t| = {t0_diff:.3f}s >= 0.5s"))

    # Peak below 0.99
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    if peak < 0.99:
        results.append(("audio_clipping", "PASS", f"Peak audio {peak:.3f} < 0.99"))
    else:
        results.append(("audio_clipping", "FAIL", f"Audio clipped: peak {peak:.3f} >= 0.99"))

    # 6. Speech activity / VAD
    cfg = load_config()
    env = compute_envelope(data.astype(np.float32), sr=16000, env_hz=cfg.audio.env_hz)
    segments = segment_speech(env, float(cfg.audio.env_hz), cfg.vad)

    if scenario == "silence":
        if len(segments) == 0:
            results.append(("speech_vad", "PASS", "No speech detected in silence scenario"))
        else:
            results.append(("speech_vad", "FAIL", f"Detected speech in silence scenario: {segments}"))
    else:
        # Require at least one segment of 0.4 s or more starting after 1.0 s
        valid_seg = any(
            (seg[1] - seg[0]) >= 0.4 and seg[0] >= 1.0 for seg in segments
        )
        if valid_seg:
            results.append(("speech_vad", "PASS", f"Speech segment found: {segments}"))
        else:
            results.append(
                ("speech_vad", "FAIL", f"No segment >= 0.4s starting after 1.0s found in {segments}")
            )

    # 7. Face tracking on every 5th frame
    if check_faces:
        if scenario == "occlusion_lowlight":
            results.append(("face_tracking", "SKIP", "Face check skipped for occlusion_lowlight"))
        else:
            tracker = FaceTracker(cfg)
            try:
                sampled_frames = 0
                face_detected_frames = 0
                multi_face_frames = 0
                f_idx = 0

                while True:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    if f_idx % 5 == 0:
                        sampled_frames += 1
                        faces = tracker.process(frame)
                        if len(faces) >= 1:
                            face_detected_frames += 1
                        if len(faces) >= 2:
                            multi_face_frames += 1
                    f_idx += 1

                if sampled_frames > 0:
                    face_frac = face_detected_frames / sampled_frames
                    if scenario == "multi_one_speaks":
                        multi_frac = multi_face_frames / sampled_frames
                        if multi_frac >= 0.60:
                            results.append(
                                ("face_tracking", "PASS", f"Multi-face fraction {multi_frac:.1%} >= 60%")
                            )
                        else:
                            results.append(
                                ("face_tracking", "FAIL", f"Multi-face fraction {multi_frac:.1%} < 60%")
                            )
                    else:
                        if face_frac >= 0.70:
                            results.append(
                                ("face_tracking", "PASS", f"Face fraction {face_frac:.1%} >= 70%")
                            )
                        else:
                            results.append(
                                ("face_tracking", "FAIL", f"Face fraction {face_frac:.1%} < 70%")
                            )
                else:
                    results.append(("face_tracking", "FAIL", "No video frames sampled"))
            finally:
                tracker.close()
    else:
        results.append(("face_tracking", "SKIP", "Face check disabled"))

    cap.release()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate recorded session directories.")
    parser.add_argument("path", nargs="?", default=None, help="Session directory or --all")
    parser.add_argument("--all", action="store_true", help="Validate all sessions in data/sessions")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sessions_to_check: list[Path] = []

    if args.all or args.path == "--all":
        sessions_dir = root / "data" / "sessions"
        if sessions_dir.is_dir():
            sessions_to_check = [p for p in sessions_dir.iterdir() if p.is_dir()]
    elif args.path:
        p = Path(args.path)
        if not p.is_absolute():
            p = root / p
        sessions_to_check = [p]
    else:
        parser.print_help()
        sys.exit(1)

    overall_fail = False
    for session_dir in sessions_to_check:
        print(f"\n--- Validating: {session_dir.name} ---")
        rows = validate(session_dir, check_faces=True)
        headers = ("CHECK", "RESULT", "DETAIL")
        all_data = [headers] + rows
        widths = [max(len(str(r[i])) for r in all_data) for i in range(3)]

        print(" | ".join(headers[i].ljust(widths[i]) for i in range(3)))
        print("-+-".join("-" * widths[i] for i in range(3)))
        for r in rows:
            print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(3)))
            if r[1] == "FAIL":
                overall_fail = True

    sys.exit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()
