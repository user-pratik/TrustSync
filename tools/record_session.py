"""Interactive session recording tool following TrustSync protocol."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import cv2

from trustsync.config import load_config
from trustsync.io.recorder import SessionRecorder
from tools.validate_session import EXPECTED_PER_SCENARIO, validate


PROMPT_COMMANDS = [
    "robot, bring me water",
    "stop",
    "turn left",
    "what time is it",
    "open the door",
    "call the nurse",
    "turn on the light",
    "go to the kitchen",
    "play some music",
    "follow me",
]


def get_next_session_id(base_prefix: str, sessions_dir: Path) -> str:
    """Find next available 3-digit counter for session directory."""
    counter = 1
    while True:
        candidate = f"{base_prefix}_{counter:03d}"
        if not (sessions_dir / candidate).exists():
            return candidate
        counter += 1


_get_next_session_id = get_next_session_id


def countdown(seconds: int = 3) -> bool:
    """Show countdown preview window. Return False if aborted with q."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return True

    t_end = time.monotonic() + seconds
    aborted = False
    try:
        while time.monotonic() < t_end:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rem = max(1, int(t_end - time.monotonic()) + 1)
            cv2.putText(
                frame,
                f"Starting in {rem}...",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("TrustSync Countdown", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                aborted = True
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return not aborted


def main() -> None:
    parser = argparse.ArgumentParser(description="Record TrustSync evaluation sessions.")
    parser.add_argument("--scenario", required=True, choices=list(EXPECTED_PER_SCENARIO.keys()))
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--speaker-side", choices=["left", "right", "center", "none"], default="center")
    parser.add_argument("--addressed", choices=["true", "false"], default="true")
    parser.add_argument("--lighting", choices=["normal", "dim"], default="normal")
    parser.add_argument("--occlusion", choices=["true", "false"], default="false")
    parser.add_argument("--spoof-device", choices=["none", "phone"], default="none")
    parser.add_argument("--duration", type=float, default=7.0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--notes", default="")
    parser.add_argument("--sessions-dir", default=None, help="Optional custom sessions directory")
    args = parser.parse_args()

    cfg = load_config()
    root = Path(__file__).resolve().parents[1]
    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else root / "data" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    first_subj = str(args.subjects[0]).lower().replace(" ", "_")
    prefix = f"{args.scenario}_{first_subj}"
    expected = EXPECTED_PER_SCENARIO[args.scenario]
    addressed = args.addressed.lower() == "true"
    occlusion = args.occlusion.lower() == "true"
    spoof_device = None if args.spoof_device == "none" else args.spoof_device

    for clip_idx in range(args.count):
        session_id = _get_next_session_id(prefix, sessions_dir)
        session_dir = sessions_dir / session_id

        # Determine prompt text
        prompt_text = ""
        if args.scenario not in ("silence", "replay_spoof"):
            prompt_text = f"Say: {PROMPT_COMMANDS[clip_idx % len(PROMPT_COMMANDS)]}"

        print(f"\n[{clip_idx + 1}/{args.count}] Preparing to record: {session_id}")
        if prompt_text:
            print(f"PROMPT: {prompt_text}")

        # 3 second countdown
        if not countdown(3):
            print("Recording aborted by user.")
            break

        meta = {
            "id": session_id,
            "scenario": args.scenario,
            "subjects": args.subjects,
            "speaker_side": args.speaker_side,
            "addressed": addressed,
            "expected": expected,
            "lighting": args.lighting,
            "occlusion": occlusion,
            "spoof_device": spoof_device,
            "notes": args.notes,
            "prompt_text": prompt_text,
        }

        recorder = SessionRecorder(cfg=cfg, out_dir=session_dir, meta=meta)
        recorder.run(duration_s=args.duration, preview=True)

        # Validate recording
        print("\nValidating recorded clip...")
        results = validate(session_dir, check_faces=True)
        has_failure = False
        for chk, res, det in results:
            print(f"  {chk}: {res} ({det})")
            if res == "FAIL":
                has_failure = True

        if has_failure:
            print("\n*** RE-RECORD THIS CLIP ***")


if __name__ == "__main__":
    main()
