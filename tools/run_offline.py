"""Offline pipeline runner for recorded session directories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import cv2

from trustsync.audio.sources import FileAudioSource
from trustsync.config import load_config
from trustsync.fusion.gate import RuleGate
from trustsync.io.video_source import FileVideoSource
from trustsync.overlay.hud import draw_hud
from trustsync.pipeline import TrustSyncPipeline
from trustsync.types import frame_result_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrustSync offline on a session.")
    parser.add_argument("--session", required=True, type=Path, help="Path to session folder")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--video-out", action="store_true", help="Render annotated.mp4 video")
    args = parser.parse_args()

    session_path = args.session.resolve()
    meta_path = session_path / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    session_id = meta.get("id", session_path.name)
    out_dir = args.out if args.out is not None else Path("out") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    video_src = FileVideoSource(session_path / "video.mp4", session_path / "frame_times.csv")
    audio_src = FileAudioSource(session_path / "audio.wav", t0=float(meta["audio_t0"]), cfg=cfg.audio)
    pipeline = TrustSyncPipeline(cfg=cfg, audio=audio_src, gate=RuleGate(cfg))

    frames_csv_path = out_dir / "frames.csv"
    utterances_json_path = out_dir / "utterances.json"

    video_writer: cv2.VideoWriter | None = None
    env_hist: list[float] = []
    utterances: list[dict] = []

    try:
        with open(frames_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "frame_idx", "t", "decision", "speech_active", "env",
                "speaker_id", "liveness", "addressee", "n_faces", "fps"
            ])

            while True:
                ok, frame, t = video_src.read()
                if not ok or frame is None:
                    break

                if args.video_out and video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(out_dir / "annotated.mp4"), fourcc, float(cfg.video.fps), (w, h)
                    )

                result = pipeline.process(frame, t)
                env_hist.append(result.env)

                writer.writerow([
                    result.frame_idx,
                    f"{result.t:.4f}",
                    result.decision,
                    result.speech_active,
                    f"{result.env:.4f}",
                    result.speaker_id if result.speaker_id is not None else "",
                    f"{result.liveness:.4f}",
                    f"{result.addressee:.4f}",
                    len(result.faces),
                    f"{result.fps:.2f}",
                ])

                if result.utterance_final:
                    utterances.append(frame_result_to_dict(result))

                if video_writer is not None:
                    annotated = draw_hud(frame, result, env_hist)
                    video_writer.write(annotated)
    finally:
        video_src.close()
        audio_src.close()
        if video_writer is not None:
            video_writer.release()

    with open(utterances_json_path, "w", encoding="utf-8") as f:
        json.dump(utterances, f, indent=2)

    decisions = [u["decision"] for u in utterances]
    expected = meta.get("expected", "")
    print(f"utterances={len(utterances)} decisions={decisions} expected={expected}")


if __name__ == "__main__":
    main()
