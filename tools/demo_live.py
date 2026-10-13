"""TrustSync live demonstration viewer."""
from __future__ import annotations

import argparse
import sys
import time
import cv2
import numpy as np

from trustsync.addressee.classifier import directed_score
from trustsync.audio.sources import LiveAudioSource
from trustsync.config import load_config
from trustsync.io.video_source import WebcamSource
from trustsync.overlay.hud import draw_hud
from trustsync.pipeline import TrustSyncPipeline
from trustsync.vision.face_tracker import FaceTracker
from trustsync.vision.gaze import gaze_off
from trustsync.vision.head_pose import estimate_pose


def main() -> None:
    """Run real-time video demo showing FaceTracker, pose, or full pipeline with HUD."""
    parser = argparse.ArgumentParser(description="Live camera pipeline demonstration.")
    parser.add_argument(
        "--stage",
        choices=["faces", "pose", "full"],
        default="faces",
        help="Pipeline processing stage to visualize.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.stage == "full":
        video_src = WebcamSource(cfg)
        audio_src = LiveAudioSource(cfg.audio)
        audio_src.start()
        pipeline = TrustSyncPipeline(cfg, audio_src)
        env_hist: list[float] = []

        try:
            while True:
                ok, frame, t = video_src.read()
                if not ok or frame is None:
                    continue

                result = pipeline.process(frame, t)
                env_hist.append(result.env)

                if result.utterance_final:
                    print(
                        f"[{result.t:.2f}] decision={result.decision} "
                        f"speaker={result.speaker_id} liveness={result.liveness:.2f} "
                        f"addressee={result.addressee:.2f} reason='{result.reason}'"
                    )

                annotated = draw_hud(frame, result, env_hist)
                cv2.imshow("TrustSync Live Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Rebuild pipeline to re-calibrate VAD
                    pipeline = TrustSyncPipeline(cfg, audio_src)
                    env_hist.clear()
                    print("Pipeline rebuilt (VAD recalibrating).")
        finally:
            video_src.close()
            audio_src.close()
            cv2.destroyAllWindows()
        return

    # Stages 'faces' and 'pose'
    cap = cv2.VideoCapture(cfg.video.camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {cfg.video.camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.height)

    tracker = FaceTracker(cfg)

    prev_time = time.monotonic()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            h, w = frame.shape[:2]
            faces = tracker.process(frame)

            for f in faces:
                x1 = int(round(f.bbox[0] * w))
                y1 = int(round(f.bbox[1] * h))
                x2 = int(round(f.bbox[2] * w))
                y2 = int(round(f.bbox[3] * h))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if args.stage == "faces":
                    label = f"id {f.track_id} mouth {f.mouth_open:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                elif args.stage == "pose":
                    yaw, pitch, roll = estimate_pose(f.pts, w, h)
                    g_off = gaze_off(f.pts, w, h)
                    dir_score = directed_score(yaw, pitch, g_off, cfg.addressee)

                    label = (
                        f"{int(round(yaw))}° {int(round(pitch))}° {int(round(roll))}° "
                        f"gaze {g_off:.2f}"
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    # Horizontal bar for directed score
                    bar_w = 80
                    bar_h = 8
                    bar_y = min(h - 12, y2 + 6)
                    filled = int(round(dir_score * bar_w))
                    cv2.rectangle(
                        frame,
                        (x1, bar_y),
                        (x1 + bar_w, bar_y + bar_h),
                        (50, 50, 50),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, bar_y),
                        (x1 + filled, bar_y + bar_h),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, bar_y),
                        (x1 + bar_w, bar_y + bar_h),
                        (200, 200, 200),
                        1,
                    )

                    # Arrow starting at landmark 1
                    lm1_x = int(round(f.pts[1, 0] * w))
                    lm1_y = int(round(f.pts[1, 1] * h))
                    dx = int(
                        np.clip(
                            3.0 * (yaw - cfg.addressee.yaw_offset_deg), -120.0, 120.0
                        )
                    )
                    dy = int(
                        np.clip(
                            3.0 * (pitch - cfg.addressee.pitch_offset_deg),
                            -120.0,
                            120.0,
                        )
                    )
                    cv2.arrowedLine(
                        frame,
                        (lm1_x, lm1_y),
                        (lm1_x + dx, lm1_y + dy),
                        (0, 255, 255),
                        2,
                    )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("TrustSync Live Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
