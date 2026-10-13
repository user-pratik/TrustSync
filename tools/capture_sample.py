"""Camera sample capture utility for test fixtures."""
import argparse
from pathlib import Path
import sys
import time
import cv2

from trustsync.config import load_config


def main() -> None:
    """Preview webcam feed, countdown 3 seconds, and save snapshot to tests/fixtures/NAME.jpg."""
    parser = argparse.ArgumentParser(description="Capture camera image sample.")
    parser.add_argument("NAME", help="Output file base name (saved as NAME.jpg)")
    args = parser.parse_args()

    cfg = load_config()
    cap = cv2.VideoCapture(cfg.video.camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera index {cfg.video.camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.height)

    out_dir = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.NAME}.jpg"

    start_time = time.monotonic()
    countdown_duration = 3.0
    saved_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            elapsed = time.monotonic() - start_time
            remaining = countdown_duration - elapsed

            preview = frame.copy()
            if remaining > 0:
                count_val = int(remaining) + 1
                prompt_text = f"Look at the camera. Saving in {count_val}..."
            else:
                prompt_text = "Saving..."

            cv2.putText(
                preview,
                prompt_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("TrustSync Capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Capture aborted by user.")
                sys.exit(0)

            if elapsed >= countdown_duration:
                saved_frame = frame
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if saved_frame is not None:
        cv2.imwrite(str(out_path), saved_frame)
        h, w = saved_frame.shape[:2]
        print(f"Saved: {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
