"""Calibrate head pose zero offsets by recording user looking directly at the camera."""
from dataclasses import replace
import math
from pathlib import Path
import shutil
import sys
import time
import cv2
import numpy as np

from trustsync.config import load_config, save_config
from trustsync.vision.face_tracker import FaceTracker
from trustsync.vision.head_pose import estimate_pose


def main() -> None:
    """Collect 5 seconds of frontal gaze, compute median yaw and pitch offsets, and write to config.yaml."""
    cfg = load_config()
    cap = cv2.VideoCapture(cfg.video.camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {cfg.video.camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.video.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.video.height)

    tracker = FaceTracker(cfg)

    print("Look straight at the robot (the camera) for 5 s")
    t_end = time.monotonic() + 5.0
    yaws: list[float] = []
    pitches: list[float] = []

    try:
        while time.monotonic() < t_end:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = tracker.process(frame)
            if len(faces) == 1:
                h, w = frame.shape[:2]
                yaw, pitch, _ = estimate_pose(faces[0].pts, w, h)
                if not math.isnan(yaw) and not math.isnan(pitch):
                    yaws.append(yaw)
                    pitches.append(pitch)
    finally:
        tracker.close()
        cap.release()

    if len(yaws) < 30:
        print(
            f"Error: Collected only {len(yaws)} valid single-face frames (< 30). "
            "Calibration aborted."
        )
        sys.exit(1)

    med_yaw = float(np.median(yaws))
    med_pitch = float(np.median(pitches))

    config_path = Path("config.yaml")
    backup_path = Path("config.yaml.bak")
    shutil.copyfile(config_path, backup_path)

    new_addressee = replace(
        cfg.addressee,
        yaw_offset_deg=round(med_yaw, 2),
        pitch_offset_deg=round(med_pitch, 2),
    )
    new_cfg = replace(cfg, addressee=new_addressee)
    save_config(new_cfg, config_path)

    print(
        f"Calibration saved ({len(yaws)} frames): "
        f"yaw_offset_deg = {new_addressee.yaw_offset_deg}, "
        f"pitch_offset_deg = {new_addressee.pitch_offset_deg}"
    )


if __name__ == "__main__":
    main()
