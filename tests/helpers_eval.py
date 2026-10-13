"""Helper fixtures and fake clip builders for evaluation and calibration tests."""
from __future__ import annotations

from trustsync.eval.metrics import ClipRecord, UttRecord
from trustsync.types import TrackSummary, UtteranceSummary


def fake_clip(
    scenario: str,
    expected: str,
    best_sync: float | None,
    lip_std: float,
    directed: float,
    quality: float = 1.0,
    n_tracks: int = 1,
    speaker_side: str = "center",
    addressed: bool = True,
    subject: str = "a",
    speaker_cx: float = 0.5,
    duration: float = 1.5,
    lag_samples: list[float] | None = None,
) -> ClipRecord:
    """Helper creating a synthetic ClipRecord for hand-computable metric testing."""
    if duration <= 0.0 or best_sync is None:
        utts = []
    else:
        tracks = [
            TrackSummary(
                track_id=0,
                sync=best_sync,
                lip_std=lip_std,
                directed=directed,
                quality=quality,
                yaw=0.0,
                pitch=0.0,
            )
        ]
        face_cx = {0: speaker_cx}
        if n_tracks == 2:
            tracks.append(
                TrackSummary(
                    track_id=1,
                    sync=0.2,
                    lip_std=0.01,
                    directed=0.8,
                    quality=1.0,
                    yaw=0.0,
                    pitch=0.0,
                )
            )
            other_cx = 0.8 if speaker_cx < 0.5 else 0.2
            face_cx[1] = other_cx

        summary = UtteranceSummary(
            t_start=1.0,
            t_end=1.0 + duration,
            duration_s=duration,
            tracks=tracks,
        )
        utts = [
            UttRecord(
                t_start=1.0,
                t_end=1.0 + duration,
                summary=summary,
                face_cx=face_cx,
            )
        ]

    lags = lag_samples if lag_samples is not None else ([0.0] if utts else [])
    return ClipRecord(
        id=f"clip_{scenario}_{subject}",
        scenario=scenario,
        subject=subject,
        subjects=[subject],
        expected=expected,
        speaker_side=speaker_side,
        addressed=addressed,
        utts=utts,
        lag_samples=lags,
    )
