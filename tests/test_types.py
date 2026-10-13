"""Tests for TrustSync core types and serialization."""
import json
import numpy as np

from trustsync.types import (
    DECISIONS,
    FaceScores,
    FrameResult,
    frame_result_to_dict,
)


def test_decisions_constants() -> None:
    """Verify DECISIONS tuple matches SPEC section 4 exactly in order."""
    expected = (
        "IDLE",
        "LISTENING",
        "ACCEPT",
        "REJECT_SPOOF",
        "REJECT_AMBIGUOUS",
        "REJECT_NOT_ADDRESSED",
        "REJECT_LOW_QUALITY",
        "REJECT_NO_FACE",
    )
    assert DECISIONS == expected


def test_frame_result_to_dict_serialization() -> None:
    """Verify frame_result_to_dict handles NaN, numpy types, and serializes to JSON."""
    face = FaceScores(
        track_id=1,
        bbox=(0.1, 0.2, 0.3, 0.4),
        sync=float(np.float32(0.85)),
        lip_std=0.012,
        best_lag_s=0.033,
        yaw=float("nan"),
        pitch=0.0,
        roll=0.0,
        gaze_off=0.1,
        directed=0.9,
        quality=0.95,
    )

    frame_res = FrameResult(
        t=10.123456,
        frame_idx=100,
        fps=29.97,
        faces=[face],
        speech_active=True,
        env=float(np.float32(0.0456)),
        noise_floor=0.002,
        speaker_id=1,
        attribution_conf=float("nan"),
        liveness=0.85,
        addressee=0.9,
        decision="ACCEPT",
        reason="Test accept",
        utterance_final=True,
    )

    res_dict = frame_result_to_dict(frame_res)

    # Verify rounding to 4 decimals
    assert res_dict["t"] == 10.1235
    # Verify NaN turned into None
    assert res_dict["attribution_conf"] is None
    assert res_dict["faces"][0]["yaw"] is None
    # Verify tuple became list
    assert isinstance(res_dict["faces"][0]["bbox"], list)
    assert res_dict["faces"][0]["bbox"] == [0.1, 0.2, 0.3, 0.4]
    # Verify numpy float32 converted to python float
    assert isinstance(res_dict["env"], float)

    # Verify json.dumps succeeds
    json_str = json.dumps(res_dict)
    assert "null" in json_str
    loaded = json.loads(json_str)
    assert loaded["decision"] == "ACCEPT"
