"""Integration tests for FastAPI application, REST endpoints, MJPEG streaming, and WebSocket."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.helpers_world import build_world
from trustsync.config import Config
from trustsync.pipeline import TrustSyncPipeline
from trustsync.server.app import create_app
from trustsync.server.runtime import Runtime, Sources
from trustsync.synth import FakeFaceTracker
from trustsync.types import FrameResult, frame_result_to_dict


class ScriptedVideoSource:
    """Mock video source streaming pre-generated frames and timestamps."""

    def __init__(self, frames: list[np.ndarray], times: list[float]) -> None:
        self.frames = frames
        self.times = times
        self.idx = 0

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.idx >= len(self.frames):
            return False, None, self.times[-1] if self.times else 0.0
        frame = self.frames[self.idx]
        t = self.times[self.idx]
        self.idx += 1
        return True, frame, t

    def close(self) -> None:
        pass


def make_test_sources(cfg: Config, duration_s: float = 8.0) -> Sources:
    rng = np.random.default_rng(1234)
    bursts = [(2.0, 4.0)]
    faces = [{
        "track_id": 0,
        "bbox": (0.3, 0.2, 0.7, 0.8),
        "yaw": 0.0,
        "quality": 1.0,
        "mode": "matched",
    }]
    pipeline, frames, times = build_world(
        rng, bursts, faces, cfg=cfg, duration_s=duration_s, fps=30.0, t0=100.0
    )
    video = ScriptedVideoSource(frames, times)
    return Sources(video=video, audio=pipeline.audio, tracker=pipeline.tracker, realtime=False)


@pytest.fixture
def test_setup(tmp_path: Path, spec_cfg: Config):
    root = tmp_path
    sessions_dir = root / "data" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy sessions s1 and s2
    for sid in ("s1", "s2"):
        s_path = sessions_dir / sid
        s_path.mkdir()
        meta = {
            "id": sid,
            "scenario": "live_single",
            "expected": "ACCEPT",
            "subjects": ["alice"],
            "speaker_side": "center",
            "addressed": True,
            "audio_t0": 100.0,
        }
        with open(s_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    failing_camera = {"fail": False}

    def test_factory(kind: str, session_id: str | None, cfg: Config, r: Path) -> Sources:
        if failing_camera["fail"] and kind == "webcam":
            raise RuntimeError("camera failed to open")
        if kind == "fast_500":
            return make_test_sources(cfg, duration_s=16.67)  # ~500 frames
        if kind == "session" and session_id not in ("synthetic", "s1", "s2"):
            raise FileNotFoundError(f"Session {session_id} not found")
        return make_test_sources(cfg, duration_s=8.0)

    runtime = Runtime(spec_cfg, root=root, factory=test_factory)
    app = create_app(spec_cfg, root=root, runtime=runtime)
    with TestClient(app) as client:
        yield client, runtime, root, failing_camera
    runtime.stop()


def test_rest_endpoints(test_setup):
    client, runtime, root, failing_camera = test_setup

    # 1. health before start has running false
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"ok": True, "running": False, "source": None}

    # 2. config GET has sections gate, addressee, sync
    res = client.get("/api/config")
    assert res.status_code == 200
    cfg_data = res.json()
    assert "gate" in cfg_data
    assert "addressee" in cfg_data
    assert "sync" in cfg_data

    # 3. config POST gate.min_sync 0.5 reflected in next GET
    res_post = client.post(
        "/api/config", json={"section": "gate", "key": "min_sync", "value": 0.5}
    )
    assert res_post.status_code == 200
    res_get = client.get("/api/config")
    assert res_get.json()["gate"]["min_sync"] == 0.5

    # 4. invalid cases give 422
    assert client.post("/api/config", json={"section": "video", "key": "fps", "value": 30}).status_code == 422
    assert client.post("/api/config", json={"section": "gate", "key": "min_sync", "value": "abc"}).status_code == 422
    assert client.post("/api/config", json={"section": "gate", "key": "min_sync", "value": 5.0}).status_code == 422
    assert client.post("/api/config", json={"section": "gate", "key": "unknown_key", "value": 0.5}).status_code == 422

    # 5. sessions lists s1 and s2
    res_sess = client.get("/api/sessions")
    assert res_sess.status_code == 200
    sess_list = res_sess.json()
    assert [s["id"] for s in sess_list] == ["s1", "s2"]

    # 6. source with invalid id gives 422 and unknown id gives 404
    assert client.post("/api/source", json={"type": "session", "id": "../x"}).status_code == 422
    assert client.post("/api/source", json={"type": "session", "id": "unknown_id"}).status_code == 404

    # 7. failing factory gives 409 and health shows running false
    failing_camera["fail"] = True
    res_fail = client.post("/api/source", json={"type": "webcam"})
    assert res_fail.status_code == 409
    assert client.get("/api/health").json()["running"] is False

    # 8. results gives 404, then 200 after writing results.json
    assert client.get("/api/results").status_code == 404
    rep_dir = root / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"test": "ok"}, f)
    res_200 = client.get("/api/results")
    assert res_200.status_code == 200
    assert res_200.json() == {"test": "ok"}


def test_websocket_flow_and_throttle(test_setup):
    client, runtime, root, *rest = test_setup

    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello"
        assert "config" in hello
        assert "decisions" in hello
        assert len(hello["decisions"]) == 8

        # Start synthetic session
        t_start = time.monotonic()
        res = client.post("/api/source", json={"type": "session", "id": "synthetic"})
        assert res.status_code == 200

        utterance_received = False
        frame_count = 0
        status_finished = False

        # Read messages until world ends (status running=False)
        start_wait = time.monotonic()
        while time.monotonic() - start_wait < 20.0:
            msg = ws.receive_json()
            mtype = msg["type"]

            if mtype == "frame":
                frame_count += 1
                # Check keys match FrameResult
                dummy_res = FrameResult(
                    t=0.0, frame_idx=0, fps=30.0, faces=[], speech_active=False,
                    env=0.0, noise_floor=0.0, speaker_id=None, attribution_conf=0.0,
                    liveness=0.0, addressee=0.0, decision="IDLE", reason="", utterance_final=False
                )
                expected_keys = set(frame_result_to_dict(dummy_res).keys())
                assert set(msg["data"].keys()) == expected_keys

            elif mtype == "utterance":
                utterance_received = True
                assert msg["data"]["decision"] == "ACCEPT"
                assert msg["data"]["speaker_id"] == 0

            elif mtype == "status":
                if msg["data"]["running"] is False and utterance_received:
                    status_finished = True
                    break

        assert utterance_received, "Never received utterance message"
        assert status_finished, "Never received terminal status message"

        elapsed = time.monotonic() - t_start
        ws_hz = runtime.cfg.server.ws_hz
        assert frame_count <= elapsed * ws_hz + 5


def test_backpressure(test_setup):
    client, runtime, root, *rest = test_setup

    q = runtime.subscribe()
    runtime.start("fast_500")

    # Wait for run to finish without draining q
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10.0:
        if not runtime.status()["running"]:
            break
        time.sleep(0.05)

    assert not runtime.status()["running"]
    assert q.qsize() <= 50
    runtime.unsubscribe(q)


def test_mjpeg_stream(test_setup):
    client, runtime, root, *rest = test_setup

    # Nothing running: placeholder is served
    with client.stream("GET", "/stream.mjpg") as resp:
        chunk = next(resp.iter_bytes(1024))
        assert b"--frame" in chunk
        assert b"\xff\xd8" in chunk  # JPEG SOI bytes

    # Running source: camera JPEG is served
    runtime.start("session", "s1")
    time.sleep(0.1)
    with client.stream("GET", "/stream.mjpg") as resp:
        chunk = next(resp.iter_bytes(2000))
        assert b"--frame" in chunk
        assert b"\xff\xd8" in chunk
    runtime.stop()


def test_lifecycle(test_setup):
    client, runtime, root, *rest = test_setup

    # Stop twice does not raise
    runtime.stop()
    runtime.stop()

    # Start while running replaces source
    runtime.start("session", "s1")
    assert runtime.status()["running"] is True
    runtime.start("session", "s2")
    assert runtime.status()["running"] is True

    # Stop joins thread within 2 s
    t0 = time.monotonic()
    runtime.stop()
    join_time = time.monotonic() - t0
    assert join_time < 2.0
    assert runtime.status()["running"] is False
