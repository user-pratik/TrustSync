"""TrustSync Gate 9: FastAPI backend, REST routes, WebSocket, and MJPEG streaming."""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path
import cv2
import numpy as np


def find_free_port() -> int:
    """Find an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 9.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    test_files = [
        "tests/test_server.py",
    ]

    for rel_path in test_files:
        test_path = root / rel_path
        if not test_path.is_file():
            results.append((f"test_{test_path.stem}", "FAIL", f"File not found: {rel_path}"))
            continue

        cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", str(test_path)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "passed"
            results.append((rel_path, "PASS", summary))
        else:
            err = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-1]
            results.append((rel_path, "FAIL", f"Exit {proc.returncode}: {err}"))

    # Hardware check
    if not hardware:
        results.append((
            "server_live_hardware",
            "SKIP",
            "Hardware check skipped (run with --hardware)",
        ))
    else:
        import httpx
        from trustsync.config import load_config

        cfg = load_config()
        port = find_free_port()
        cmd_srv = [sys.executable, "-m", "trustsync.server.app", "--port", str(port)]
        srv_proc = subprocess.Popen(
            cmd_srv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(root)
        )

        try:
            base_url = f"http://127.0.0.1:{port}"
            # 1. Wait for health
            started = False
            t_wait = time.monotonic()
            while time.monotonic() - t_wait < 5.0:
                try:
                    r = httpx.get(f"{base_url}/api/health", timeout=1.0)
                    if r.status_code == 200:
                        started = True
                        break
                except Exception:
                    time.sleep(0.1)

            if not started:
                results.append(("server_live_hardware", "FAIL", "Server failed to start within 5s"))
                return results

            # 2. POST /api/source webcam
            r_start = httpx.post(f"{base_url}/api/source", json={"type": "webcam"}, timeout=3.0)
            if r_start.status_code != 200:
                results.append((
                    "server_live_hardware",
                    "FAIL",
                    f"POST /api/source failed: {r_start.status_code} {r_start.text}",
                ))
                return results

            # 3. Read 20 WebSocket frame messages within 3 s
            import websockets
            import asyncio

            async def read_ws():
                uri = f"ws://127.0.0.1:{port}/ws"
                count = 0
                async with websockets.connect(uri) as ws:
                    t_start = time.monotonic()
                    while count < 20 and (time.monotonic() - t_start) < 3.0:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "frame":
                            count += 1
                return count

            loop = asyncio.new_event_loop()
            try:
                frame_count = loop.run_until_complete(read_ws())
            finally:
                loop.close()

            if frame_count < 20:
                results.append((
                    "server_live_hardware",
                    "FAIL",
                    f"Received only {frame_count}/20 frames within 3s",
                ))
                return results

            # 4. Decode one JPEG from /stream.mjpg and verify shape
            with httpx.stream("GET", f"{base_url}/stream.mjpg", timeout=3.0) as stream_resp:
                chunk = b""
                for part in stream_resp.iter_bytes():
                    chunk += part
                    if b"\xff\xd9" in chunk:  # JPEG EOI marker
                        break

            start_idx = chunk.find(b"\xff\xd8")
            end_idx = chunk.find(b"\xff\xd9")
            if start_idx != -1 and end_idx != -1:
                jpeg_bytes = chunk[start_idx : end_idx + 2]
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                expected_shape = (cfg.video.height, cfg.video.width, 3)
                if img is None or img.shape != expected_shape:
                    results.append((
                        "server_live_hardware",
                        "FAIL",
                        f"Stream JPEG shape {img.shape if img is not None else None} != {expected_shape}",
                    ))
                    return results
            else:
                results.append(("server_live_hardware", "FAIL", "Failed parsing JPEG from /stream.mjpg"))
                return results

            # 5. POST /api/stop
            httpx.post(f"{base_url}/api/stop", timeout=2.0)
            results.append((
                "server_live_hardware",
                "PASS",
                f"Webcam stream, 20 WS frames, and {cfg.video.width}x{cfg.video.height} JPEG verified",
            ))

        finally:
            srv_proc.terminate()
            try:
                srv_proc.wait(timeout=2.0)
            except Exception:
                srv_proc.kill()

    return results
