"""Phase 11 gate: Web integration, Replay mode, Settings, Results, and E2E WebSocket test."""
from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import websockets


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    web_dir = root / "web"
    results: list[tuple[str, str, str]] = []

    # 1. Re-run Gate 10 web checks
    try:
        from tools.gates import gate_10
        g10_rows = gate_10.run(hardware=hardware)
        # Include non-hardware rows from gate 10
        for row in g10_rows:
            if row[0] != "browser_check":
                results.append((f"g10_{row[0]}", row[1], row[2]))
    except Exception as exc:
        results.append(("g10_checks", "FAIL", f"Failed running gate 10 checks: {exc}"))

    # 2. Required files exist
    required_files = [
        "src/lib/streamReducer.ts",
        "src/lib/api.ts",
        "src/components/SourceSelector.tsx",
        "src/components/Toast.tsx",
        "src/pages/ReplayPage.tsx",
        "src/pages/SettingsPage.tsx",
        "src/lib/matchExpected.ts",
        "src/pages/ResultsPage.tsx",
        "src/lib/results.ts",
        "public/mock-results.json",
    ]

    missing = [f for f in required_files if not (web_dir / f).exists()]
    if missing:
        results.append(("phase11_files_exist", "FAIL", f"Missing files: {', '.join(missing)}"))
    else:
        results.append(("phase11_files_exist", "PASS", f"All {len(required_files)} Phase 11 files exist"))

    # 3. Check public/mock-results.json validity
    mock_res_file = web_dir / "public" / "mock-results.json"
    if not mock_res_file.is_file():
        results.append(("mock_results_valid", "FAIL", "public/mock-results.json missing"))
    else:
        try:
            with open(mock_res_file, "r", encoding="utf-8") as f:
                res_data = json.load(f)
            req_keys = {
                "generated_at", "dataset", "config", "current", "calibrated",
                "loso_rule", "loso_learned", "learned_kind", "ablation",
                "av_lag", "fps", "figures", "notes"
            }
            missing_keys = req_keys - set(res_data.keys())
            if missing_keys:
                results.append(("mock_results_valid", "FAIL", f"Missing keys: {missing_keys}"))
            else:
                results.append(("mock_results_valid", "PASS", "Valid mock results JSON with all required keys"))
        except Exception as exc:
            results.append(("mock_results_valid", "FAIL", str(exc)))

    # 4. tools/make_demo_session.py runs and validates
    frontal_path = root / "tests" / "fixtures" / "face_frontal.jpg"
    if not frontal_path.is_file():
        results.append(("demo_session_validation", "SKIP", "face_frontal.jpg missing"))
    else:
        demo_dir = root / "data" / "sessions" / "demo_replay_000"
        if not demo_dir.exists():
            proc_demo = subprocess.run([sys.executable, str(root / "tools" / "make_demo_session.py")], capture_output=True, text=True)
            if proc_demo.returncode != 0:
                results.append(("demo_session_validation", "FAIL", f"make_demo_session failed: {proc_demo.stderr}"))
                return results

        try:
            from tools.validate_session import validate
            val_rows = validate(demo_dir, check_faces=True)
            failed_val = [r for r in val_rows if r[1] == "FAIL"]
            if failed_val:
                results.append(("demo_session_validation", "FAIL", f"Validation failures: {failed_val}"))
            else:
                results.append(("demo_session_validation", "PASS", "demo_replay_000 passed session validator"))
        except Exception as exc:
            results.append(("demo_session_validation", "FAIL", str(exc)))

    # 5. Integration row with uvicorn server subprocess
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd_srv = [sys.executable, "-m", "trustsync.server.app", "--port", str(port)]
    srv_proc = subprocess.Popen(cmd_srv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(root))

    try:
        # Wait for server health
        started = False
        t_wait = time.monotonic()
        while time.monotonic() - t_wait < 8.0:
            try:
                r = httpx.get(f"{base_url}/api/health", timeout=1.0)
                if r.status_code == 200:
                    started = True
                    break
            except Exception:
                time.sleep(0.1)

        if not started:
            results.append(("server_integration", "FAIL", "FastAPI server failed to start within 8 s"))
            return results

        # Check GET / returns HTML with <div id="root">
        r_root = httpx.get(f"{base_url}/", timeout=3.0)
        if r_root.status_code != 200 or '<div id="root">' not in r_root.text:
            results.append(("server_integration", "FAIL", f"GET / status {r_root.status_code}, root div missing"))
            return results

        # Check GET /api/sessions returns list
        r_sess = httpx.get(f"{base_url}/api/sessions", timeout=3.0)
        if r_sess.status_code != 200 or not isinstance(r_sess.json(), list):
            results.append(("server_integration", "FAIL", f"GET /api/sessions failed: {r_sess.status_code}"))
            return results

        # Check GET /api/results returns 200 or 404
        r_res = httpx.get(f"{base_url}/api/results", timeout=3.0)
        if r_res.status_code not in (200, 404):
            results.append(("server_integration", "FAIL", f"GET /api/results unexpected status {r_res.status_code}"))
            return results

        # If demo session exists, test WebSocket utterance decision
        demo_dir = root / "data" / "sessions" / "demo_replay_000"
        if not demo_dir.is_dir():
            results.append(("demo_replay_ws_test", "SKIP", "demo_replay_000 session folder missing"))
        else:
            async def run_ws_e2e() -> tuple[bool, str]:
                uri = f"ws://127.0.0.1:{port}/ws"
                async with websockets.connect(uri) as ws:
                    # Expect hello message
                    raw_hello = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg_hello = json.loads(raw_hello)
                    if msg_hello.get("type") != "hello":
                        return False, f"Expected hello message, got {msg_hello.get('type')}"

                    # Trigger session
                    async with httpx.AsyncClient() as client:
                        r_post = await client.post(
                            f"{base_url}/api/source",
                            json={"type": "session", "id": "demo_replay_000"},
                            timeout=5.0,
                        )
                        if r_post.status_code != 200:
                            return False, f"POST /api/source failed: {r_post.status_code} {r_post.text}"

                    # Listen for utterance message
                    t0 = time.monotonic()
                    while time.monotonic() - t0 < 30.0:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "utterance":
                            dec = msg.get("data", {}).get("decision")
                            if dec == "REJECT_SPOOF":
                                return True, "Utterance decision REJECT_SPOOF verified"
                            return False, f"Expected decision REJECT_SPOOF, got {dec}"
                    return False, "Timeout waiting for utterance message (30 s)"

                return False, "Unknown socket termination"

            loop = asyncio.new_event_loop()
            try:
                ok, detail = loop.run_until_complete(run_ws_e2e())
                results.append(("demo_replay_ws_test", "PASS" if ok else "FAIL", detail))
            except Exception as exc:
                results.append(("demo_replay_ws_test", "FAIL", f"WS E2E error: {exc}"))
            finally:
                loop.close()

        # Stop source
        httpx.post(f"{base_url}/api/stop", timeout=2.0)
        results.append(("server_integration", "PASS", "Static dist mount, REST endpoints, and WebSocket connected"))

    finally:
        srv_proc.terminate()
        try:
            srv_proc.wait(timeout=2.0)
        except Exception:
            srv_proc.kill()

    return results
