"""TrustSync Gate 12: Robustness hardening, benchmark, and degradation verification."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from trustsync.config import load_config
from trustsync.pipeline import TrustSyncPipeline
from trustsync.io.video_source import WebcamSource
from trustsync.audio.sources import LiveAudioSource


def run(hardware: bool = False, slow: bool = False) -> list[tuple[str, str, str]]:
    """Execute Gate 12 checks.

    Args:
        hardware: If True, execute live webcam and microphone degradation checks.
        slow: If True, slow tests flag enabled.

    Returns:
        list of (check, result, detail) tuples where result is PASS, FAIL, or SKIP.
    """
    checks: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # 1. pytest tests/test_robustness.py
    cmd_robust = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", "tests/test_robustness.py"]
    try:
        proc = subprocess.run(cmd_robust, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0:
            detail = proc.stdout.strip().splitlines()[-1] if proc.stdout else "passed"
            checks.append(("pytest tests/test_robustness.py", "PASS", detail))
        else:
            out = (proc.stdout + "\n" + proc.stderr).strip()
            detail = out.splitlines()[-1] if out else f"exit code {proc.returncode}"
            checks.append(("pytest tests/test_robustness.py", "FAIL", detail))
    except Exception as e:
        checks.append(("pytest tests/test_robustness.py", "FAIL", str(e)))

    # 2. pytest tests/test_benchmark_logic.py
    cmd_bench_logic = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", "tests/test_benchmark_logic.py"]
    try:
        proc = subprocess.run(cmd_bench_logic, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0:
            detail = proc.stdout.strip().splitlines()[-1] if proc.stdout else "passed"
            checks.append(("pytest tests/test_benchmark_logic.py", "PASS", detail))
        else:
            out = (proc.stdout + "\n" + proc.stderr).strip()
            detail = out.splitlines()[-1] if out else f"exit code {proc.returncode}"
            checks.append(("pytest tests/test_benchmark_logic.py", "FAIL", detail))
    except Exception as e:
        checks.append(("pytest tests/test_benchmark_logic.py", "FAIL", str(e)))

    # 3. Benchmark photo run
    fixture_path = root / "tests" / "fixtures" / "face_frontal.jpg"
    bench_json_path = root / "reports" / "benchmark.json"

    if not fixture_path.is_file():
        checks.append(("benchmark photo run", "SKIP", "tests/fixtures/face_frontal.jpg missing"))
        checks.append(("GOAL fps.mean >= 15", "SKIP", "benchmark photo run was skipped"))
        checks.append(("GOAL decision_latency_s.max <= 0.8", "SKIP", "benchmark photo run was skipped"))
    else:
        cmd_bench = [sys.executable, "tools/benchmark.py", "--source", "photo", "--frames", "150"]
        try:
            b_proc = subprocess.run(cmd_bench, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if b_proc.returncode != 0:
                checks.append(("benchmark photo run", "FAIL", f"benchmark.py exited {b_proc.returncode}"))
                checks.append(("GOAL fps.mean >= 15", "FAIL", "benchmark failed"))
                checks.append(("GOAL decision_latency_s.max <= 0.8", "FAIL", "benchmark failed"))
            else:
                if not bench_json_path.is_file():
                    checks.append(("benchmark photo run", "FAIL", "reports/benchmark.json not generated"))
                    checks.append(("GOAL fps.mean >= 15", "FAIL", "benchmark.json missing"))
                    checks.append(("GOAL decision_latency_s.max <= 0.8", "FAIL", "benchmark.json missing"))
                else:
                    bench_data = json.loads(bench_json_path.read_text(encoding="utf-8"))
                    required_keys = ["machine", "source", "n_frames", "stages", "fps", "decision_latency_s"]
                    machine_keys = ["platform", "processor", "python", "cpu_count", "opencv", "mediapipe"]
                    has_keys = all(k in bench_data for k in required_keys) and all(k in bench_data.get("machine", {}) for k in machine_keys)

                    if has_keys:
                        checks.append(("benchmark photo run", "PASS", "150 frames processed, reports/benchmark.json valid"))
                    else:
                        checks.append(("benchmark photo run", "FAIL", "reports/benchmark.json missing required keys"))

                    fps_mean = float(bench_data.get("fps", {}).get("mean", 0.0))
                    if fps_mean >= 15.0:
                        checks.append(("GOAL fps.mean >= 15", "PASS", f"fps.mean = {fps_mean:.2f} >= 15"))
                    else:
                        checks.append((
                            "GOAL fps.mean >= 15",
                            "FAIL",
                            f"fps.mean = {fps_mean:.2f} < 15; suggest setting video.max_faces to 2 and video.width/height to 480x360 in config.yaml and re-running",
                        ))

                    dec_max = float(bench_data.get("decision_latency_s", {}).get("max", 0.0))
                    if dec_max <= 0.8:
                        checks.append(("GOAL decision_latency_s.max <= 0.8", "PASS", f"decision_latency_s.max = {dec_max:.4f} s <= 0.8"))
                    else:
                        checks.append(("GOAL decision_latency_s.max <= 0.8", "FAIL", f"decision_latency_s.max = {dec_max:.4f} s > 0.8"))
        except Exception as e:
            checks.append(("benchmark photo run", "FAIL", str(e)))
            checks.append(("GOAL fps.mean >= 15", "FAIL", str(e)))
            checks.append(("GOAL decision_latency_s.max <= 0.8", "FAIL", str(e)))

    # 4. Hardware rows
    if not hardware:
        checks.append(("live check mouth covered", "SKIP", "needs --hardware"))
        checks.append(("live check low light / half lens", "SKIP", "needs --hardware"))
    else:
        # Live check 1: mouth covered
        print("\n[HARDWARE CHECK 1/2] Cover your mouth completely with your hand and say a command (8 s recording)...")
        cfg = load_config()
        try:
            cam = WebcamSource(cfg)
            mic = LiveAudioSource(cfg.audio)
            mic.start()
            pipe = TrustSyncPipeline(cfg, audio=mic)
            start_t = time.monotonic()
            decisions_seen: list[str] = []
            while time.monotonic() - start_t < 8.0:
                ok, frame, t = cam.read()
                if not ok or frame is None:
                    break
                res = pipe.process(frame, t)
                if res.utterance_final:
                    decisions_seen.append(res.decision)
            cam.close()
            mic.close()

            if any(d == "ACCEPT" for d in decisions_seen):
                checks.append(("live check mouth covered", "FAIL", f"Utterance falsely ACCEPTed: {decisions_seen}"))
            else:
                checks.append(("live check mouth covered", "PASS", f"No ACCEPT decision seen (decisions: {decisions_seen or ['none']})"))
        except Exception as e:
            checks.append(("live check mouth covered", "FAIL", str(e)))

        # Live check 2: low light or covered lens
        print("\n[HARDWARE CHECK 2/2] Turn the room lights low or cover half the camera lens and say a command (8 s recording)...")
        try:
            cam = WebcamSource(cfg)
            mic = LiveAudioSource(cfg.audio)
            mic.start()
            pipe = TrustSyncPipeline(cfg, audio=mic)
            start_t = time.monotonic()
            decisions_seen = []
            while time.monotonic() - start_t < 8.0:
                ok, frame, t = cam.read()
                if not ok or frame is None:
                    break
                res = pipe.process(frame, t)
                if res.utterance_final:
                    decisions_seen.append(res.decision)
            cam.close()
            mic.close()

            if any(d == "ACCEPT" for d in decisions_seen):
                checks.append(("live check low light / half lens", "FAIL", f"Utterance falsely ACCEPTed: {decisions_seen}"))
            else:
                checks.append(("live check low light / half lens", "PASS", f"No ACCEPT decision seen (decisions: {decisions_seen or ['none']})"))
        except Exception as e:
            checks.append(("live check low light / half lens", "FAIL", str(e)))

    return checks
