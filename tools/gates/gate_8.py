"""TrustSync Gate 8: Metrics, caching, calibration, learned gate, and evaluation pipeline."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 8.

    Returns:
        List of (check_name, result, detail) tuples.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    test_files = [
        "tests/test_eval_metrics.py",
        "tests/test_eval_cache.py",
        "tests/test_calibrate.py",
        "tests/test_learned.py",
        "tests/test_plots.py",
        "tests/test_eval_run.py",
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

    # Dataset check
    sessions_dir = root / "data" / "sessions"
    valid_sessions_count = 0
    if sessions_dir.is_dir():
        from tools.validate_session import validate
        for s_dir in sessions_dir.iterdir():
            if s_dir.is_dir() and (s_dir / "meta.json").is_file():
                v_res = validate(s_dir, check_faces=True)
                if all(r[1] in ("PASS", "SKIP") for r in v_res):
                    valid_sessions_count += 1

    if valid_sessions_count < 40:
        results.append((
            "dataset_present",
            "SKIP",
            f"record more clips ({valid_sessions_count}/40 valid sessions present)",
        ))
    else:
        results.append((
            "dataset_present",
            "PASS",
            f"{valid_sessions_count} valid sessions present",
        ))

        # Run eval end to end into a temp folder
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_reports = Path(tmp_dir) / "reports"
            cmd_eval = [
                sys.executable,
                "-m",
                "trustsync.eval.run_eval",
                "--sessions",
                str(sessions_dir),
                "--out",
                str(tmp_reports),
                "--calibrate",
                "--learned",
            ]
            p_eval = subprocess.run(cmd_eval, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(root))
            res_json_file = tmp_reports / "results.json"
            if p_eval.returncode == 0 and res_json_file.is_file():
                with open(res_json_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

                results.append(("run_eval_end_to_end", "PASS", "run_eval succeeded and produced results.json"))

                # AV lag saturation
                av_lag = eval_data.get("av_lag", {})
                sat_frac = av_lag.get("saturated_fraction", 0.0)
                med_lag = av_lag.get("median", 0.0)
                if sat_frac < 0.2:
                    results.append((
                        "av_lag_not_saturated",
                        "PASS",
                        f"median lag {med_lag}s, sat_frac {sat_frac:.1%}",
                    ))
                else:
                    results.append((
                        "av_lag_not_saturated",
                        "FAIL",
                        f"median lag {med_lag}s, sat_frac {sat_frac:.1%}: set audio.latency_s to the negative median if it is large",
                    ))

                # Read LOSO pooled numbers (prefer loso_learned, fall back to loso_rule)
                loso_dict = eval_data.get("loso_learned") or eval_data.get("loso_rule") or {}
                pooled = loso_dict.get("pooled", {})
                far_all = pooled.get("far_all_reject", 1.0)
                frr = pooled.get("frr", 1.0)
                attr_acc = pooled.get("attribution_accuracy", 0.0)
                addr_acc = pooled.get("addressee_accuracy", 0.0)

                # Goal rows
                res_far = "PASS" if far_all <= 0.10 else "FAIL"
                results.append(("goal_far_all_reject", res_far, f"far_all_reject {far_all:.3f} <= 0.10"))

                res_frr = "PASS" if frr <= 0.30 else "FAIL"
                results.append(("goal_frr", res_frr, f"frr {frr:.3f} <= 0.30"))

                res_attr = "PASS" if attr_acc >= 0.85 else "FAIL"
                results.append(("goal_attribution_accuracy", res_attr, f"attribution_accuracy {attr_acc:.3f} >= 0.85"))

                res_addr = "PASS" if addr_acc >= 0.80 else "FAIL"
                results.append(("goal_addressee_accuracy", res_addr, f"addressee_accuracy {addr_acc:.3f} >= 0.80"))
            else:
                results.append(("run_eval_end_to_end", "FAIL", f"run_eval failed: {p_eval.stderr}"))

    return results
