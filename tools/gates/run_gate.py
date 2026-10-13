"""TrustSync gate runner.

Usage: python tools/gates/run_gate.py N [--hardware]
"""
import argparse
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TrustSync phase gates.")
    parser.add_argument("N", type=int, help="Phase number to run gates up to")
    parser.add_argument("--hardware", action="store_true", help="Enable hardware checks")
    parser.add_argument("--slow", action="store_true", help="Enable slow tests and checks")
    args = parser.parse_args()

    N: int = args.N
    hardware: bool = args.hardware
    slow: bool = args.slow

    rows: list[tuple[str, str, str, str]] = []

    # (a) First run pytest subprocess
    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware"]
    try:
        proc = subprocess.run(
            pytest_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            pytest_result = "PASS"
            detail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "pytest passed"
        elif proc.returncode == 5:
            if N == 0:
                pytest_result = "PASS"
                detail = "exit code 5 (no tests collected; allowed for Phase 0)"
            else:
                pytest_result = "FAIL"
                detail = "exit code 5 (no tests collected)"
        else:
            pytest_result = "FAIL"
            out = (proc.stdout + "\n" + proc.stderr).strip()
            detail = out.splitlines()[-1] if out else f"exit code {proc.returncode}"
    except Exception as e:
        pytest_result = "FAIL"
        detail = str(e)

    rows.append(("pytest", 'pytest -q -m "not hardware"', pytest_result, detail))

    if slow:
        slow_cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not hardware",
            "--runslow",
            "tests/test_robustness.py",
        ]
        try:
            sproc = subprocess.run(
                slow_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if sproc.returncode == 0:
                s_result = "PASS"
                s_detail = (
                    sproc.stdout.strip().splitlines()[-1]
                    if sproc.stdout.strip()
                    else "pytest slow passed"
                )
            else:
                s_result = "FAIL"
                s_out = (sproc.stdout + "\n" + sproc.stderr).strip()
                s_detail = s_out.splitlines()[-1] if s_out else f"exit code {sproc.returncode}"
        except Exception as e:
            s_result = "FAIL"
            s_detail = str(e)
        rows.append(
            (
                "pytest",
                'pytest -q -m "not hardware" --runslow tests/test_robustness.py',
                s_result,
                s_detail,
            )
        )

    # (b) For K from 0 to N import tools/gates/gate_K.py
    gates_dir = Path(__file__).resolve().parent
    for K in range(N + 1):
        gate_file = gates_dir / f"gate_{K}.py"
        gate_name = f"gate_{K}"
        if not gate_file.is_file():
            rows.append((gate_name, f"gate_{K} missing", "FAIL", f"File {gate_file.name} not found"))
            continue

        try:
            spec = importlib.util.spec_from_file_location(f"tools.gates.gate_{K}", str(gate_file))
            if spec is None or spec.loader is None:
                rows.append((gate_name, f"gate_{K} import", "FAIL", f"Cannot load spec for {gate_file}"))
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, "run"):
                rows.append((gate_name, f"gate_{K} run()", "FAIL", "run() function missing"))
                continue
            sig = inspect.signature(mod.run)
            kwargs: dict[str, bool] = {"hardware": hardware}
            if "slow" in sig.parameters:
                kwargs["slow"] = slow
            checks = mod.run(**kwargs)
            for check, result, check_detail in checks:
                rows.append((gate_name, check, result, check_detail))
        except Exception as e:
            rows.append((gate_name, f"gate_{K} execution", "FAIL", str(e)))

    # (c) Print aligned table with columns GATE | CHECK | RESULT | DETAIL
    headers = ("GATE", "CHECK", "RESULT", "DETAIL")
    all_data = [headers] + rows
    col_widths = [max(len(str(r[i])) for r in all_data) for i in range(4)]

    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(4))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(4))
    print(header_line)
    print(separator_line)
    for r in rows:
        row_line = " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(4))
        print(row_line)

    # (d) Exit code and (e) Last line
    failures = sum(1 for r in rows if r[2] == "FAIL")
    has_skip = any(r[2] == "SKIP" for r in rows)

    if failures > 0:
        summary = f"GATE {N}: FAIL ({failures} failures)"
    else:
        summary = f"GATE {N}: PASS"

    if has_skip:
        summary += " [hardware checks pending]"

    print(summary)
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
