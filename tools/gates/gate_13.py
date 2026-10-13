"""TrustSync Gate 13: Documentation, release kit, and clean-clone verification."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(hardware: bool = False, slow: bool = False) -> list[tuple[str, str, str]]:
    """Execute Gate 13 checks.

    Args:
        hardware: If True, execute hardware checks (none in Gate 13).
        slow: If True, execute slow clean clone check.

    Returns:
        list of (check, result, detail) tuples.
    """
    checks: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # 1. pytest tests/test_report_assets.py
    cmd_test = [sys.executable, "-m", "pytest", "-q", "-m", "not hardware", "tests/test_report_assets.py"]
    try:
        proc = subprocess.run(cmd_test, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode == 0:
            detail = proc.stdout.strip().splitlines()[-1] if proc.stdout else "passed"
            checks.append(("pytest tests/test_report_assets.py", "PASS", detail))
        else:
            out = (proc.stdout + "\n" + proc.stderr).strip()
            detail = out.splitlines()[-1] if out else f"exit code {proc.returncode}"
            checks.append(("pytest tests/test_report_assets.py", "FAIL", detail))
    except Exception as e:
        checks.append(("pytest tests/test_report_assets.py", "FAIL", str(e)))

    # 2. File-exist rows
    required_docs = [
        ("README.md", root / "README.md"),
        ("docs/ARCHITECTURE.md", root / "docs" / "ARCHITECTURE.md"),
        ("docs/RECORDING.md", root / "docs" / "RECORDING.md"),
        ("report/limitations.md", root / "report" / "limitations.md"),
    ]
    for name, path in required_docs:
        if path.is_file():
            checks.append((f"{name} exists", "PASS", f"{name} present ({path.stat().st_size} bytes)"))
        else:
            checks.append((f"{name} exists", "FAIL", f"Missing {name}"))

    # 3. README.md command mention check
    readme_path = root / "README.md"
    if not readme_path.is_file():
        checks.append(("README.md mentions required commands", "FAIL", "README.md not found"))
    else:
        readme_text = readme_path.read_text(encoding="utf-8")
        required_cmds = [
            "pip install mediapipe==0.10.14",
            "pip install -r requirements.txt",
            "pip install -e .",
            "tools/gates/run_gate.py 6",
            "tools/run_demo.py",
        ]
        missing_cmds = [cmd for cmd in required_cmds if cmd not in readme_text]
        if not missing_cmds:
            checks.append(("README.md mentions required commands", "PASS", "All installation and demo commands documented"))
        else:
            checks.append(("README.md mentions required commands", "FAIL", f"Missing commands in README: {missing_cmds}"))

    # 4. Slow row: clean clone
    if not slow:
        checks.append(("clean clone", "SKIP", "clean clone check skipped (needs --slow)"))
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="trustsync_clone_"))
        try:
            clone_dir = temp_dir / "trustsync"
            # Clone repo locally
            git_bin = shutil.which("git") or "git"
            clone_res = subprocess.run(
                [git_bin, "clone", str(root), str(clone_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if clone_res.returncode != 0:
                checks.append(("clean clone", "FAIL", f"git clone failed: {clone_res.stderr.strip()}"))
            else:
                # Create clean venv
                venv_dir = clone_dir / ".venv"
                venv_res = subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if venv_res.returncode != 0:
                    checks.append(("clean clone", "FAIL", f"venv creation failed: {venv_res.stderr.strip()}"))
                else:
                    if os.name == "nt":
                        venv_py = venv_dir / "Scripts" / "python.exe"
                    else:
                        venv_py = venv_dir / "bin" / "python"

                    # Run documented install order
                    install_cmds = [
                        [str(venv_py), "-m", "pip", "install", "mediapipe==0.10.14"],
                        [str(venv_py), "-m", "pip", "install", "-r", "requirements.txt"],
                        [str(venv_py), "-m", "pip", "install", "-e", "."],
                    ]
                    install_ok = True
                    err_msg = ""
                    for icmd in install_cmds:
                        ires = subprocess.run(icmd, cwd=clone_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
                        if ires.returncode != 0:
                            install_ok = False
                            err_msg = (ires.stdout + "\n" + ires.stderr).strip().splitlines()[-1]
                            break

                    if not install_ok:
                        checks.append(("clean clone", "FAIL", f"pip install failed: {err_msg}"))
                    else:
                        # Run default pytest
                        pt_res = subprocess.run(
                            [str(venv_py), "-m", "pytest", "-q", "-m", "not hardware"],
                            cwd=clone_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                        )
                        if pt_res.returncode == 0:
                            checks.append(("clean clone", "PASS", f"clean clone installed and pytest passed ({pt_res.stdout.strip().splitlines()[-1]})"))
                        else:
                            out = (pt_res.stdout + "\n" + pt_res.stderr).strip()
                            checks.append(("clean clone", "FAIL", f"pytest failed in clean clone: {out.splitlines()[-1]}"))
        except Exception as e:
            checks.append(("clean clone", "FAIL", str(e)))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return checks
