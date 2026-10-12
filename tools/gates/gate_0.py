"""TrustSync Gate 0: Project scaffold and environment verification."""
from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    """Run all checks for Gate 0.

    Returns:
        List of (check_name, result, detail) where result is 'PASS', 'FAIL', or 'SKIP'.
    """
    results: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]

    # 1. Python version 3.10 or 3.11
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info.major == 3 and sys.version_info.minor in (10, 11):
        results.append(("python_version", "PASS", f"Python {py_ver}"))
    else:
        results.append(("python_version", "FAIL", f"Python {py_ver} (must be 3.10 or 3.11)"))

    # 2. Imports succeed: numpy, scipy, cv2, mediapipe, sounddevice, soundfile,
    #    yaml, sklearn, joblib, pandas, matplotlib, fastapi, uvicorn, httpx
    pkgs = [
        "numpy",
        "scipy",
        "cv2",
        "mediapipe",
        "sounddevice",
        "soundfile",
        "yaml",
        "sklearn",
        "joblib",
        "pandas",
        "matplotlib",
        "fastapi",
        "uvicorn",
        "httpx",
    ]
    missing = []
    for pkg in pkgs:
        try:
            importlib.import_module(pkg)
        except Exception as e:
            missing.append(f"{pkg} ({e})")
    if not missing:
        results.append(("package_imports", "PASS", f"All {len(pkgs)} packages imported"))
    else:
        results.append(("package_imports", "FAIL", f"Failed: {', '.join(missing)}"))

    # 3. 'import cv2' came from opencv-contrib-python and pip has no distribution named opencv-python
    try:
        has_contrib = False
        try:
            importlib.metadata.distribution("opencv-contrib-python")
            has_contrib = True
        except importlib.metadata.PackageNotFoundError:
            pass

        has_base = False
        try:
            importlib.metadata.distribution("opencv-python")
            has_base = True
        except importlib.metadata.PackageNotFoundError:
            pass

        dist_map = importlib.metadata.packages_distributions()
        cv2_dists = dist_map.get("cv2", [])
        is_contrib = any("contrib" in d for d in cv2_dists) or has_contrib

        if is_contrib and not has_base:
            results.append(
                (
                    "opencv_distribution",
                    "PASS",
                    "cv2 from opencv-contrib-python, opencv-python not installed",
                )
            )
        else:
            results.append(
                (
                    "opencv_distribution",
                    "FAIL",
                    f"has_contrib={has_contrib}, has_base={has_base}, cv2_dists={cv2_dists}",
                )
            )
    except Exception as e:
        results.append(("opencv_distribution", "FAIL", str(e)))

    # 4. mediapipe.solutions.face_mesh accessible and mediapipe version is 0.10.14
    try:
        import mediapipe as mp

        mp_ver = getattr(mp, "__version__", "unknown")
        has_face_mesh = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
        if mp_ver == "0.10.14" and has_face_mesh:
            results.append(
                ("mediapipe_version_and_mesh", "PASS", f"Version {mp_ver}, face_mesh accessible")
            )
        else:
            results.append(
                (
                    "mediapipe_version_and_mesh",
                    "FAIL",
                    f"Version={mp_ver} (need 0.10.14), face_mesh={has_face_mesh}",
                )
            )
    except Exception as e:
        results.append(("mediapipe_version_and_mesh", "FAIL", str(e)))

    # 5. Folders from SPEC section 2 and required files exist
    spec_folders = [
        root / "docs",
        root / ".agents" / "rules",
        root / ".agents" / "skills",
        root / ".agents" / "workflows",
        root / "src" / "trustsync",
        root / "src" / "trustsync" / "audio",
        root / "src" / "trustsync" / "vision",
        root / "src" / "trustsync" / "addressee",
        root / "src" / "trustsync" / "sync",
        root / "src" / "trustsync" / "fusion",
        root / "src" / "trustsync" / "io",
        root / "src" / "trustsync" / "overlay",
        root / "src" / "trustsync" / "eval",
        root / "src" / "trustsync" / "server",
        root / "tools",
        root / "tools" / "gates",
        tests_dir := root / "tests",
        root / "tests" / "fixtures",
        root / "report",
        root / "out",
        root / "data" / "sessions",
        root / "models",
        root / "reports" / "figures",
        root / "web",
    ]
    spec_files = [
        root / "AGENTS.md",
        root / "PROGRESS.md",
        root / "README.md",
        root / "requirements.txt",
        root / "pyproject.toml",
        root / "docs" / "SPEC.md",
        root / "tools" / "gates" / "run_gate.py",
        root / "tools" / "gates" / "gate_0.py",
    ]
    missing_dirs = [str(f.relative_to(root)) for f in spec_folders if not f.is_dir()]
    missing_files = [str(f.relative_to(root)) for f in spec_files if not f.is_file()]
    if not missing_dirs and not missing_files:
        results.append(
            (
                "spec_folders_and_files",
                "PASS",
                f"{len(spec_folders)} folders and {len(spec_files)} files verified",
            )
        )
    else:
        results.append(
            (
                "spec_folders_and_files",
                "FAIL",
                f"Missing dirs: {missing_dirs}; missing files: {missing_files}",
            )
        )

    # 6. AGENTS.md is shorter than 12000 characters
    try:
        agents_txt = (root / "AGENTS.md").read_text(encoding="utf-8")
        if len(agents_txt) < 12000:
            results.append(("agents_md_length", "PASS", f"{len(agents_txt)} chars (< 12000)"))
        else:
            results.append(("agents_md_length", "FAIL", f"{len(agents_txt)} chars (>= 12000)"))
    except Exception as e:
        results.append(("agents_md_length", "FAIL", str(e)))

    # 7. Each of the 4 skills has SKILL.md whose frontmatter name equals its folder name
    skills_dir = root / ".agents" / "skills"
    skill_folders = [d for d in skills_dir.iterdir() if d.is_dir()]
    skill_errors = []
    expected_skills = {
        "phase-gate",
        "synthetic-signal-testing",
        "modern-ui-design",
        "debug-root-cause",
    }
    found_skills = {d.name for d in skill_folders}
    if not expected_skills.issubset(found_skills):
        skill_errors.append(f"Missing skill folders: {expected_skills - found_skills}")

    for d in skill_folders:
        skill_file = d / "SKILL.md"
        if not skill_file.is_file():
            skill_errors.append(f"{d.name}: missing SKILL.md")
            continue
        content = skill_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        name_val = None
        if lines and lines[0].strip() == "---":
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                if line.startswith("name:"):
                    name_val = line.split("name:", 1)[1].strip()
        if name_val != d.name:
            skill_errors.append(f"{d.name}: frontmatter name '{name_val}' != folder '{d.name}'")

    if not skill_errors and len(skill_folders) >= 4:
        results.append(("skills_frontmatter", "PASS", f"{len(skill_folders)} skills verified"))
    else:
        results.append(("skills_frontmatter", "FAIL", "; ".join(skill_errors)))

    # 8. Git repository exists
    if (root / ".git").is_dir():
        results.append(("git_repository", "PASS", "Git repository exists"))
    else:
        results.append(("git_repository", "FAIL", ".git directory not found"))

    # 9. Hardware check: VideoCapture(0) opens and returns 5 frames
    if not hardware:
        results.append(("camera_hardware", "SKIP", "Hardware check skipped (run with --hardware)"))
    else:
        try:
            import time
            import cv2

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                results.append(("camera_hardware", "FAIL", "Cannot open cv2.VideoCapture(0)"))
            else:
                frames = 0
                t0 = time.monotonic()
                w, h = 0, 0
                for _ in range(5):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames += 1
                    h, w = frame.shape[:2]
                elapsed = time.monotonic() - t0
                cap.release()
                if frames == 5:
                    fps = frames / elapsed if elapsed > 0 else 0.0
                    results.append(("camera_hardware", "PASS", f"{w}x{h} @ {fps:.1f} FPS"))
                else:
                    results.append(
                        ("camera_hardware", "FAIL", f"Captured only {frames}/5 frames")
                    )
        except Exception as e:
            results.append(("camera_hardware", "FAIL", str(e)))

    # 10. Hardware check: sounddevice input device and 1 second recording RMS > 0
    if not hardware:
        results.append(
            ("microphone_hardware", "SKIP", "Hardware check skipped (run with --hardware)")
        )
    else:
        try:
            import numpy as np
            import sounddevice as sd

            dev = sd.query_devices(kind="input")
            dev_name = dev.get("name", "Default")
            rec = sd.rec(int(16000 * 1.0), samplerate=16000, channels=1, dtype="float32")
            sd.wait()
            rms = float(np.sqrt(np.mean(rec**2)))
            if rms > 0.0:
                results.append(
                    ("microphone_hardware", "PASS", f"{dev_name}, RMS {rms:.5f}")
                )
            else:
                results.append(("microphone_hardware", "FAIL", f"{dev_name}, RMS is 0.0"))
        except Exception as e:
            results.append(("microphone_hardware", "FAIL", str(e)))

    return results
