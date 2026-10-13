"""Phase 10 gate: Modern dashboard on mock data."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(hardware: bool = False) -> list[tuple[str, str, str]]:
    root = Path(__file__).resolve().parent.parent.parent
    web_dir = root / "web"
    results: list[tuple[str, str, str]] = []

    # 1. npm executable exists
    npm_cmd = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_cmd:
        results.append(("npm_available", "FAIL", "npm executable not found in PATH"))
        return results
    results.append(("npm_available", "PASS", f"Found npm at {npm_cmd}"))

    # 2. Required files exist
    required_files = [
        "src/types.ts",
        "src/lib/decision.ts",
        "src/lib/format.ts",
        "src/components/Card.tsx",
        "src/components/Chip.tsx",
        "src/components/Skeleton.tsx",
        "src/components/EmptyState.tsx",
        "src/components/ErrorState.tsx",
        "src/components/AppShell.tsx",
        "src/components/ConnectionChip.tsx",
        "src/lib/mock.ts",
        "src/lib/useTrustStream.ts",
        "src/components/VideoStage.tsx",
        "src/components/EnvelopeStrip.tsx",
        "src/components/DecisionCard.tsx",
        "src/components/TrustGauges.tsx",
        "src/components/Gauge.tsx",
        "src/components/SyncTimeline.tsx",
        "src/components/EventLog.tsx",
        "src/pages/LivePage.tsx",
    ]

    missing = [f for f in required_files if not (web_dir / f).exists()]
    if missing:
        results.append(("required_files_exist", "FAIL", f"Missing files: {', '.join(missing)}"))
    else:
        results.append(("required_files_exist", "PASS", f"All {len(required_files)} expected UI files present"))

    # 3. tailwind.config.js mentions all 11 tokens
    tw_config = web_dir / "tailwind.config.js"
    if not tw_config.exists():
        results.append(("tailwind_tokens", "FAIL", "tailwind.config.js missing"))
    else:
        content = tw_config.read_text(encoding="utf-8")
        tokens = [
            "bg", "elev", "card", "border", "text", "muted",
            "accept", "reject", "listen", "info", "idle"
        ]
        missing_tokens = [tok for tok in tokens if f"{tok}:" not in content and f"'{tok}'" not in content and f'"{tok}"' not in content]
        if missing_tokens:
            results.append(("tailwind_tokens", "FAIL", f"Missing tokens in tailwind.config.js: {missing_tokens}"))
        else:
            results.append(("tailwind_tokens", "PASS", "All 11 tokens present in tailwind.config.js"))

    # 4. Offline URL check: no external URLs in web/src or web/index.html
    url_pattern = re.compile(r"https?://([^\s\"'<>]+)")
    allowed_domains = {"www.w3.org", "localhost", "127.0.0.1"}
    external_violations: list[str] = []

    files_to_check = list((web_dir / "src").rglob("*.*"))
    if (web_dir / "index.html").exists():
        files_to_check.append(web_dir / "index.html")

    for f in files_to_check:
        if f.is_file() and f.suffix in {".ts", ".tsx", ".js", ".jsx", ".html", ".css"}:
            text = f.read_text(encoding="utf-8", errors="ignore")
            for match in url_pattern.finditer(text):
                full_url = match.group(0)
                domain = match.group(1).split("/")[0].split(":")[0]
                if domain not in allowed_domains:
                    external_violations.append(f"{f.name}: {full_url}")

    if external_violations:
        results.append(("offline_rule", "FAIL", f"Found external URLs: {'; '.join(external_violations[:3])}"))
    else:
        results.append(("offline_rule", "PASS", "No external CDN/HTTP links in src/ or index.html"))

    # 5. npm run typecheck
    try:
        proc = subprocess.run(
            [npm_cmd, "run", "typecheck"],
            cwd=str(web_dir),
            capture_output=True,
            text=True,
            timeout=60,
            shell=True if sys.platform == "win32" else False,
        )
        if proc.returncode == 0:
            results.append(("npm_typecheck", "PASS", "tsc -b passed with 0 errors"))
        else:
            results.append(("npm_typecheck", "FAIL", (proc.stdout + proc.stderr).strip()[-200:]))
    except Exception as exc:
        results.append(("npm_typecheck", "FAIL", str(exc)))

    # 6. npm test (vitest run)
    try:
        proc = subprocess.run(
            [npm_cmd, "test"],
            cwd=str(web_dir),
            capture_output=True,
            text=True,
            timeout=60,
            shell=True if sys.platform == "win32" else False,
        )
        if proc.returncode == 0:
            results.append(("npm_test", "PASS", "vitest unit tests passed"))
        else:
            results.append(("npm_test", "FAIL", (proc.stdout + proc.stderr).strip()[-200:]))
    except Exception as exc:
        results.append(("npm_test", "FAIL", str(exc)))

    # 7. npm run build
    try:
        proc = subprocess.run(
            [npm_cmd, "run", "build"],
            cwd=str(web_dir),
            capture_output=True,
            text=True,
            timeout=60,
            shell=True if sys.platform == "win32" else False,
        )
        if proc.returncode == 0:
            results.append(("npm_build", "PASS", "Vite production build succeeded"))
        else:
            results.append(("npm_build", "FAIL", (proc.stdout + proc.stderr).strip()[-200:]))
    except Exception as exc:
        results.append(("npm_build", "FAIL", str(exc)))

    # 8. dist/index.html exists
    if (web_dir / "dist" / "index.html").exists():
        results.append(("dist_index_html", "PASS", "dist/index.html generated"))
    else:
        results.append(("dist_index_html", "FAIL", "dist/index.html missing after build"))

    # 9. Hardware row
    results.append((
        "browser_check",
        "PASS" if hardware else "SKIP",
        "BROWSER: run npm run dev in web/ and open http://localhost:5173",
    ))

    return results
