"""Launch TrustSync full-stack demo with web dashboard."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    web_dir = root / "web"
    dist_dir = web_dir / "dist"

    # Build web frontend if dist is missing
    if not (dist_dir / "index.html").is_file():
        print("web/dist missing. Building web frontend...")
        npm_bin = shutil.which("npm") or shutil.which("npm.cmd")
        if npm_bin is None:
            print("ERROR: npm not found in PATH.")
            return 1

        ci_res = subprocess.run([npm_bin, "ci"], cwd=web_dir, check=False)
        if ci_res.returncode != 0:
            # Fall back to npm install if npm ci fails
            subprocess.run([npm_bin, "install"], cwd=web_dir, check=True)

        subprocess.run([npm_bin, "run", "build"], cwd=web_dir, check=True)

    # Start FastAPI server
    print("Starting TrustSync server...")
    proc = subprocess.Popen([sys.executable, "-m", "trustsync.server.app"], cwd=root)

    # Wait until /api/health answers
    health_url = "http://127.0.0.1:8000/api/health"
    app_url = "http://127.0.0.1:8000"
    connected = False
    for _ in range(60):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0) as resp:
                if resp.status == 200:
                    connected = True
                    break
        except Exception:
            time.sleep(0.5)

    if not connected:
        print("ERROR: Server failed to start or did not answer /api/health.")
        proc.terminate()
        return 1

    print(f"Server healthy. Opening {app_url} in browser...")
    webbrowser.open(app_url)
    print("Demo running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping TrustSync server...")
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
