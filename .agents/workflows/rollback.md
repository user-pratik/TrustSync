---
description: Return to the last passing phase after failed attempts
---
1. Ask which phase N failed.
2. Run: git stash -u
3. Run: git checkout -b retry-phase-N phase-(N-1)-pass
4. Run: python tools/gates/run_gate.py (N-1) and print the table.
5. Do not start any new work.
