---
description: Run the cumulative gate for a phase and show the result table
---
1. Ask which phase number N, unless the user already gave it.
2. Run: python tools/gates/run_gate.py N
3. Run: pytest -q -m "not hardware"
4. Paste both final tables. If anything is FAIL, stop and list the failing rows. Do not fix anything.
