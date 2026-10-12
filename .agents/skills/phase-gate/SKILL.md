---
name: phase-gate
description: Use when finishing, verifying or starting any TrustSync phase, or when asked to run a gate, check for regressions, or report phase status.
---
# Phase gate procedure
1. python tools/gates/run_gate.py N runs gate_0.py through gate_N.py in order and stops at the first failure. Each gate_K.py prints a table with columns CHECK | RESULT | DETAIL and exits 0 only if every row is PASS.
2. Always also run: pytest -q -m "not hardware"
3. Hardware rows (camera, microphone, live FPS) print SKIP unless the script is run with --hardware. A phase is tag-ready only after the human has run it once with --hardware and every row is PASS.
4. Never say a phase is done unless run_gate.py N exited 0. Paste the final table in your answer.
5. After a pass, append one line to PROGRESS.md: phase N | PASS | date | oneline summary.
6. Later phases may ADD checks to earlier gate files; never weaken, skip or delete an existing check.
