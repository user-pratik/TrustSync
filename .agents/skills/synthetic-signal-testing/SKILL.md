---
name: synthetic-signal-testing
description: Use when writing or reviewing tests for audio envelopes, VAD, lip signals, sync scoring, head pose or any numeric code in TrustSync. Build deterministic synthetic inputs instead of needing a camera or microphone.
---
# Rules for numeric tests
- Every test creates rng = np.random.default_rng(1234). Never use unseeded randomness.
- Build inputs with the helpers in src/trustsync/synth.py.
- Each numeric assertion states its tolerance and the reason in a comment (example: within 2 frames = 67 ms at 30 fps).
- Always write three cases: positive (must detect), negative (must not detect), degenerate (empty, constant, NaN, too short).
- Statistical claims use at least 200 seeds and assert on percentiles, never on a single draw.
- Assert a margin around a threshold, not just the threshold itself.
- Never loosen a tolerance to make a test pass. Fix the code, or report STUCK.
- Known bug sources to test on purpose: lag sign, BGR vs RGB, normalized vs pixel coordinates, seconds vs milliseconds, division by zero, off-by-one at window edges.
