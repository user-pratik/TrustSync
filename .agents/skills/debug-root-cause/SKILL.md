---
name: debug-root-cause
description: Use when a test, gate, build or run fails in TrustSync, before changing any code.
---
# Debug protocol
1. Reproduce: re-run the failing command and keep the last 40 lines of output.
2. Localize: name the file and function where the first wrong value appears. Print intermediate values; do not guess.
3. One hypothesis at a time. Change one thing, re-run, record the result.
4. Fix the cause in src/, never in the test, never by loosening a tolerance, never by catching the exception.
5. After the fix, run the whole gate, not only the failing test.
6. After two failed attempts write STUCK, list what you ruled out, and stop.
