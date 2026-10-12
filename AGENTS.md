# TrustSync agent rules (always on)
Project: a real-time perception module for a social robot. A camera and a
microphone feed a pipeline that decides whether a spoken command was (1)
spoken live, (2) spoken by which visible face, (3) directed at the robot.
Output is ACCEPT or a REJECT_* reason.
## Source of truth
- docs/SPEC.md is frozen. Never change it. Never rename or re-sign a file,
class, function, field or config key it defines. If SPEC.md looks wrong, STOP
and ask the human.
- Build only what the current prompt asks for. No extra features. No
refactors of earlier phases.
## Working rules
1. Python 3.11. Type hints on every function. Document numpy array shapes in
docstrings.
2. Every threshold and constant comes from config.yaml through
src/trustsync/config.py. No magic numbers in code.
3. No global mutable state. No sleep() in library code. Use logging, never
print(), outside tools/.
4. Time: every timestamp is float seconds from time.monotonic(). Video and
audio share this clock.
5. Library code must run without a camera or microphone. Hardware is touched
only in src/trustsync/io/ and tools/.
6. Tests: pytest, seeded randomness (np.random.default_rng(1234)), no
network, no single test over 5 s. Hardware tests carry @pytest.mark.hardware.
7. Never edit a test to make it pass. Never delete a failing test. Never hide
an error with try/except.
8. Never install a package that is not in requirements.txt without asking.
9. Run tests: pytest -q -m "not hardware" Run gates: python
tools/gates/run_gate.py N
10. Tests never read live values from config.yaml (calibration edits it).
They use the spec_cfg fixture in tests/conftest.py, which loads
tests/fixtures/config_spec_defaults.yaml. Never edit that fixture.
## Report-back format (end of every task)
FILES CREATED OR CHANGED: list
COMMANDS RUN: each with PASS or FAIL
GATE RESULT: the final table
OPEN PROBLEMS: list, or none
Never claim a command passed unless you ran it and saw its output.
## Things that fail silently in this project
- OpenCV frames are BGR; MediaPipe needs RGB.
- MediaPipe landmarks are normalized 0..1; multiply x by width and y by
height for pixels.
- Audio is 16000 Hz, mono, float32 in [-1, 1].
- Pearson correlation of a constant signal is NaN; guard std < 1e-8.
- Windows: use pathlib.Path, never hard-coded slashes.
