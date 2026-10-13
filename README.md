# TrustSync: Real-Time Multi-Modal Perception for Social Robots

TrustSync is a real-time multi-modal perception module designed for social robots to verify spoken voice commands before execution. When a speech command is heard by the robot, TrustSync answers three fundamental physical questions: (1) was the command spoken **live** by a physical human in front of the robot or replayed through an external loudspeaker, (2) **who** among the visible faces produced the acoustic signal, and (3) was the command **directed at the robot** or uttered during side conversations. The system fuses facial landmark tracking, lip aperture dynamics, audio envelope energy, and head pose estimation into an interpretable gating decision (`ACCEPT` or a specific `REJECT_*` reason).

---

## Quick-Start Installation

Prerequisites: Python 3.11 and Node.js 20 LTS.

To set up the development environment, execute the installation steps in this exact order:

```bash
pip install mediapipe==0.10.14
pip install -r requirements.txt
pip install -e .
```

Verify your setup with the Phase 6 smoke test:

```bash
python tools/gates/run_gate.py 6
```

---

## Running the Live Demo

TrustSync includes a full-stack dashboard featuring a FastAPI backend with raw camera MJPEG streaming and a modern React control interface:

```bash
python tools/run_demo.py
```

This launches the local server and automatically opens the dashboard in your web browser at `http://127.0.0.1:8000`.

---

## Recording Evaluation Data

To collect balanced multi-modal datasets across live, spoof, multi-speaker, and unaddressed scenarios, follow the comprehensive protocol in [docs/RECORDING.md](docs/RECORDING.md):

```bash
# Example: record live single-speaker session
python tools/record_session.py --scenario live_single --subjects pratik --count 8

# Validate recorded clips
python tools/validate_session.py --all
```

---

## Evaluation & Calibration

To run end-to-end evaluation, leave-one-subject-out (LOSO) cross-validation, threshold sweeps, and train the learned gating classifier:

```bash
python -m trustsync.eval.run_eval --calibrate --learned
```

Evaluation metrics, ablation results, and publication-ready figures are generated directly into `reports/` and `report/`.

---

## Repository Layout

```
trustsync/
├── AGENTS.md               # Agent pairing rules & project invariants
├── PROGRESS.md             # Phase progression log
├── README.md               # Project documentation and quick-start
├── requirements.txt        # Pinned Python package dependencies
├── pyproject.toml          # Build configuration and pytest markers
├── config.yaml             # Thresholds and hardware configuration
├── docs/
│   ├── SPEC.md             # Frozen system contract and algorithms
│   ├── ARCHITECTURE.md     # Pipeline block diagrams and formulas
│   └── RECORDING.md        # Recording protocol and data ethics
├── src/trustsync/
│   ├── config.py           # Dataclass configuration loader and validator
│   ├── types.py            # Typed perception and decision dataclasses
│   ├── synth.py            # Hardware-free deterministic signal generators
│   ├── pipeline.py         # End-to-end perception orchestration
│   ├── utterance.py        # Utterance state machine aggregator
│   ├── audio/              # RMS envelope extraction and VAD
│   ├── vision/             # Face tracking, lip aperture, quality, pose, gaze
│   ├── addressee/          # Addressee angle classification
│   ├── sync/               # Multi-lag Pearson synchrony scoring
│   ├── fusion/             # Rule-based and learned gating fusion
│   ├── io/                 # Video/audio streaming and recording
│   ├── overlay/            # OpenCV HUD visualization overlay
│   ├── eval/               # Evaluation metrics, calibration, and plotting
│   └── server/             # FastAPI backend and WebSocket runtime
├── tools/                  # Offline tools, recording, benchmarks, demos
├── web/                    # Vite + React 18 + Tailwind modern dashboard
└── tests/                  # Deterministic unit, integration, and robustness tests
```

---

## Phases and Release Tags

| Phase | Description | Release Tag |
| :---: | :--- | :--- |
| Phase 0 | Environment, scaffold, and gate framework | `phase-0-pass` |
| Phase 1 | Typed config, data structures, synthetic generators | `phase-1-pass` |
| Phase 2 | Audio pipeline, RMS envelope, adaptive VAD | `phase-2-pass` |
| Phase 3 | Face tracking, lip aperture extraction, quality metrics | `phase-3-pass` |
| Phase 4 | Head pose estimation, eye gaze, addressee scoring | `phase-4-pass` |
| Phase 5 | Lip-audio synchrony scoring and attribution | `phase-5-pass` |
| Phase 6 | End-to-end pipeline, rule gate, HUD overlay | `phase-6-pass` |
| Phase 7 | Session recorder, format validator, protocol | `phase-7-pass` |
| Phase 8 | Evaluation framework, threshold calibration, learned gate | `phase-8-pass` |
| Phase 9 | FastAPI backend, raw MJPEG stream, WebSocket API | `phase-9-pass` |
| Phase 10 | Modern responsive dark dashboard (mock stream) | `phase-10-pass` |
| Phase 11 | Full web integration, replay runner, settings, results | `phase-11-pass` |
| Phase 12 | Hardening, robustness fuzzing, latency benchmark | `phase-12-pass` |
| Phase 13 | Documentation, demo kit, release verification | `phase-13-pass` |
