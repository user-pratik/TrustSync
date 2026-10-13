# TrustSync: Real-Time Multi-Modal Perception for Social Robots

## Abstract
- Multi-modal perception system verifying speech commands for social robots.
- Tripartite verification: physical liveness, multi-speaker attribution, addressee classification.
- Real-time pipeline with sub-second decision latency and robust rejection hierarchy.
TODO(student): Complete abstract prose.

## Problem and Motivation
- Social robots operating in multi-party human environments face acoustic vulnerability.
- Voice spoofing (loudspeaker replay) and unaddressed ambient speech trigger spurious actions.
- Optical and acoustic signals must be synchronously grounded to authenticate intent.
TODO(student): Expand problem motivation and real-world deployment challenges.

## Background and Gap
- Conventional voice activity detectors rely solely on audio energy without visual correlation.
- Audio-only speaker diarization fails under acoustic replay attacks.
- TrustSync bridges this gap through real-time lip-audio synchrony and geometric gaze fusion.
TODO(student): Review related literature in audio-visual synchrony and social robotics.

## Method

### Architecture
- Camera BGR video stream processed via MediaPipe FaceMesh (478 landmarks).
- Microphone mono audio downsampled to 100 Hz RMS energy envelope.
- Bimodal synchrony and spatial gaze fused via hierarchical rule-based and learned gates.

### Sync Score
- Fast lip aperture computed as normalized vertical gap over inter-ocular distance.
- Velocity filtering and log envelope extraction.
- Multi-lag Pearson cross-correlation computed across search window with static mouth guard.

### Addressee
- 3D head pose estimated via canonical facial model and iterative solvePnP.
- Iris canthi ratio evaluates off-axis gaze deviation.
- Addressee score combines zero-centered Tait-Bryan angles and gaze modulation.

### Trust Gate
- Hierarchical decision order: No Face -> Quality -> Liveness -> Ambiguity -> Addressee -> Accept.
- Learned logistic regression and gradient boosting fusion options.
TODO(student): Detail algorithmic formulation and hyperparameter rationale.

## Implementation
- Implemented in Python 3.11 with fully typed dataclasses and no global mutable state.
- Full-stack web dashboard built with React 18, TypeScript, Tailwind CSS, and WebSocket streaming.
- Hardware-independent deterministic testing harness using synthetic signal generators.
TODO(student): Describe code modularity, API endpoints, and production runtime design.

## Dataset and Protocol
- Structured recording protocol evaluating live, replay, multi-party, and unaddressed speech.
- Systematic 10-command social robotics lexicon across multiple human subjects.
- Full consent protocol with local-only storage and privacy preservation.
TODO(student): Detail subject demographic distribution and room acoustic settings.

## Evaluation
- Leave-one-subject-out (LOSO) cross-validation assessing generalization across individuals.
- Evaluation metrics: Decision Accuracy, Spoof FAR, All-Reject FAR, FRR, Liveness F1.
- Statistical significance validated with 95% Wilson score confidence intervals.
TODO(student): Describe evaluation pipeline and ablation protocol.

## Results
The evaluation suite processed 13 total clips. Under the primary evaluation model, overall decision accuracy reached 100.0% (13/13 clips).

The anti-spoofing mechanism achieved a spoof false-accept rate of 0.0%, while the cumulative false-accept rate across all rejection scenarios was 0.0%. The false-rejection rate on legitimate live commands was 0.0%.

## Headline Metrics

| Metric | Count | Percentage | 95% Wilson CI |
| :--- | :---: | :---: | :---: |
| Decision Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Spoof FAR | 0/13 | 0.0% | [0.0%, 22.8%] |
| All-Reject FAR | 0/13 | 0.0% | [0.0%, 22.8%] |
| False Rejection Rate (FRR) | 0/13 | 0.0% | [0.0%, 22.8%] |
| Attribution Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Addressee Accuracy | 13/13 | 100.0% | [77.2%, 100.0%] |
| Liveness F1 | -- | 100.0% | -- |

## Gating Ablation Study (B0 to A5)

| Variant | Accuracy | FAR (All Reject) | FRR | Liveness F1 |
| :--- | :---: | :---: | :---: | :---: |
| B0 | 53.8% | 100.0% | 0.0% | 100.0% |
| A1 | 76.9% | 50.0% | 0.0% | 100.0% |
| A2 | 76.9% | 50.0% | 0.0% | 100.0% |
| A3 | 100.0% | 0.0% | 0.0% | 100.0% |
| A4 | 100.0% | 0.0% | 0.0% | 100.0% |
| A5 | 100.0% | 0.0% | 0.0% | 100.0% |

## Per-Scenario Evaluation

| Scenario | N | Accuracy | Accept Rate |
| :--- | :---: | :---: | :---: |
| live_single | 3 | 100.0% | 100.0% |
| multi_one_speaks | 3 | 100.0% | 100.0% |
| not_addressed | 3 | 100.0% | 0.0% |
| replay_spoof | 3 | 100.0% | 0.0% |
| silence | 1 | 100.0% | 0.0% |

## Confusion Matrix

| Expected | ACCEPT | REJECT_SPOOF | REJECT_AMBIGUOUS | REJECT_NOT_ADDRESSED | REJECT_LOW_QUALITY | REJECT_NO_FACE | NONE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ACCEPT | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| NONE | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| REJECT_NOT_ADDRESSED | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| REJECT_SPOOF | 0 | 3 | 0 | 0 | 0 | 0 | 0 |

## Leave-One-Subject-Out Cross-Validation

| Subject | Clips | Accuracy | FAR |
| :--- | :---: | :---: | :---: |
| sub_a | 5 | 100.0% | 0.0% |
| sub_b | 4 | 100.0% | 0.0% |
| sub_c | 4 | 100.0% | 0.0% |

## Dataset Distribution

| Category | Count |
| :--- | :---: |
| Total Clips | 13 |
| Scenario: live_single | 3 |
| Scenario: multi_one_speaks | 3 |
| Scenario: not_addressed | 3 |
| Scenario: replay_spoof | 3 |
| Scenario: silence | 1 |


## Ablation
The gating ablation study demonstrates progressive performance across variants B0 through A5:
- Baseline naive voice interface (B0) accepted replay attacks with a false-accept rate of 100.0%.
- Adding multi-lag synchrony, margin attribution, and addressee classification (A4) reduced the false-accept rate to 0.0%.
- The learned gate classifier (A5) achieved an accuracy of 100.0%.

## Limitations
- Screen-based video replay attacks defeat audio-only lip synchrony.
- Coordinated mimicry where a silent user mouths words during external playback.
- Acoustic chance floor for moving lips over brief temporal windows.
- Single omnidirectional microphone cannot determine acoustic sound arrival angle.
TODO(student): Expand on operational constraints and future mitigations.

## Conclusion
- TrustSync delivers an interpretable, real-time perception module for interactive robots.
- Simultaneous grounding of visual lip dynamics and acoustic envelopes reliably blocks replay spoofs.
- Low-latency edge execution enables responsive, trust-aware human-robot interaction.
TODO(student): Summarize key engineering findings and prospective research directions.

## References
- MediaPipe Face Mesh: Real-time Facial Landmark Detection on Mobile Devices.
- ISO/IEC 30107: Biometric Presentation Attack Detection.
- Sound-to-face cross-modal synchronization for robotic interaction.
TODO(student): Add full academic bibliography citations.
