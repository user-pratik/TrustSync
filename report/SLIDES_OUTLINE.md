# TrustSync: Presentation Slides Outline (12 Slides)

## Slide 1: Title & Overview
- **Title**: TrustSync: Multi-Modal Perception for Social Robot Command Authentication
- Bullet 1: The challenge of voice authentication in shared human environments.
- Bullet 2: Tripartite verification: Live human? Which speaker? Directed at robot?
- Bullet 3: Real-time, explainable edge decision framework.
- *Figure*: None (Title slide)

## Slide 2: Motivation & Attack Vectors
- **Title**: Acoustic Vulnerabilities in Social Robotics
- Bullet 1: Smart speakers and robots accept voice commands indiscriminately.
- Bullet 2: Loudspeaker replay spoofs: playing recorded owner speech from mobile phones.
- Bullet 3: Incidental speech: background conversations triggering accidental actions.
- *Figure*: `figures/timeline_replay_spoof.png`

## Slide 3: System Architecture
- **Title**: TrustSync Multi-Modal Perception Pipeline
- Bullet 1: Synchronized RGB video and 16 kHz mono microphone capture.
- Bullet 2: Parallel visual tracking (MediaPipe 478 Mesh) and 100 Hz RMS audio envelope.
- Bullet 3: Sub-second utterance state machine yielding interpretable gate decisions.
- *Figure*: Pipeline ASCII architecture block diagram

## Slide 4: Lip-Audio Synchrony Scoring
- **Title**: Multi-Lag Pearson Correlation for Liveness
- Bullet 1: Scale-free mouth aperture: normalized vertical gap over inter-ocular distance.
- Bullet 2: Feature extraction: dynamic aperture signal, velocity, and log audio envelope.
- Bullet 3: Fast cross-correlation search across ±200 ms lag window.
- *Figure*: `figures/roc_sync.png`

## Slide 5: Multi-Speaker Attribution
- **Title**: Resolving Who Spoke in Multi-Party Settings
- Bullet 1: Multi-face tracking with persistent Hungarian track ID assignment.
- Bullet 2: Simultaneous synchrony estimation for all visible faces.
- Bullet 3: Attribution confidence margin: top speaker score minus second candidate.
- *Figure*: `figures/sync_by_scenario.png`

## Slide 6: Addressee Detection
- **Title**: Head Pose and Gaze Classification
- Bullet 1: 3D head pose via canonical facial geometry and iterative solvePnP.
- Bullet 2: Iris displacement ratio estimating off-axis gaze angle.
- Bullet 3: Geometric directedness score distinguishing direct vs side conversations.
- *Figure*: `figures/roc_directed.png`

## Slide 7: Hierarchical Trust Gating
- **Title**: Transparent Rule Gate & Learned Classifier
- Bullet 1: Strict precedence ordering: No Face -> Quality -> Spoof -> Ambiguity -> Addressee.
- Bullet 2: Every rejection provides a human-interpretable quantitative rationale.
- Bullet 3: Trainable scikit-learn classifier with Leave-One-Subject-Out calibration.
- *Figure*: `figures/ablation.png`

## Slide 8: Dataset & Evaluation Protocol
- **Title**: Balanced Multi-Modal Evaluation Dataset
- Bullet 1: 6 structured scenarios: live single, replay spoof, multi-party, unaddressed, occlusion, silence.
- Bullet 2: Controlled 10-command robotics vocabulary across multiple human subjects.
- Bullet 3: Ethical governance: written participant consent, zero network egress.
- *Figure*: `figures/timeline_live_single.png`

## Slide 9: Headline Performance Results
- **Title**: Benchmark Accuracy & Anti-Spoofing Efficacy
- Bullet 1: Near-zero false accept rate on loudspeaker replay attacks.
- Bullet 2: Robust generalization validated via Leave-One-Subject-Out (LOSO) cross-validation.
- Bullet 3: Statistical rigor demonstrated with 95% Wilson confidence intervals.
- *Figure*: `figures/confusion.png`

## Slide 10: Ablation Analysis
- **Title**: Component Ablation: B0 to A5
- Bullet 1: Naive voice interfaces fail completely against presentation attacks (100% FAR).
- Bullet 2: Lip synchrony eliminates acoustic replay vulnerabilities.
- Bullet 3: Addressee scoring and margin gating successfully resolve multi-party ambiguity.
- *Figure*: `figures/ablation.png`

## Slide 11: Real-Time Performance & Benchmarking
- **Title**: Low-Latency Edge Runtime
- Bullet 1: Throughput exceeding 25+ FPS on standard CPU without dedicated GPU acceleration.
- Bullet 2: Sub-second end-to-end command decision latency upon speech completion.
- Bullet 3: Browser-based full-stack MJPEG streaming and interactive dashboard.
- *Figure*: `figures/lag_hist.png`

## Slide 12: Limitations & Future Directions
- **Title**: Operational Limitations and Future Roadmap
- Bullet 1: Defense boundary: audio-only loudspeaker attacks vs screen video replay.
- Bullet 2: Active challenge-response tokens and IR depth sensing for 3D liveness.
- Bullet 3: Conclusion: TrustSync enables secure, intuitive voice control for interactive robotics.
- *Figure*: None (Conclusion slide)
