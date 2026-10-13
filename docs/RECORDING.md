# TrustSync Recording Protocol and Guidelines

This document outlines the session capture protocol, scenario specifications, and ethical data governance procedures for the TrustSync evaluation dataset.

## Setup and Environment

Record at a fixed recording station:
- **Webcam Position**: At eye level, approximately 60 cm distance from the subject.
- **Acoustic Environment**: Quiet room with low reverberation and controlled background noise.
- **Replay Setup**: Phone speaker positioned roughly 10 cm below chin level at normal speech volume.
- **Clip Duration**: 7 seconds per clip.
  - Initial 1.5 seconds: Remain silent while the Voice Activity Detector (VAD) calibrates the acoustic noise floor.
  - Subsequent 5.5 seconds: Speak the prompted command clearly once.
- **Workflow Order**: Always record live clips before replay clips, as replay attacks playback the subject's own recorded audio.

---

## Recording Protocol Matrix

| Scenario | Capture Instructions | Minimum Clips | Full Dataset | Expected Gate Decision |
| :--- | :--- | :---: | :---: | :--- |
| `live_single` | Single subject faces the camera directly and speaks the command. | 5 per subject (15) | 8 per subject (24) | `ACCEPT` |
| `replay_spoof` | Same subject remains in frame with mouth firmly closed and motionless. A phone held ~10 cm below chin plays back the subject's own `live_single` audio. | 5 per subject (15) | 8 per subject (24) | `REJECT_SPOOF` |
| `multi_one_speaks` | Two subjects sit side by side in frame. Exactly one speaks while the other remains silent but naturally alive (blinking, head shift). Record with `--speaker-side left` or `right`. Swap roles and seating positions. | 2 per pair (6) | 4 per pair (12) | `ACCEPT` (with matching `speaker_side`) |
| `not_addressed` | Single subject speaks the command while turned 45° to 60° away from the camera (facing a side wall or partner). Recorded with `--addressed false`. | 5 per subject (15) | 8 per subject (24) | `REJECT_NOT_ADDRESSED` |
| `occlusion_lowlight` | Mouth fully occluded by hand/mask or dim ambient lighting (lit only by screen). Recorded with `--occlusion true` and `--lighting dim`. | 3 per subject (9) | 6 per subject (18) | `REJECT_ANY` |
| `silence` | Subjects visible in frame but nobody speaks. Normal ambient room sounds and keyboard typing only. | 4 | 6 | `NONE` (no utterance produced) |
| **Total** | | **64 clips** | **108 clips** | |

---

## Command Prompts

Prompt texts cycle deterministically through this standard 10-item social robotics lexicon:
1. `robot, bring me water`
2. `stop`
3. `turn left`
4. `what time is it`
5. `open the door`
6. `call the nurse`
7. `turn on the light`
8. `go to the kitchen`
9. `play some music`
10. `follow me`

---

## Example Recording CLI Invocations

```bash
# Record live single-subject commands
python tools/record_session.py --scenario live_single --subjects pratik --count 8

# Record multi-speaker session with speaker on the left
python tools/record_session.py --scenario multi_one_speaks --subjects pratik suchet --speaker-side left --count 2

# Record unaddressed commands (head turned away)
python tools/record_session.py --scenario not_addressed --subjects suchet --addressed false --count 8

# Validate all recorded sessions
python tools/validate_session.py --all
```

---

## Ethical Consent and Privacy

All recorded people gave written consent, data stays local, and faces are not published.
- All session recordings (video frames, raw audio waveforms, metadata) remain strictly on the local machine and are never uploaded to remote servers.
- Bounding boxes and landmarks are processed locally; face imagery is restricted to evaluation and debug verification.
