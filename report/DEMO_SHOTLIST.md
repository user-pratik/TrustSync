# TrustSync: 3-Minute Demonstration Video Shot List

This shot list details the visual sequence, spoken script, and dashboard state transitions for the four core proposal demonstrations.

---

### Shot 1: System Introduction & Live Command Acceptance (0:00 - 0:45)
- **Visual / Screen**:
  - Full-screen view of the TrustSync web dashboard at `http://127.0.0.1:8000`.
  - Live video hero card showing user sitting frontal at 60 cm distance with corner-bracket face box in teal.
  - Decision Card transitions from `IDLE` (slate) to `LISTENING` (amber) as voice begins, then snaps to `ACCEPTED` (teal).
  - Trust gauges show high Liveness (>0.70), high Attribution (>0.20), high Addressee (>0.90).
- **Spoken Narration**:
  - *"Welcome to the demonstration of TrustSync, a real-time multi-modal perception module that verifies spoken voice commands for social robots."*
  - User speaks: *"Robot, bring me water."*
  - *"Notice the system tracks facial motion, correlates my lip aperture with speech acoustics, and confirms frontal gaze. The command is immediately accepted."*

---

### Shot 2: Loudspeaker Replay Spoof Rejection (0:45 - 1:30)
- **Visual / Screen**:
  - User sits facing camera with lips completely still.
  - A mobile phone positioned 10 cm below the chin plays back the user's previously recorded voice saying *"Open the door"*.
  - Envelope strip shows acoustic energy and VAD indicates speech active.
  - Face overlay turns red; Decision Card displays `REJECTED: NOT LIVE` (`REJECT_SPOOF`).
  - Liveness gauge drops to zero; reason states: *"best sync below min threshold and lip motion static"*.
- **Spoken Narration**:
  - *"Now we demonstrate defense against presentation attacks. I will remain silent while my phone plays back a recorded voice command beside my mouth."*
  - Mobile phone plays: *"Open the door."*
  - *"Although loud audio was captured, the lip aperture remained motionless. TrustSync flags the static mouth and rejects the spoof command instantly."*

---

### Shot 3: Multi-Speaker Attribution & Turn-Taking (1:30 - 2:15)
- **Visual / Screen**:
  - Two subjects sit side by side in frame (Track 0 left, Track 1 right).
  - Track 0 remains silent while smiling/shifting. Track 1 speaks *"Turn left"*.
  - Dashboard shows two corner-bracket boxes: Track 1 in teal (attributed speaker), Track 0 in slate (idle).
  - Sync timeline shows two live correlation curves with clear separation between speakers.
  - Event log records `ACCEPTED` with `Speaker ID: 1`.
- **Spoken Narration**:
  - *"In shared social environments, multiple people are frequently visible. Here, two subjects are present, but only the person on the right speaks."*
  - Subject on right speaks: *"Turn left."*
  - *"By cross-correlating each person's unique mouth movement with the shared acoustic envelope, TrustSync correctly attributes the speech to Speaker 1."*

---

### Shot 4: Unaddressed Speech Rejection (2:15 - 2:45)
- **Visual / Screen**:
  - Single subject speaks while turned 50° away toward a side partner or wall.
  - Gaze arrow on face overlay points sideways; head pose displays `yaw ~ 45°`.
  - Decision Card renders red `REJECTED: NOT FOR THE ROBOT` (`REJECT_NOT_ADDRESSED`).
  - Addressee gauge falls below threshold; reason notes: *"directed score below min threshold"*.
- **Spoken Narration**:
  - *"Finally, robots must ignore side conversations not intended for them. I will now speak while turning my head away."*
  - User speaks turned away: *"What time is it?"*
  - *"The speech is live and clearly spoken, but 3D head pose and gaze estimation detect that I am not looking at the robot. The command is politely discarded."*

---

### Shot 5: Conclusion & Summary (2:45 - 3:00)
- **Visual / Screen**:
  - Replay and Results pages shown, highlighting the KPI overview cards and confusion matrix.
- **Spoken Narration**:
  - *"TrustSync provides social robots with reliable, explainable perception—ensuring commands are live, attributed, and addressed. Thank you."*
