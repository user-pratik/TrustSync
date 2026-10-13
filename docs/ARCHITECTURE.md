# TrustSync Architecture & Algorithmic Specification

TrustSync is a real-time multi-modal perception module designed for social robotics. It evaluates spoken user commands by determining whether the utterance was:
1. **Spoken live** (rejecting replay attacks from external phone speakers),
2. **Spoken by which visible person** (resolving multi-party speaker attribution),
3. **Directed at the robot** (distinguishing commands directed at the robot from side conversations).

The pipeline outputs either `ACCEPT` or an explanatory rejection decision (`REJECT_SPOOF`, `REJECT_AMBIGUOUS`, `REJECT_NOT_ADDRESSED`, `REJECT_LOW_QUALITY`, `REJECT_NO_FACE`).

---

## High-Level Pipeline Architecture

```
           +--------------------+              +--------------------+
           |  RGB Video Camera  |              |  Microphone Audio  |
           +---------+----------+              +---------+----------+
                     | (BGR -> RGB)                      | (16 kHz mono)
                     v                                   v
           +--------------------+              +--------------------+
           | MediaPipe 478 Mesh |              | 100 Hz Envelope RMS|
           | Face Tracker + ID  |              | VAD State Machine  |
           +---------+----------+              +---------+----------+
                     |                                   |
         +-----------+-----------+                       |
         |                       |                       |
         v                       v                       |
+-----------------+     +-----------------+              |
| Head Pose & Gaze|     | Mouth Aperture  |              |
| 3D PnP + Iris   |     | Landmark 13/14  |              |
+--------+--------+     +--------+--------+              |
         |                       |                       |
         v                       +----------+ +----------+
+-----------------+                         | |
| Directed Score  |                         v v
| Yaw/Pitch/Gaze  |               +--------------------+
+--------+--------+               | Multi-Lag Pearson  |
         |                        | Synchrony Scorer   |
         |                        +---------+----------+
         |                                  |
         +----------------+ +---------------+
                          | |
                          v v
               +-----------------------+
               |  Trust Gate Fusion    |
               |  Rule-Based / Learned |
               +-----------+-----------+
                           |
                           v
               +-----------------------+
               | Final Utterance State |
               | ACCEPT / REJECT_*     |
               +-----------------------+
```

---

## Algorithmic Subsystems and Mathematical Formulations

### 1. Vision & Face Tracking (`src/trustsync/vision/`)

#### Face Tracking & ID Association (`face_tracker.py`, `id_tracker.py`)
Incoming video frames in BGR format are converted to RGB and processed through MediaPipe FaceMesh (`refine_landmarks=True`, up to `video.max_faces`).
Centroids $c_i = (\frac{x_1+x_2}{2}, \frac{y_1+y_2}{2})$ of detected face bounding boxes are associated with existing tracks using the Hungarian matching algorithm (`scipy.optimize.linear_sum_assignment`) over Euclidean distances:
- Pairwise distances exceeding `tracker.max_dist` are rejected.
- Unmatched detections receive monotonically increasing unique IDs that are never reused.
- Tracks inactive for more than `tracker.max_missed` consecutive frames are pruned.
- **Config keys**: `video.fps`, `video.width`, `video.height`, `video.max_faces`, `tracker.max_dist`, `tracker.max_missed`.

#### Scale-Free Lip Aperture (`lip.py`)
Mouth opening is computed in pixel space from inner lip landmarks 13 (upper) and 14 (lower), normalized by the inter-ocular distance (IOD) between outer eye corners 33 and 263:
$$\text{gap} = \|\mathbf{p}_{13} - \mathbf{p}_{14}\|_2, \quad \text{iod} = \|\mathbf{p}_{33} - \mathbf{p}_{263}\|_2$$
$$\text{mouth\_open} = \begin{cases} \frac{\text{gap}}{\text{iod}}, & \text{if } \text{iod} \ge 10^{-6} \\ 0.0, & \text{otherwise} \end{cases}$$
- **Config keys**: `lip.buffer_s`.

#### Image Quality Assessment (`quality.py`)
A composite quality metric in $[0, 1]$ is evaluated from face bounding box dimensions, mean grayscale luminance, and boundary proximity:
$$s_{\text{size}} = \text{clip}\left(\frac{h_{\text{face}}}{cfg.min\_face\_px}, 0, 1\right)$$
$$s_{\text{bright}} = \text{clip}\left(\frac{\text{mean}(\text{gray}_{\text{crop}})}{cfg.min\_brightness}, 0, 1\right)$$
$$s_{\text{edge}} = \begin{cases} 1.0, & \text{if } \text{frac}(\text{landmarks outside } [0.02, 0.98]) \le cfg.max\_edge\_frac \\ 0.0, & \text{otherwise} \end{cases}$$
$$\text{face\_quality} = s_{\text{size}} \cdot s_{\text{bright}} \cdot s_{\text{edge}}$$
- **Config keys**: `quality.min_face_px`, `quality.min_brightness`, `quality.max_edge_frac`.

---

### 2. Audio Processing & Voice Activity Detection (`src/trustsync/audio/`)

#### Envelope Extraction (`envelope.py`)
Audio sampled at 16000 Hz is downsampled to an energy envelope at `audio.env_hz` (100 Hz). For hop size $H = \lfloor \frac{sr}{env\_hz} \rfloor$, envelope samples are computed as the root-mean-square (RMS) over centered $2H$-sample windows:
$$\text{env}[i] = \sqrt{\frac{1}{2H} \sum_{m=-H}^{H-1} x[i \cdot H + m]^2}$$

#### Adaptive Voice Activity Detection (`envelope.py`)
During the first `vad.calib_s` seconds, the VAD computes baseline noise floor $\bar{n} = \text{mean}(\text{env})$.
For subsequent samples, speech activity is triggered when:
$$\text{speech} = \text{env} > \max(\bar{n} \cdot \text{vad.on\_factor}, \text{vad.min\_abs})$$
Upon speech offset, activity status persists for `vad.hangover_s` seconds. While quiet, the noise floor updates adaptively:
$$\bar{n}_{t+1} = (1 - \alpha) \bar{n}_t + \alpha \cdot \text{env}_t \quad (\alpha = \text{vad.adapt})$$
- **Config keys**: `audio.sample_rate`, `audio.env_hz`, `vad.calib_s`, `vad.on_factor`, `vad.min_abs`, `vad.hangover_s`, `vad.adapt`.

---

### 3. Head Pose, Gaze, and Addressee Estimation (`src/trustsync/vision/`, `src/trustsync/addressee/`)

#### 3D Head Pose (`head_pose.py`)
Standard 3D facial landmarks $[1, 152, 33, 263, 61, 291]$ are mapped to a canonical 3D facial frame and solved using `cv2.solvePnP` (`SOLVEPNP_ITERATIVE`) with zero optical distortion:
Rotation matrix $R = \text{Rodrigues}(\mathbf{r}_{\text{vec}})$ yields Tait-Bryan angles:
$$\text{pitch} = \text{atan2}(R_{2,1}, R_{2,2}), \quad \text{yaw} = \text{atan2}(-R_{2,0}, \sqrt{R_{2,1}^2 + R_{2,2}^2}), \quad \text{roll} = \text{atan2}(R_{1,0}, R_{0,0})$$

#### Eye Gaze Off-Axis (`gaze.py`)
Iris landmark positions (468 right, 473 left) are compared relative to the inner and outer canthi (33, 133 and 362, 263):
$$r_1 = \frac{x_{468} - x_{33}}{x_{133} - x_{33}}, \quad r_2 = \frac{x_{473} - x_{362}}{x_{263} - x_{362}}$$
$$\text{gaze\_off} = \text{clip}\left(2 \cdot \text{mean}(|r_1 - 0.5|, |r_2 - 0.5|), 0, 1\right)$$

#### Directedness Score (`classifier.py`)
Pose angles are zero-centered using calibrated baseline offsets:
$$y_c = \text{yaw} - \text{yaw\_offset\_deg}, \quad p_c = \text{pitch} - \text{pitch\_offset\_deg}$$
$$d = \max\left(\frac{|y_c|}{\text{max\_yaw\_deg}}, \frac{|p_c|}{\text{max\_pitch\_deg}}\right)$$
$$\text{pose} = \begin{cases} 1.0, & \text{if } d \le 1.0 \\ \max(0.0, 2.0 - d), & \text{otherwise} \end{cases}$$
$$\text{directed} = \text{pose} \cdot \left(1.0 - \text{gaze\_weight} \cdot \min\left(1.0, \frac{\text{gaze\_off}}{\text{gaze\_max\_off}}\right)\right)$$
- **Config keys**: `addressee.max_yaw_deg`, `addressee.max_pitch_deg`, `addressee.yaw_offset_deg`, `addressee.pitch_offset_deg`, `addressee.gaze_weight`, `addressee.gaze_max_off`, `addressee.use_gaze`.

---

### 4. Lip-Audio Synchrony Scoring & Speaker Attribution (`src/trustsync/sync/scorer.py`)

1. **Detrending**: Lip aperture $\mathbf{l}$ is smoothed with a 1-second uniform filter $\mathbf{b}$ to isolate fast dynamic movement $\mathbf{l}_d = \mathbf{l} - \mathbf{b}$. If $\text{std}(\mathbf{l}_d) < \text{min\_lip\_std}$, mouth motion is marked static and returns score $0.0$.
2. **Feature Extraction**:
   - Aperture feature: $f_{\text{open}} = \mathbf{l}_d$
   - Velocity feature: $f_{\text{vel}} = \text{uniform\_filter1d}(|\nabla \mathbf{l}_d \cdot \text{fps}|, 3)$
   - Log audio envelope: $e = \log(\text{env} + \text{env\_eps})$
3. **Multi-Lag Pearson Search**: For lag $j \in [-K, K]$ where $K = \text{round}(\text{max\_lag\_s} \cdot \text{fps})$:
   $$r(j) = \frac{\sum (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum (a_i - \bar{a})^2 \sum (b_i - \bar{b})^2}}$$
   where positive $j$ indicates audio occurs later than lip movement.
4. **Attribution**: The candidate speaker is the track with the maximum correlation score. Attribution confidence is defined as the score margin between the highest and second-highest tracks:
   $$\text{attribution\_conf} = s_{(1)} - s_{(2)}$$
- **Config keys**: `sync.window_s`, `sync.min_window_s`, `sync.max_lag_s`, `sync.min_lip_std`, `sync.env_eps`.

---

### 5. Utterance State Machine & Trust Gate Decision Order (`src/trustsync/fusion/`)

#### Utterance Tracker (`utterance.py`)
- Initiates upon `speech == True`.
- Accumulates per-frame face observations (directedness, quality, yaw, pitch).
- Closes upon `speech == False` or when elapsed time reaches `utterance.max_s`.
- Discards utterances shorter than `utterance.min_s`.
- For retained utterances, computes median metrics across active frames and evaluates synchrony across the padded span $[t_{\text{start}} - \text{pad\_s}, t_{\text{end}} + \text{pad\_s}]$.

#### Rule Gate Precedence Hierarchy (`gate.py`)
Decisions are evaluated in strict priority order:
1. **Rule 1 (No Faces)**: If no tracks are visible $\to$ `REJECT_NO_FACE`.
2. **Rule 2 (General Low Quality)**: If $\max_{i}(\text{quality}_i) < \text{min\_quality}$ $\to$ `REJECT_LOW_QUALITY`.
3. **Rule 3 (Replay Spoof)**: For candidate track with highest sync, if $\text{sync} < \text{min\_sync}$ or $\text{lip\_std} < \text{min\_lip\_std}$ $\to$ `REJECT_SPOOF`.
4. **Rule 4 (Ambiguous Speaker)**: If multiple tracks exist and $s_{(1)} - s_{(2)} < \text{min\_margin}$ $\to$ `REJECT_AMBIGUOUS`.
5. **Rule 5 (Speaker Low Quality)**: If candidate speaker quality $< \text{min\_quality}$ $\to$ `REJECT_LOW_QUALITY`.
6. **Rule 6 (Unaddressed Command)**: If candidate speaker directedness $< \text{min\_directed}$ $\to$ `REJECT_NOT_ADDRESSED`.
7. **Rule 7 (Acceptance)**: Otherwise $\to$ `ACCEPT` with speaker identified.
- **Config keys**: `gate.mode`, `gate.min_sync`, `gate.min_margin`, `gate.min_directed`, `gate.min_quality`, `gate.model_path`.
