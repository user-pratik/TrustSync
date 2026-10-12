# TrustSync SPEC v1 (FROZEN)

## 1. Stack
Python 3.11. requirements.txt (install mediapipe FIRST; it brings its own
opencv-contrib-python; never install opencv-python):
mediapipe==0.10.14
numpy==1.26.4
scipy==1.11.4
sounddevice==0.4.7
soundfile==0.12.1
PyYAML==6.0.1
scikit-learn==1.4.2
joblib==1.4.2
pandas==2.2.2
matplotlib==3.8.4
fastapi==0.111.0
uvicorn[standard]==0.30.1
httpx==0.27.0
pytest==8.2.2
pyproject.toml: package 'trustsync' in src/, pytest pythonpath=['src'],
marker 'hardware'. Install with pip install -e .

## 2. Tree
trustsync/
AGENTS.md PROGRESS.md README.md requirements.txt pyproject.toml config.yaml
.gitignore
docs/SPEC.md
.agents/rules .agents/skills .agents/workflows
src/trustsync/
__init__.py config.py types.py synth.py pipeline.py utterance.py
audio/{__init__,envelope,sources}.py
vision/{__init__,face_tracker,id_tracker,lip,quality,head_pose,gaze}.py
addressee/{__init__,classifier}.py
sync/{__init__,scorer}.py
fusion/{__init__,gate,learned}.py
io/{__init__,video_source,recorder}.py
overlay/{__init__,hud}.py
eval/{__init__,metrics,calibrate,run_eval,plots}.py
server/{__init__,app,runtime}.py
tools/{run_offline,record_session,validate_session,capture_sample,calibrate_pose,benchmark,demo_live,audio_meter,make_demo_session,run_demo,make_report_assets,make_report_skeleton}.py
tools/gates/{run_gate.py, gate_0.py ... gate_13.py}
tests/conftest.py tests/helpers_world.py tests/helpers_eval.py
tests/test_*.py tests/fixtures/ (config_spec_defaults.yaml and your face
jpgs)
docs/ARCHITECTURE.md docs/RECORDING.md report/ (generated in Phase 13)
out/ (scratch, git-ignored)
data/sessions/<session_id>/{video.mp4,audio.wav,frame_times.csv,meta.json}
models/ reports/figures/ reports/results.json web/

## 3. config.yaml (exact keys; write in normal block style; * = re-calibrated in Phase 8)
video: fps 30, width 640, height 480, camera_index 0, max_faces 4
audio: sample_rate 16000, env_hz 100, device null, latency_s 0.0, ring_s 10.0
vad: calib_s 1.0, on_factor 3.0, min_abs 0.004, hangover_s 0.30, adapt 0.005
utterance: min_s 0.40, max_s 4.0, pad_s 0.25
tracker: max_dist 0.15, max_missed 15
lip: buffer_s 6.0
sync: window_s 1.5, min_window_s 0.6, max_lag_s 0.20, min_lip_std 0.004*,
env_eps 0.0001
quality: min_face_px 80, min_brightness 40, max_edge_frac 0.05
addressee: max_yaw_deg 25*, max_pitch_deg 20*, yaw_offset_deg 0.0,
pitch_offset_deg 0.0, gaze_weight 0.3, gaze_max_off 0.6, use_gaze true
gate: mode rule, min_sync 0.35*, min_margin 0.10*, min_directed 0.5*,
min_quality 0.5*, model_path models/gate.joblib
server: host 127.0.0.1, port 8000, ws_hz 20, jpeg_quality 80
config.py: one frozen dataclass per section (VideoCfg, AudioCfg, VadCfg,
UtteranceCfg, TrackerCfg, LipCfg, SyncCfg, QualityCfg, AddresseeCfg, GateCfg,
ServerCfg) whose field names equal the keys, and Config holding them under
the section names. load_config(path: Path|None=None) -> Config (default:
repo-root config.yaml); save_config(cfg, path); Config.to_dict().

## 4. Types (src/trustsync/types.py, all dataclasses)
DECISIONS =
('IDLE','LISTENING','ACCEPT','REJECT_SPOOF','REJECT_AMBIGUOUS','REJECT_NOT_ADDRESSED','REJECT_LOW_QUALITY','REJECT_NO_FACE')
FaceObs(frozen): track_id:int, bbox:tuple[float,float,float,float]
(x1,y1,x2,y2 normalized 0..1), pts:np.ndarray (478,3) normalized x,y plus
relative z, mouth_open:float, quality:float
FaceScores: track_id:int, bbox, sync:float, lip_std:float, best_lag_s:float,
yaw:float, pitch:float, roll:float, gaze_off:float, directed:float,
quality:float
SyncResult: score:float, lag_s:float, lip_std:float, valid:bool,
r_open:float, r_vel:float
TrackSummary: track_id:int, sync:float, lip_std:float, directed:float,
quality:float, yaw:float, pitch:float
UtteranceSummary: t_start:float, t_end:float, duration_s:float,
tracks:list[TrackSummary]
GateDecision: decision:str, reason:str, speaker_id:int|None, liveness:float,
attribution_conf:float, addressee:float
FrameResult: t:float, frame_idx:int, fps:float, faces:list[FaceScores],
speech_active:bool, env:float, noise_floor:float, speaker_id:int|None,
attribution_conf:float, liveness:float, addressee:float, decision:str,
reason:str, utterance_final:bool
frame_result_to_dict(r: FrameResult) -> dict: JSON-safe (python floats
rounded to 4 decimals, nan becomes null, tuples become lists).

## 5. Modules, signatures and exact algorithms
### audio/envelope.py
compute_envelope(x: np.ndarray, sr: int, env_hz: int) -> np.ndarray: float32
RMS; hop = sr//env_hz; window = 2*hop centred on each hop; one value per hop,
length ceil(len(x)/hop); zero-pad the end.
class VAD(cfg: VadCfg, dt: float): update(env: float) -> bool; properties
noise_floor, calibrated. For the first calib_s seconds (count updates*dt)
noise_floor = mean(env) and it returns False. Afterwards on = env >
max(noise_floor*on_factor, min_abs); output stays True for hangover_s after
the last on sample; while output is False, noise_floor = (1-
adapt)*noise_floor + adapt*env.
segment_speech(env: np.ndarray, env_hz: float, cfg: VadCfg) ->
list[tuple[float,float]]: runs VAD with dt=1/env_hz; each segment is (time of
first on sample, time of last on sample); trailing hangover is trimmed.

### audio/sources.py
class AudioSource: envelope_at(times: np.ndarray) -> np.ndarray (linear
interpolation on the stored envelope timeline; 0.0 outside the covered
range); latest() -> tuple[float,float] (t, env); close().
NumpyAudioSource(env: np.ndarray, env_hz: float, t0: float) : env[i] is at
time t0 + i/env_hz.
FileAudioSource(wav_path: Path, t0: float, cfg: AudioCfg): soundfile read,
mix to mono, require sample rate == cfg.sample_rate (else ValueError),
compute_envelope, delegate to NumpyAudioSource.
LiveAudioSource(cfg: AudioCfg, record_to: Path|None=None):
sounddevice.InputStream(samplerate, channels=1, blocksize=sr//env_hz, dtype
float32, device cfg.device). Callback computes ONE envelope value (block RMS)
with timestamp time.monotonic() - cfg.latency_s - 0.5*block_s; keeps the last
ring_s seconds in a lock-protected deque. If record_to is set it also writes
raw samples to that WAV and exposes t0 (monotonic time of the first sample).
start(), close().

### vision
lip.mouth_open(pts, w, h) -> float: pixel coords (x*w, y*h); gap =
dist(landmark 13, landmark 14); iod = dist(landmark 33, landmark 263); return
gap/iod, or 0.0 if iod < 1e-6.
id_tracker.IdTracker(max_dist: float, max_missed: int).assign(centroids:
np.ndarray (n,2) normalized) -> list[int]: Hungarian matching
(scipy.optimize.linear_sum_assignment) on euclidean distance to existing
tracks; a pair with distance > max_dist is rejected; unmatched detections get
new ids 0,1,2,... that are never reused; a track unmatched for more than
max_missed consecutive calls is deleted. Centroid = centre of the landmark
bounding box.
quality.face_quality(frame_bgr, bbox, pts, cfg: QualityCfg) -> float =
s_size*s_bright*s_edge. s_size = clip(face_height_px/min_face_px,0,1).
s_bright = clip(mean grayscale inside bbox / min_brightness,0,1). s_edge =
1.0 if the fraction of landmarks with x or y outside [0.02,0.98] is <=
max_edge_frac else 0.0.
face_tracker.FaceTracker(cfg: Config): process(frame_bgr) -> list[FaceObs];
close(). Uses mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=False,
max_num_faces=cfg.video.max_faces, refine_landmarks=True,
min_detection_confidence=0.5, min_tracking_confidence=0.5) on the RGB frame.
Returns [] when no face.
head_pose.estimate_pose(pts, w, h) -> tuple[float,float,float] (yaw, pitch,
roll in degrees; NaN triple on failure). Landmarks [1,152,33,263,61,291] map
to this 3D model (x right, y down, z away from camera, so a frontal face
gives angles near 0): 1:(0,0,0) 152:(0,330,65) 33:(-225,-170,135) 263:
(225,-170,135) 61:(-150,150,125) 291:(150,150,125). Camera matrix f=w,
cx=w/2, cy=h/2, zero distortion, cv2.solvePnP with SOLVEPNP_ITERATIVE. R =
cv2.Rodrigues(rvec)[0]; pitch = atan2(R[2,1], R[2,2]); yaw = atan2(-R[2,0],
sqrt(R[2,1]**2+R[2,2]**2)); roll = atan2(R[1,0], R[0,0]); convert to degrees.
gaze.gaze_off(pts, w, h) -> float: right eye r1 = (x468 - x33)/(x133 - x33);
left eye r2 = (x473 - x362)/(x263 - x362); both are 0.5 when the iris is
centred. gaze_off = clip(mean(|r1-0.5|,|r2-0.5|)*2, 0, 1). Return 0.0 if a
denominator is below 1e-6.

### addressee/classifier.py
directed_score(yaw, pitch, gaze_off, cfg: AddresseeCfg) -> float: yc = yaw -
yaw_offset_deg; pc = pitch - pitch_offset_deg; d = max(|yc|/max_yaw_deg,
|pc|/max_pitch_deg); pose = 1.0 if d <= 1 else max(0.0, 2.0 - d); if
use_gaze: pose *= 1 - gaze_weight*min(1, gaze_off/gaze_max_off); any NaN
input gives 0.0.

### sync/scorer.py
sync_score(lip: np.ndarray, env: np.ndarray, fps: float, cfg: SyncCfg) ->
SyncResult. Both arrays have the same length N and are sampled on the same
uniform grid at fps.
1. If N < round(min_window_s*fps): SyncResult(0,0,0,valid=False,0,0).
2. k = round(fps); lip_d = lip - scipy.ndimage.uniform_filter1d(lip, k,
mode='nearest'); lip_std = std(lip_d). If lip_std < min_lip_std: return score
0.0, valid=True (static mouth).
3. Features: f_open = lip_d; f_vel =
uniform_filter1d(abs(gradient(lip_d))*fps, 3, mode='nearest'). Audio: e =
log(env + env_eps).
4. K = round(max_lag_s*fps). For each lag j in -K..K: if j >= 0 pair a =
f[:N-j] with b = e[j:], else a = f[-j:] with b = e[:N+j]. r = Pearson(a, b)
computed on that overlap (0.0 if either std < 1e-8). Positive lag means the
audio is later than the lip motion.
5. For each feature take the maximum r over lags and its lag. score = max(0,
r_open_best, r_vel_best); lag_s = lag of the winning feature / fps. Fill
r_open, r_vel with the per-feature maxima.
attribute(scores: dict[int,float]) -> tuple[int|None, float]: (id of the
highest score, highest minus second highest where second = 0.0 if only one
id); empty dict gives (None, 0.0).

### fusion/gate.py
class Gate: decide(summary: UtteranceSummary) -> GateDecision. RuleGate(cfg:
Config) follows section 6. LearnedGate is defined in Phase 8 and then frozen.

### pipeline.py and utterance.py
TrustSyncPipeline(cfg: Config, audio: AudioSource, gate: Gate|None=None,
tracker=None): process(frame_bgr: np.ndarray, t: float) -> FrameResult.
'tracker' is injectable (anything with process(frame_bgr)->list[FaceObs]);
default FaceTracker. Per call:
1. faces = tracker.process(frame). 2. env_now = audio.envelope_at([t])[0];
speech = vad.update(env_now), VAD dt = 1/video.fps.
3. For every face: append (t, mouth_open) to that track's buffer (keep
lip.buffer_s seconds); yaw,pitch,roll = estimate_pose; gaze = gaze_off if
use_gaze else 0; directed = directed_score; running sync = sync_score on the
last window_s (grid at video.fps, lip via np.interp, env via
audio.envelope_at); if the buffer covers less than min_window_s the running
sync is 0. Running sync is for display only.
4. Utterance state machine (UtteranceTracker): starts when speech becomes
True; while active the decision is LISTENING; ends when speech becomes False
or elapsed >= utterance.max_s. If duration < utterance.min_s: discard
(decision IDLE). Otherwise build UtteranceSummary: per track, sync =
sync_score over the grid [t_start - pad_s, t_end + pad_s] at video.fps (a
track whose buffer covers < 80 percent of that span gets sync 0 and lip_std
0); directed, quality, yaw, pitch = median over the utterance frames in which
the track was visible. gate.decide(summary) gives the final decision; that
frame has utterance_final=True. The next frame returns to IDLE.
5. fps = exponential moving average of 1/(time between process calls), alpha
0.1.

### io, overlay, synth
io/video_source.py: read() -> tuple[bool, np.ndarray|None, float] (ok, frame,
t). WebcamSource(cfg) (cv2.VideoCapture; t = time.monotonic() right after
read). FileVideoSource(video_path, frame_times_csv) (t from csv column t).
overlay/hud.py: draw_hud(frame_bgr, result: FrameResult, env_hist:
list[float]) -> np.ndarray (returns a copy).
synth.py (test helpers, no hardware): make_env(duration_s, env_hz, bursts:
list[tuple[float,float]], rng) -> np.ndarray; make_lip(env, env_hz, fps,
lag_s, gain, noise, rng) -> np.ndarray; make_static_lip(duration_s, fps,
noise, rng) -> np.ndarray; make_pts(yaw=0, pitch=0, roll=0, w=640, h=480,
mouth_gap_ratio=0.05, iris_ratio=0.5, center=(0.5,0.5), scale=1.0) ->
np.ndarray (478,3); class FakeFaceTracker(script) (script: list of
list[FaceObs], one entry per frame).

## 6. RuleGate decision order
1. no tracks: REJECT_NO_FACE.
2. max quality over tracks < min_quality: REJECT_LOW_QUALITY.
3. best = track with the highest sync. If best.sync < min_sync or
best.lip_std < sync.min_lip_std: REJECT_SPOOF.
4. two or more tracks and best.sync - second.sync < min_margin:
REJECT_AMBIGUOUS.
5. best.quality < min_quality: REJECT_LOW_QUALITY.
6. best.directed < min_directed: REJECT_NOT_ADDRESSED.
7. otherwise ACCEPT with speaker_id = best.track_id.
liveness = best.sync; attribution_conf = best.sync - second.sync (second = 0
if one track); addressee = best.directed. reason is a short human sentence
naming the numbers.

## 7. Session files and labels
meta.json keys: id, scenario, subjects (list of str), speaker_side
('left'|'right'|'center'|'none'), addressed (bool), expected
('ACCEPT'|'REJECT_SPOOF'|'REJECT_NOT_ADDRESSED'|'REJECT_ANY'|'NONE'),
lighting ('normal'|'dim'), occlusion (bool), spoof_device (null|'phone'),
audio_t0 (float, monotonic), video_fps (float), notes (str).
Scenarios and expected: live_single ACCEPT; replay_spoof REJECT_SPOOF;
multi_one_speaks ACCEPT (speaker_side set); not_addressed
REJECT_NOT_ADDRESSED; occlusion_lowlight REJECT_ANY; silence NONE (no
utterance may be produced).
frame_times.csv: header frame_idx,t. A track's side = left if bbox centre x <
0.4, right if > 0.6, else center.

## 8. HTTP API (Phase 9)
GET /api/health -> {ok, running, source}
GET /api/config -> config dict. POST /api/config {section,key,value} ->
updated dict (allowed only for sections gate and addressee, and
sync.min_lip_std).
GET /api/sessions -> [{id, scenario, expected}] from data/sessions.
POST /api/source {type:'webcam'|'session', id?} -> {ok}. POST /api/stop ->
{ok}.
GET /api/results -> reports/results.json, or 404.
GET /stream.mjpg -> multipart/x-mixed-replace, boundary=frame, raw camera
JPEG without overlay.
WS /ws -> first message {type:'hello', config, decisions}; then
{type:'frame', data: frame_result_to_dict} at server.ws_hz; on
utterance_final also {type:'utterance', data: same dict}; and {type:'status',
data:{running, source, error}} whenever the runtime starts, stops or fails
(source is 'webcam', 'session:<id>' or null; error is a string or null).

## 9. UI contract (Phases 10-11)
web/: Vite + React 18 + TypeScript + Tailwind 3.4 + lucide-react + recharts +
framer-motion, hand-written components (no shadcn CLI). web/src/types.ts
mirrors section 4. Mock mode is the URL query ?mock=1.
