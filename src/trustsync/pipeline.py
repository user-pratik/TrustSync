"""TrustSync end-to-end multi-modal perception pipeline."""
from __future__ import annotations

import collections
import logging
import time
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

from trustsync.audio.envelope import VAD
from trustsync.audio.sources import AudioSource
from trustsync.config import Config
from trustsync.fusion.gate import Gate, RuleGate
from trustsync.sync.scorer import attribute, sync_score
from trustsync.types import FaceObs, FaceScores, FrameResult, SyncResult, TrackSummary, UtteranceSummary
from trustsync.utterance import UtteranceTracker
from trustsync.addressee import classifier as addressee_cls
from trustsync.vision.face_tracker import FaceTracker
from trustsync.vision.gaze import gaze_off
from trustsync.vision.head_pose import estimate_pose


def _window_sync(
    buffer: deque[tuple[float, float]],
    t_now: float,
    audio: AudioSource,
    cfg: Config,
) -> SyncResult:
    """Compute running sync score on the last window_s seconds."""
    if not buffer:
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    t0 = t_now - cfg.sync.window_s
    t1 = t_now
    n = round(cfg.sync.window_s * cfg.video.fps)
    if n < round(cfg.sync.min_window_s * cfg.video.fps):
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    # Require buffer to cover at least min_window_s
    buf_span = buffer[-1][0] - buffer[0][0]
    if buf_span < cfg.sync.min_window_s:
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    grid = t0 + np.arange(n, dtype=np.float64) / cfg.video.fps
    buf_arr = np.array(buffer, dtype=np.float64)
    buf_t = buf_arr[:, 0]
    buf_lip = buf_arr[:, 1]

    interp_lip = np.interp(grid, buf_t, buf_lip, left=buf_lip[0], right=buf_lip[-1])
    env = audio.envelope_at(grid)
    return sync_score(interp_lip, env, float(cfg.video.fps), cfg.sync)


def _span_sync(
    buffer: deque[tuple[float, float]],
    t0: float,
    t1: float,
    audio: AudioSource,
    cfg: Config,
) -> SyncResult:
    """Compute sync score over a span [t0, t1], checking 80% buffer coverage."""
    if not buffer:
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    dur = max(0.0, t1 - t0)
    n = round(dur * cfg.video.fps)
    if n < round(cfg.sync.min_window_s * cfg.video.fps):
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    grid = t0 + np.arange(n, dtype=np.float64) / cfg.video.fps

    # A track whose buffer covers < 80% of span gets sync 0 and lip_std 0
    t_min = buffer[0][0]
    t_max = buffer[-1][0]
    covered = np.count_nonzero((grid >= t_min) & (grid <= t_max))
    if (covered / n) < 0.80:
        return SyncResult(0.0, 0.0, 0.0, False, 0.0, 0.0)

    buf_arr = np.array(buffer, dtype=np.float64)
    buf_t = buf_arr[:, 0]
    buf_lip = buf_arr[:, 1]

    interp_lip = np.interp(grid, buf_t, buf_lip, left=buf_lip[0], right=buf_lip[-1])
    env = audio.envelope_at(grid)
    return sync_score(interp_lip, env, float(cfg.video.fps), cfg.sync)


class TrustSyncPipeline:
    """End-to-end multi-modal pipeline coordinating vision, audio, sync, and gating."""

    def __init__(
        self,
        cfg: Config,
        audio: AudioSource,
        gate: Gate | None = None,
        tracker: FaceTracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.audio = audio
        self.gate: Gate = gate if gate is not None else RuleGate(cfg)
        self.tracker = tracker if tracker is not None else FaceTracker(cfg)

        self._vad = VAD(cfg.vad, dt=1.0 / cfg.video.fps)
        self._utterance_tracker = UtteranceTracker(
            cfg.utterance, trim_s=cfg.vad.hangover_s
        )
        self._buffers: dict[int, deque[tuple[float, float]]] = {}
        self._fps = float(cfg.video.fps)
        self._last_perf: float | None = None
        self._last_t: float | None = None
        self._frame_idx = 0

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameResult:
        """Process one video frame and timestamp through the pipeline.

        Args:
            frame_bgr: BGR video frame image array.
            t: Monotonic timestamp of the frame in seconds.

        Returns:
            FrameResult: Aggregated perception state and decision for current frame.
        """
        # Hardening input checks
        if (
            not isinstance(frame_bgr, np.ndarray)
            or frame_bgr.ndim != 3
            or frame_bgr.shape[2] != 3
            or frame_bgr.dtype != np.uint8
            or frame_bgr.shape[0] < 1
            or frame_bgr.shape[1] < 1
        ):
            raise ValueError("frame must be HxWx3 uint8")

        if not np.isfinite(t):
            raise ValueError("t must be finite")

        if self._last_t is not None and t < (self._last_t - 1e-6):
            logger.warning(
                "Monotonic clock went backwards: %f < %f. Resetting state.",
                t,
                self._last_t,
            )
            self._buffers.clear()
            self._utterance_tracker = UtteranceTracker(
                self.cfg.utterance, trim_s=self.cfg.vad.hangover_s
            )
            self._vad = VAD(self.cfg.vad, dt=1.0 / self.cfg.video.fps)

        self._last_t = t

        # 1. Update FPS
        now_perf = time.perf_counter()
        if self._last_perf is not None:
            dt_perf = now_perf - self._last_perf
            if dt_perf > 0.0:
                inst_fps = 1.0 / dt_perf
                self._fps = 0.9 * self._fps + 0.1 * inst_fps
        self._last_perf = now_perf

        h, w = frame_bgr.shape[:2]

        # 2. Vision: detect faces
        faces: list[FaceObs] = self.tracker.process(frame_bgr)

        # 3. Audio: sample envelope and update VAD
        env_now = float(self.audio.envelope_at(np.array([t], dtype=np.float64))[0])
        speech = self._vad.update(env_now)

        # 4. Per-face tracking, pose, gaze, running sync
        faces_scores: list[FaceScores] = []
        obs_map: dict[int, dict[str, float]] = {}

        for face in faces:
            if not all(np.isfinite(c) for c in face.bbox):
                logger.debug("Dropping face %d with non-finite bbox", face.track_id)
                continue
            if np.isnan(face.pts).any():
                logger.debug("Face %d contains NaN landmarks", face.track_id)

            tid = face.track_id
            if tid not in self._buffers:
                self._buffers[tid] = deque()
            buf = self._buffers[tid]
            buf.append((t, face.mouth_open))

            # Prune older than buffer_s
            cutoff = t - self.cfg.lip.buffer_s
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            yaw, pitch, roll = estimate_pose(face.pts, w, h)
            gaze = (
                gaze_off(face.pts, w, h)
                if self.cfg.addressee.use_gaze
                else 0.0
            )
            directed = addressee_cls.directed_score(
                yaw, pitch, gaze, self.cfg.addressee
            )

            sync_res = _window_sync(buf, t, self.audio, self.cfg)
            run_sync = sync_res.score if sync_res.valid else 0.0

            faces_scores.append(
                FaceScores(
                    track_id=tid,
                    bbox=face.bbox,
                    sync=run_sync,
                    lip_std=sync_res.lip_std if sync_res.valid else 0.0,
                    best_lag_s=sync_res.lag_s if sync_res.valid else 0.0,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    gaze_off=gaze,
                    directed=directed,
                    quality=face.quality,
                )
            )

            obs_map[tid] = {
                "directed": directed,
                "quality": face.quality,
                "yaw": yaw,
                "pitch": pitch,
            }

        # Delete dead buffers
        dead_tids = [
            tid
            for tid, buf in self._buffers.items()
            if buf and buf[-1][0] < (t - self.cfg.lip.buffer_s)
        ]
        for tid in dead_tids:
            del self._buffers[tid]

        # Running speaker attribution
        running_sync_dict = {fs.track_id: fs.sync for fs in faces_scores}
        spk_id, attr_conf = attribute(running_sync_dict)

        # 5. Utterance state machine update
        event = self._utterance_tracker.update(t, speech, obs_map)

        decision = "IDLE"
        reason = ""
        speaker_id: int | None = None
        liveness = 0.0
        attribution_conf = 0.0
        addressee = 0.0
        utterance_final = False

        if event is not None:
            if not event.kept:
                # Discarded short utterance -> returns IDLE
                decision = "IDLE"
                reason = ""
                utterance_final = False
            else:
                # Kept utterance -> finalize GateDecision
                t_start_pad = event.t_start - self.cfg.utterance.pad_s
                t_end_pad = event.t_end + self.cfg.utterance.pad_s

                track_summaries: list[TrackSummary] = []
                for tid, obs_dict in event.per_track.items():
                    buf = self._buffers.get(tid, deque())
                    s_res = _span_sync(buf, t_start_pad, t_end_pad, self.audio, self.cfg)

                    d_med = float(np.median(obs_dict["directed"])) if obs_dict["directed"] else 0.0
                    q_med = float(np.median(obs_dict["quality"])) if obs_dict["quality"] else 0.0
                    y_med = float(np.median(obs_dict["yaw"])) if obs_dict["yaw"] else 0.0
                    p_med = float(np.median(obs_dict["pitch"])) if obs_dict["pitch"] else 0.0

                    track_summaries.append(
                        TrackSummary(
                            track_id=tid,
                            sync=s_res.score if s_res.valid else 0.0,
                            lip_std=s_res.lip_std if s_res.valid else 0.0,
                            directed=d_med,
                            quality=q_med,
                            yaw=y_med,
                            pitch=p_med,
                        )
                    )

                summary = UtteranceSummary(
                    t_start=event.t_start,
                    t_end=event.t_end,
                    duration_s=event.t_end - event.t_start,
                    tracks=track_summaries,
                )

                gate_decision = self.gate.decide(summary)
                decision = gate_decision.decision
                reason = gate_decision.reason
                speaker_id = gate_decision.speaker_id
                liveness = gate_decision.liveness
                attribution_conf = gate_decision.attribution_conf
                addressee = gate_decision.addressee
                utterance_final = True
        elif self._utterance_tracker.active:
            decision = "LISTENING"
            reason = "listening"
            speaker_id = spk_id
            liveness = max((fs.sync for fs in faces_scores), default=0.0)
            attribution_conf = attr_conf
            if spk_id is not None:
                spk_fs = next((fs for fs in faces_scores if fs.track_id == spk_id), None)
                addressee = spk_fs.directed if spk_fs is not None else 0.0
            else:
                addressee = 0.0

        result = FrameResult(
            t=t,
            frame_idx=self._frame_idx,
            fps=float(self._fps),
            faces=faces_scores,
            speech_active=speech,
            env=env_now,
            noise_floor=float(self._vad.noise_floor),
            speaker_id=speaker_id,
            attribution_conf=attribution_conf,
            liveness=liveness,
            addressee=addressee,
            decision=decision,
            reason=reason,
            utterance_final=utterance_final,
        )

        self._frame_idx += 1
        return result
