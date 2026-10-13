"""Unit tests for UtteranceTracker state machine."""
from __future__ import annotations

from trustsync.config import Config
from trustsync.utterance import UtteranceTracker


def test_utterance_kept_run(spec_cfg: Config) -> None:
    """A 1.0 s speech run is kept and t_end is last speech frame minus trim_s."""
    trim_s = 0.25
    tracker = UtteranceTracker(spec_cfg.utterance, trim_s=trim_s)
    obs = {0: {"directed": 0.9, "quality": 1.0, "yaw": 0.0, "pitch": 0.0}}

    dt = 1.0 / 30.0
    t = 100.0

    # 1.0 s of speech (30 frames)
    for _ in range(30):
        res = tracker.update(t, True, obs)
        assert res is None
        assert tracker.active is True
        t += dt

    # Speech ends
    event = tracker.update(t, False, obs)
    assert event is not None
    assert event.kept is True
    # t_end equals last speech frame minus trim_s
    last_speech_t = t - dt
    assert abs(event.t_end - (last_speech_t - trim_s)) < 1e-6
    assert abs(event.t_start - 100.0) < 1e-6
    assert tracker.active is False


def test_utterance_short_run_discarded(spec_cfg: Config) -> None:
    """A 0.2 s run is not kept (< min_s = 0.40 s)."""
    trim_s = 0.25
    tracker = UtteranceTracker(spec_cfg.utterance, trim_s=trim_s)
    obs = {0: {"directed": 0.9, "quality": 1.0, "yaw": 0.0, "pitch": 0.0}}

    dt = 1.0 / 30.0
    t = 100.0

    # 6 frames = 0.2 s
    for _ in range(6):
        tracker.update(t, True, obs)
        t += dt

    event = tracker.update(t, False, obs)
    assert event is not None
    assert event.kept is False


def test_utterance_forced_end_at_max_s(spec_cfg: Config) -> None:
    """Run longer than max_s is force-ended, and second event ignores speech until False once."""
    trim_s = 0.25
    tracker = UtteranceTracker(spec_cfg.utterance, trim_s=trim_s)
    obs = {0: {"directed": 0.9, "quality": 1.0, "yaw": 0.0, "pitch": 0.0}}

    dt = 1.0 / 30.0
    t = 100.0
    max_frames = round(spec_cfg.utterance.max_s / dt)

    event = None
    for _ in range(max_frames + 5):
        res = tracker.update(t, True, obs)
        if res is not None:
            event = res
            break
        t += dt

    assert event is not None
    assert event.kept is True
    assert tracker.active is False

    # While speech stays True, no new event starts
    t += dt
    res2 = tracker.update(t, True, obs)
    assert res2 is None
    assert tracker.active is False

    # Once speech goes False, tracker unlatches
    t += dt
    tracker.update(t, False, obs)

    # Next speech can start a new utterance
    t += dt
    tracker.update(t, True, obs)
    assert tracker.active is True


def test_utterance_two_separate_runs(spec_cfg: Config) -> None:
    """Two separate runs produce two distinct UtteranceEvent objects."""
    trim_s = 0.25
    tracker = UtteranceTracker(spec_cfg.utterance, trim_s=trim_s)
    obs = {0: {"directed": 0.9, "quality": 1.0, "yaw": 0.0, "pitch": 0.0}}

    dt = 1.0 / 30.0
    t = 100.0

    # Run 1: 1.0 s speech
    for _ in range(30):
        tracker.update(t, True, obs)
        t += dt
    ev1 = tracker.update(t, False, obs)
    assert ev1 is not None and ev1.kept is True

    # Silence for 1.0 s
    for _ in range(30):
        t += dt
        tracker.update(t, False, obs)

    # Run 2: 1.0 s speech
    for _ in range(30):
        tracker.update(t, True, obs)
        t += dt
    ev2 = tracker.update(t, False, obs)
    assert ev2 is not None and ev2.kept is True

    assert ev2.t_start > ev1.t_end
