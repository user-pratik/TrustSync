"""TrustSync audio meter hardware diagnostic tool."""
import time

from trustsync.audio.envelope import VAD
from trustsync.audio.sources import LiveAudioSource
from trustsync.config import load_config


def main() -> None:
    """Run real-time audio volume and VAD meter for 15 seconds."""
    cfg = load_config()
    source = LiveAudioSource(cfg.audio)
    source.start()
    dt = 1.0 / cfg.audio.env_hz
    vad = VAD(cfg.vad, dt=dt)

    t_end = time.monotonic() + 15.0
    try:
        while time.monotonic() < t_end:
            _t, env = source.latest()
            is_speech = vad.update(env)

            bar_len = 10
            fill = min(bar_len, max(0, int(round((env / 0.05) * bar_len))))
            bar = f"[{'#' * fill}{' ' * (bar_len - fill)}]"
            state = "SPEECH" if is_speech else "silence"
            print(f"{bar} {env:.3f} {state}")
            time.sleep(0.1)
    finally:
        source.close()


if __name__ == "__main__":
    main()
