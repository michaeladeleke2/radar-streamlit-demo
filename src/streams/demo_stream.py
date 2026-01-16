# src/streams/demo_stream.py
import numpy as np


class DemoRadarStream:
    """
    Synthetic radar stream used for demo mode.
    Generates a 1D signal chunk that roughly imitates gesture energy patterns.
    """

    def __init__(self, fs_hz: int = 2000, seed: int = 123):
        self.fs_hz = int(fs_hz)
        self.rng = np.random.default_rng(seed)

    def read_chunk(self, n_samples: int, mode: str = "pushing") -> np.ndarray:
        n = int(n_samples)
        t = np.arange(n, dtype=np.float32) / float(self.fs_hz)

        mode = (mode or "").strip().lower()

        # base noise
        x = 0.05 * self.rng.standard_normal(n).astype(np.float32)

        if mode in ("rest", "resting", "idle"):
            return x

        if mode in ("push", "pushing"):
            # bursty energy, slightly increasing then decreasing
            env = np.exp(-((t - t.mean()) ** 2) / (2 * (0.12 ** 2))).astype(np.float32)
            carrier = np.sin(2 * np.pi * (30 + 10 * np.sin(2 * np.pi * 1.2 * t)) * t).astype(np.float32)
            x += 0.8 * env * carrier
            x += 0.15 * env * self.rng.standard_normal(n).astype(np.float32)
            return x

        if mode in ("up", "forward"):
            # ramp-like frequency rise
            f0, f1 = 10.0, 90.0
            k = (f1 - f0) / max(t[-1], 1e-6)
            phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
            env = np.clip(t / max(t[-1], 1e-6), 0, 1).astype(np.float32)
            x += 0.7 * env * np.sin(phase).astype(np.float32)
            return x

        if mode in ("right", "swipe_right"):
            # oscillatory mid-band with two pulses
            env1 = np.exp(-((t - 0.35 * t[-1]) ** 2) / (2 * (0.08 ** 2))).astype(np.float32)
            env2 = np.exp(-((t - 0.70 * t[-1]) ** 2) / (2 * (0.08 ** 2))).astype(np.float32)
            env = env1 + env2
            carrier = np.sin(2 * np.pi * 55.0 * t).astype(np.float32)
            x += 0.65 * env * carrier
            return x

        # fallback
        return x