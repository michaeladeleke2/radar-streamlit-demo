import numpy as np

def mti_filter(x: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """
    Simple MTI-like high-pass along time.
    x: [T, ...]
    """
    y = np.zeros_like(x, dtype=np.float32)
    bg = np.zeros_like(x[0], dtype=np.float32)
    for t in range(x.shape[0]):
        bg = alpha * bg + (1 - alpha) * x[t]
        y[t] = x[t] - bg
    return y

def stft_slow_time(x: np.ndarray, n_fft: int = 128, hop: int = 8, win: str = "hann") -> np.ndarray:
    """
    STFT over slow-time axis to create Doppler-vs-time.
    x: [T] (1D slow-time signal)
    returns: [F, TT] complex
    """
    T = x.shape[0]
    if win == "hann":
        w = np.hanning(n_fft).astype(np.float32)
    else:
        w = np.ones(n_fft, dtype=np.float32)

    frames = []
    for start in range(0, T - n_fft + 1, hop):
        seg = x[start:start+n_fft] * w
        X = np.fft.fft(seg, n=n_fft)
        frames.append(X)

    S = np.stack(frames, axis=1)  # [F, TT]
    S = np.fft.fftshift(S, axes=0)  # center zero Doppler
    return S

def to_db_and_normalize(S: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """
    Convert magnitude to dB, clamp, and normalize to [0,1] for stable rendering.
    """
    mag = np.abs(S).astype(np.float32)
    mag = np.maximum(mag, 1e-8)
    db = 20.0 * np.log10(mag)
    db = np.clip(db, floor_db, db.max())
    db = (db - floor_db) / (db.max() - floor_db + 1e-8)
    return db

def synth_slow_time_signal(
    T: int,
    mode: str,
    fs_slow: float = 20.0,
    seed: int | None = None
) -> np.ndarray:
    """
    Create a radar-like slow-time signal that produces micro-doppler-like bursts.
    - resting: mostly static/clutter + tiny noise
    - pushing: adds short bursts of Doppler energy
    """
    rng = np.random.default_rng(seed)

    t = np.arange(T) / fs_slow

    # strong static clutter component (creates center line after Doppler processing)
    clutter = 1.0 + 0.02 * rng.standard_normal(T)

    # low noise
    noise = 0.02 * rng.standard_normal(T)

    x = clutter + noise

    if mode == "pushing":
        # add 1–3 short “gesture bursts”
        n_bursts = rng.integers(1, 4)
        for _ in range(n_bursts):
            center = rng.integers(int(0.15*T), int(0.85*T))
            width = rng.integers(int(0.03*T), int(0.08*T))
            amp = rng.uniform(0.6, 1.2)

            # Doppler-like oscillation (a moving target produces a tone in slow-time)
            f0 = rng.uniform(1.5, 5.0)  # "doppler frequency" in Hz (slow-time)
            phase = rng.uniform(0, 2*np.pi)

            idx = np.arange(T)
            env = np.exp(-0.5 * ((idx - center) / (width + 1e-6))**2)

            burst = amp * env * np.cos(2*np.pi*f0*t + phase)

            # also add a second component to create richer “blobs”
            f1 = f0 * rng.uniform(1.3, 1.8)
            burst += 0.4 * amp * env * np.cos(2*np.pi*f1*t + phase*0.5)

            x += burst

    return x.astype(np.float32)

def demo_microdoppler_image(
    mode: str,
    seconds: float = 2.0,
    fs_slow: float = 20.0,
    n_fft: int = 128,
    hop: int = 8,
    mti_alpha: float = 0.98,
    seed: int | None = None
) -> np.ndarray:
    """
    Returns an image array [F, TT] normalized to [0,1] (ready for imshow).
    """
    T = int(seconds * fs_slow)
    x = synth_slow_time_signal(T=T, mode=mode, fs_slow=fs_slow, seed=seed)

    # MTI-like high-pass (removes background slowly-varying clutter)
    x_mti = mti_filter(x[:, None], alpha=mti_alpha)[:, 0]

    S = stft_slow_time(x_mti, n_fft=n_fft, hop=hop, win="hann")
    img = to_db_and_normalize(S, floor_db=-60.0)

    # Increase sparsity / contrast to look more like your real plots
    img = np.power(img, 1.6)  # gamma
    img[img < 0.08] = 0.0     # threshold low energy

    return img