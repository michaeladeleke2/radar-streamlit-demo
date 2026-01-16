import numpy as np
from matplotlib import cm

EPS = 1e-8


def _stft_custom(vec: np.ndarray, window: int, nfft: int, shift: int) -> np.ndarray:
    """
    Matches the custom STFT in your processing_utils.py (Hanning + FFT per hop).
    Returns shape: (nfft, n_time)
    """
    vec = np.asarray(vec).reshape(-1)
    n = (len(vec) - window - 1) // shift
    if n <= 0:
        raise ValueError("Signal chunk too small for STFT settings. Increase chunk_size or reduce window/noverlap.")

    out = np.zeros((nfft, n), dtype=np.complex64)
    win = np.hanning(window).astype(np.float32)

    for i in range(n):
        seg = vec[i * shift : i * shift + window].astype(np.float32)
        seg = seg * win
        out[:, i] = np.fft.fft(seg, n=nfft)

    return out


def compute_microdoppler_spectrogram_like_physical(
    signal_1d: np.ndarray,
    prf_hz: float,
    window: int = 256,
    noverlap: int = 200,
    nfft: int = 2**10,
) -> np.ndarray:
    """
    Reproduces the 'look' from processing_utils.spectrogram():
    - STFT
    - fftshift on frequency axis
    - returns dB image (freq x time) in [-20, 0] clipped range (relative to max)
    """
    shift = window - noverlap
    Z = _stft_custom(signal_1d, window=window, nfft=nfft, shift=shift)
    S = np.abs(np.fft.fftshift(Z, axes=0)).astype(np.float32)

    maxval = float(np.max(S) + EPS)
    db = 20.0 * np.log10((S / maxval) + EPS)

    # Match the Normalize(vmin=-20, clip=True) behavior
    db = np.clip(db, -20.0, 0.0)
    return db


def db_to_jet_rgb_uint8(db_img: np.ndarray, vmin_db: float = -20.0, vmax_db: float = 0.0) -> np.ndarray:
    """
    Convert dB image to an RGB uint8 image using jet colormap (like your saved PNGs).
    """
    x = (db_img - vmin_db) / (vmax_db - vmin_db + 1e-9)
    x = np.clip(x, 0.0, 1.0)

    cmap = cm.get_cmap("jet")
    rgba = cmap(x)  # float RGBA [0..1]
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb