from __future__ import annotations
import numpy as np
from scipy.signal import welch, find_peaks

def psd_welch(y: np.ndarray, fs: float, nperseg: int | None = None):
    """
    Welch PSD for a 1D series y sampled at rate fs (samples per unit time).
    Returns (f, Pxx). f is nonnegative freq axis.
    """
    y = np.asarray(y, dtype=float)
    f, Pxx = welch(y, fs=fs, nperseg=nperseg)
    return f, Pxx


def dominant_peaks(f: np.ndarray, Pxx: np.ndarray, k: int = 3, prominence: float = 0.0):
    """
    Return the top-k peaks by height with optional prominence filter.
    """
    idx, props = find_peaks(Pxx, prominence=prominence)
    if idx.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(Pxx[idx])[::-1]
    idx = idx[order][:k]
    return f[idx], Pxx[idx]