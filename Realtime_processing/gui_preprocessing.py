from __future__ import annotations

from typing import Tuple

import numpy as np

from Realtime_processing.preprocessing import bandpass_filter, notch_filter


def run_gui_filter_pipeline(
    data: np.ndarray,
    sample_rate: int,
    low_cutoff: float,
    high_cutoff: float,
    notch_hz: float,
) -> np.ndarray:
    """Run GUI display filter chain: bandpass -> notch (CAR disabled)."""
    step1 = bandpass_filter(data, low_cutoff, high_cutoff, sample_rate)
    return notch_filter(step1, notch_hz, sample_rate)


def compute_gui_fft_amplitude(data: np.ndarray, sample_rate: int, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT amplitude (uV) with mean removal and Hamming window."""
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.shape[1] < nfft:
        raise ValueError("Not enough samples for requested nfft.")
    windowed = matrix[:, -nfft:].copy()
    windowed -= np.mean(windowed, axis=1, keepdims=True)
    windowed *= np.hamming(nfft)
    spectrum = np.fft.rfft(windowed, axis=1)
    amplitude = np.abs(spectrum) / nfft
    if amplitude.shape[1] > 2:
        amplitude[:, 1:-1] *= 2.0
    freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sample_rate))
    return freqs, amplitude


def smooth_fft_amplitude_openbci(
    current: np.ndarray,
    previous: np.ndarray | None,
    alpha: float,
    min_amp: float,
) -> np.ndarray:
    """Smooth FFT amplitude using OpenBCI geometric blend in log-power space."""
    cur = np.maximum(np.asarray(current, dtype=np.float64), min_amp)
    if previous is None:
        return cur
    prev = np.maximum(np.asarray(previous, dtype=np.float64), min_amp)
    log_power = (1.0 - alpha) * np.log(cur**2) + alpha * np.log(prev**2)
    return np.sqrt(np.exp(log_power))
