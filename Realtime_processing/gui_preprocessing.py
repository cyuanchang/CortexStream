from __future__ import annotations

from typing import Sequence, Tuple

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


def _head_intensity_tail(
    filtered: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> np.ndarray:
    """Return the last window_seconds of filtered EEG per channel."""
    matrix = np.asarray(filtered, dtype=np.float64)
    window_samples = max(int(round(float(sample_rate) * window_seconds)), 1)
    if matrix.shape[1] < window_samples:
        raise ValueError("Not enough samples for requested intensity window.")
    return matrix[:, -window_samples:]


def compute_head_intensity_std(
    filtered: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> np.ndarray:
    """Compute per-channel std (uV) over the last window_seconds of filtered EEG."""
    tail = _head_intensity_tail(filtered, sample_rate, window_seconds)
    return np.std(tail, axis=1)


def compute_head_polarity_openbci(
    filtered: np.ndarray,
    sample_rate: int,
    window_seconds: float,
) -> tuple[np.ndarray, int]:
    """Compute +/-1 polarity per channel relative to the strongest channel (OpenBCI)."""
    tail = _head_intensity_tail(filtered, sample_rate, window_seconds)
    std_per_ch = np.std(tail, axis=1)
    ref_idx = int(np.argmax(std_per_ch))
    ref_trace = tail[ref_idx]
    polarity = np.ones(tail.shape[0], dtype=np.float64)
    for ch in range(tail.shape[0]):
        dot_prod = float(np.dot(tail[ch], ref_trace))
        polarity[ch] = 1.0 if dot_prod >= 0.0 else -1.0
    return polarity, ref_idx


def compute_band_powers_openbci(
    amplitude_uv: np.ndarray,
    freqs: np.ndarray,
    nfft: int,
    sample_rate: int,
    band_edges: Sequence[Tuple[float, float]],
) -> np.ndarray:
    """Sum single-sided PSD from smoothed FFT amplitude per OpenBCI DataProcessing.pde."""
    amplitude = np.asarray(amplitude_uv, dtype=np.float64)
    freq_axis = np.asarray(freqs, dtype=np.float64)
    n_channels = amplitude.shape[0]
    n_bands = len(band_edges)
    powers = np.zeros((n_channels, n_bands), dtype=np.float64)
    nyquist_idx = nfft // 2

    for band_idx, (band_low, band_high) in enumerate(band_edges):
        mask = (freq_axis >= band_low) & (freq_axis < band_high)
        if not np.any(mask):
            continue
        bin_indices = np.where(mask)[0]
        for ch in range(n_channels):
            band_sum = 0.0
            for bin_idx in bin_indices:
                mag = amplitude[ch, bin_idx]
                if bin_idx != 0 and bin_idx != nyquist_idx:
                    psd = (mag * mag) * nfft / float(sample_rate) / 4.0
                else:
                    psd = (mag * mag) * nfft / float(sample_rate)
                band_sum += psd
            powers[ch, band_idx] = band_sum
    return powers


def normalize_band_powers_per_channel(powers: np.ndarray, eps: float) -> np.ndarray:
    """Normalize each channel so its band powers sum to 1."""
    matrix = np.asarray(powers, dtype=np.float64)
    totals = np.maximum(np.sum(matrix, axis=1, keepdims=True), eps)
    return matrix / totals
