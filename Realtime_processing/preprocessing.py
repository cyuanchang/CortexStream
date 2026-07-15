from __future__ import annotations

import numpy as np
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes


def bandpass_filter(data: np.ndarray, low_cutoff: float, high_cutoff: float, sample_rate: int) -> np.ndarray:
    """Apply channel-wise bandpass filtering to EEG matrix [n_channels, n_samples]."""
    filtered = np.asarray(data, dtype=np.float64).copy()
    nyquist = 0.5 * float(sample_rate)
    if not (0.0 < low_cutoff < high_cutoff < nyquist):
        raise ValueError("Invalid bandpass range. Require 0 < low_cutoff < high_cutoff < sample_rate/2.")
    for ch in range(filtered.shape[0]):
        DataFilter.perform_bandpass(
            filtered[ch],
            sample_rate,
            low_cutoff,
            high_cutoff,
            4,
            FilterTypes.BUTTERWORTH.value,
            0.0,
        )
    return filtered


def notch_filter(data: np.ndarray, notch_freq: float, sample_rate: int) -> np.ndarray:
    """Apply channel-wise notch filtering for line noise removal."""
    filtered = np.asarray(data, dtype=np.float64).copy()
    if int(round(notch_freq)) == 50:
        noise_type = NoiseTypes.FIFTY.value
    elif int(round(notch_freq)) == 60:
        noise_type = NoiseTypes.SIXTY.value
    else:
        noise_type = NoiseTypes.FIFTY_AND_SIXTY.value
    for ch in range(filtered.shape[0]):
        DataFilter.remove_environmental_noise(filtered[ch], sample_rate, noise_type)
    return filtered


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply common average reference across channels at each sample."""
    matrix = np.asarray(data, dtype=np.float64)
    return matrix - np.mean(matrix, axis=0, keepdims=True)


def compute_band_power(psd: np.ndarray, freqs: np.ndarray, band_low: float, band_high: float) -> np.ndarray:
    """Compute per-channel mean power inside requested frequency band."""
    mask = (freqs >= band_low) & (freqs <= band_high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.mean(psd[:, mask], axis=1)
