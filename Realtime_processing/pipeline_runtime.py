from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from Realtime_processing.gui_preprocessing import (
    compute_band_powers_openbci,
    compute_gui_fft_amplitude,
    compute_head_intensity_std,
    compute_head_polarity_openbci,
    normalize_band_powers_per_channel,
    run_gui_filter_pipeline,
    smooth_fft_amplitude_openbci,
)


@dataclass
class DisplaySnapshot:
    filtered_processing: np.ndarray
    window_data: np.ndarray


@dataclass
class SpectralSnapshot:
    freqs: np.ndarray
    smoothed_fft_amplitude: np.ndarray
    head_intensity: np.ndarray
    head_polarity: np.ndarray
    head_ref_idx: int
    normalized_band_powers: np.ndarray


def build_display_snapshot(
    raw_processing_buffer: np.ndarray,
    sample_rate: int,
    window_samples: int,
    low_cutoff_hz: float,
    high_cutoff_hz: float,
    notch_hz: float,
) -> DisplaySnapshot:
    """Compute filtered processing buffer and visible time-series tail."""
    filtered = run_gui_filter_pipeline(
        raw_processing_buffer,
        sample_rate,
        low_cutoff_hz,
        high_cutoff_hz,
        notch_hz,
    )
    return DisplaySnapshot(
        filtered_processing=filtered,
        window_data=filtered[:, -window_samples:],
    )


def build_spectral_snapshot(
    filtered_processing: np.ndarray,
    sample_rate: int,
    nfft: int,
    previous_smoothed_fft: np.ndarray | None,
    fft_smoothing_alpha: float,
    fft_min_amplitude_uv: float,
    head_window_seconds: float,
    band_edges: Sequence[Tuple[float, float]],
    band_power_eps: float,
) -> SpectralSnapshot:
    """Compute FFT/head/band outputs from a single filtered processing buffer."""
    freqs, amplitude = compute_gui_fft_amplitude(filtered_processing, sample_rate, nfft)
    smoothed_fft = smooth_fft_amplitude_openbci(
        amplitude,
        previous_smoothed_fft,
        fft_smoothing_alpha,
        fft_min_amplitude_uv,
    )
    intensity = compute_head_intensity_std(
        filtered_processing,
        sample_rate,
        head_window_seconds,
    )
    polarity, ref_idx = compute_head_polarity_openbci(
        filtered_processing,
        sample_rate,
        head_window_seconds,
    )
    band_powers = compute_band_powers_openbci(
        smoothed_fft,
        freqs,
        nfft,
        sample_rate,
        band_edges,
    )
    normalized = normalize_band_powers_per_channel(band_powers, band_power_eps)
    return SpectralSnapshot(
        freqs=freqs,
        smoothed_fft_amplitude=smoothed_fft,
        head_intensity=intensity,
        head_polarity=polarity,
        head_ref_idx=ref_idx,
        normalized_band_powers=normalized,
    )
