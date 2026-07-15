from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Realtime_processing.preprocessing import bandpass_filter, common_average_reference, notch_filter


@dataclass(frozen=True)
class DecoderPreprocessConfig:
    enabled: bool = True
    bandpass_low_hz: float = 4.0
    bandpass_high_hz: float = 40.0
    apply_notch: bool = True
    notch_hz: float = 60.0
    apply_car: bool = False
    zscore_per_channel: bool = True
    zscore_eps: float = 1e-6
    channel_order: tuple[int, ...] = ()


def run_decoder_preprocessing(
    window: np.ndarray,
    sample_rate_hz: int,
    config: DecoderPreprocessConfig,
) -> np.ndarray:
    """Apply decoder-side preprocessing to one (channels, samples) window."""
    matrix = np.asarray(window, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("decoder window must be 2D")
    if not config.enabled:
        return matrix

    out = matrix.copy()
    if config.channel_order:
        order = np.asarray(config.channel_order, dtype=np.int64)
        if order.size != out.shape[0]:
            raise ValueError("channel_order length does not match channel count")
        out = out[order, :]

    out = bandpass_filter(out, config.bandpass_low_hz, config.bandpass_high_hz, sample_rate_hz)
    if config.apply_notch:
        out = notch_filter(out, config.notch_hz, sample_rate_hz)
    if config.apply_car:
        out = common_average_reference(out)
    if config.zscore_per_channel:
        mu = np.mean(out, axis=1, keepdims=True)
        sigma = np.std(out, axis=1, keepdims=True)
        out = (out - mu) / np.maximum(sigma, config.zscore_eps)
    return out
