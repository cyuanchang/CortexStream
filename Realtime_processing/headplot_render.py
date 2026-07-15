from __future__ import annotations

import numpy as np


def _log10_intensity_bounds(intense_min_uv: float, intense_max_uv: float) -> tuple[float, float]:
    return float(np.log10(intense_min_uv)), float(np.log10(intense_max_uv))


def quantize_rgb(rgb: np.ndarray, levels: int = 12) -> np.ndarray:
    """OpenBCI quantizeColor: posterize RGB to contour bands."""
    ticks_per_color = 256 // (levels + 1)
    if ticks_per_color <= 0:
        return rgb
    out = np.asarray(rgb, dtype=np.float64)
    out = np.floor(out / ticks_per_color) * ticks_per_color
    return np.clip(out, 0.0, 255.0)


def _map_log_intensity(
    intensity_uv: np.ndarray,
    intense_min_uv: float,
    intense_max_uv: float,
) -> np.ndarray:
    log_min, log_max = _log10_intensity_bounds(intense_min_uv, intense_max_uv)
    clipped = np.clip(intensity_uv, intense_min_uv, intense_max_uv)
    log_vals = np.log10(clipped)
    mapped = (log_vals - log_min) / max(log_max - log_min, 1e-12)
    return np.clip(mapped, 0.0, 1.0)


def pixel_voltage_to_rgb(
    pixel_volt_uv: np.ndarray,
    intense_min_uv: float,
    intense_max_uv: float,
    contour_levels: int = 12,
) -> np.ndarray:
    """OpenBCI calcPixelColor: red (+) / blue (-), white fade, log scale, quantize."""
    volt = np.asarray(pixel_volt_uv, dtype=np.float64)
    is_negative = volt < 0.0
    intensity = np.clip(np.abs(volt), intense_min_uv, intense_max_uv)
    mapped = _map_log_intensity(intensity, intense_min_uv, intense_max_uv)

    rgb = np.zeros((*volt.shape, 3), dtype=np.float64)
    rgb[..., 0] = 224.0
    rgb[..., 1] = 56.0
    rgb[..., 2] = 45.0
    neg_rgb = np.array([54.0, 87.0, 158.0], dtype=np.float64)
    rgb[is_negative] = neg_rgb

    fade = (1.0 - mapped)[..., np.newaxis]
    base = rgb / 255.0
    rgb = (base + (1.0 - base) * fade) * 255.0
    rgb = np.clip(rgb, 0.0, 255.0)
    rgb = quantize_rgb(rgb, levels=contour_levels)
    return rgb.astype(np.uint8)


def electrode_intensity_to_rgb(
    intensity_uv: np.ndarray,
    intense_min_uv: float,
    intense_max_uv: float,
) -> np.ndarray:
    """OpenBCI updateElectrodeColors: white-to-red by unsigned std (no polarity on disks)."""
    intensity = np.clip(np.asarray(intensity_uv, dtype=np.float64), intense_min_uv, intense_max_uv)
    mapped = _map_log_intensity(intensity, intense_min_uv, intense_max_uv)
    base = np.array([255.0, 0.0, 0.0], dtype=np.float64)
    fade = 1.0 - mapped
    rgb = np.zeros((intensity.size, 3), dtype=np.float64)
    for i in range(3):
        val = base[i] / 255.0
        rgb[:, i] = (val + (1.0 - val) * fade) * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def render_head_image(
    pixel_volt_grid: np.ndarray,
    mask: np.ndarray,
    intense_min_uv: float,
    intense_max_uv: float,
    contour_levels: int = 12,
) -> np.ndarray:
    """Build RGBA image (H, W, 4) for the scalp field; outside mask is transparent white."""
    grid = np.asarray(pixel_volt_grid, dtype=np.float64)
    head_mask = np.asarray(mask, dtype=bool)
    h, w = grid.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = 0

    if not np.any(head_mask):
        return rgba

    rgb = pixel_voltage_to_rgb(
        grid[head_mask],
        intense_min_uv,
        intense_max_uv,
        contour_levels=contour_levels,
    )
    rgba[head_mask, :3] = rgb
    rgba[head_mask, 3] = 255
    return rgba


def scalp_rgba_for_display(rgba: np.ndarray) -> np.ndarray:
    """Map grid RGBA [x_index, y_index] to pyqtgraph col-major ImageItem (x, y)."""
    display = np.flip(rgba, axis=1).copy()
    transparent = display[..., 3] == 0
    display[transparent, :3] = 0
    return np.ascontiguousarray(display)
