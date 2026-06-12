from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from Realtime_processing.montage import ChannelPosition


@dataclass
class IDWHeadmapModel:
    grid_size: int
    mask: np.ndarray
    weights: np.ndarray

    def interpolate(self, channel_values: np.ndarray) -> np.ndarray:
        """Interpolate channel values to dense scalp grid."""
        values = np.asarray(channel_values, dtype=np.float64)
        valid_values = values[: self.weights.shape[1]]
        interpolated = self.weights @ valid_values
        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        heatmap[self.mask] = interpolated
        return heatmap


def build_idw_headmap_model(
    channel_positions: List[ChannelPosition],
    grid_size: int,
    power: float,
    eps: float,
) -> IDWHeadmapModel:
    """Precompute IDW interpolation weights for all scalp pixels."""
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis)
    mask = (grid_x**2 + grid_y**2) <= 1.0

    points = np.column_stack((grid_x[mask], grid_y[mask]))
    elec_xy = np.array([(pos.x, pos.y) for pos in channel_positions], dtype=np.float64)

    diff = points[:, None, :] - elec_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, eps)
    weights = 1.0 / np.power(dist, power)
    weights /= np.sum(weights, axis=1, keepdims=True)

    return IDWHeadmapModel(grid_size=grid_size, mask=mask, weights=weights)
