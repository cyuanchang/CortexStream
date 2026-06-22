from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from Realtime_processing.montage import ChannelPosition

_IDW_POWER = 2.0
_IDW_EPS = 1e-6
_WEIGHT_EPS = 1e-12


@dataclass
class OpenBCIHeadmapModel:
    grid_size: int
    mask: np.ndarray
    weights: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    electrode_xy: np.ndarray

    def interpolate_voltage(self, signed_elec_uv: np.ndarray) -> np.ndarray:
        """Diffuse signed electrode voltages to scalp grid (OpenBCI calcPixelVoltage without EMA)."""
        values = np.asarray(signed_elec_uv, dtype=np.float64)
        n_elec = min(values.size, self.weights.shape[0])
        if n_elec == 0:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        flat = self.weights[:n_elec].T @ values[:n_elec]
        grid = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float64)
        grid[self.mask] = flat
        return grid


def _bilinear_upscale(
    coarse: np.ndarray,
    n_wide_full: int,
    n_tall_full: int,
    decimation: int,
) -> np.ndarray:
    n_elec, n_wide_small, n_tall_small = coarse.shape
    fine = np.zeros((n_elec, n_wide_full, n_tall_full), dtype=np.float64)
    for ix in range(n_wide_full):
        ix_source = ix // decimation
        dx_frac = float(ix - ix_source * decimation) / float(decimation)
        for iy in range(n_tall_full):
            iy_source = iy // decimation
            dy_frac = float(iy - iy_source * decimation) / float(decimation)
            if ix_source < n_wide_small - 1 and iy_source < n_tall_small - 1:
                v00 = coarse[:, ix_source, iy_source]
                v10 = coarse[:, ix_source + 1, iy_source]
                v01 = coarse[:, ix_source, iy_source + 1]
                v11 = coarse[:, ix_source + 1, iy_source + 1]
                foo1 = (v10 - v00) * dx_frac + v00
                foo2 = (v11 - v01) * dx_frac + v01
                fine[:, ix, iy] = (foo2 - foo1) * dy_frac + foo1
            elif ix_source < n_wide_small - 1:
                v00 = coarse[:, ix_source, iy_source]
                v10 = coarse[:, ix_source + 1, iy_source]
                fine[:, ix, iy] = (v10 - v00) * dx_frac + v00
            elif iy_source < n_tall_small - 1:
                v00 = coarse[:, ix_source, iy_source]
                v01 = coarse[:, ix_source, iy_source + 1]
                fine[:, ix, iy] = (v01 - v00) * dy_frac + v00
            else:
                fine[:, ix, iy] = coarse[:, ix_source, iy_source]
    return fine


def _where_are_the_pixels(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    electrode_xy: np.ndarray,
    elec_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_wide, n_tall = grid_x.shape
    n_elec = electrode_xy.shape[0]
    within_head = (grid_x**2 + grid_y**2) <= 1.0
    within_electrode = np.full((n_wide, n_tall), -1, dtype=np.int32)

    for ix in range(n_wide):
        for iy in range(n_tall):
            px, py = grid_x[ix, iy], grid_y[ix, iy]
            best_elec = -1
            for ielec in range(n_elec):
                ex, ey = electrode_xy[ielec]
                dist = max(_IDW_EPS, float(np.hypot(px - ex, py - ey)))
                if dist < elec_radius:
                    best_elec = ielec
            within_electrode[ix, iy] = best_elec

    for ielec in range(n_elec):
        ex, ey = electrode_xy[ielec]
        dists = np.hypot(grid_x - ex, grid_y - ey)
        best_flat = int(np.argmin(dists))
        best_ix, best_iy = np.unravel_index(best_flat, dists.shape)
        within_electrode[best_ix, best_iy] = ielec

    return within_head, within_electrode


def _make_all_the_connections(
    within_head: np.ndarray,
    within_electrode: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_wide, n_tall = within_head.shape
    to_pixels = np.full((n_wide, n_tall, 4, 2), -1, dtype=np.int32)
    to_electrodes = np.full((n_wide, n_tall, 4), -1, dtype=np.int32)
    offsets = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for iy in range(n_tall):
        for ix in range(n_wide):
            for idir, (dx, dy) in enumerate(offsets):
                ix_try = ix + dx
                iy_try = iy + dy
                if 0 <= ix_try < n_wide and 0 <= iy_try < n_tall:
                    elec = within_electrode[ix_try, iy_try]
                    if elec >= 0:
                        to_electrodes[ix, iy, idir] = elec
                    elif within_head[ix_try, iy_try]:
                        to_pixels[ix, iy, idir, 0] = ix_try
                        to_pixels[ix, iy, idir, 1] = iy_try
    return to_pixels, to_electrodes


def _compute_weight_factors_one_electrode(
    to_pixels: np.ndarray,
    to_electrodes: np.ndarray,
    ielec: int,
    pixel_val: np.ndarray,
    lim_iter_count: int = 2000,
    dval_threshold: float = 1e-5,
    change_fac: float = 0.2,
) -> None:
    n_wide, n_tall = to_pixels.shape[0], to_pixels.shape[1]
    n_dir = 4
    prev_val = np.zeros((n_wide, n_tall), dtype=np.float64)
    max_dval = 10.0 * dval_threshold
    iter_count = 0

    while iter_count < lim_iter_count and max_dval > dval_threshold:
        iter_count += 1
        prev_val[:, :] = pixel_val[ielec]
        max_dval = 0.0

        for ix in range(n_wide):
            for iy in range(n_tall):
                total = 0.0
                any_connections = False
                for idir in range(n_dir):
                    tx = to_pixels[ix, iy, idir, 0]
                    ty = to_pixels[ix, iy, idir, 1]
                    if tx > -1:
                        total += prev_val[tx, ty] - prev_val[ix, iy]
                        any_connections = True
                    te = to_electrodes[ix, iy, idir]
                    if te > -1:
                        target = 1.0 if te == ielec else 0.0
                        total += target - prev_val[ix, iy]
                        any_connections = True
                if any_connections:
                    dval = change_fac * total
                    pixel_val[ielec, ix, iy] = prev_val[ix, iy] + dval
                    max_dval = max(max_dval, abs(dval))
                else:
                    pixel_val[ielec, ix, iy] = -1.0


def _get_closest_weight_fac(weight_fac: np.ndarray, ix: int, iy: int) -> float:
    n_wide, n_tall = weight_fac.shape
    step = 1
    while step < max(n_wide, n_tall):
        sum_val = 0.0
        n_sum = 0
        any_within = False
        for ix_test in range(ix - step, ix + step + 1):
            for iy_test in range(iy - step, iy + step + 1):
                if 0 <= ix_test < n_wide and 0 <= iy_test < n_tall:
                    any_within = True
                    if weight_fac[ix_test, iy_test] >= 0.0:
                        sum_val += weight_fac[ix_test, iy_test]
                        n_sum += 1
        if n_sum > 0:
            return sum_val / n_sum
        step += 1
        if not any_within:
            break
    return -1.0


def _clean_up_boundaries(
    within_head: np.ndarray,
    within_electrode: np.ndarray,
    weight_fac: np.ndarray,
) -> None:
    n_elec, n_wide, n_tall = weight_fac.shape
    for ix in range(n_wide):
        for iy in range(n_tall):
            if not within_head[ix, iy]:
                weight_fac[:, ix, iy] = -1.0
                continue
            for ielec in range(n_elec):
                if weight_fac[ielec, ix, iy] < 0.0:
                    weight_fac[ielec, ix, iy] = _get_closest_weight_fac(
                        weight_fac[ielec], ix, iy
                    )
            elec = within_electrode[ix, iy]
            if elec >= 0:
                weight_fac[:, ix, iy] = 0.0
                weight_fac[elec, ix, iy] = 1.0


def _compute_true_average_weights(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    electrode_xy: np.ndarray,
    elec_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_elec = electrode_xy.shape[0]
    within_head, within_electrode = _where_are_the_pixels(
        grid_x, grid_y, electrode_xy, elec_radius
    )
    to_pixels, to_electrodes = _make_all_the_connections(within_head, within_electrode)
    weight_fac = np.zeros((n_elec, *within_head.shape), dtype=np.float64)
    for ielec in range(n_elec):
        _compute_weight_factors_one_electrode(to_pixels, to_electrodes, ielec, weight_fac)
    _clean_up_boundaries(within_head, within_electrode, weight_fac)
    return within_head, weight_fac


def _idw_weights_at_points(
    points_xy: np.ndarray,
    electrode_xy: np.ndarray,
    power: float = _IDW_POWER,
    eps: float = _IDW_EPS,
) -> np.ndarray:
    """Return (n_points, n_elec) IDW weights, row-normalized."""
    diff = points_xy[:, None, :] - electrode_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist = np.maximum(dist, eps)
    raw = 1.0 / np.power(dist, power)
    totals = np.maximum(np.sum(raw, axis=1, keepdims=True), _WEIGHT_EPS)
    return raw / totals


def _build_scalp_weights(
    scalp_mask: np.ndarray,
    fine_weights: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    electrode_xy: np.ndarray,
) -> np.ndarray:
    """Build (n_elec, n_scalp_pixels) weights with diffusion + IDW fallback, normalized per pixel."""
    n_elec = fine_weights.shape[0]
    scalp_indices = np.argwhere(scalp_mask)
    n_scalp = scalp_indices.shape[0]
    points_xy = np.column_stack(
        (grid_x[scalp_mask], grid_y[scalp_mask]),
    )
    idw = _idw_weights_at_points(points_xy, electrode_xy).T

    out = np.zeros((n_elec, n_scalp), dtype=np.float64)
    for p_idx, (ix, iy) in enumerate(scalp_indices):
        diff_w = fine_weights[:, ix, iy].copy()
        diff_w = np.where(diff_w < 0.0, 0.0, diff_w)
        if np.sum(diff_w) >= _WEIGHT_EPS:
            out[:, p_idx] = diff_w
        else:
            out[:, p_idx] = idw[:, p_idx]

    col_sums = np.maximum(np.sum(out, axis=0, keepdims=True), _WEIGHT_EPS)
    return out / col_sums


def build_openbci_headmap_model(
    channel_positions: List[ChannelPosition],
    grid_size: int,
    electrode_rel_diam: float,
    decimation: int = 4,
) -> OpenBCIHeadmapModel:
    """Precompute OpenBCI-style diffusion interpolation weights on a scalp grid."""
    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="ij")
    electrode_xy = np.array([(pos.x, pos.y) for pos in channel_positions], dtype=np.float64)
    head_diameter = 2.0
    elec_radius = 0.5 * electrode_rel_diam * head_diameter

    n_wide_small = grid_size // decimation + 1
    axis_small = np.linspace(-1.0, 1.0, n_wide_small, dtype=np.float64)
    coarse_x, coarse_y = np.meshgrid(axis_small, axis_small, indexing="ij")

    _, coarse_weights = _compute_true_average_weights(
        coarse_x, coarse_y, electrode_xy, elec_radius
    )
    fine_weights = _bilinear_upscale(coarse_weights, grid_size, grid_size, decimation)

    scalp_mask, _ = _where_are_the_pixels(grid_x, grid_y, electrode_xy, elec_radius)
    within_electrode_dummy = np.full(scalp_mask.shape, -1, dtype=np.int32)
    _clean_up_boundaries(scalp_mask, within_electrode_dummy, fine_weights)

    weights = _build_scalp_weights(
        scalp_mask, fine_weights, grid_x, grid_y, electrode_xy
    )

    return OpenBCIHeadmapModel(
        grid_size=grid_size,
        mask=scalp_mask,
        weights=weights,
        grid_x=grid_x,
        grid_y=grid_y,
        electrode_xy=electrode_xy,
    )
