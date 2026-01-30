#!/usr/bin/env python3
"""Pure math/numerical utilities (no business logic).

Only numpy and scipy are allowed here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def standardize_time_grid(t: np.ndarray, x: np.ndarray, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Map irregularly-sampled time series to a fixed-step grid using CubicSpline.

    Steps:
    1) Ensure time is strictly increasing
    2) Handle duplicate timestamps (merge by averaging values)
    3) Shift time so that t[0] = 0

    Args:
        t: Time array (1D)
        x: Value array aligned with t (1D)
        dt: Target time step for uniform grid

    Returns:
        t_grid: Uniform time grid starting at 0
        x_grid: Interpolated values on t_grid
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    if t.size != x.size:
        raise ValueError("t and x must have the same length")
    if t.size < 2:
        return t - (t[0] if t.size else 0.0), x.copy()
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")

    # Sort by time to enforce monotonicity
    order = np.argsort(t)
    t_sorted = t[order]
    x_sorted = x[order]

    # Merge duplicate timestamps by averaging their values
    unique_t, inverse = np.unique(t_sorted, return_inverse=True)
    if unique_t.size < 2:
        # All timestamps are identical; cannot build a spline
        return np.array([0.0]), np.array([float(np.mean(x_sorted))])

    sums = np.zeros_like(unique_t, dtype=float)
    counts = np.zeros_like(unique_t, dtype=float)
    np.add.at(sums, inverse, x_sorted)
    np.add.at(counts, inverse, 1.0)
    x_unique = sums / np.maximum(counts, 1.0)

    # Shift time so that it starts at 0
    t0 = unique_t[0]
    t_unique = unique_t - t0

    # Build uniform time grid
    t_end = float(t_unique[-1])
    n_steps = int(np.floor(t_end / dt)) + 1
    t_grid = np.arange(n_steps, dtype=float) * dt
    if t_grid[-1] < t_end:
        t_grid = np.append(t_grid, t_end)

    # Cubic spline interpolation
    spline = CubicSpline(t_unique, x_unique, bc_type="natural")
    x_grid = spline(t_grid)

    return t_grid, x_grid


def savgol_derivatives(
    x: np.ndarray,
    dt: float,
    window_rate: float = 0.08,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute first and second derivatives via Savitzky-Golay filtering.

    Window length is computed as a fraction of data length and forced to be odd.

    Args:
        x: Value array (1D)
        dt: Sampling interval (assumed uniform)
        window_rate: Fraction of data length used as window length
        polyorder: Polynomial order for SG filter

    Returns:
        dx: First derivative array
        ddx: Second derivative array
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 3:
        dx = np.gradient(x, dt) if x.size else x.copy()
        ddx = np.gradient(dx, dt) if x.size else x.copy()
        return dx, ddx

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number")
    if not np.isfinite(window_rate) or window_rate <= 0.0:
        raise ValueError("window_rate must be a positive finite number")

    n = x.size
    window = int(np.floor(n * window_rate))
    window = max(5, window)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        window = 3
    if window > n:
        window = n if n % 2 == 1 else n - 1

    polyorder = int(polyorder)
    if polyorder < 1:
        polyorder = 1
    if polyorder >= window:
        polyorder = window - 1

    dx = savgol_filter(x, window, polyorder, deriv=1, delta=dt, mode="interp")
    ddx = savgol_filter(x, window, polyorder, deriv=2, delta=dt, mode="interp")
    return dx, ddx


def spline_derivative_analytical(
    t: np.ndarray,
    x: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """Compute analytical derivatives using CubicSpline.

    Args:
        t: Time array (1D, strictly increasing)
        x: Value array aligned with t (1D)
        order: Derivative order (1 for velocity, 2 for acceleration)

    Returns:
        dx: Derivative array of given order
    """
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if t.size != x.size:
        raise ValueError("t and x must have the same length")
    if t.size < 2:
        return np.zeros_like(x)
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3")

    spline = CubicSpline(t, x, bc_type="natural")
    return spline.derivative(order)(t)


def magnitude(vec_array: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute vector magnitude along a given axis.

    Args:
        vec_array: Array of vectors, e.g. shape (..., 3)
        axis: Axis corresponding to vector components

    Returns:
        Magnitudes array
    """
    vec_array = np.asarray(vec_array, dtype=float)
    return np.linalg.norm(vec_array, axis=axis)

