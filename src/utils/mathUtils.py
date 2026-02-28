#!/usr/bin/env python3
"""Pure math/numerical utilities (no business logic).

Only numpy and scipy are allowed here.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


def safe_unit_vector(vec: np.ndarray, fallback: np.ndarray = None) -> np.ndarray:  # type: ignore
    """Compute unit vector with fallback for zero/near-zero vectors.

    Args:
        vec: Input vector (1D array)
        fallback: Fallback unit vector if input is zero/near-zero

    Returns:
        Unit vector, or fallback if input is invalid
    """
    vec = np.asarray(vec, dtype=float)
    if vec.ndim != 1:
        raise ValueError("Input must be a 1D vector")

    nrm = float(np.linalg.norm(vec))
    if not np.isfinite(nrm) or nrm < 1e-12:
        if fallback is None:
            fallback = np.zeros_like(vec)
            fallback[0] = 1.0  # Default to x-axis
        return fallback
    return vec / nrm


def savgol_window(n: int, desired: int = 21, poly: int = 3) -> int | None:
    """Pick a valid Savitzky-Golay window length for n samples."""
    if n <= 0:
        return None
    w = min(desired, n)
    if w % 2 == 0:
        w -= 1
    if w < poly + 2:
        return None
    return w


def derivatives_smooth(
    t: np.ndarray, y: np.ndarray, poly: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dy/dt, d2y/dt2) using Savitzky-Golay when possible.

    Falls back to numpy.gradient for short or irregular sequences.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size != y.size or t.size < 3:
        dy = np.gradient(y)
        d2y = np.gradient(dy)
        return dy, d2y

    dt = np.diff(t)
    dt_med = float(np.median(dt)) if dt.size else 0.0
    if (not np.isfinite(dt_med)) or dt_med <= 0:
        dy = np.gradient(y, t)
        d2y = np.gradient(dy, t)
        return dy, d2y

    window = savgol_window(int(t.size), desired=21, poly=poly)
    if window is None:
        dy = np.gradient(y, t)
        d2y = np.gradient(dy, t)
        return dy, d2y

    dy = savgol_filter(y, window, poly, deriv=1, delta=dt_med)
    d2y = savgol_filter(y, window, poly, deriv=2, delta=dt_med)
    return dy, d2y


def unwrap_angle_rad(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    return np.unwrap(theta)


def standardize_time_grid(
    t: np.ndarray, x: np.ndarray, dt: float = 0.01666667
) -> Tuple[np.ndarray, np.ndarray]:
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


def linear_fit_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float] | None:
    """Fit y = kx + b and return parameter CIs.

    Args:
        x: Independent variable samples
        y: Dependent variable samples
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Dict with slope/intercept, standard errors, CI bounds, R², n, dof;
        returns None if there are not enough valid samples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = int(x.size)
    if n < 3:
        return None

    design = np.column_stack((x, np.ones_like(x)))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    k = float(beta[0])
    b = float(beta[1])

    y_hat = k * x + b
    resid = y - y_hat
    dof = n - 2
    if dof <= 0:
        return None

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    s2 = ss_res / dof
    cov = s2 * np.linalg.inv(design.T @ design)
    se_k = float(np.sqrt(cov[0, 0]))
    se_b = float(np.sqrt(cov[1, 1]))

    try:
        from scipy.stats import t as student_t  # lazy import

        t_crit = float(student_t.ppf(1.0 - alpha / 2.0, dof))
    except Exception:
        t_crit = 1.96

    k_lo, k_hi = k - t_crit * se_k, k + t_crit * se_k
    b_lo, b_hi = b - t_crit * se_b, b + t_crit * se_b

    return {
        "n": float(n),
        "dof": float(dof),
        "k": k,
        "b": b,
        "se_k": se_k,
        "se_b": se_b,
        "k_lo": float(k_lo),
        "k_hi": float(k_hi),
        "b_lo": float(b_lo),
        "b_hi": float(b_hi),
        "r2": r2,
    }


def local_initial_derivative(
    t: np.ndarray, x: np.ndarray, n_points: int = 7, poly_deg: int = 2
) -> float:
    """Estimate the initial derivative (v at t=0) using a local polynomial fit.

    This avoids the boundary effects often seen with global splines (e.g. natural BC zeroing out curvature).
    For the start of a throw, a local quadratic fit is usually more physically realistic than a spline end-knot condition.

    Args:
        t: Time-series array (1D). Should be sorted.
        x: Value array (1D).
        n_points: Number of initial points to use for the fit (default=7).
        poly_deg: Degree of polynomial (2=quadratic => constant accel assumption).

    Returns:
        v0: Estimated derivative dx/dt at the first time point.
    """
    if len(t) < n_points:
        n_points = len(t)
    if n_points < poly_deg + 1:
        # Fallback to simple finite diff if not enough points
        if len(t) >= 2:
            return (x[1] - x[0]) / (t[1] - t[0])
        return 0.0

    # Extract the local window
    t_local = t[:n_points]
    x_local = x[:n_points]

    # Shift time for numerical stability (fit relative to t[0])
    dt_local = t_local - t_local[0]

    # Fit polynomial: x(dt) = c0 + c1*dt + c2*dt^2 ...
    # deriv at dt=0 is just c1
    coeffs = np.polyfit(dt_local, x_local, poly_deg)

    # polyfit returns [c_deg, ..., c1, c0] (highest power first)
    # The coefficient of the linear term (power 1) is at index -(2) => coeffs[-2]
    # For a degree 2 fit: [a, b, c] -> ax^2 + bx + c. derivative is 2ax + b. at x=0, derivative is b.

    return coeffs[-2]
