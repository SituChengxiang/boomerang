#!/usr/bin/env python3
"""Comprehensive derivatives module for velocity and acceleration computation.

Provides multiple numerical differentiation methods for trajectory data:
1. np.gradient - Basic numerical differentiation
2. Savitzky-Golay - Smooth polynomial-based derivatives
3. Cubic spline - Interpolated derivatives with natural boundaries
4. Lagrange - Three-point Lagrange polynomial derivatives
5. Center difference - Manual central difference with boundary handling
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter


class Derivatives:
    """Container for velocity and acceleration results."""

    def __init__(
        self,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
    ) -> None:
        self.t = t
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.ax = ax
        self.ay = ay
        self.az = az

    @property
    def speed(self) -> np.ndarray:
        """Calculate instantaneous speed sqrt(vx² + vy² + vz²)."""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    def to_dict(self) -> dict:
        """Convert to dictionary for easy export."""
        return {
            "t": self.t,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
            "speed": self.speed,
        }


def gradient_method(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, edge_order: int = 2
) -> Derivatives:
    """Compute derivatives using numpy's gradient function.

    Args:
        t: Time array
        x, y, z: Position arrays
        edge_order: Order of accuracy at boundaries (1 or 2)

    Returns:
        Derivatives object with velocities and accelerations
    """
    # Validate edge_order
    if edge_order not in (1, 2):
        edge_order = 2

    # First derivatives (velocity)
    vx = np.gradient(x, t, edge_order=edge_order)
    vy = np.gradient(y, t, edge_order=edge_order)
    vz = np.gradient(z, t, edge_order=edge_order)

    # Second derivatives (acceleration)
    ax = np.gradient(vx, t, edge_order=edge_order)
    ay = np.gradient(vy, t, edge_order=edge_order)
    az = np.gradient(vz, t, edge_order=edge_order)

    return Derivatives(t, x, y, z, vx, vy, vz, ax, ay, az)


def savgol_method(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    window: Optional[int] = None,
    polyorder: int = 3,
) -> Derivatives:
    """Compute derivatives using Savitzky-Golay filter.

    Args:
        t: Time array
        x, y, z: Position arrays
        window: Window size (auto-determined if None)
        polyorder: Polynomial order for fitting

    Returns:
        Derivatives object with velocities and accelerations
        :param y:
        :param t:
        :param x:
    """
    n = len(t)
    if n < 3:
        # Fallback to gradient for insufficient data
        return gradient_method(t, x, y, z, edge_order=2 if n >= 3 else 1)

    # Calculate time step (assume roughly uniform)
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        dt = 1e-6

    # Determine window size
    if window is None:
        window = min(11, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = max(5, n if n % 2 else n - 1)
    window = min(window, n)
    if window % 2 == 0:
        window -= 1

    # Adjust polynomial orders
    poly_order_vel = min(2, window - 1)  # For 1st derivative
    poly_order_acc = min(3, window - 1)  # For 2nd derivative

    try:
        # Calculate velocities
        vx = savgol_filter(x, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vy = savgol_filter(y, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vz = savgol_filter(z, window, poly_order_vel, deriv=1, delta=dt, mode="interp")

        # Calculate accelerations
        ax = savgol_filter(x, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        ay = savgol_filter(y, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        az = savgol_filter(z, window, poly_order_acc, deriv=2, delta=dt, mode="interp")

        return Derivatives(t, x, y, z, vx, vy, vz, ax, ay, az)

    except Exception:
        # Fallback to gradient if Savitzky-Golay fails
        return gradient_method(t, x, y, z, edge_order=2 if n >= 3 else 1)


def cubic_spline_method(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Derivatives:
    """Compute derivatives using cubic spline interpolation.

    Uses natural boundary conditions for smooth interpolation.

    Args:
        t: Time array
        x, y, z: Position arrays

    Returns:
        Derivatives object with velocities and accelerations
    """
    n = len(t)
    if n < 4:
        # Need at least 4 points for cubic spline
        # Fallback to center difference method
        return center_difference_method(t, x, y, z)

    try:
        # Create cubic splines with natural boundary conditions
        sx = CubicSpline(t, x, bc_type="natural")
        sy = CubicSpline(t, y, bc_type="natural")
        sz = CubicSpline(t, z, bc_type="natural")

        # Compute derivatives
        vx = sx.derivative(1)(t)  # First derivative
        vy = sy.derivative(1)(t)
        vz = sz.derivative(1)(t)

        ax = sx.derivative(2)(t)  # Second derivative
        ay = sy.derivative(2)(t)
        az = sz.derivative(2)(t)

        return Derivatives(t, x, y, z, vx, vy, vz, ax, ay, az)

    except Exception:
        # Fallback to gradient if spline fails
        return gradient_method(t, x, y, z, edge_order=2 if n >= 3 else 1)


def lagrange_method(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Derivatives:
    """Compute derivatives using three-point Lagrange polynomial interpolation.

    Fits a quadratic through each set of 3 consecutive points and evaluates
    derivatives at the center point.

    Args:
        t: Time array
        x, y, z: Position arrays

    Returns:
        Derivatives object with velocities and accelerations
    """
    n = len(t)
    if n < 3:
        # Fallback to gradient for insufficient data
        return gradient_method(t, x, y, z, edge_order=1 if n < 3 else 2)

    # Initialize arrays
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)
    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)
    az = np.zeros(n, dtype=float)

    # Internal points (3-point Lagrange)
    for i in range(1, n - 1):
        t0, t1, t2 = t[i - 1], t[i], t[i + 1]
        dt01, dt02, dt12 = t0 - t1, t0 - t2, t1 - t2
        dt10, dt20, dt21 = t1 - t0, t2 - t0, t2 - t1

        # Precompute common denominator for numeric stability
        d0 = dt01 * dt02
        d1 = dt10 * dt12
        d2 = dt20 * dt21

        # X component
        x0, x1, x2 = x[i - 1], x[i], x[i + 1]
        vx[i] = (x0 * (t1 - t2) + x1 * (t2 - t0) + x2 * (t0 - t1)) / (d0 + d1 + d2)
        ax[i] = 2 * (x0 * (t1 - t2) + x1 * (t2 - t0) + x2 * (t0 - t1)) / (d0 * d1 * d2)

        # Y component
        y0, y1, y2 = y[i - 1], y[i], y[i + 1]
        vy[i] = (y0 * (t1 - t2) + y1 * (t2 - t0) + y2 * (t0 - t1)) / (d0 + d1 + d2)
        ay[i] = 2 * (y0 * (t1 - t2) + y1 * (t2 - t0) + y2 * (t0 - t1)) / (d0 * d1 * d2)

        # Z component
        z0, z1, z2 = z[i - 1], z[i], z[i + 1]
        vz[i] = (z0 * (t1 - t2) + z1 * (t2 - t0) + z2 * (t0 - t1)) / (d0 + d1 + d2)
        az[i] = 2 * (z0 * (t1 - t2) + z1 * (t2 - t0) + z2 * (t0 - t1)) / (d0 * d1 * d2)

    # Boundary points (simple finite differences)
    dt_forward = t[1] - t[0]
    dt_backward = t[-1] - t[-2]

    # Forward/backward difference for first derivatives
    vx[0] = (x[1] - x[0]) / dt_forward
    vy[0] = (y[1] - y[0]) / dt_forward
    vz[0] = (z[1] - z[0]) / dt_forward

    vx[-1] = (x[-1] - x[-2]) / dt_backward
    vy[-1] = (y[-1] - y[-2]) / dt_backward
    vz[-1] = (z[-1] - z[-2]) / dt_backward

    # Second derivative boundaries (if enough points)
    if n >= 3:
        ax[0] = (x[2] - 2 * x[1] + x[0]) / (dt_forward**2)
        ay[0] = (y[2] - 2 * y[1] + y[0]) / (dt_forward**2)
        az[0] = (z[2] - 2 * z[1] + z[0]) / (dt_forward**2)

        ax[-1] = (x[-1] - 2 * x[-2] + x[-3]) / (dt_backward**2)
        ay[-1] = (y[-1] - 2 * y[-2] + y[-3]) / (dt_backward**2)
        az[-1] = (z[-1] - 2 * z[-2] + z[-3]) / (dt_backward**2)

    return Derivatives(t, x, y, z, vx, vy, vz, ax, ay, az)


def center_difference_method(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Derivatives:
    """Compute derivatives using central difference with manual boundary handling.

    Uses central difference for internal points and forward/backward difference
    for boundary points.

    Args:
        t: Time array
        x, y, z: Position arrays

    Returns:
        Derivatives object with velocities and accelerations
    """
    n = len(t)
    if n < 2:
        # Return zeros for insufficient data
        return Derivatives(
            t,
            x,
            y,
            z,
            np.zeros_like(x),
            np.zeros_like(y),
            np.zeros_like(z),
            np.zeros_like(x),
            np.zeros_like(y),
            np.zeros_like(z),
        )

    # Initialize arrays
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)
    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)
    az = np.zeros(n, dtype=float)

    # Time differences
    dt = np.diff(t)
    dt = np.where(dt <= 1e-12, 1e-12, dt)  # Avoid division by zero

    # First derivatives (velocity)
    if n >= 3:
        # Center points
        vx[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
        vy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
        vz[1:-1] = (z[2:] - z[:-2]) / (t[2:] - t[:-2])

    # Boundary points
    vx[0] = (x[1] - x[0]) / dt[0]
    vy[0] = (y[1] - y[0]) / dt[0]
    vz[0] = (z[1] - z[0]) / dt[0]

    if n >= 2:
        vx[-1] = (x[-1] - x[-2]) / dt[-1]
        vy[-1] = (y[-1] - y[-2]) / dt[-1]
        vz[-1] = (z[-1] - z[-2]) / dt[-1]

    # Second derivatives (acceleration)
    if n >= 3:
        # Center points
        ax[1:-1] = (vx[2:] - vx[:-2]) / (t[2:] - t[:-2])
        ay[1:-1] = (vy[2:] - vy[:-2]) / (t[2:] - t[:-2])
        az[1:-1] = (vz[2:] - vz[:-2]) / (t[2:] - t[:-2])

    # Boundary points
    if n >= 2:
        ax[0] = (vx[1] - vx[0]) / dt[0]
        ay[0] = (vy[1] - vy[0]) / dt[0]
        az[0] = (vz[1] - vz[0]) / dt[0]

        if n >= 3:
            ax[-1] = (vx[-1] - vx[-2]) / dt[-1]
            ay[-1] = (vy[-1] - vy[-2]) / dt[-1]
            az[-1] = (vz[-1] - vz[-2]) / dt[-1]

    return Derivatives(t, x, y, z, vx, vy, vz, ax, ay, az)


def compute_derivatives(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    method: str = "auto",
    **kwargs,
) -> Derivatives:
    """Main function to compute velocity and acceleration using various methods.

    Args:
        t: Time array (must be strictly increasing)
        x, y, z: Position arrays
        method: One of:
            - "auto": Choose the best method based on data
            - "gradient": numpy.gradient (fast, robust)
            - "savgol": Savitzky-Golay (smooth, for uniform dt)
            - "spline": Cubic spline (smooth, for curve fitting)
            - "lagrange": Lagrange polynomial (3-point fitting)
            - "center": Manual center difference (robust)
        **kwargs: Method-specific parameters
            - window: For savgol (int, default 11)
            - polyorder: For savgol (int, default 3)
            - edge_order: For gradient (1 or 2, default 2)

    Returns:
        Derivatives object with all computed fields
    """
    n = len(t)

    # Auto-select method
    if method == "auto":
        if n < 4:
            method = "gradient"
        else:
            # Check if time is roughly uniform
            dt = np.diff(t)
            med = float(np.median(dt))
            if med > 0:
                rel_spread = float(np.max(np.abs(dt - med)) / med)
                if rel_spread > 0.01:
                    # Non-uniform time -> gradient or center difference
                    method = "gradient"
                else:
                    # Uniform time -> can use savgol or spline
                    method = "savgol"
            else:
                method = "gradient"

    # Execute chosen method
    if method == "gradient":
        edge_order = kwargs.get("edge_order", 2)
        return gradient_method(t, x, y, z, edge_order=edge_order)

    elif method == "savgol":
        window = kwargs.get("window", None)
        polyorder = kwargs.get("polyorder", 3)
        return savgol_method(t, x, y, z, window=window, polyorder=polyorder)

    elif method == "spline":
        return cubic_spline_method(t, x, y, z)

    elif method == "lagrange":
        return lagrange_method(t, x, y, z)

    elif method == "center":
        return center_difference_method(t, x, y, z)

    else:
        raise ValueError(f"Unknown method: {method}")


def get_derivatives_config() -> dict:
    """Return default configuration for derivatives computation."""
    return {
        "gradient": {"edge_order": 2},
        "savgol": {"window": 11, "polyorder": 3},
        "spline": {},
        "lagrange": {},
        "center": {},
        "auto": {},
    }


def compare_methods(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    methods: tuple = ("gradient", "savgol", "spline", "lagrange", "center"),
) -> dict:
    """Compute derivatives using multiple methods and compare results.

    Args:
        t, x, y, z: Trajectory data
        methods: Tuple of method names to compare

    Returns:
        Dictionary mapping method names to Derivatives objects
    """
    results = {}
    for method in methods:
        try:
            results[method] = compute_derivatives(t, x, y, z, method=method)
        except Exception as e:
            results[method] = None
            print(f"Warning: {method} failed: {e}")
    return results


# ==========================================
# Utility Functions
# ==========================================


def ensure_strictly_increasing_time(
    t: np.ndarray,
    *series: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, ...]:
    """Ensure time is finite, sorted, and strictly increasing.

    Tracker exports sometimes contain duplicate timestamps; numerical
    differentiation may emit warnings in that case. We drop non-increasing
    timestamps while keeping the first occurrence.

    Args:
        t: Time array
        *series: Additional arrays to keep in sync
        eps: Minimum time difference between consecutive points

    Returns:
        Cleaned time and series arrays
    """
    t = np.asarray(t, dtype=float)
    if any(len(s) != len(t) for s in series):
        raise ValueError("t and series must have the same length")

    # Filter finite values
    finite_mask = np.isfinite(t)
    for s in series:
        finite_mask &= np.isfinite(s)

    t = t[finite_mask]
    cleaned = [np.asarray(s, dtype=float)[finite_mask] for s in series]

    if len(t) == 0:
        return t, *cleaned

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    cleaned = [s[order] for s in cleaned]

    # Keep only strictly increasing times
    keep_idx = [0]
    last_t = float(t[0])
    for i in range(1, len(t)):
        ti = float(t[i])
        if ti > last_t + eps:
            keep_idx.append(i)
            last_t = ti

    keep = np.asarray(keep_idx, dtype=int)
    t_out = t[keep]
    series_out = [s[keep] for s in cleaned]
    return t_out, *series_out
