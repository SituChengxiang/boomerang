#!/usr/bin/env python3
"""Simple filter wrapper for Savitzky-Golay, gradient, and Kalman filtering.

Provides basic filtering functionality for trajectory data:
- Savitzky-Golay filter for smoothing and derivatives
- Gradient-based derivative calculation
- Kalman filter (forward filter only, no smoothing)
- Window handling utilities

"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pykalman import KalmanFilter as PyKalmanFilter
from scipy.signal import savgol_filter


def ensure_odd_window(window: int, n: int) -> int:
    """Ensure window size is odd and within bounds.

    Args:
        window: Desired window size
        n: Number of data points

    Returns:
        Valid odd window size
    """
    w = int(window)
    if w < 3:
        w = 3
    if w > n:
        w = n
    if w % 2 == 0:
        w -= 1
    if w < 3:
        w = max(3, n if n % 2 else n - 1)
    return w


def check_uniform_time(t: np.ndarray, tol: float = 1e-3) -> bool:
    """Check if time array is roughly uniformly sampled.

    Args:
        t: Time array
        tol: Relative tolerance for uniformity check

    Returns:
        True if time is roughly uniform, False otherwise
    """
    if len(t) < 3:
        return True
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0:
        return False
    max_rel_diff = float(np.max(np.abs(dt - med)) / med)
    return max_rel_diff <= tol


def gradient_filter(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate velocity using numpy gradient.

    Args:
        t: Time array
        x, y, z: Position arrays

    Returns:
        Tuple of (vx, vy, vz)
    """
    edge_order = 2 if len(t) >= 3 else 1
    vx = np.gradient(x, t, edge_order=edge_order)
    vy = np.gradient(y, t, edge_order=edge_order)
    vz = np.gradient(z, t, edge_order=edge_order)
    return vx, vy, vz


def savgol_filter_1d(
    signal: np.ndarray,
    t: np.ndarray,
    window: Optional[int] = None,
    polyorder: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to 1D signal.

    Args:
        signal: Input signal array
        t: Time array (for calculating delta)
        window: Window size (auto-determined if None)
        polyorder: Polynomial order
        deriv: Derivative order (0=smooth, 1=first, 2=second)

    Returns:
        Filtered signal
    """
    n = len(signal)
    if n < 3:
        return signal.copy()

    # Calculate time step
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        dt = 1e-6

    # Determine window size
    if window is None:
        window = min(11, n)
    window = ensure_odd_window(window, n)

    # Adjust polynomial order if needed
    polyorder = min(polyorder, window - 1)

    return savgol_filter(
        signal, window, polyorder, deriv=deriv, delta=dt, mode="interp"
    )


def savgol_filter_3d(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    window: Optional[int] = None,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate smoothed velocities and accelerations using Savitzky-Golay filter.

    Args:
        t: Time array
        x, y, z: Position arrays
        window: Window size (auto-determined if None)
        polyorder: Polynomial order for derivatives

    Returns:
        Tuple of (vx, vy, vz, ax, ay, az) - velocities and accelerations
    """
    n = len(t)
    if n < 3:
        # Return zeros for insufficient data
        return (
            np.zeros_like(x),
            np.zeros_like(y),
            np.zeros_like(z),
            np.zeros_like(x),
            np.zeros_like(y),
            np.zeros_like(z),
        )

    # Calculate time step
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
    poly_order_vel = min(2, window - 1)  # 1st derivative
    poly_order_acc = (
        min(3, window - 1) if polyorder > 2 else min(2, window - 1)
    )  # 2nd derivative

    try:
        # Calculate velocities
        vx = savgol_filter(x, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vy = savgol_filter(y, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vz = savgol_filter(z, window, poly_order_vel, deriv=1, delta=dt, mode="interp")

        # Calculate accelerations
        ax = savgol_filter(x, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        ay = savgol_filter(y, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        az = savgol_filter(z, window, poly_order_acc, deriv=2, delta=dt, mode="interp")

        return vx, vy, vz, ax, ay, az

    except Exception:
        # Fallback to gradient if Savitzky-Golay fails
        edge_order = 2 if n >= 3 else 1
        vx = np.gradient(x, t, edge_order=edge_order)
        vy = np.gradient(y, t, edge_order=edge_order)
        vz = np.gradient(z, t, edge_order=edge_order)
        ax = np.gradient(vx, t, edge_order=edge_order)
        ay = np.gradient(vy, t, edge_order=edge_order)
        az = np.gradient(vz, t, edge_order=edge_order)
        return vx, vy, vz, ax, ay, az


def smooth_signal(
    signal: np.ndarray, t: np.ndarray, window: Optional[int] = None, polyorder: int = 3
) -> np.ndarray:
    """Simplify smooth 1D signal using Savitzky-Golay or gradient.

    Args:
        signal: Input signal
        t: Time array
        window: Window size for Savitzky-Golay
        polyorder: Polynomial order

    Returns:
        Smoothed signal
    """
    n = len(signal)
    if n < 3:
        return signal.copy()

    if check_uniform_time(t) and n >= 5:
        return savgol_filter_1d(signal, t, window, polyorder, deriv=0)
    else:
        # For non-uniform or small datasets, use simple mean smoothing
        return np.convolve(signal, np.ones(min(3, n)) / min(3, n), mode="same")


def get_filter_config() -> dict:
    """Return default filter configuration for easy customization."""
    return {
        "savgol_window": 11,
        "savgol_polyorder": 3,
        "uniform_time_tol": 1e-3,
        "min_points": 3,
        "kalman_process_noise": 1e-4,
        "kalman_measurement_noise": 1e-2,
    }


class KalmanFilter:
    """Kalman filter using pykalman library (forward filter only).

    Provides state estimation for trajectory data using forward Kalman filter.
    Uses constant-velocity model: state = [position, velocity].
    For RTS smoothing, use the RTSKalmanSmoother in smoother.py.

    Note: This filter implementation intentionally does not include smoothing.
    For RTS smoothing functionality, the repository provides RTSKalmanSmoother
    in the smoother.py module, separating filtering from smoothing as they
    serve different purposes (real-time estimation vs. offline optimization).
    """

    def __init__(
        self, process_noise: float = 1e-4, measurement_noise: float = 1e-2
    ) -> None:
        """Initialize Kalman filter.

        Args:
            process_noise: Process variance (model uncertainty)
            measurement_noise: Measurement variance (sensor uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def _create_kf(self, dt: float) -> PyKalmanFilter:
        """Create a pykalman KalmanFilter for given time step.

        Args:
            dt: Time step between measurements

        Returns:
            Configured pykalman.KalmanFilter
        """
        # Constant-velocity model: state = [position, velocity]
        # State transition: x_k = x_{k-1} + v * dt, v_k = v_{k-1}
        transition_matrix = np.array([[1.0, dt], [0.0, 1.0]])
        observation_matrix = np.array([[1.0, 0.0]])  # Only observe position

        # Process noise covariance
        Q = np.array([[self.process_noise, 0.0], [0.0, self.process_noise]])

        # Measurement noise covariance
        R = np.array([[self.measurement_noise]])

        # Initial state and covariance
        initial_state_mean = np.array([0.0, 0.0])
        initial_state_covariance = np.eye(2) * 10.0  # Large initial uncertainty

        return PyKalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )

    def filter(
        self, times: np.ndarray, measurements: np.ndarray
    ) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Apply forward Kalman filter only.

        Args:
            times: Time array
            measurements: Measurement array

        Returns:
            Tuple of (filtered_state_means, filtered_state_covariances)
            Returns (measurements.copy(), None) if n < 2.
        """
        n = len(measurements)
        if n < 2:
            return measurements.copy(), None

        # Use average dt for simplicity
        dt = float(np.median(np.diff(times)))
        if dt <= 0:
            dt = 1e-6

        kf = self._create_kf(dt)

        # Apply filter
        measurements_2d = measurements.reshape(-1, 1)
        filtered_state_means, filtered_state_covariances = kf.filter(measurements_2d)

        return filtered_state_means, filtered_state_covariances

    def filter_3d(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply Kalman filtering to 3D trajectory.

        Args:
            t: Time array
            x, y, z: Position arrays

        Returns:
            Tuple of (vx, vy, vz, ax, ay, az) - velocities and accelerations
        """
        # Filter each dimension
        state_x, _ = self.filter(t, x)
        state_y, _ = self.filter(t, y)
        state_z, _ = self.filter(t, z)

        # Extract velocities (index 1 in state) and approximate accelerations
        vx = state_x[:, 1]
        vy = state_y[:, 1]
        vz = state_z[:, 1]

        # Calculate acceleration from velocity
        edge_order = 2 if len(t) >= 3 else 1
        ax = np.gradient(vx, t, edge_order=edge_order)
        ay = np.gradient(vy, t, edge_order=edge_order)
        az = np.gradient(vz, t, edge_order=edge_order)

        return vx, vy, vz, ax, ay, az
