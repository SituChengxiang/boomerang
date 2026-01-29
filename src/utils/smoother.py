#!/usr/bin/env python3
"""Simple smoother wrapper for RTT Kalman and Savitzky-Golay smoothing.

Provides smoothing functionality for trajectory data:
- RTS Kalman smoother for state estimation and trajectory refinement
- Savitzky-Golay smoothing for polynomial-based trajectory smoothing
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter


class RTSKalmanSmoother:
    """RTS Kalman smoother for trajectory smoothing with constant-velocity model.

    Combines forward Kalman filtering and backward Rauch-Tung-Striebel smoothing.
    """

    def __init__(
        self, process_noise: float = 1e-4, measurement_noise: float = 1e-2
    ) -> None:
        """Initialize RTS Kalman smoother.

        Args:
            process_noise: Process variance (model uncertainty)
            measurement_noise: Measurement variance (sensor uncertainty)
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def _create_kf(self, dt: float) -> PyKalmanFilter:
        """Create pykalman KalmanFilter for given time step.

        Args:
            dt: Time step between measurements

        Returns:
            Configured pykalman.KalmanFilter
        """
        # Constant-velocity model: state = [position, velocity]
        transition_matrix = np.array([[1.0, dt], [0.0, 1.0]])
        observation_matrix = np.array([[1.0, 0.0]])

        # Covariances
        Q = np.array([[self.process_noise, 0.0], [0.0, self.process_noise]])
        R = np.array([[self.measurement_noise]])

        # Initial state
        initial_state_mean = np.array([0.0, 0.0])
        initial_state_covariance = np.eye(2) * 10.0

        return PyKalmanFilter(
            transition_matrices=transition_matrix,
            observation_matrices=observation_matrix,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
        )

    def smooth(self, times: np.ndarray, measurements: np.ndarray) -> np.ndarray:
        """Apply RTS smoother to 1D measurements.

        Args:
            times: Time array
            measurements: Measurement array

        Returns:
            Smoothed states (position, velocity)
        """
        n = len(measurements)
        if n < 2:
            return measurements.copy()

        dt = float(np.median(np.diff(times)))
        if dt <= 0:
            dt = 1e-6

        kf = self._create_kf(dt)
        measurements_2d = measurements.reshape(-1, 1)
        smoothed_state_means, _ = kf.smooth(measurements_2d)

        return smoothed_state_means

    def smooth_3d(
        self, t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply RTS smoother to 3D trajectory.

        Args:
            t: Time array
            x, y, z: Position arrays

        Returns:
            Tuple of (vx, vy, vz, ax, ay, az) - velocities and accelerations
        """
        state_x = self.smooth(t, x)
        state_y = self.smooth(t, y)
        state_z = self.smooth(t, z)

        vx = state_x[:, 1]
        vy = state_y[:, 1]
        vz = state_z[:, 1]

        edge_order = 2 if len(t) >= 3 else 1
        ax = np.gradient(vx, t, edge_order=edge_order)
        ay = np.gradient(vy, t, edge_order=edge_order)
        az = np.gradient(vz, t, edge_order=edge_order)

        return vx, vy, vz, ax, ay, az


def smooth_savgol_1d(
    signal: np.ndarray, t: np.ndarray, window: Optional[int] = None, polyorder: int = 3
) -> np.ndarray:
    """Smooth 1D signal using Savitzky-Golay filter.

    Args:
        signal: Input signal
        t: Time array
        window: Window size (auto-determined if None)
        polyorder: Polynomial order

    Returns:
        Smoothed signal
    """
    from scipy.signal import savgol_filter

    n = len(signal)
    if n < 3:
        return signal.copy()

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        dt = 1e-6

    if window is None:
        window = min(11, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = max(5, n if n % 2 else n - 1)
    window = min(window, n)

    polyorder = min(polyorder, window - 1)

    return savgol_filter(signal, window, polyorder, deriv=0, delta=dt, mode="interp")


def smooth_savgol_3d(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    window: Optional[int] = None,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth 3D positions using Savitzky-Golay filter.

    Args:
        t: Time array
        x, y, z: Position arrays
        window: Window size
        polyorder: Polynomial order

    Returns:
        Tuple of (x_smooth, y_smooth, z_smooth)
    """
    x_s = smooth_savgol_1d(x, t, window, polyorder)
    y_s = smooth_savgol_1d(y, t, window, polyorder)
    z_s = smooth_savgol_1d(z, t, window, polyorder)
    return x_s, y_s, z_s


def get_smoother_config() -> dict:
    """Return default smoother configuration."""
    return {
        "kalman_process_noise": 1e-4,
        "kalman_measurement_noise": 1e-2,
        "savgol_window": 11,
        "savgol_polyorder": 3,
    }
