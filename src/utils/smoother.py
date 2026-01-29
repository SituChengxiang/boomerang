#!/usr/bin/env python3
"""Minimal trajectory smoothing utilities.

Pipeline:
1) Kalman filter (forward only)
2) Time normalization (uniform dt)
3) Startup local smoothing for the first 0.2s
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from pykalman import KalmanFilter as PyKalmanFilter
from scipy.signal import savgol_filter

# =========================
# Tunable parameters (edit here)
# =========================
KALMAN_PROCESS_NOISE = 0.001
KALMAN_MEASUREMENT_NOISE = 0.005

STARTUP_DURATION = 0.2  # seconds
STARTUP_WINDOW = 1     # points (odd)
STARTUP_POLYORDER =  2  # polynomial order


def normalize_time(t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Normalize time to start at 0 with uniform dt (median dt)."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return t - (t[0] if len(t) else 0.0), 0.0

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1e-6

    t_norm = np.arange(len(t), dtype=float) * dt
    return t_norm, dt


def _kalman_filter_1d(
    t: np.ndarray,
    measurements: np.ndarray,
    process_noise: float,
    measurement_noise: float,
) -> np.ndarray:
    """Forward-only Kalman filter. Returns position states."""
    n = len(measurements)
    if n < 2:
        return measurements.copy()

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1e-6

    transition_matrix = np.array([[1.0, dt], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    Q = np.array([[process_noise, 0.0], [0.0, process_noise]])
    R = np.array([[measurement_noise]])

    dt0 = float(t[1] - t[0]) if n >= 2 else dt
    if not np.isfinite(dt0) or dt0 <= 0.0:
        dt0 = dt

    init_pos = float(measurements[0])
    init_vel = float((measurements[1] - measurements[0]) / dt0) if n >= 2 else 0.0

    kf = PyKalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=np.array([init_pos, init_vel]),
        initial_state_covariance=np.eye(2),
    )

    filtered_state_means, _ = kf.filter(measurements.reshape(-1, 1))
    return filtered_state_means[:, 0]


def _startup_local_smooth(
    t: np.ndarray,
    values: np.ndarray,
    duration: float,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing only to the first `duration` seconds."""
    values = values.copy()
    if len(t) < 3:
        return values

    cutoff_idx = int(np.searchsorted(t, t[0] + duration))
    if cutoff_idx < 2:
        return values

    local_window = min(window, cutoff_idx * 2 - 1)
    local_window = max(3, local_window)
    if local_window % 2 == 0:
        local_window -= 1

    polyorder = min(polyorder, local_window - 1)
    values[:cutoff_idx] = savgol_filter(
        values[:cutoff_idx], local_window, polyorder, mode="interp"
    )
    return values


def smooth_trajectory(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    process_noise: float = KALMAN_PROCESS_NOISE,
    measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
    startup_duration: float = STARTUP_DURATION,
    startup_window: int = STARTUP_WINDOW,
    startup_polyorder: int = STARTUP_POLYORDER,
) -> Dict[str, np.ndarray]:
    """Kalman filter -> time normalization -> startup local smoothing.

    Returns dict with t, x, y, z, vx, vy, vz.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # 1) Kalman filter (forward-only)
    x_s = _kalman_filter_1d(t, x, process_noise, measurement_noise)
    y_s = _kalman_filter_1d(t, y, process_noise, measurement_noise)
    z_s = _kalman_filter_1d(t, z, process_noise, measurement_noise)

    # 2) Time normalization (after Kalman)
    t_norm, _ = normalize_time(t)

    # 3) Startup local smoothing (first 0.2s)
    x_s = _startup_local_smooth(t_norm, x_s, startup_duration, startup_window, startup_polyorder)
    y_s = _startup_local_smooth(t_norm, y_s, startup_duration, startup_window, startup_polyorder)
    z_s = _startup_local_smooth(t_norm, z_s, startup_duration, startup_window, startup_polyorder)

    # 4) Velocities on normalized time
    edge_order = 2 if len(t_norm) >= 3 else 1
    vx = np.gradient(x_s, t_norm, edge_order=edge_order)
    vy = np.gradient(y_s, t_norm, edge_order=edge_order)
    vz = np.gradient(z_s, t_norm, edge_order=edge_order)

    return {
        "t": t_norm,
        "x": x_s,
        "y": y_s,
        "z": z_s,
        "vx": vx,
        "vy": vy,
        "vz": vz,
    }


def get_smoother_config() -> Dict[str, float]:
    """Expose current default parameters."""
    return {
        "kalman_process_noise": KALMAN_PROCESS_NOISE,
        "kalman_measurement_noise": KALMAN_MEASUREMENT_NOISE,
        "startup_duration": STARTUP_DURATION,
        "startup_window": STARTUP_WINDOW,
        "startup_polyorder": STARTUP_POLYORDER,
    }
