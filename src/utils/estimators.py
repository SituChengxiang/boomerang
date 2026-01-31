#!/usr/bin/env python3
"""State estimators for trajectory preprocessing.

Transforms noisy raw samples (t, x, y, z) into physically consistent
state vectors [p, v, a] on a standardized time grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
from pykalman import KalmanFilter
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from . import mathUtils


@dataclass
class EstimatorOutput:
    """Structured output for all estimators."""

    t_std: np.ndarray
    pos: np.ndarray  # shape (N, 3)
    vel: np.ndarray  # shape (N, 3)
    acc: np.ndarray  # shape (N, 3)
    sigma: np.ndarray  # shape (N, 3) or (N,)


class BaseEstimator(Protocol):
    """Base interface for all estimators."""

    def estimate(self, t_raw: np.ndarray, pos_raw: np.ndarray) -> EstimatorOutput: ...


def _validate_inputs(
    t_raw: np.ndarray, pos_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_raw, dtype=float).ravel()
    pos = np.asarray(pos_raw, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos_raw must have shape (N, 3)")
    if t.size != pos.shape[0]:
        raise ValueError("t_raw length must match pos_raw rows")
    if t.size < 2:
        raise ValueError("Need at least 2 samples")
    return t, pos


def _constant_acceleration_matrices(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    F = np.array(
        [
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    H = np.array([[1.0, 0.0, 0.0]])
    return F, H


class KalmanEstimator:
    """RTS Smoother with constant-acceleration model and EM-fitted noise."""

    def __init__(
        self,
        em_iters: int = 10,
        process_noise: float = 0.1,
        measurement_noise: float = 0.001,
    ) -> None:
        self.em_iters = int(em_iters)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

    def _fit_axis(self, t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-6

        F, H = _constant_acceleration_matrices(dt)
        # White-noise jerk model -> allows acceleration to drift smoothly
        q = float(self.process_noise)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Q = q * np.array(
            [
                [dt4 / 4.0, dt3 / 2.0, dt2 / 2.0],
                [dt3 / 2.0, dt2, dt],
                [dt2 / 2.0, dt, 1.0],
            ]
        )
        R = np.array([[self.measurement_noise]])

        # Robust initial state estimation using weighted quadratic least squares
        # Fit polynomial y = a*x^2 + b*x + c to first 7 points
        n_init = min(7, len(x))
        t_init = t[:n_init]
        x_init = x[:n_init]

        # Weighting: exponential decay for older points
        weights = np.exp(
            -np.linspace(0, 2, n_init)
        )  # weights: ~0.1 to 1.0 for far to near

        # Build Vandermonde matrix for quadratic fit: [1, t, t^2]
        A = np.column_stack([t_init**2, t_init, np.ones(n_init)])

        # Weighted least squares: (A^T W A) theta = A^T W x
        W = np.diag(weights)
        AW = A.T @ W @ A
        bW = A.T @ W @ x_init
        theta = np.linalg.solve(AW, bW)  # [a, b, c]

        # Initial velocity = derivative at t[0] = 2*a*t[0] + b
        v_init = 2 * theta[0] * t_init[0] + theta[1]

        # Clamp velocity to plausible range
        # For boomerang telemetry, velocity typically 0-20 m/s
        v_init = float(np.clip(v_init, -30.0, 30.0))

        init_state = np.array([x[0], v_init, 0.0])

        # Tight initial covariance to prevent Kalman from diverging
        # Position: 0.01m, Velocity: 1 m/s, Acceleration: 1 m/s²
        init_cov = np.diag([0.01, 1.0, 1.0])

        kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=init_state,
            initial_state_covariance=init_cov,
        )

        # EM iterations - fewer iterations to prevent overfitting noise
        kf = kf.em(x.reshape(-1, 1), n_iter=min(self.em_iters, 5))
        state_means, state_covs = kf.smooth(x.reshape(-1, 1))

        # Post-process: Clip unrealistic velocities
        # Replace any velocity > 100 m/s (impossible for boomerang)
        state_means[:, 1] = np.clip(state_means[:, 1], -100.0, 100.0)

        return state_means, state_covs

    def estimate(self, t_raw: np.ndarray, pos_raw: np.ndarray) -> EstimatorOutput:
        t, pos = _validate_inputs(t_raw, pos_raw)

        state_means = []
        state_sigmas = []
        for axis in range(3):
            means, covs = self._fit_axis(t, pos[:, axis])
            state_means.append(means)
            state_sigmas.append(np.sqrt(np.maximum(covs[:, 0, 0], 0.0)))

        means = np.stack(state_means, axis=2)  # (N, 3, 3)
        sigma = np.stack(state_sigmas, axis=1)  # (N, 3)

        # Use Kalman smoothed results directly (already uniformly sampled in time)
        t_std = t
        pos_std = means[:, 0, :]
        vel_std = means[:, 1, :]
        acc_std = means[:, 2, :]

        return EstimatorOutput(
            t_std=t_std, pos=pos_std, vel=vel_std, acc=acc_std, sigma=sigma
        )


class GPREstimator:
    """Gaussian Process estimator with analytical derivatives."""

    def __init__(
        self,
        length_scale: float = 0.2,
        noise_level: float = 1e-3,
        alpha: float = 0.0,
        n_restarts_optimizer: int = 2,
    ) -> None:
        self.length_scale = float(length_scale)
        self.noise_level = float(noise_level)
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)

    def _fit_axis(
        self, t: np.ndarray, x: np.ndarray
    ) -> Tuple[GaussianProcessRegressor, np.ndarray, np.ndarray, np.ndarray]:
        kernel = RBF(length_scale=self.length_scale) + WhiteKernel(
            noise_level=self.noise_level
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        gp.fit(t[:, None], x)
        return (
            gp,
            np.asarray(gp.alpha_, dtype=float),
            np.asarray(gp.kernel_.k1.length_scale, dtype=float),  # type: ignore[attr-defined]  sklearn kernels have inconsistent type stubs
            np.asarray(gp.kernel_.k2.noise_level, dtype=float),  # type: ignore[attr-defined]  sklearn kernels have inconsistent type stubs
        )

    @staticmethod
    def _rbf_kernel_and_derivatives(
        t_pred: np.ndarray, t_train: np.ndarray, length_scale: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_pred = t_pred[:, None]
        t_train = t_train[None, :]
        r = t_pred - t_train
        l2 = float(length_scale) ** 2
        k = np.exp(-0.5 * (r**2) / l2)
        dk = -k * (r / l2)
        d2k = k * ((r**2) / (l2**2) - 1.0 / l2)
        return k, dk, d2k

    def estimate(self, t_raw: np.ndarray, pos_raw: np.ndarray) -> EstimatorOutput:
        t, pos = _validate_inputs(t_raw, pos_raw)
        t0 = t[0]
        t_shift = t - t0

        # Use standardize_time_grid to define the unified grid
        t_std, _ = mathUtils.standardize_time_grid(t_shift, pos[:, 0])
        r = t_std[:, None] - t_shift[None, :]
        r2 = r * r

        pos_pred = []
        vel_pred = []
        acc_pred = []
        sig_pred = []

        for axis in range(3):
            gp, alpha, length_scale, _ = self._fit_axis(t_shift, pos[:, axis])
            l2 = float(np.ravel(length_scale)[0]) ** 2
            k = np.exp(-0.5 * r2 / l2)
            dk = -k * (r / l2)
            d2k = k * ((r2 / (l2**2)) - 1.0 / l2)
            mean = k @ alpha
            dmean = dk @ alpha
            d2mean = d2k @ alpha

            std = gp.predict(t_std[:, None], return_std=True)[1]

            pos_pred.append(mean)
            vel_pred.append(dmean)
            acc_pred.append(d2mean)
            sig_pred.append(std)

        pos_std = np.stack(pos_pred, axis=1)
        vel_std = np.stack(vel_pred, axis=1)
        acc_std = np.stack(acc_pred, axis=1)
        sigma_std = np.stack(sig_pred, axis=1)

        return EstimatorOutput(
            t_std=t_std, pos=pos_std, vel=vel_std, acc=acc_std, sigma=sigma_std
        )


class SplineEstimator:
    """Lightweight estimator using spline and Savitzky-Golay derivatives."""

    def __init__(
        self, dt: float = 0.01, window_rate: float = 0.08, polyorder: int = 3
    ) -> None:
        self.dt = float(dt)
        self.window_rate = float(window_rate)
        self.polyorder = int(polyorder)

    def estimate(self, t_raw: np.ndarray, pos_raw: np.ndarray) -> EstimatorOutput:
        t, pos = _validate_inputs(t_raw, pos_raw)

        t_std, x_std = mathUtils.standardize_time_grid(t, pos[:, 0], dt=self.dt)
        y_std = CubicSpline(t - t[0], pos[:, 1], bc_type="natural")(t_std)
        z_std = CubicSpline(t - t[0], pos[:, 2], bc_type="natural")(t_std)
        pos_std = np.stack([x_std, y_std, z_std], axis=1)

        vx = mathUtils.spline_derivative_analytical(t_std, x_std, order=1)
        vy = mathUtils.spline_derivative_analytical(t_std, y_std, order=1)
        vz = mathUtils.spline_derivative_analytical(t_std, z_std, order=1)

        # Fix start point velocity using local quadratic fit (avoids spline boundary artifacts)
        vx[0] = mathUtils.local_initial_derivative(t_std, x_std)
        vy[0] = mathUtils.local_initial_derivative(t_std, y_std)
        vz[0] = mathUtils.local_initial_derivative(t_std, z_std)

        vel_std = np.stack([vx, vy, vz], axis=1)

        ax = mathUtils.spline_derivative_analytical(t_std, x_std, order=2)
        ay = mathUtils.spline_derivative_analytical(t_std, y_std, order=2)
        az = mathUtils.spline_derivative_analytical(t_std, z_std, order=2)
        acc_std = np.stack([ax, ay, az], axis=1)

        # sigma from residuals between spline and raw data, projected to t_std
        spline_x = CubicSpline(t - t[0], pos[:, 0], bc_type="natural")
        spline_y = CubicSpline(t - t[0], pos[:, 1], bc_type="natural")
        spline_z = CubicSpline(t - t[0], pos[:, 2], bc_type="natural")
        res = np.stack(
            [
                pos[:, 0] - spline_x(t - t[0]),
                pos[:, 1] - spline_y(t - t[0]),
                pos[:, 2] - spline_z(t - t[0]),
            ],
            axis=1,
        )
        sigma_axis = np.std(res, axis=0)
        sigma_std = np.tile(sigma_axis, (t_std.size, 1))

        # Optional smoothing removed to prevent double-differentiation error
        # (Original code wrongly called savgol_derivatives on velocity, resulting in acceleration)

        return EstimatorOutput(
            t_std=t_std, pos=pos_std, vel=vel_std, acc=acc_std, sigma=sigma_std
        )


__all__ = [
    "EstimatorOutput",
    "BaseEstimator",
    "KalmanEstimator",
    "GPREstimator",
    "SplineEstimator",
]
