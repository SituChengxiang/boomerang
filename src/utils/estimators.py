#!/usr/bin/env python3
"""State estimators for trajectory preprocessing.

Transforms noisy raw samples (t, x, y, z) into physically consistent
state vectors [p, v, a] on a standardized time grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
from mathUtils import (
    magnitude,
    savgol_derivatives,
    spline_derivative_analytical,
    standardize_time_grid,
)
from pykalman import KalmanFilter
from scipy.interpolate import CubicSpline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


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
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> None:
        self.em_iters = int(em_iters)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

    def _fit_axis(self, t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1e-6

        F, H = _constant_acceleration_matrices(dt)
        Q = np.eye(3) * self.process_noise
        R = np.array([[self.measurement_noise]])

        init_state = np.array([x[0], (x[1] - x[0]) / dt, 0.0])
        init_cov = np.eye(3)

        kf = KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=init_state,
            initial_state_covariance=init_cov,
        )

        kf = kf.em(x.reshape(-1, 1), n_iter=self.em_iters)
        state_means, state_covs = kf.smooth(x.reshape(-1, 1))

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

        # Align to uniform time grid
        t_std, x_std = standardize_time_grid(t, means[:, 0, 0])
        y_std = CubicSpline(t - t[0], means[:, 0, 1], bc_type="natural")(t_std)
        z_std = CubicSpline(t - t[0], means[:, 0, 2], bc_type="natural")(t_std)
        pos_std = np.stack([x_std, y_std, z_std], axis=1)

        vx_std = CubicSpline(t - t[0], means[:, 1, 0], bc_type="natural")(t_std)
        vy_std = CubicSpline(t - t[0], means[:, 1, 1], bc_type="natural")(t_std)
        vz_std = CubicSpline(t - t[0], means[:, 1, 2], bc_type="natural")(t_std)
        vel_std = np.stack([vx_std, vy_std, vz_std], axis=1)

        ax_std = CubicSpline(t - t[0], means[:, 2, 0], bc_type="natural")(t_std)
        ay_std = CubicSpline(t - t[0], means[:, 2, 1], bc_type="natural")(t_std)
        az_std = CubicSpline(t - t[0], means[:, 2, 2], bc_type="natural")(t_std)
        acc_std = np.stack([ax_std, ay_std, az_std], axis=1)

        sigma_std = CubicSpline(t - t[0], sigma, bc_type="natural")(t_std)

        return EstimatorOutput(
            t_std=t_std, pos=pos_std, vel=vel_std, acc=acc_std, sigma=sigma_std
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
        t_std, _ = standardize_time_grid(t_shift, pos[:, 0])
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

        t_std, x_std = standardize_time_grid(t, pos[:, 0], dt=self.dt)
        y_std = CubicSpline(t - t[0], pos[:, 1], bc_type="natural")(t_std)
        z_std = CubicSpline(t - t[0], pos[:, 2], bc_type="natural")(t_std)
        pos_std = np.stack([x_std, y_std, z_std], axis=1)

        vx = spline_derivative_analytical(t_std, x_std, order=1)
        vy = spline_derivative_analytical(t_std, y_std, order=1)
        vz = spline_derivative_analytical(t_std, z_std, order=1)
        vel_std = np.stack([vx, vy, vz], axis=1)

        ax = spline_derivative_analytical(t_std, x_std, order=2)
        ay = spline_derivative_analytical(t_std, y_std, order=2)
        az = spline_derivative_analytical(t_std, z_std, order=2)
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

        # Optional smoothing of derivatives for stability
        vx_s, ax_s = savgol_derivatives(
            vel_std[:, 0], self.dt, self.window_rate, self.polyorder
        )
        vy_s, ay_s = savgol_derivatives(
            vel_std[:, 1], self.dt, self.window_rate, self.polyorder
        )
        vz_s, az_s = savgol_derivatives(
            vel_std[:, 2], self.dt, self.window_rate, self.polyorder
        )
        vel_std = np.stack([vx_s, vy_s, vz_s], axis=1)
        acc_std = np.stack([ax_s, ay_s, az_s], axis=1)

        return EstimatorOutput(
            t_std=t_std, pos=pos_std, vel=vel_std, acc=acc_std, sigma=sigma_std
        )


__all__ = [
    "EstimatorOutput",
    "BaseEstimator",
    "KalmanEstimator",
    "GPREstimator",
    "SplineEstimator",
    "magnitude",
]
