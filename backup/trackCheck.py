#!/usr/bin/env python3
"""Enhanced preprocessing for boomerang trajectory data with RTS smoothing, energy-based truncation, and derivative optimization."""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# Constants
G = 9.793  # Gravity constant (m/s^2)
MIN_VALID_FLIGHT_TIME = 0.6  # Minimum valid flight time in seconds
ENERGY_TOLERANCE = 10  # Tolerance for dE/dt > 0 detection (m^2/s^3)


class RTSKalmanSmoother:
    """Rauch-Tung-Striebel (RTS) Kalman Smoother for constant-velocity model."""

    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise]], dtype=float)
        self.R = float(measurement_noise)

    def smooth(self, times: np.ndarray, measurements: np.ndarray) -> np.ndarray:
        """Apply forward Kalman filter and backward RTS smoother."""
        n = len(measurements)
        if n < 2:
            return measurements.copy()

        # Forward Kalman filter
        x_f = np.zeros((n, 2), dtype=float)  # Forward states: [position, velocity]
        P_f = np.zeros((n, 2, 2), dtype=float)  # Forward covariances

        # Initialize
        x_f[0] = np.array([measurements[0], 0.0])
        P_f[0] = np.eye(2)

        # Forward pass
        for i in range(1, n):
            dt = float(times[i] - times[i - 1])
            if dt <= 0:
                dt = 1e-6

            # State transition matrix
            F = np.array([[1.0, dt], [0.0, 1.0]])

            # Predict
            x_pred = F @ x_f[i - 1]
            P_pred = F @ P_f[i - 1] @ F.T + self.Q

            # Update
            y = measurements[i] - x_pred[0]
            S = P_pred[0, 0] + self.R
            K = np.array([P_pred[0, 0] / S, P_pred[1, 0] / S])

            x_f[i] = x_pred + K * y
            P_f[i] = (np.eye(2) - np.outer(K, np.array([1.0, 0.0]))) @ P_pred

        # Backward RTS smoother
        x_s = np.zeros((n, 2), dtype=float)  # Smoothed states
        P_s = np.zeros((n, 2, 2), dtype=float)  # Smoothed covariances

        # Initialize with last forward estimate
        x_s[-1] = x_f[-1]
        P_s[-1] = P_f[-1]

        # Backward pass
        for i in range(n - 2, -1, -1):
            dt = float(times[i + 1] - times[i])
            if dt <= 0:
                dt = 1e-6

            F = np.array([[1.0, dt], [0.0, 1.0]])

            # Smoothing gain
            C = P_f[i] @ F.T @ np.linalg.inv(F @ P_f[i] @ F.T + self.Q)

            # Smooth state and covariance
            x_s[i] = x_f[i] + C @ (x_s[i + 1] - F @ x_f[i])
            P_s[i] = P_f[i] + C @ (P_s[i + 1] - F @ P_f[i] @ F.T - self.Q) @ C.T

        return x_s


def calculate_energy(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate mechanical energy per unit mass and its derivative using robust numerical methods.

    Args:
        t: Time array (must be strictly increasing)
        x, y, z: Position arrays

    Returns:
        Tuple of (energy_per_mass, dE_dt) where:
            energy_per_mass = 0.5*(vx² + vy² + vz²) + g*z
            dE_dt = derivative of energy_per_mass (v·a + g*vz)
    """
    # 确保时间严格递增且无重复（使用已有的辅助函数）
    t, x, y, z = _ensure_strictly_increasing_time(t, x, y, z)

    n = len(t)
    if n < 2:
        # 数据点太少，返回NaN数组
        energy = np.full_like(t, np.nan, dtype=float)
        dE_dt = np.full_like(t, np.nan, dtype=float)
        return energy, dE_dt

    # 方法1：如果数据点足够，使用样条导数（更平滑）
    use_spline = n >= 4
    vx = np.zeros(n, dtype=float)
    vy = np.zeros(n, dtype=float)
    vz = np.zeros(n, dtype=float)
    ax = np.zeros(n, dtype=float)
    ay = np.zeros(n, dtype=float)
    az = np.zeros(n, dtype=float)

    if use_spline:
        try:
            # 使用三次样条计算导数和二阶导数
            sx = CubicSpline(t, x, bc_type="natural")
            sy = CubicSpline(t, y, bc_type="natural")
            sz = CubicSpline(t, z, bc_type="natural")

            vx = sx.derivative(1)(t)
            vy = sy.derivative(1)(t)
            vz = sz.derivative(1)(t)

            ax = sx.derivative(2)(t)
            ay = sy.derivative(2)(t)
            az = sz.derivative(2)(t)
        except Exception:
            # 样条失败，回退到数值差分
            use_spline = False

    if not use_spline:
        # 方法2：使用中心差分法（内部点）和前向/后向差分（边界点）
        # 计算时间间隔
        dt = np.diff(t)
        # 避免除以零
        dt = np.where(dt <= 1e-12, 1e-12, dt)

        # 内部点使用中心差分
        if n >= 3:
            vx[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
            vy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
            vz[1:-1] = (z[2:] - z[:-2]) / (t[2:] - t[:-2])

        # 边界点使用前向/后向差分
        vx[0] = (x[1] - x[0]) / dt[0]
        vy[0] = (y[1] - y[0]) / dt[0]
        vz[0] = (z[1] - z[0]) / dt[0]

        if n >= 2:
            vx[-1] = (x[-1] - x[-2]) / dt[-1]
            vy[-1] = (y[-1] - y[-2]) / dt[-1]
            vz[-1] = (z[-1] - z[-2]) / dt[-1]

        # 计算加速度（使用同样的差分方法）
        if n >= 3:
            # 内部点
            ax[1:-1] = (vx[2:] - vx[:-2]) / (t[2:] - t[:-2])
            ay[1:-1] = (vy[2:] - vy[:-2]) / (t[2:] - t[:-2])
            az[1:-1] = (vz[2:] - vz[:-2]) / (t[2:] - t[:-2])

        # 边界点
        if n >= 2:
            ax[0] = (vx[1] - vx[0]) / dt[0]
            ay[0] = (vy[1] - vy[0]) / dt[0]
            az[0] = (vz[1] - vz[0]) / dt[0]

            if n >= 3:
                ax[-1] = (vx[-1] - vx[-2]) / dt[-1]
                ay[-1] = (vy[-1] - vy[-2]) / dt[-1]
                az[-1] = (vz[-1] - vz[-2]) / dt[-1]

    # 动能: 0.5*v²
    kinetic = 0.5 * (vx**2 + vy**2 + vz**2)

    # 势能: g*z
    potential = G * z

    # 总机械能
    energy = kinetic + potential

    # 能量变化率: dE/dt = v·a + g*vz
    dE_dt = vx * ax + vy * ay + vz * az + G * vz

    # 可选：对dE/dt进行轻微平滑以减少噪声
    if n >= 5:
        try:
            window = min(5, n)
            if window % 2 == 0:
                window -= 1
            if window >= 3:
                dE_dt = savgol_filter(dE_dt, window, 2, mode="interp")
        except Exception:
            # 平滑失败，保持原值
            pass

    return energy, dE_dt


def _ensure_strictly_increasing_time(
    t: np.ndarray,
    *series: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, ...]:
    """Ensure time is finite, sorted, and strictly increasing.

    Tracker exports sometimes contain duplicate timestamps; numerical
    differentiation (np.gradient) will emit divide-by-zero warnings in that case.
    We drop non-increasing time samples while keeping the first occurrence.
    """
    t = np.asarray(t, dtype=float)
    if any(len(s) != len(t) for s in series):
        raise ValueError("t and series must have the same length")

    finite_mask = np.isfinite(t)
    for s in series:
        finite_mask &= np.isfinite(s)

    t = t[finite_mask]
    cleaned = [np.asarray(s, dtype=float)[finite_mask] for s in series]
    if len(t) == 0:
        return (t, *cleaned)

    order = np.argsort(t)
    t = t[order]
    cleaned = [s[order] for s in cleaned]

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
    return (t_out, *series_out)


def calculate_energy_from_smoother(
    t: np.ndarray,
    smoother_states: Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ],  # (x_state, y_state, z_state)
) -> Tuple[np.ndarray, np.ndarray]:
    """使用RTS平滑器的状态计算能量，避免数值微分"""
    x_state, y_state, z_state = smoother_states

    # 从状态中提取位置和速度
    # 假设每个state是(n, 2)数组，[position, velocity]
    vx = x_state[:, 1]  # x方向速度
    vy = y_state[:, 1]  # y方向速度
    vz = z_state[:, 1]  # z方向速度

    # 动能: 0.5*v²
    kinetic = 0.5 * (vx**2 + vy**2 + vz**2)

    # 势能: g*z (使用平滑后的位置)
    potential = G * x_state[:, 0]  # 假设x_state[:, 0]是z位置

    energy = kinetic + potential

    # 计算能量变化率: dE/dt = v·a
    # 可以从平滑器状态估计加速度，或使用Savitzky-Golay平滑后的导数
    ax = np.gradient(vx, t, edge_order=1)  # 使用更稳定的边界处理
    ay = np.gradient(vy, t, edge_order=1)
    az = np.gradient(vz, t, edge_order=1)

    dE_dt = vx * ax + vy * ay + vz * az + G * vz

    return energy, dE_dt


def truncate_bad_data(
    t: np.ndarray, x_state: np.ndarray, y_state: np.ndarray, z_state: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    # 提取位置（无论输入是状态还是位置）
    if x_state.ndim == 2:  # 是状态数组
        x = x_state[:, 0]
        y = y_state[:, 0]
        z = z_state[:, 0]
    else:  # 是位置数组
        x, y, z = x_state, y_state, z_state

    if len(t) < 3:
        return t, x, y, z, -1

    # 计算能量 - 使用正确的参数
    try:
        # 检查是否是状态数组
        if x_state.ndim == 2:
            # 传递状态数组
            energy, dE_dt = calculate_energy_from_smoother(
                t, (x_state, y_state, z_state)
            )
        else:
            # 如果是位置数组，回退到原始的能量计算方法
            # 需要导入原始的calculate_energy函数
            energy, dE_dt = calculate_energy(t, x_state, y_state, z_state)
    except Exception as e:
        print(f"  Warning: Energy calculation failed: {e}")
        print("  Falling back to simple truncation check")
        return t, x, y, z, -1

    # Find first point where dE/dt > tolerance (energy increasing)
    # Allow small tolerance for numerical noise
    violation_mask = dE_dt > ENERGY_TOLERANCE

    # Also check for NaN or infinite values
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    # Also treat non-finite energy as bad
    energy_mask = np.isfinite(energy) & np.isfinite(dE_dt)

    # Combine masks
    bad_mask = violation_mask | ~valid_mask | ~energy_mask

    # Find first bad point, but avoid truncating at the very beginning.
    # Edge derivatives are naturally noisier; require a small warm-up and
    # (by default) at least two consecutive bad samples.
    warmup = min(3, len(t) - 1)
    candidate = bad_mask.copy()
    candidate[:warmup] = False

    for i in np.where(candidate)[0]:
        # Require consecutive confirmation when possible
        if i + 1 < len(candidate) and candidate[i + 1]:
            trunc_idx = int(i)
            return t[:trunc_idx], x[:trunc_idx], y[:trunc_idx], z[:trunc_idx], trunc_idx

    bad_indices = np.where(candidate)[0]
    if len(bad_indices) > 0:
        trunc_idx = int(bad_indices[0])
        return t[:trunc_idx], x[:trunc_idx], y[:trunc_idx], z[:trunc_idx], trunc_idx

    return t, x, y, z, -1


def smooth_derivatives(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate and smooth velocities and accelerations using Savitzky-Golay filter.

    Args:
        t, x, y, z: Time and position arrays

    Returns:
        Dictionary containing smoothed velocities and accelerations
    """
    n = len(t)
    if n < 5:
        # Not enough points for proper smoothing
        edge_order = 2 if n >= 3 else 1
        vx = np.gradient(x, t, edge_order=edge_order)
        vy = np.gradient(y, t, edge_order=edge_order)
        vz = np.gradient(z, t, edge_order=edge_order)
        ax = np.gradient(vx, t, edge_order=edge_order)
        ay = np.gradient(vy, t, edge_order=edge_order)
        az = np.gradient(vz, t, edge_order=edge_order)
        return {"vx": vx, "vy": vy, "vz": vz, "ax": ax, "ay": ay, "az": az}

    # Determine window size for Savitzky-Golay filter
    window = min(11, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = 5 if n >= 5 else 3

    # Ensure window is odd and <= n
    window = min(window, n)
    if window % 2 == 0:
        window -= 1

    # Calculate derivatives using Savitzky-Golay filter
    # Polynomial order 2 for velocity (1st derivative), 3 for acceleration (2nd derivative)
    poly_order_vel = min(2, window - 1)
    poly_order_acc = min(3, window - 1)

    # Calculate derivatives
    dt: float = float(np.mean(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 1e-6

    try:
        vx = savgol_filter(x, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vy = savgol_filter(y, window, poly_order_vel, deriv=1, delta=dt, mode="interp")
        vz = savgol_filter(z, window, poly_order_vel, deriv=1, delta=dt, mode="interp")

        ax = savgol_filter(x, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        ay = savgol_filter(y, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
        az = savgol_filter(z, window, poly_order_acc, deriv=2, delta=dt, mode="interp")
    except Exception:
        # Fallback to simple gradient if Savitzky-Golay fails
        edge_order = 2 if n >= 3 else 1
        vx = np.gradient(x, t, edge_order=edge_order)
        vy = np.gradient(y, t, edge_order=edge_order)
        vz = np.gradient(z, t, edge_order=edge_order)
        ax = np.gradient(vx, t, edge_order=edge_order)
        ay = np.gradient(vy, t, edge_order=edge_order)
        az = np.gradient(vz, t, edge_order=edge_order)

    return {"vx": vx, "vy": vy, "vz": vz, "ax": ax, "ay": ay, "az": az}


def load_track(
    path: pathlib.Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load track data from CSV file."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        raise ValueError("CSV must contain at least one row of data")
    try:
        t = np.asarray(data["t"], dtype=float)
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    except ValueError as exc:
        raise ValueError("CSV columns must be named t,x,y,z") from exc

    # Sort by time
    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]

    # Remove duplicate / non-increasing timestamps to prevent derivative warnings.
    t, x, y, z = _ensure_strictly_increasing_time(t, x, y, z)
    if len(t) == 0:
        raise ValueError("Track contains no finite samples")
    return t, x, y, z


def spline_interpolate(
    t: np.ndarray, values: np.ndarray, factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply cubic spline interpolation."""
    factor = max(1, int(factor))
    count = max(len(t), len(t) * factor)
    t_new = np.linspace(float(t[0]), float(t[-1]), count)
    spline = CubicSpline(t, values, bc_type="natural")
    return t_new, spline(t_new)


def export_track(
    csv_path: pathlib.Path,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    velocities: Optional[Dict[str, np.ndarray]] = None,
    accelerations: Optional[Dict[str, np.ndarray]] = None,
) -> pathlib.Path:
    """Export processed track data to CSV."""
    out_path = csv_path.with_name(f"{csv_path.stem}opt{csv_path.suffix}")

    # Prepare data for export
    data_to_export = [t, x, y, z]
    headers = ["t", "x", "y", "z"]

    # Add velocities if available
    if velocities is not None:
        data_to_export.extend([velocities["vx"], velocities["vy"], velocities["vz"]])
        headers.extend(["vx", "vy", "vz"])

    # Add accelerations if available
    if accelerations is not None:
        data_to_export.extend(
            [accelerations["ax"], accelerations["ay"], accelerations["az"]]
        )
        headers.extend(["ax", "ay", "az"])

    # Stack and save
    stacked = np.column_stack(data_to_export)
    header_str = ",".join(headers)
    np.savetxt(
        out_path, stacked, delimiter=",", header=header_str, comments="", fmt="%.8f"
    )

    return out_path


def plot_tracks(
    raw: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    smoothed: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    spline: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    energy_info: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    output: pathlib.Path,
    show: bool,
) -> None:
    """Plot tracks with energy information."""
    t_raw, x_raw, y_raw, z_raw = raw
    _, x_s, y_s, z_s = smoothed
    spline_data = spline

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]

    # Create figure with subplots
    if energy_info is not None:
        fig = plt.figure(figsize=(15, 10))

        # 3D trajectory plot
        ax1 = fig.add_subplot(231, projection="3d")
        ax1.plot(x_raw, y_raw, z_raw, label="Raw", color="#d9534f", alpha=0.7)
        ax1.plot(x_s, y_s, z_s, label="RTS Smoothed", color="#4285f4", linewidth=2.0)
        if spline_data is not None:
            _, xs, ys, zs = spline_data
            ax1.plot(
                xs,
                ys,
                zs,
                label="Spline",
                color="#5b8e7d",
                linewidth=1.8,
                linestyle="--",
            )
        ax1.scatter(
            x_raw[0],
            y_raw[0],
            z_raw[0],
            color="#5cb85c",
            marker="o",
            s=50,
            label="Start",
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("3D Trajectory")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # Energy plots
        t_energy, energy, dE_dt = energy_info

        ax2 = fig.add_subplot(232)
        ax2.plot(t_energy, energy, "b-", linewidth=2, label="Energy/mass")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Energy per mass (m²/s²)")
        ax2.set_title("Mechanical Energy")
        ax2.grid(True)
        ax2.legend()

        ax3 = fig.add_subplot(233)
        ax3.plot(t_energy, dE_dt, "r-", linewidth=2, label="dE/dt")
        ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax3.axhline(
            y=ENERGY_TOLERANCE,
            color="g",
            linestyle=":",
            alpha=0.5,
            label=f"Tolerance ({ENERGY_TOLERANCE})",
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("dE/dt (m²/s³)")
        ax3.set_title("Energy Change Rate")
        ax3.grid(True)
        ax3.legend()

        # XY, XZ, YZ projections
        ax4 = fig.add_subplot(234)
        ax4.plot(x_raw, y_raw, "r-", alpha=0.5, label="Raw")
        ax4.plot(x_s, y_s, "b-", label="Smoothed")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_title("XY Projection")
        ax4.grid(True)
        ax4.legend()

        ax5 = fig.add_subplot(235)
        ax5.plot(x_raw, z_raw, "r-", alpha=0.5, label="Raw")
        ax5.plot(x_s, z_s, "b-", label="Smoothed")
        ax5.set_xlabel("X")
        ax5.set_ylabel("Z")
        ax5.set_title("XZ Projection")
        ax5.grid(True)
        ax5.legend()

        ax6 = fig.add_subplot(236)
        ax6.plot(y_raw, z_raw, "r-", alpha=0.5, label="Raw")
        ax6.plot(y_s, z_s, "b-", label="Smoothed")
        ax6.set_xlabel("Y")
        ax6.set_ylabel("Z")
        ax6.set_title("YZ Projection")
        ax6.grid(True)
        ax6.legend()

    else:
        # Simple 3D plot if no energy info
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_raw, y_raw, z_raw, label="Raw", color="#d9534f", alpha=0.7)
        ax.plot(x_s, y_s, z_s, label="RTS Smoothed", color="#4285f4", linewidth=2.0)
        if spline_data is not None:
            _, xs, ys, zs = spline_data
            ax.plot(
                xs,
                ys,
                zs,
                label="Spline",
                color="#5b8e7d",
                linewidth=1.8,
                linestyle="--",
            )
        ax.scatter(
            x_raw[0],
            y_raw[0],
            z_raw[0],
            color="#5cb85c",
            marker="o",
            s=50,
            label="Start",
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Trajectory")
        ax.legend(loc="upper left")
        ax.grid(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced preprocessing for boomerang trajectory data with RTS smoothing, energy-based truncation, and derivative optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=pathlib.Path, help="Track CSV file")
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("plot.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--process-noise",
        type=float,
        default=0.01,
        help="Process noise variance (Q) for RTS smoother",
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=0.1,
        help="Measurement noise variance (R) for RTS smoother",
    )
    parser.add_argument(
        "--mode",
        choices=["opt-only", "full"],
        default="opt-only",
        help=(
            "Processing mode. opt-only: only RTS smoothing and export t,x,y,z to *opt.csv (no truncation). "
            "full: run the full pipeline (energy truncation, time normalization, spline, derivatives, etc)."
        ),
    )
    parser.add_argument(
        "--spline-factor",
        type=int,
        default=2,
        help="Interpolation factor; 2 doubles the number of points before export",
    )
    parser.add_argument(
        "--normalize-time",
        action="store_true",
        help="(opt-only) Replace t with a uniform grid using average dt (keeps sample count)",
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Skip writing the *opt.csv output"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the plot interactively"
    )

    args = parser.parse_args()

    print(f"Processing: {args.csv_path}")
    print(f"Mode: {args.mode}")
    if args.mode == "full":
        print(f"Energy tolerance: {ENERGY_TOLERANCE} m²/s³")
        print(f"Minimum flight time: {MIN_VALID_FLIGHT_TIME} s")

    # Load raw data
    t_raw, x_raw, y_raw, z_raw = load_track(args.csv_path)
    print(f"  Raw data points: {len(t_raw)}")
    print(f"  Time range: {t_raw[0]:.3f} to {t_raw[-1]:.3f} s")
    print(f"  Duration: {t_raw[-1] - t_raw[0]:.3f} s")

    # Step 1: Apply RTS smoother
    print("\nStep 1: Applying RTS smoother...")
    smoother_x = RTSKalmanSmoother(args.process_noise, args.measurement_noise)
    smoother_y = RTSKalmanSmoother(args.process_noise, args.measurement_noise)
    smoother_z = RTSKalmanSmoother(args.process_noise, args.measurement_noise)

    # 获取完整状态
    x_state = smoother_x.smooth(t_raw, x_raw)
    y_state = smoother_y.smooth(t_raw, y_raw)
    z_state = smoother_z.smooth(t_raw, z_raw)

    # 提取平滑后的位置
    x_smooth = x_state[:, 0]
    y_smooth = y_state[:, 0]
    z_smooth = z_state[:, 0]

    # Simple mode: export RTS-smoothed positions (no truncation / energy checks).
    if args.mode == "opt-only":
        t_out = t_raw
        if args.normalize_time and len(t_out) > 1:
            avg_dt = float(np.mean(np.diff(t_out)))
            if not np.isfinite(avg_dt) or avg_dt <= 0.0:
                avg_dt = 0.0166667
            t_out = np.arange(len(t_out), dtype=float) * avg_dt

        if not args.no_export:
            print("\nExport: Writing RTS-smoothed track (*opt.csv)...")
            out_csv = export_track(
                args.csv_path,
                t_out,
                x_smooth,
                y_smooth,
                z_smooth,
                velocities=None,
                accelerations=None,
            )
            print(f"  Exported: {out_csv}")
        else:
            print("\nExport: Skipping export (--no-export flag)")

        # Keep visualization lightweight (no energy panels)
        print("\nPlot: Generating visualization...")
        plot_tracks(
            raw=(t_raw, x_raw, y_raw, z_raw),
            smoothed=(t_out, x_smooth, y_smooth, z_smooth),
            spline=None,
            energy_info=None,
            title=f"{args.csv_path.name} (RTS smoothed)",
            output=args.output,
            show=args.show,
        )
        print(f"  Plot saved: {args.output}")
        print("\n=== Preprocessing Complete (opt-only) ===")
        print(f"Final data points: {len(t_out)}")
        if len(t_out) > 1:
            print(f"Duration: {t_out[-1] - t_out[0]:.3f} s")
        return

    # Step 2: Energy-based truncation
    print("\nStep 2: Energy-based truncation...")
    t_trunc, x_trunc, y_trunc, z_trunc, trunc_idx = truncate_bad_data(
        t_raw,
        x_state,
        y_state,
        z_state,  # 传递状态而不是位置
    )

    if trunc_idx >= 0:
        print(f"  Truncated at index {trunc_idx} (t = {t_raw[trunc_idx]:.3f} s)")
        print(f"  Removed {len(t_raw) - len(t_trunc)} points")
        print(f"  Remaining points: {len(t_trunc)}")
        print(f"  Remaining duration: {t_trunc[-1] - t_trunc[0]:.3f} s")
    else:
        print("  No truncation needed")

    # Step 3: Quality assessment
    print("\nStep 3: Quality assessment...")
    flight_time = t_trunc[-1] - t_trunc[0] if len(t_trunc) > 1 else 0.0
    print(f"  Effective flight time: {flight_time:.3f} s")

    if flight_time < MIN_VALID_FLIGHT_TIME:
        print(
            f"  WARNING: Flight time ({flight_time:.3f} s) < minimum ({MIN_VALID_FLIGHT_TIME} s)"
        )
        print("  This track may be invalid for analysis")
        valid_track = False
    else:
        print("✓ Flight time meets minimum requirement")
        valid_track = True

    # Step 4: Normalize time (start at 0 with fixed step)
    print("\nStep 4: Time normalization...")
    if len(t_trunc) > 1:
        # Use original time step or default to 0.0166667 (60 Hz)
        avg_dt = np.mean(np.diff(t_trunc))
        if avg_dt <= 0:
            avg_dt = 0.0166667
        t_norm = np.arange(len(t_trunc), dtype=float) * avg_dt
        print(f"  Normalized time step: {avg_dt:.6f} s ({1 / avg_dt:.1f} Hz)")
    else:
        t_norm = t_trunc.copy()
        print("  Not enough points for normalization")

    # Step 5: Spline interpolation
    print("\nStep 5: Spline interpolation...")
    if len(t_norm) >= 4:  # Need at least 4 points for cubic spline
        t_spline, xs = spline_interpolate(t_norm, x_trunc, args.spline_factor)
        _, ys = spline_interpolate(t_norm, y_trunc, args.spline_factor)
        _, zs = spline_interpolate(t_norm, z_trunc, args.spline_factor)
        spline_data = (t_spline, xs, ys, zs)
        print(f"  Interpolated from {len(t_norm)} to {len(t_spline)} points")
    else:
        print(f"  WARNING: Not enough points ({len(t_norm)}) for spline interpolation")
        spline_data = None

    # Step 6: Smooth derivatives
    print("\nStep 6: Smoothing derivatives...")
    velocities = None
    accelerations = None

    if spline_data is not None and len(spline_data[0]) >= 5:
        t_final, x_final, y_final, z_final = spline_data
        derivatives = smooth_derivatives(t_final, x_final, y_final, z_final)
        velocities = {
            "vx": derivatives["vx"],
            "vy": derivatives["vy"],
            "vz": derivatives["vz"],
        }
        accelerations = {
            "ax": derivatives["ax"],
            "ay": derivatives["ay"],
            "az": derivatives["az"],
        }
        print("  ✓ Velocities and accelerations calculated and smoothed")
    elif spline_data is None and len(t_norm) >= 5:
        derivatives = smooth_derivatives(t_norm, x_trunc, y_trunc, z_trunc)
        velocities = {
            "vx": derivatives["vx"],
            "vy": derivatives["vy"],
            "vz": derivatives["vz"],
        }
        accelerations = {
            "ax": derivatives["ax"],
            "ay": derivatives["ay"],
            "az": derivatives["az"],
        }
        print("  ✓ Velocities and accelerations calculated (using normalized time)")
    else:
        print("WARNING: Not enough points for derivative calculation")

    # Step 7: Export processed data
    if not args.no_export and valid_track:
        print("\nStep 7: Exporting processed data...")
        if spline_data is not None:
            t_final, x_final, y_final, z_final = spline_data
            out_csv = export_track(
                args.csv_path,
                t_final,
                x_final,
                y_final,
                z_final,
                velocities,
                accelerations,
            )
        else:
            out_csv = export_track(
                args.csv_path,
                t_norm,
                x_trunc,
                y_trunc,
                z_trunc,
                velocities,
                accelerations,
            )
        print(f"  Exported: {out_csv}")
    elif args.no_export:
        print("\nStep 7: Skipping export (--no-export flag)")
    elif not valid_track:
        print("\nStep 7: Skipping export (invalid track)")

    # Step 8: Visualization
    print("\nStep 8: Generating visualization...")
    # Calculate energy for plotting
    energy_info = None
    if len(t_trunc) >= 3:
        energy, dE_dt = calculate_energy_from_smoother(
            t_trunc, (x_trunc, y_trunc, z_trunc)
        )
        energy_info = (t_trunc, energy, dE_dt)

    plot_tracks(
        raw=(t_raw, x_raw, y_raw, z_raw),
        smoothed=(t_trunc, x_trunc, y_trunc, z_trunc),
        spline=spline_data,
        energy_info=energy_info,
        title=f"{args.csv_path.name} (Flight time: {flight_time:.3f}s)",
        output=args.output,
        show=args.show,
    )
    print(f"  Plot saved: {args.output}")

    print("\n=== Preprocessing Complete ===")
    print(f"Track validity: {'VALID' if valid_track else 'INVALID'}")
    print(f"Final data points: {len(t_trunc)}")
    print(f"Effective flight time: {flight_time:.3f} s")


if __name__ == "__main__":
    main()
