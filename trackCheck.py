#!/usr/bin/env python3
"""Plot and export raw vs. Kalman + spline-smoothed 3D track from CSV."""
from __future__ import annotations

import argparse
import pathlib
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
try:
    from scipy.interpolate import CubicSpline
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required: pip install scipy") from exc


class Kalman1D:
    """Constant-velocity Kalman filter for a single axis."""

    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise]], dtype=float)
        self.R = float(measurement_noise)
        self.x = np.zeros(2, dtype=float)  # state: [position, velocity]
        self.P = np.eye(2, dtype=float)
        self._initialized = False

    def _init_state(self, initial_position: float) -> None:
        self.x[0] = initial_position
        self.x[1] = 0.0
        self._initialized = True

    def step(self, measurement: float, dt: float) -> float:
        if not self._initialized:
            self._init_state(measurement)
            return measurement

        # Predict
        F = np.array([[1.0, dt], [0.0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Update
        y = measurement - self.x[0]
        S = self.P[0, 0] + self.R
        K = np.array([self.P[0, 0] / S, self.P[1, 0] / S])
        self.x[0] += K[0] * y
        self.x[1] += K[1] * y
        I_KH = np.array([[1.0 - K[0], 0.0], [-K[1], 1.0]])
        self.P = I_KH @ self.P
        return self.x[0]


def run_filter(times: np.ndarray, values: np.ndarray, q: float, r: float) -> np.ndarray:
    """Apply Kalman filter to one coordinate series."""
    kf = Kalman1D(q, r)
    filtered = np.zeros_like(values, dtype=float)
    prev_t = float(times[0])
    for i, (t, v) in enumerate(zip(times, values)):
        dt = float(t - prev_t) if i else 0.0
        if dt <= 0.0 and i:
            dt = 1e-6
        prev_t = float(t)
        filtered[i] = kf.step(float(v), dt)
    return filtered


def load_track(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    order = np.argsort(t)
    return t[order], x[order], y[order], z[order]


def plot_tracks(raw: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                filtered: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                spline: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
                title: str,
                output: pathlib.Path,
                show: bool) -> None:
    t_raw, x_raw, y_raw, z_raw = raw
    _, x_f, y_f, z_f = filtered
    spline_data = spline

    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "DejaVu Sans", "Arial"]

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_raw, y_raw, z_raw, label="Raw", color="#d9534f", alpha=0.7)
    ax.plot(x_f, y_f, z_f, label="Kalman", color="#4285f4", linewidth=2.0)
    if spline_data is not None:
        _, xs, ys, zs = spline_data
        ax.plot(xs, ys, zs, label="Kalman+Spline", color="#5b8e7d", linewidth=1.8, linestyle="--")
    ax.scatter(x_raw[0], y_raw[0], z_raw[0], color="#5cb85c", marker="o", s=50, label="Start")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    if show:
        plt.show()
    plt.close(fig)


def spline_interpolate(t: np.ndarray, values: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    factor = max(1, int(factor))
    count = max(len(t), len(t) * factor)
    t_new = np.linspace(float(t[0]), float(t[-1]), count)
    spline = CubicSpline(t, values, bc_type="natural")
    return t_new, spline(t_new)


def export_track(csv_path: pathlib.Path,
                 t: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 z: np.ndarray) -> pathlib.Path:
    out_path = csv_path.with_name(f"{csv_path.stem}opt{csv_path.suffix}")
    stacked = np.column_stack([t, x, y, z])
    np.savetxt(out_path, stacked, delimiter=",", header="t,x,y,z", comments="", fmt="%.8f")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot and export raw vs. Kalman + spline-smoothed 3D track from CSV (columns: t,x,y,z).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=pathlib.Path, help="Track CSV file")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("plot.png"),
                        help="Output image path")
    parser.add_argument("--process-noise", type=float, default=0.01, help="Process noise variance (Q)")
    parser.add_argument("--measurement-noise", type=float, default=0.1, help="Measurement noise variance (R)")
    parser.add_argument("--spline-factor", type=int, default=1,
                        help="Interpolation factor; 2 doubles the number of points before export")
    parser.add_argument("--no-export", action="store_true", help="Skip writing the *opt.csv output")
    parser.add_argument("--show", action="store_true", help="Display the plot interactively")
    args = parser.parse_args()

    t, x, y, z = load_track(args.csv_path)
    kx = run_filter(t, x, args.process_noise, args.measurement_noise)
    ky = run_filter(t, y, args.process_noise, args.measurement_noise)
    kz = run_filter(t, z, args.process_noise, args.measurement_noise)

    t_spline, xs = spline_interpolate(t, kx, args.spline_factor)
    _, ys = spline_interpolate(t, ky, args.spline_factor)
    _, zs = spline_interpolate(t, kz, args.spline_factor)

    spline_data = (t_spline, xs, ys, zs)
    if not args.no_export:
        out_csv = export_track(args.csv_path, t_spline, xs, ys, zs)
        print(f"Exported: {out_csv}")

    plot_tracks((t, x, y, z), (t, kx, ky, kz), spline_data,
                title=str(args.csv_path), output=args.output, show=args.show)


if __name__ == "__main__":
    main()
