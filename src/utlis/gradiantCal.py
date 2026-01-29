#!/usr/bin/env python3
"""Compute descent rate from track opt CSV using Savitzky-Golay filter."""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
try:
    from scipy.interpolate import CubicSpline
    from scipy.signal import savgol_filter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("scipy is required: pip install scipy") from exc


def load_track(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        raise ValueError("CSV must contain at least one row of data")
    try:
        t = np.asarray(data["t"], dtype=float)
        z = np.asarray(data["z"], dtype=float)
    except ValueError as exc:
        raise ValueError("CSV columns must be named t,x,y,z") from exc
    order = np.argsort(t)
    return t[order], z[order]


def savgolDerivatives(t: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dz/dt and d2z/dt2 via Savitzky-Golay filter.

    This fits a local polynomial (order 2 or 3) to a window of points and evaluates
    its derivatives, providing robust smoothing against noise.
    """
    n = len(t)
    dt = float(np.mean(np.diff(t)))  # Assume roughly uniform sampling

    # Choose window size: must be odd, ~15-25% of data length or fixed small number
    # For trajectories ~100 points, window_length=11 is usually good smoothing.
    window_length = min(11, n)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 5:
        window_length = max(3, n if n % 2 else n - 1)

    # Order 3 allows accurate capture of jerk, Order 2 captures const acceleration.
    # We use order 3 to allow changing acceleration (jerk) but smooth out high freq.
    polyorder = min(3, window_length - 1)

    # Calculate 1st derivative (velocity)
    dz_dt = savgol_filter(z, window_length, polyorder, deriv=1, delta=dt)

    # Calculate 2nd derivative (acceleration)
    d2z_dt2 = savgol_filter(z, window_length, polyorder, deriv=2, delta=dt)

    return dz_dt, d2z_dt2


def plotZT(t: np.ndarray, z: np.ndarray, dz_dt: np.ndarray, title: str) -> None:
    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "DejaVu Sans", "Arial"]

    points = np.array([t, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = dz_dt[:-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    lc = LineCollection(list(segments), cmap="coolwarm", norm=Normalize(vmin=colors.min(), vmax=colors.max()))
    lc.set_array(colors)
    lc.set_linewidth(2.0)
    ax.add_collection(lc)

    ax.scatter(t, z, s=12, color="black", alpha=0.4, label="Samples")
    ax.set_xlabel("t")
    ax.set_ylabel("z")
    ax.set_title(title)
    ax.grid(True)
    ax.autoscale()

    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("dz/dt (下降率)")

    plt.tight_layout()
    plt.show()


def lagrangeDerivatives(t: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dz/dt and d2z/dt2 via Lagrange interpolation.

    This fits a polynomial through each set of 3 consecutive points and evaluates
    its derivatives at the center point, providing smooth estimates.
    """
    n = len(t)
    dz_dt = np.zeros(n)
    d2z_dt2 = np.zeros(n)

    for i in range(1, n - 1):
        t0, t1, t2 = t[i - 1], t[i], t[i + 1]
        z0, z1, z2 = z[i - 1], z[i], z[i + 1]

        # Compute first derivative (dz/dt) at t1
        dz_dt[i] = ((z0 * (t1 - t2) + z1 * (t2 - t0) + z2 * (t0 - t1)) /
                     ((t0 - t1) * (t0 - t2) + (t1 - t0) * (t1 - t2) + (t2 - t0) * (t2 - t1)))

        # Compute second derivative (d2z/dt2) at t1
        d2z_dt2[i] = (2 * (z0 * (t1 - t2) + z1 * (t2 - t0) + z2 * (t0 - t1)) /
                       ((t0 - t1) * (t0 - t2) * (t1 - t2)))

    # Handle boundaries with simple finite differences
    dz_dt[0] = (z[1] - z[0]) / (t[1] - t[0])
    dz_dt[-1] = (z[-1] - z[-2]) / (t[-1] - t[-2])
    d2z_dt2[0] = (z[2] - 2 * z[1] + z[0]) / ((t[1] - t[0]) ** 2)
    d2z_dt2[-1] = (z[-1] - 2 * z[-2] + z[-3]) / ((t[-1] - t[-2]) ** 2)

    return dz_dt, d2z_dt2

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute descent rate using Savitzky-Golay filter and plot z-t.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=pathlib.Path, help="Track opt CSV file (columns: t,x,y,z)")
    args = parser.parse_args()

    t, z = load_track(args.csv_path)
    dz_dt, d2z_dt2 = savgolDerivatives(t, z)

    print("Descent rate stats (dz/dt):")
    print(f"  min: {dz_dt.min():.6f}")
    print(f"  max: {dz_dt.max():.6f}")
    print(f"  mean: {dz_dt.mean():.6f}")
    print("Descent trend stats (d2z/dt2):")
    print(f"  min: {d2z_dt2.min():.6f}")
    print(f"  max: {d2z_dt2.max():.6f}")
    print(f"  mean: {d2z_dt2.mean():.6f}")

    plotZT(t, z, dz_dt, title=str(args.csv_path))


if __name__ == "__main__":
    main()
