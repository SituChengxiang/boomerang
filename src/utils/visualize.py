#!/usr/bin/env python3
"""Lightweight visualization utilities for boomerang trajectory debugging.

Essential plotting functions:
1. Setup debug styling
2. 3D trajectory comparison (Raw/Smooth/Fit)
3. Time series plotting (multi-track)
4. 3D trajectory with color-coded values (energy/speed/etc)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def setup_debug_style() -> None:
    """Configure matplotlib style for debugging/tracking visualization.

    Features:
    - Large fonts for clarity
    - Clean grid for reading values
    - Clear color scheme
    """
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.grid": True,
            "figure.figsize": (10, 8),
        }
    )


def plot_3d_trajectory_compare(
    t: NDArray[np.float64],
    x_raw: NDArray[np.float64],
    y_raw: NDArray[np.float64],
    z_raw: NDArray[np.float64],
    x_smooth: Optional[NDArray[np.float64]] = None,
    y_smooth: Optional[NDArray[np.float64]] = None,
    z_smooth: Optional[NDArray[np.float64]] = None,
    x_fit: Optional[NDArray[np.float64]] = None,
    y_fit: Optional[NDArray[np.float64]] = None,
    z_fit: Optional[NDArray[np.float64]] = None,
    x_final: Optional[NDArray[np.float64]] = None,
    y_final: Optional[NDArray[np.float64]] = None,
    z_final: Optional[NDArray[np.float64]] = None,
    labels: Tuple[str, str, str] = ("Raw", "Smoothed", "Fitted"),
    final_label: str = "Final Points",
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Plot 3D trajectory comparison: Raw data vs Smoothed vs Fitted.

    Args:
        t: Time array
        x_raw, y_raw, z_raw: Raw trajectory data
        x_smooth, y_smooth, z_smooth: Smoothed trajectory (optional)
        x_fit, y_fit, z_fit: Fitted/Integrated trajectory (optional)
        labels: Labels for the three trajectories
        x_final, y_final, z_final: Final (trimmed) points to overlay (optional)
        final_label: Label for final points

    Returns:
        Matplotlib figure with 3D plot

    Example:
        >>> plot_3d_trajectory_compare(t, x_raw, y_raw, z_raw,
        ...                            x_smooth=x_s, y_smooth=y_s, z_smooth=z_s)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot raw data (scatter points to show data density)
    ax.scatter(
        x_raw,
        y_raw,
        z_raw,  # pyright: ignore[reportArgumentType]
        c="black",
        s=20,
        alpha=0.4,
        marker="o",
        edgecolors="none",
        label=labels[0],
    )

    # Plot smoothed data
    if x_smooth is not None and y_smooth is not None and z_smooth is not None:
        ax.plot(
            x_smooth,
            y_smooth,
            z_smooth,
            c="#4285f4",
            linewidth=2.5,
            alpha=0.9,
            label=labels[1],
        )

    # Plot fitted data
    if x_fit is not None and y_fit is not None and z_fit is not None:
        ax.plot(
            x_fit,
            y_fit,
            z_fit,
            c="#ea4335",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label=labels[2],
        )

    # Plot final trimmed points
    if x_final is not None and y_final is not None and z_final is not None:
        ax.scatter(
            x_final,
            y_final,
            z_final,  # pyright: ignore[reportArgumentType]
            c="#fff235",
            s=28,
            alpha=0.9,
            marker="^",
            linewidth=0.4,
            label=final_label,
        )

    # Mark start point
    ax.scatter(
        x_raw[0],
        y_raw[0],
        z_raw[0],
        c="#34a853",
        s=150,
        marker="*",
        linewidth=1,
        label="Start",
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory Comparison")
    ax.legend(loc="best")
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_3d_trajectory_with_colors(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    values: NDArray[np.float64],
    colormap: str = "viridis",
    title: str = "3D Trajectory with Colored Values",
    label: str = "Value",
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Plot 3D trajectory with colors representing values (energy/speed/etc).

    Args:
        t, x, y, z: Trajectory data
        values: Values to color by (e.g., energy, speed, dE/dt)
        colormap: Matplotlib colormap name
        title: Plot title
        label: Colorbar label

    Returns:
        Matplotlib figure with colored 3D trajectory

    Example:
        >>> energy, _ = calculate_total_energy(t, x, y, z)
        >>> plot_3d_trajectory_with_colors(t, x, y, z, energy,
        >>>                                 colormap="plasma", label="Energy/S")
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory with colors
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))  # pyright: ignore[reportPrivateImportUsage]
    cmap = plt.get_cmap(colormap)

    # Create line collection for coloring
    from matplotlib.collections import LineCollection

    lc = LineCollection(
        segments,  # pyright: ignore[reportArgumentType]
        cmap=cmap,
        norm=norm,
        linewidth=3,
        alpha=0.9,
        joinstyle="round",
    )
    lc.set_array((values[:-1] + values[1:]) / 2)  # Average values for segments
    ax.add_collection3d(lc)

    # Also plot start and end points prominently
    ax.scatter(
        x[0],
        y[0],
        z[0],
        c="green",
        s=150,
        marker="*",
        edgecolors="black",
        linewidth=1,
        label="Start",
    )
    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        c="red",
        s=100,
        marker="o",
        edgecolors="black",
        linewidth=1,
        label="End",
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, fraction=0.03)
    cbar.set_label(label)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_time_series_multiple(
    tracks: Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]],
    ylabel: str = "",
    title: str = "Time Series",
    xlabel: str = "Time (s)",
    show_zero_line: bool = False,
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Plot time series for multiple tracks in one windows (multi-line).

    Args:
        tracks: Dictionary mapping track name to (time, values) tuple
        ylabel: Y-axis label
        title: Plot title
        xlabel: X-axis label
        show_zero_line: Whether to draw zero reference line

    Returns:
        Matplotlib figure with multiple time series

    Example:
        >>> plot_time_series_multiple(
        ...     {"track1": (t1, energy1), "track2": (t2, energy2)},
        ...     ylabel="Energy (J/kg)", title="Energy vs Time"
        ... )
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    cmap = plt.get_cmap("tab10")

    for i, (track_name, (t_data, y_data)) in enumerate(tracks.items()):
        color = cmap(i % 10)
        ax.plot(t_data, y_data, color=color, linewidth=2, alpha=0.9, label=track_name)
        ax.scatter(t_data, y_data, s=30, color=color, alpha=0.7)

    if show_zero_line:
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    return fig


def plot_3d_trajectory_plus_energy_overlay(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    z: NDArray[np.float64],
    energy: NDArray[np.float64],
    show_grid: bool = True,
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Plot 3D trajectory with energy as color.

    Args:
        t, x, y, z: Trajectory data
        energy: Energy per unit mass (J/kg) or other scalar values
        show_grid: Show 3D grid lines

    Returns:
        Matplotlib figure

    Note: This is essentially a wrapper for plot_3d_trajectory_with_colors
          with energy-specific labeling.
    """
    return plot_3d_trajectory_with_colors(
        t,
        x,
        y,
        z,
        energy,
        colormap="coolwarm",
        title="3D Trajectory - Energy Overlay",
        label="Energy per mass (J/kg)",
    )


def plot_velocity_comparison(
    t: NDArray[np.float64],
    tracks: Dict[str, NDArray[np.float64]],
    velocity_component: str = "vz",
    title: Optional[str] = None,
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Plot velocity component comparison across tracks.

    Args:
        t: Time array (must be the same for all tracks)
        tracks: Dictionary mapping track name to velocity array
        velocity_component: Name used in title ("vx", "vy", "vz", or "speed")
        title: Custom title (default auto-generated)

    Returns:
        Matplotlib figure
    """
    if title is None:
        title = f"Velocity Component: {velocity_component}"

    # Reuse plot_time_series_multiple
    track_dict = {name: (t, velocity) for name, velocity in tracks.items()}

    return plot_time_series_multiple(
        track_dict,
        ylabel=f"{velocity_component} (m/s)",
        title=title,
        xlabel="Time (s)",
        show_zero_line=True,
    )


# Simpler alternative for highly specific one-off plots
def quick_plot(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    marker: str = "",
    show: bool = True,
) -> plt.Figure:  # pyright: ignore[reportPrivateImportUsage]
    """Quick utility plot - one line, simple.

    Args:
        x, y: Data arrays
        title, xlabel, ylabel: Plot labels
        marker: Marker style ("" for line only)
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if marker:
        ax.plot(x, y, marker=marker, linewidth=2, markersize=5, alpha=0.8)
    else:
        ax.plot(x, y, linewidth=2, alpha=0.8)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return fig


# Export list
__all__ = [
    "setup_debug_style",
    "plot_3d_trajectory_compare",
    "plot_3d_trajectory_with_colors",
    "plot_time_series_multiple",
    "plot_3d_trajectory_plus_energy_overlay",
    "plot_velocity_comparison",
    "quick_plot",
]


if __name__ == "__main__":
    # Demo
    setup_debug_style()

    # Create demo data
    t = np.linspace(0, 10, 100)
    x_raw = t + 0.1 * np.random.randn(100)
    y_raw = np.sin(t) + 0.1 * np.random.randn(100)
    z_raw = -t + 0.1 * np.random.randn(100)

    # Simulate smoothed data
    x_s = t
    y_s = np.sin(t)
    z_s = -t

    print("Creating demo 3D trajectory comparison...")
    fig = plot_3d_trajectory_compare(
        t, x_raw, y_raw, z_raw, x_s, y_s, z_s, labels=("Raw Data", "Smoothed", "")
    )
    fig.savefig("/tmp/demo_3d_trajectory.png", dpi=150)
    print("Saved: /tmp/demo_3d_trajectory.png")

    # Simulate energy data
    energy = 0.5 * (t**2 + np.sin(t) ** 2 + (-t) ** 2)
    print("\nCreating demo with energy color...")
    fig2 = plot_3d_trajectory_with_colors(
        t, x_s, y_s, z_s, energy, "plasma", "3D Path Colored by Energy", "Energy (J/kg)"
    )
    fig2.savefig("/tmp/demo_energy_colored.png", dpi=150)
    print("Saved: /tmp/demo_energy_colored.png")

    # Multi-track
    tracks = {
        "Track 1": (t, energy),
        "Track 2": (t, energy * 0.8),
        "Track 3": (t, energy * 1.2),
    }
    print("\nCreating demo multi-track plot...")
    fig3 = plot_time_series_multiple(
        tracks, ylabel="Energy (J/kg)", title="Energy Comparison"
    )
    fig3.savefig("/tmp/demo_multi_track.png", dpi=150)
    print("Saved: /tmp/demo_multi_track.png")
