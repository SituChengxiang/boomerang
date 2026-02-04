#!/usr/bin/env python3
"""
Object-Oriented Visualization System for Boomerang Trajectory Analysis.

This module provides a comprehensive OOP-based visualization framework that
replaces the procedural plotting functions with encapsulated, reusable classes.

Key Features:
- Base visualization classes with common functionality
- Specialized visualizers for different analysis domains
- Consistent styling and data handling
- Easy extensibility for new visualization types
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

# Add project root to Python path for absolute imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.aerodynamics import (
    calculate_effective_coefficients,
    calculate_net_aerodynamic_force,
    decompose_aerodynamic_force,
    find_optimal_v_rot,
)
from src.utils.energy import calculate_energy_from_dataframe
from src.utils.kinematics import analyze_track
from src.utils.physicsCons import MASS


class TrackDataWrapper:
    """Data container class for track data with computed properties.

    This class wraps raw track data and provides computed properties
    for easy access in visualizations.
    """

    def __init__(self, df: pd.DataFrame, track_name: str = ""):
        """Initialize with DataFrame containing track data.

        Args:
            df: DataFrame with columns: t, vx, vy, vz, (x, y, z optional)
            track_name: Optional track name for identification
        """
        self.df = df
        self.track_name = track_name
        self._computed_data = None

    def __str__(self) -> str:
        return f"TrackDataWrapper({self.track_name})"

    def compute_analysis_data(self) -> tuple:
        """Compute and cache analysis data using kinematics utilities."""
        if self._computed_data is None:
            self._computed_data = analyze_track(self.df, self.track_name)
        return self._computed_data

    def compute_energy_data(self) -> tuple:
        """Compute energy data using energy utilities."""
        return calculate_energy_from_dataframe(self.df)

    @property
    def t(self) -> NDArray[np.float64]:
        """Time array."""
        return self.df.t.values # pyright: ignore[reportReturnType]

    @property
    def vx(self) -> NDArray[np.float64]:
        """X velocity array."""
        return self.df.vx.values # pyright: ignore[reportReturnType]

    @property
    def vy(self) -> NDArray[np.float64]:
        """Y velocity array."""
        return self.df.vy.values # pyright: ignore[reportReturnType]

    @property
    def vz(self) -> NDArray[np.float64]:
        """Z velocity array."""
        return self.df.vz.values # pyright: ignore[reportReturnType]

    @property
    def x(self) -> Optional[NDArray[np.float64]]:
        """X position array (if available)."""
        return self.df.x.values if "x" in self.df.columns else None # pyright: ignore[reportReturnType]

    @property
    def y(self) -> Optional[NDArray[np.float64]]:
        """Y position array (if available)."""
        return self.df.y.values if "y" in self.df.columns else None # pyright: ignore[reportReturnType]

    @property
    def z(self) -> Optional[NDArray[np.float64]]:
        """Z position array (if available)."""
        return self.df.z.values if "z" in self.df.columns else None # pyright: ignore[reportReturnType]


class VisualizationBase:
    """Base class for all visualizations with common functionality."""

    def __init__(self, title: str = "", figsize: Tuple[int, int] = (12, 8)):
        """Initialize base visualization.

        Args:
            title: Plot title
            figsize: Figure size
        """
        self.title = title
        self.figsize = figsize
        self._setup_styles()

    def _setup_styles(self) -> None:
        """Configure matplotlib styles for consistent appearance."""
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.labelsize": 14,
                "axes.titlesize": 16,
                "legend.fontsize": 11,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "axes.grid": True,
                "figure.figsize": self.figsize,
            }
        )

    def _create_figure(self, title: Optional[str] = None) -> Figure:
        """Create a new figure with consistent styling."""
        fig = plt.figure(figsize=self.figsize)
        if title or self.title:
            fig.suptitle(title or self.title, fontsize=16, fontweight="bold")
        return fig

    def _get_color_map(self, num_colors: int = 10):
        """Get a color map for multi-track plotting."""
        # Use appropriate tab colormap based on number of colors needed
        if num_colors <= 10:
            return plt.get_cmap("tab10")
        elif num_colors <= 20:
            return plt.get_cmap("tab20")
        else:
            return plt.get_cmap("tab20c")

    def show(self) -> None:
        """Display the plot."""
        plt.tight_layout()
        plt.show()

    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the plot to file."""
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        print(f"Saved: {filename}")


class TrajectoryVisualizer(VisualizationBase):
    """Specialized visualizer for 3D trajectory plots."""

    def __init__(self, title: str = "3D Trajectory Visualization"):
        super().__init__(title, figsize=(14, 10))

    def plot_3d_trajectory_compare(
        self,
        track_data: Dict[str, TrackDataWrapper],
        show_raw: bool = True,
        show_smooth: bool = True,
        show_fit: bool = True,
        show_final: bool = True,
    ) -> Figure:
        """Plot 3D trajectory comparison for multiple tracks.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects
            show_raw: Show raw data points
            show_smooth: Show smoothed trajectory
            show_fit: Show fitted trajectory
            show_final: Show final trimmed points

        Returns:
            Matplotlib figure
        """
        fig = self._create_figure("3D Trajectory Comparison")
        ax: Any = fig.add_subplot(111, projection="3d")
        color_map = self._get_color_map(len(track_data))

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)  # type:ignore

            x = track_wrapper.x
            y = track_wrapper.y
            z = track_wrapper.z
            if x is None or y is None or z is None:
                continue

            # Plot raw data
            if show_raw:
                ax.scatter(
                    x,
                    y,
                    z,
                    c=color,
                    s=20,
                    alpha=0.4,
                    marker="o",
                    edgecolors="none",
                    label=f"{track_name} Raw",
                )

            # TODO: Add smoothed and fitted data when available
            # This would come from the data processing pipeline

            # Mark start point
            ax.scatter(
                float(x[0]),
                float(y[0]),
                float(z[0]),
                c="green",
                s=150,
                marker="*",
                linewidth=1,
                label="Start",
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="best")
        ax.grid(True)

        return fig

    def plot_3d_trajectory_with_energy(
        self, track_data: Dict[str, TrackDataWrapper], color_map: str = "viridis"
    ) -> Figure:
        """Plot 3D trajectory with energy as color.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects
            color_map: Matplotlib color_map name

        Returns:
            Matplotlib figure
        """
        fig = self._create_figure("3D Trajectory - Energy Overlay")
        ax: Any = fig.add_subplot(111, projection="3d")

        for track_name, track_wrapper in track_data.items():
            x = track_wrapper.x
            y = track_wrapper.y
            z = track_wrapper.z
            if x is None or y is None or z is None:
                continue

            # Calculate energy
            t, total_energy, kinetic_energy, potential_energy = (
                track_wrapper.compute_energy_data()
            )

            # Simple approach: plot line with single color for now
            # (3D LineCollection with colors is complex, would need custom implementation)
            color = self._get_color_map(len(track_data))(0)  # type:ignore
            ax.plot(
                x,
                y,
                z,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=f"{track_name} Trajectory",
            )

            # Plot start and end points
            ax.scatter(
                float(x[0]),
                float(y[0]),
                float(z[0]),
                c="green",
                s=150,
                marker="*",
                edgecolors="black",
                linewidth=1,
                label=f"{track_name} Start",
            )
            ax.scatter(
                float(x[-1]),
                float(y[-1]),
                float(z[-1]),
                c="red",
                s=100,
                marker="o",
                edgecolors="black",
                linewidth=1,
                label=f"{track_name} End",
            )

        # Note: Colorbar removed since we're using single color for simplicity
        # For full energy coloring, a custom 3D LineCollection would be needed

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="best")
        ax.grid(True)

        return fig


class TimeSeriesVisualizer(VisualizationBase):
    """Specialized visualizer for time-series plots."""

    def __init__(self, title: str = "Time Series Visualization"):
        super().__init__(title, figsize=(12, 8))

    def plot_velocity_components(
        self,
        track_data: Dict[str, TrackDataWrapper],
        components: List[str] = ["vx", "vy", "vz"],
    ) -> Figure:
        """Plot velocity components vs time.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects
            components: List of velocity components to plot (vx, vy, vz)

        Returns:
            Matplotlib figure
        """
        num_components = len(components)
        fig, axes = plt.subplots(
            num_components, 1, figsize=(12, 4 * num_components), sharex=True
        )
        if num_components == 1:
            axes = [axes]

        color_map = self._get_color_map(len(track_data))

        # Collect data for consistent y-limits
        all_values = {comp: [] for comp in components}

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            for j, component in enumerate(components):
                if component == "vx":
                    values = track_wrapper.vx
                    ylabel = "$v_x$ (m/s)"
                elif component == "vy":
                    values = track_wrapper.vy
                    ylabel = "$v_y$ (m/s)"
                elif component == "vz":
                    values = track_wrapper.vz
                    ylabel = "$v_z$ (m/s)"
                else:
                    continue

                axes[j].plot(
                    track_wrapper.t,
                    values,
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"{track_name}" if j == 0 else "",
                )
                all_values[component].extend(values)

                if j == num_components - 1:
                    axes[j].set_xlabel("Time (s)")
                axes[j].set_ylabel(ylabel)
                axes[j].grid(True, alpha=0.5)
                axes[j].axhline(0, color="k", linestyle="--", alpha=0.3)

        # Set consistent y-limits
        for j, component in enumerate(components):
            if all_values[component]:
                axes[j].set_ylim(min(all_values[component]), max(all_values[component]))

        # Add legend to first subplot only
        if len(track_data) > 0:
            axes[0].legend(fontsize=8, loc="best")

        fig.suptitle("Velocity Components vs Time", fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig

    def plot_energy_components(self, track_data: Dict[str, TrackDataWrapper]) -> Figure:
        """Plot energy components (kinetic, potential, total) vs time.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        color_map = self._get_color_map(len(track_data))

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            # Calculate energy data
            t, total_energy, kinetic_energy, potential_energy = (
                track_wrapper.compute_energy_data()
            )

            # Plot total energy
            ax.plot(
                t,
                total_energy,
                color=color,
                linewidth=2,
                alpha=0.9,
                label=f"{track_name} Total",
            )

            # Plot kinetic and potential energy
            ax.plot(
                t,
                kinetic_energy,
                "--",
                color=color,
                alpha=0.7,
                linewidth=1.5,
                label=f"{track_name} Kinetic",
            )
            ax.plot(
                t,
                potential_energy,
                ":",
                color=color,
                alpha=0.7,
                linewidth=1.5,
                label=f"{track_name} Potential",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy per mass ($m^2/s^2$)")
        ax.set_title("Energy Components vs Time")
        ax.grid(True)
        ax.legend(fontsize=8, loc="best")

        return fig


class ForceAnalysisVisualizer(VisualizationBase):
    """Specialized visualizer for aerodynamic force analysis."""

    def __init__(self, title: str = "Force Analysis Visualization"):
        super().__init__(title, figsize=(14, 10))

    def plot_vertical_aero_vs_horizontal_speed(
        self, track_data: Dict[str, TrackDataWrapper]
    ) -> Figure:
        """Plot vertical aerodynamic force vs horizontal speed squared.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        color_map = self._get_color_map(len(track_data))

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            # Get analysis data
            analysis_data = track_wrapper.compute_analysis_data()
            t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp, power_per_mass, *_ = (
                analysis_data
            )

            ax.scatter(v_xy_sq, f_z_aero, s=5, alpha=0.5, label=track_name, color=color)

        ax.set_title("Vertical Aero Acceleration (Lift_z) vs $v_{xy}^2$")
        ax.set_xlabel("Horizontal Speed Squared ($m^2/s^2$)")
        ax.set_ylabel("Vertical Acceleration + G ($m/s^2$)")
        ax.grid(True)
        ax.axhline(9.8, color="k", linestyle="--", alpha=0.3, label="1G Hover")
        ax.legend(fontsize=8, loc="best")

        return fig

    def plot_aerodynamic_forces_decomposition(
        self, track_data: Dict[str, TrackDataWrapper]
    ) -> Figure:
        """Plot aerodynamic forces decomposition and effective coefficients.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))

        # Legend strategy: color encodes track, linestyle encodes component.
        color_map = self._get_color_map(len(track_data))
        track_handles: list[Line2D] = []
        for i, track_name in enumerate(track_data.keys()):
            track_handles.append(
                Line2D([0], [0], color=color_map(i % 10), lw=2, label=track_name)
            )
        component_handles = [
            Line2D([0], [0], color="k", lw=2, linestyle="--", label="tangential (t)"),
            Line2D([0], [0], color="k", lw=2, linestyle="-", label="h-normal (n)"),
            Line2D([0], [0], color="k", lw=2, linestyle=":", label="vertical (z)"),
        ]

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            # Extract kinematic data from the original dataframe
            acc = np.column_stack(
                (track_wrapper.df.ax, track_wrapper.df.ay, track_wrapper.df.az)
            )
            vel = np.column_stack(
                (track_wrapper.vx, track_wrapper.vy, track_wrapper.vz)
            )

            # Calculate aerodynamic forces
            F_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
            F_components = decompose_aerodynamic_force(F_aero, vel, speed)

            # Find optimal rotational velocity
            optimal_v_rot, coeffs = find_optimal_v_rot(F_components, speed)

            # Plot 1: Aerodynamic force components (tangential / horizontal-normal / vertical)
            axes[0].plot(
                track_wrapper.t,
                F_components["F_t"],
                linestyle="--",
                color=color,
                alpha=0.8,
            )
            axes[0].plot(
                track_wrapper.t,
                F_components["F_n"],
                linestyle="-",
                color=color,
                alpha=0.8,
            )
            axes[0].plot(
                track_wrapper.t,
                F_components["F_z"],
                linestyle=":",
                color=color,
                alpha=0.75,
            )
            axes[0].set_ylabel("Force (N)")
            axes[0].set_title("Aerodynamic Force Components")
            axes[0].grid(True)

            # Plot 2: Effective coefficients without rotation correction
            coeffs_no_rot = calculate_effective_coefficients(F_components, speed, 0.0)
            axes[1].plot(
                track_wrapper.t,
                coeffs_no_rot["C_n_eff"],
                linestyle="-",
                color=color,
                alpha=0.8,
            )
            axes[1].plot(
                track_wrapper.t,
                coeffs_no_rot["C_t_eff"],
                linestyle="--",
                color=color,
                alpha=0.8,
            )
            axes[1].plot(
                track_wrapper.t,
                coeffs_no_rot["C_z_eff"],
                linestyle=":",
                color=color,
                alpha=0.8,
            )
            axes[1].set_ylabel("Effective Coefficient")
            axes[1].set_title("Effective Coefficients (No Rotation Correction)")
            axes[1].grid(True)

            # Plot 3: Effective coefficients with optimal rotation correction
            axes[2].plot(
                track_wrapper.t,
                coeffs["C_n_eff"],
                linestyle="-",
                color=color,
                alpha=0.85,
            )
            axes[2].plot(
                track_wrapper.t,
                coeffs["C_t_eff"],
                linestyle="--",
                color=color,
                alpha=0.85,
            )
            axes[2].plot(
                track_wrapper.t,
                coeffs["C_z_eff"],
                linestyle=":",
                color=color,
                alpha=0.85,
            )
            axes[2].set_ylabel("Effective Coefficient")
            axes[2].set_title(
                f"Effective Coefficients (Optimal v_rot={optimal_v_rot:.2f} m/s)"
            )
            axes[2].grid(True)

            # Plot 4: Speed vs time for reference
            axes[3].plot(
                track_wrapper.t, speed, "-", alpha=0.7, label=f"{track_name} speed"
            )
            axes[3].set_ylabel("Speed (m/s)")
            axes[3].set_title("Speed vs Time")
            axes[3].grid(True)
            axes[3].set_xlabel("Time (s)")

        # Shared legends (avoid repeating long legend 3x and squeezing subplot height).
        fig.subplots_adjust(right=0.80)
        fig.legend(
            handles=track_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=8,
            frameon=False,
            title="Tracks",
        )
        axes[0].legend(
            handles=component_handles,
            loc="upper left",
            fontsize=8,
            frameon=False,
            title="Components",
        )
        plt.tight_layout(rect=(0, 0, 0.80, 1))
        return fig

    def plot_effective_coefficients_vs_speed(
        self,
        track_data: Dict[str, TrackDataWrapper],
        min_speed: float = 0.2,
        n_bins: int = 20,
    ) -> Figure:
        """Validation plot: effective coefficients vs speed.

        This is the most direct diagnostic for the low-speed strong-turn stage:
        if q uses only v^2, coefficients often "blow up" near low speed;
        rotational correction (v^2+v_rot^2) should flatten them if it is physically needed.

        Args:
            track_data: mapping of track name -> TrackDataWrapper
            min_speed: minimum speed threshold for plotting
            n_bins: number of bins for a binned-median trend line

        Returns:
            Matplotlib figure
        """

        def binned_median(x: np.ndarray, y: np.ndarray, bins: int):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if x.size < 10:
                return np.array([]), np.array([])
            edges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            meds = np.full_like(centers, np.nan, dtype=float)
            for i in range(centers.size):
                m = (x >= edges[i]) & (x < edges[i + 1])
                if np.any(m):
                    meds[i] = float(np.nanmedian(y[m]))
            keep = np.isfinite(meds)
            return centers[keep], meds[keep]

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axes[0].set_title("Effective Coefficients vs Speed (low-speed validation)")

        color_map = self._get_color_map(len(track_data))

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            acc = np.column_stack(
                (track_wrapper.df.ax, track_wrapper.df.ay, track_wrapper.df.az)
            )
            vel = np.column_stack((track_wrapper.vx, track_wrapper.vy, track_wrapper.vz))

            F_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
            F_components = decompose_aerodynamic_force(F_aero, vel, speed)

            # No-rotation coefficients
            coeffs0 = calculate_effective_coefficients(F_components, speed, 0.0)

            # Optimal-rotation coefficients (per-track)
            vrot_opt, _ = find_optimal_v_rot(F_components, speed)
            coeffs1 = calculate_effective_coefficients(F_components, speed, vrot_opt)

            mask = np.isfinite(speed) & (speed >= float(min_speed))
            sp = speed[mask]

            for ax, key, label in (
                (axes[0], "C_n_eff", "C_n_eff"),
                (axes[1], "C_z_eff", "C_z_eff"),
                (axes[2], "C_t_eff", "C_t_eff"),
            ):
                y0 = coeffs0[key][mask]
                y1 = coeffs1[key][mask]

                ax.scatter(sp, y0, s=8, alpha=0.18, color=color, label="_nolegend_")
                ax.scatter(sp, y1, s=10, alpha=0.18, color=color, marker="x", label="_nolegend_")

                cx0, my0 = binned_median(sp, y0, n_bins)
                cx1, my1 = binned_median(sp, y1, n_bins)
                if cx0.size:
                    ax.plot(cx0, my0, color=color, alpha=0.6, linewidth=1.5, label=f"{track_name} {label} (no rot)" if key == "C_n_eff" else "_nolegend_")
                if cx1.size:
                    ax.plot(cx1, my1, color=color, alpha=0.9, linewidth=1.8, linestyle="--", label=f"{track_name} {label} (v_rot={vrot_opt:.2f})" if key == "C_n_eff" else "_nolegend_")

        axes[0].set_ylabel("C_n_eff")
        axes[1].set_ylabel("C_z_eff")
        axes[2].set_ylabel("C_t_eff")
        axes[2].set_xlabel("Speed |v| (m/s)")

        for ax in axes:
            ax.grid(True, alpha=0.3)

        # Legend only on first axis to avoid clutter
        axes[0].legend(fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        return fig

    def plot_centripetal_force_closure(
        self,
        track_data: Dict[str, TrackDataWrapper],
        min_horizontal_speed: float = 0.3,
    ) -> Figure:
        """Closed-loop check: required centripetal force vs inferred F_n.

        In the horizontal plane, gravity has no component, so the horizontal
        normal aerodynamic force should satisfy (approximately):
            F_n \approx m * a_{\perp,h}
        and also
            F_n \approx m * v_h * \dot\psi

        Args:
            track_data: mapping of track name -> TrackDataWrapper
            min_horizontal_speed: mask out samples with very small v_h to avoid
                unstable direction/heading-rate estimates.

        Returns:
            Matplotlib figure
        """

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].set_title("Centripetal Force Closure: $F_n$ vs $m a_{\\perp,h}$ / $m v_h \\dot\\psi$")

        color_map = self._get_color_map(len(track_data))
        track_handles: list[Line2D] = []
        for i, track_name in enumerate(track_data.keys()):
            track_handles.append(
                Line2D([0], [0], color=color_map(i % 10), lw=2, label=track_name)
            )

        component_handles = [
            Line2D([0], [0], color="k", lw=2, linestyle="-", label="$F_n$ (from $F_{aero}$)"),
            Line2D([0], [0], color="k", lw=2, linestyle="--", label="$m a_{\\perp,h}$"),
            Line2D([0], [0], color="k", lw=2, linestyle=":", label="$m v_h \\dot\\psi$"),
        ]

        all_fn: list[np.ndarray] = []
        all_req: list[np.ndarray] = []

        for i, (track_name, track_wrapper) in enumerate(track_data.items()):
            color = color_map(i % 10)

            analysis = track_wrapper.compute_analysis_data()
            t = np.asarray(analysis[0], dtype=float)
            a_perp_signed = np.asarray(analysis[8], dtype=float)
            v_h = np.asarray(analysis[7], dtype=float)
            heading_rate_from_cross_deg = analysis[14]
            if heading_rate_from_cross_deg is None:
                psi_dot = np.full_like(v_h, np.nan, dtype=float)
            else:
                psi_dot = np.deg2rad(np.asarray(heading_rate_from_cross_deg, dtype=float))

            acc = np.column_stack(
                (track_wrapper.df.ax, track_wrapper.df.ay, track_wrapper.df.az)
            )
            vel = np.column_stack((track_wrapper.vx, track_wrapper.vy, track_wrapper.vz))

            F_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
            F_components = decompose_aerodynamic_force(F_aero, vel, speed)
            F_n = np.asarray(F_components["F_n"], dtype=float)

            req1 = MASS * a_perp_signed
            req2 = MASS * v_h * psi_dot

            mask = (
                np.isfinite(t)
                & np.isfinite(F_n)
                & np.isfinite(req1)
                & (np.isfinite(v_h))
                & (v_h >= float(min_horizontal_speed))
            )

            # Time-series overlay
            axes[0].plot(t[mask], F_n[mask], linestyle="-", color=color, alpha=0.85)
            axes[0].plot(t[mask], req1[mask], linestyle="--", color=color, alpha=0.75)
            axes[0].plot(t[mask], req2[mask], linestyle=":", color=color, alpha=0.65)

            # Scatter for closure
            axes[1].scatter(
                req1[mask],
                F_n[mask],
                s=10,
                alpha=0.20,
                color=color,
                label="_nolegend_",
            )

            all_fn.append(F_n[mask])
            all_req.append(req1[mask])

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Force (N)")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Scatter: $F_n$ vs $m a_{\\perp,h}$ (ideal: y=x)")
        axes[1].set_xlabel("$m a_{\\perp,h}$ (N)")
        axes[1].set_ylabel("$F_n$ (N)")
        axes[1].grid(True, alpha=0.3)

        # y=x reference line over combined range
        if all_fn and all_req:
            x_all = np.concatenate(all_req)
            y_all = np.concatenate(all_fn)
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            if np.any(finite):
                lo = float(np.nanpercentile(np.concatenate([x_all[finite], y_all[finite]]), 2))
                hi = float(np.nanpercentile(np.concatenate([x_all[finite], y_all[finite]]), 98))
                axes[1].plot([lo, hi], [lo, hi], color="k", linestyle="--", alpha=0.4)
                axes[1].set_xlim(lo, hi)
                axes[1].set_ylim(lo, hi)

        # Shared legends
        fig.subplots_adjust(right=0.80)
        fig.legend(
            handles=track_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=8,
            frameon=False,
            title="Tracks",
        )
        axes[0].legend(
            handles=component_handles,
            loc="upper left",
            fontsize=8,
            frameon=False,
            title="Signals",
        )

        plt.tight_layout(rect=(0, 0, 0.80, 1))
        return fig


class CompositeVisualizer:
    """Composite visualizer that combines multiple visualization types."""

    def __init__(self):
        """Initialize composite visualizer."""
        self.visualizers = []

    def add_visualizer(self, visualizer: VisualizationBase) -> None:
        """Add a visualizer to the composite.

        Args:
            visualizer: Visualization object to add
        """
        self.visualizers.append(visualizer)

    def generate_all_plots(
        self, track_data: Dict[str, TrackDataWrapper], output_dir: str = "out/"
    ) -> None:
        """Generate all plots from all visualizers.

        Args:
            track_data: Dictionary mapping track names to TrackDataWrapper objects
            output_dir: Directory to save plots
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        for i, visualizer in enumerate(self.visualizers):
            print(f"\nGenerating plots from {visualizer.__class__.__name__}...")

            # Generate appropriate plots based on visualizer type
            if isinstance(visualizer, TrajectoryVisualizer):
                fig1 = visualizer.plot_3d_trajectory_compare(track_data)
                fig1.savefig(f"{output_dir}trajectory_compare_{i}.png", dpi=150)
                plt.close(fig1)

                fig2 = visualizer.plot_3d_trajectory_with_energy(track_data)
                fig2.savefig(f"{output_dir}trajectory_energy_{i}.png", dpi=150)
                plt.close(fig2)

            elif isinstance(visualizer, TimeSeriesVisualizer):
                fig1 = visualizer.plot_velocity_components(track_data)
                fig1.savefig(f"{output_dir}velocity_components_{i}.png", dpi=150)
                plt.close(fig1)

                fig2 = visualizer.plot_energy_components(track_data)
                fig2.savefig(f"{output_dir}energy_components_{i}.png", dpi=150)
                plt.close(fig2)

            elif isinstance(visualizer, ForceAnalysisVisualizer):
                fig1 = visualizer.plot_vertical_aero_vs_horizontal_speed(track_data)
                fig1.savefig(f"{output_dir}vertical_aero_{i}.png", dpi=150)
                plt.close(fig1)

                fig2 = visualizer.plot_aerodynamic_forces_decomposition(track_data)
                fig2.savefig(f"{output_dir}force_decomposition_{i}.png", dpi=150)
                plt.close(fig2)

                fig3 = visualizer.plot_effective_coefficients_vs_speed(track_data)
                fig3.savefig(f"{output_dir}coeffs_vs_speed_{i}.png", dpi=150)
                plt.close(fig3)

                fig4 = visualizer.plot_centripetal_force_closure(track_data)
                fig4.savefig(f"{output_dir}centripetal_closure_{i}.png", dpi=150)
                plt.close(fig4)

        print(f"\nAll plots saved to {output_dir}")


# Convenience function to create a standard visualization suite
def create_standard_visualization_suite() -> CompositeVisualizer:
    """Create a composite visualizer with standard visualization components.

    Returns:
        CompositeVisualizer with standard visualizers
    """
    composite = CompositeVisualizer()

    # Add standard visualizers
    composite.add_visualizer(TrajectoryVisualizer())
    composite.add_visualizer(TimeSeriesVisualizer())
    composite.add_visualizer(ForceAnalysisVisualizer())

    return composite


if __name__ == "__main__":
    # Demo and testing
    print("Testing OOP Visualization System...")

    # Create some dummy data for testing
    t = np.linspace(0, 10, 100)
    x = t + 0.1 * np.random.randn(100)
    y = np.sin(t) + 0.1 * np.random.randn(100)
    z = -t + 0.1 * np.random.randn(100)
    vx = np.cos(t) + 0.05 * np.random.randn(100)
    vy = np.sin(t) + 0.05 * np.random.randn(100)
    vz = -0.5 * np.ones(100) + 0.05 * np.random.randn(100)

    # Create dummy DataFrame
    df = pd.DataFrame(
        {
            "t": t,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "ax": -np.sin(t) + 0.01 * np.random.randn(100),
            "ay": np.cos(t) + 0.01 * np.random.randn(100),
            "az": 0.01 * np.random.randn(100),
        }
    )

    # Create track data wrapper
    track_data = {"test_track": TrackDataWrapper(df, "Test Track")}

    # Test individual visualizers
    print("\n1. Testing TrajectoryVisualizer...")
    traj_vis = TrajectoryVisualizer()
    fig1 = traj_vis.plot_3d_trajectory_compare(track_data)
    fig1.savefig("/tmp/test_trajectory_compare.png", dpi=150)
    plt.close(fig1)
    print("✓ Trajectory comparison plot saved")

    print("\n2. Testing TimeSeriesVisualizer...")
    time_vis = TimeSeriesVisualizer()
    fig2 = time_vis.plot_velocity_components(track_data)
    fig2.savefig("/tmp/test_velocity_components.png", dpi=150)
    plt.close(fig2)
    print("✓ Velocity components plot saved")

    print("\n3. Testing ForceAnalysisVisualizer...")
    force_vis = ForceAnalysisVisualizer()
    fig3 = force_vis.plot_vertical_aero_vs_horizontal_speed(track_data)
    fig3.savefig("/tmp/test_vertical_aero.png", dpi=150)
    plt.close(fig3)
    print("✓ Vertical aero plot saved")

    print("\n4. Testing CompositeVisualizer...")
    composite = create_standard_visualization_suite()
    composite.generate_all_plots(track_data, "/tmp/test_composite_")
    print("✓ Composite visualization test completed")

    print("\n All OOP visualization tests completed successfully!")
    print("Plots saved to /tmp/ directory")
