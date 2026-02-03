#!/usr/bin/env python3
"""
Visualize Data
Reads velocity.csv, computes accelerations, and plots forces to validate physical assumptions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.utils.aerodynamics import (
    calculate_effective_coefficients,
    calculate_net_aerodynamic_force,
    decompose_aerodynamic_force,
    find_optimal_v_rot,
)
from src.utils.energy import calculate_energy_from_dataframe
from src.utils.kinematics import analyze_track

# Use shared constants and utilities
from src.utils.physicsCons import MASS


def plot_vertical_aero_vs_horizontal_speed(tracks_data):
    """Plot 1: Vertical Aero Force vs Horizontal Speed Squared"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp, power_per_mass, *_ = data
        color = cmap(i % 10)
        ax.scatter(v_xy_sq, f_z_aero, s=5, alpha=0.5, label=track, color=color)

    ax.set_title("Vertical Aero Acceleration (Lift_z) vs $v_{xy}^2$")
    ax.set_xlabel("Horizontal Speed Squared ($m^2/s^2$)")
    ax.set_ylabel("Vertical Acceleration + G ($m/s^2$)")
    ax.grid(True)
    ax.axhline(9.8, color="k", linestyle="--", alpha=0.3, label="1G Hover")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_drag_deceleration_vs_total_speed(tracks_data):
    """Plot 2: Drag Deceleration vs Total Speed Squared"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp, power_per_mass, *_ = data
        color = cmap(i % 10)
        ax.scatter(v_total_sq, -a_drag_est, s=5, alpha=0.5, color=color, label=track)

    ax.set_title("Drag Deceleration vs $v_{total}^2$")
    ax.set_xlabel("Total Speed Squared ($m^2/s^2$)")
    ax.set_ylabel("Drag Deceleration ($m/s^2$)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_vertical_velocity_vs_time(tracks_data, df_all):
    """Plot 3: Vertical Velocity vs Time"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, track in enumerate(tracks_data.keys()):
        df_track = df_all[df_all.track == track].sort_values(by="t")  # type: ignore[attr-defined]
        if len(df_track) < 10:
            continue
        color = cmap(i % 10)
        ax.plot(df_track.t, df_track.vz, label=track, color=color)

    ax.set_title("Vertical Velocity ($v_z$) vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("$v_z$ (m/s)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_vertical_aero_force_vs_time(tracks_data):
    """Plot 4: Vertical Aero Force vs Time"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp, power_per_mass, *_ = data
        color = cmap(i % 10)
        ax.plot(t, f_z_aero, color=color, label=track)

    ax.set_title("Vertical Aero Force (Lift) vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lift Acceleration ($m/s^2$)")
    ax.grid(True)
    ax.axhline(9.8, color="k", linestyle="--", label="Gravity Offset")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_mechanical_energy_vs_time(energy_data):
    """Plot 5: Mechanical Energy vs Time"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(energy_data.items()):
        t, total_energy, kinetic_energy, potential_energy = data
        color = cmap(i % 10)
        ax.plot(t, total_energy, label=track, color=color, linewidth=1.5)

    ax.set_title("Mechanical Energy vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy($m^2/s^2$)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_energy_components_vs_time(energy_data):
    """Plot 6: Energy Components (Kinetic and Potential) vs Time"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(energy_data.items()):
        t, total_energy, kinetic_energy, potential_energy = data
        color = cmap(i % 10)
        ax.plot(
            t,
            kinetic_energy,
            "--",
            alpha=0.7,
            color=color,
            linewidth=1,
            label=f"{track} (Kinetic)",
        )
        ax.plot(
            t,
            potential_energy,
            ":",
            alpha=0.7,
            color=color,
            linewidth=1,
            label=f"{track} (Potential)",
        )

    ax.set_title("Energy Components: Kinetic and Potential vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy per mass ($m^2/s^2$)")
    ax.grid(True)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_energy_change_rate(energy_data):
    """Plot 7: Energy Change Rate vs Time"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(energy_data.items()):
        t, total_energy, kinetic_energy, potential_energy = data
        color = cmap(i % 10)

        # Calculate energy change rate (dE/dt)
        # dt = np.mean(np.diff(t))
        dE_dt = np.gradient(total_energy, t)

        # Smooth if possible
        window = min(11, len(t))
        if window % 2 == 0:
            window -= 1
        if window >= 5:
            dE_dt_smooth = savgol_filter(dE_dt, window, 3)
        else:
            dE_dt_smooth = dE_dt

        ax.plot(t, dE_dt_smooth, label=track, color=color, linewidth=1.5)

    ax.set_title("Energy Change Rate (dE/dt) vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dE/dt per mass ($m^2/s^3$)")
    ax.grid(True)
    ax.axhline(
        0, color="k", linestyle="--", alpha=0.3, label="Energy Conservation Line"
    )
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.show()


def plot_all_velocity_components_vs_time(tracks_data, df_all):
    """Plot 8: All Velocity Components (vx, vy, vz) vs Time"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    cmap = plt.get_cmap("tab10")

    # Collect all velocities to get consistent y-limits
    all_vx, all_vy, all_vz = [], [], []

    for i, track in enumerate(tracks_data.keys()):
        df_track = df_all[df_all.track == track].sort_values(by="t")  # type: ignore[attr-defined]
        if len(df_track) < 10:
            continue
        color = cmap(i % 10)

        # Plot vx
        axes[0].plot(df_track.t, df_track.vx, label=track, color=color, linewidth=1.5)
        all_vx.extend(df_track.vx)
        # Plot vy
        axes[1].plot(df_track.t, df_track.vy, label=track, color=color, linewidth=1.5)
        all_vy.extend(df_track.vy)
        # Plot vz
        axes[2].plot(df_track.t, df_track.vz, label=track, color=color, linewidth=1.5)
        all_vz.extend(df_track.vz)

    # Style settings for better readability
    axes[0].set_title("Velocity Components vs Time", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("$v_x$ (m/s)", fontsize=11)
    axes[0].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[0].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    axes[1].set_ylabel("$v_y$ (m/s)", fontsize=11)
    axes[1].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[1].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_ylabel("$v_z$ (m/s)", fontsize=11)
    axes[2].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[2].axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=1)
    axes[2].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    # Add y-axis bounds for clarity
    if all_vx and all_vy and all_vz:
        axes[0].set_ylim(min(all_vx), max(all_vx))
        axes[1].set_ylim(min(all_vy), max(all_vy))
        axes[2].set_ylim(min(all_vz), max(all_vz))

    plt.tight_layout()
    plt.show()


def plot_pitch_angle_vs_time(tracks_data, df_all):
    """Plot 10: Pitch Angle (θ) vs Time, where tan(θ) = vz / sqrt(vx² + vy²)"""
    _, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    valid_tracks = []
    for track in tracks_data.keys():
        df_track = df_all[df_all.track == track].sort_values(by="t")
        if len(df_track) < 10:
            continue

        vx = df_track.vx.values
        vy = df_track.vy.values
        vz = df_track.vz.values

        # Calculate horizontal speed
        v_horiz = np.sqrt(
            vx**2 + vy**2 + 1e-10
        )  # Add small epsilon to avoid division by zero

        # Calculate pitch angle in radians: θ = arctan(vz / v_horiz)
        theta = np.arctan2(vz, v_horiz)

        # Convert to degrees for display
        theta_deg = np.degrees(theta)

        color = cmap(len(valid_tracks) % 10)
        ax.plot(df_track.t, theta_deg, label=track, color=color, linewidth=1.5)
        valid_tracks.append(track)

    ax.set_title("Pitch Angle vs Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Pitch Angle θ (degrees)", fontsize=11)
    ax.grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=1, label="Horizontal")
    ax.legend(fontsize=8, loc="best", ncol=2, framealpha=0.9)

    plt.tight_layout()
    plt.show()


def plot_all_acceleration_components_vs_time(tracks_data, df_all):
    """Plot 9: All Acceleration Components (ax, ay, az) vs Time for multiple tracks"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    cmap = plt.get_cmap("tab10")

    # Collect all velocities to get consistent y-limits
    all_ax, all_ay, all_az = [], [], []

    for i, track in enumerate(tracks_data.keys()):
        df_track = df_all[df_all.track == track].sort_values(by="t")  # type: ignore[attr-defined]
        if len(df_track) < 10:
            continue
        color = cmap(i % 10)

        # Plot ax
        axes[0].plot(df_track.t, df_track.ax, label=track, color=color, linewidth=1.5)
        all_ax.extend(df_track.ax)
        # Plot ay
        axes[1].plot(df_track.t, df_track.ay, label=track, color=color, linewidth=1.5)
        all_ay.extend(df_track.ay)
        # Plot az
        axes[2].plot(df_track.t, df_track.az, label=track, color=color, linewidth=1.5)
        all_az.extend(df_track.az)

    # Style settings for better readability
    axes[0].set_title("Acceleration Components vs Time", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("$a_x$ (m/s)", fontsize=11)
    axes[0].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[0].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    axes[1].set_ylabel("$a_y$ (m/s)", fontsize=11)
    axes[1].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[1].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_ylabel("$a_z$ (m/s)", fontsize=11)
    axes[2].grid(True, alpha=0.5, linestyle="-", linewidth=0.5)
    axes[2].axhline(0, color="k", linestyle="--", alpha=0.3, linewidth=1)
    axes[2].legend(fontsize=7, loc="best", ncol=2, framealpha=0.9)

    # Add y-axis bounds for clarity
    if all_ax and all_ay and all_az:
        axes[0].set_ylim(min(all_ax), max(all_ax))
        axes[1].set_ylim(min(all_ay), max(all_ay))
        axes[2].set_ylim(min(all_az), max(all_az))

    plt.tight_layout()
    plt.show()


def plot_perpendicular_acceleration_vs_time(tracks_data):
    """Plot Perpendicular Acceleration (Centripetal Acceleration) vs Time"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax_vh = ax.twinx()
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        (
            t,
            a_perp_h,
            v_h,
            a_perp_h_curv,
        ) = data
        color = cmap(i % 10)
        ax.plot(
            t, a_perp_h, label=f"{track} (v×a)", color=color, alpha=0.75, linewidth=1.5
        )
        if a_perp_h_curv is not None:
            ax.plot(
                t,
                a_perp_h_curv,
                color=color,
                linestyle=":",
                alpha=0.75,
                linewidth=2.0,
                label=f"{track} (curvature)",
            )

        # Overlay horizontal speed on right axis to diagnose sensitivity when v_h is small
        label_vh = r"$v_h$ (right axis)" if i == 0 else "_nolegend_"
        ax_vh.plot(
            t,
            v_h,
            color=color,
            linestyle="--",
            alpha=0.22,
            linewidth=1.0,
            label=label_vh,
        )

    ax.set_title(
        "Horizontal Centripetal Acceleration (Two Estimates) & Horizontal Speed"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Perpendicular Acceleration ($m/s^2$)")
    ax_vh.set_ylabel(r"Horizontal Speed $v_h$ (m/s)")
    ax.grid(True, alpha=0.3)
    # Unified legend (avoid duplicating per-track v_h entries)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_vh.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    ax.set_ylim(bottom=0)  # Perpendicular acceleration should be non-negative

    # Add explanation text
    ax.text(
        0.02,
        0.98,
        r"(1) $a_{\perp,h}=|v_x a_y - v_y a_x|/v_h$   (2) curvature from $x(t),y(t)$",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_power_vs_time(tracks_data):
    """Plot Power vs Time"""
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp_h, power_per_mass, *_ = (
            data
        )
        color = cmap(i % 10)
        # Convert power per mass to actual power (W)
        power = MASS * power_per_mass
        ax.plot(t, power, label=track, color=color, alpha=0.7, linewidth=1.5)

    ax.set_title("Power vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="k", linestyle="--", alpha=0.5, linewidth=0.8)

    # Add explanation text
    ax.text(
        0.02,
        0.98,
        r"$P = \mathbf{F} \cdot \mathbf{v} = m(\mathbf{a} \cdot \mathbf{v} + g v_z)$",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


def plot_heading_rate_consistency(tracks_data, df_all, max_tracks: int = 4):
    """Validation: heading rate computed three ways should agree.

    - dpsi/dt from unwrapped atan2(vy, vx)
    - dpsi/dt from (vx*ay - vy*ax)/(vx^2+vy^2)
    - dpsi/dt from x(t),y(t) derivatives (if available)

    This helps determine whether spikes in a_perp,h are physical or a low-v_h amplification artifact.
    """
    tracks = sorted(list(tracks_data.keys()))[: max(1, int(max_tracks))]
    if not tracks:
        return

    _, axes = plt.subplots(
        len(tracks), 1, figsize=(12, 3.4 * len(tracks)), sharex=False
    )
    if len(tracks) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for idx, track in enumerate(tracks):
        ax = axes[idx]
        df_track = df_all[df_all.track == track].sort_values(by="t")  # type: ignore[attr-defined]
        if len(df_track) < 10:
            continue
        data = tracks_data[track]
        (
            t,
            v_h,
            heading_rate_deg,
            heading_rate_from_cross_deg,
            heading_rate_xy_deg,
        ) = data

        color = cmap(idx % 10)
        ax.plot(
            t,
            heading_rate_deg,
            color=color,
            linewidth=1.6,
            label=r"$d\psi/dt$ from heading(v)",
        )
        ax.plot(
            t,
            heading_rate_from_cross_deg,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label=r"$d\psi/dt=(v_x a_y-v_y a_x)/(v_x^2+v_y^2)$",
        )
        if heading_rate_xy_deg is not None:
            ax.plot(
                t,
                heading_rate_xy_deg,
                color=color,
                linestyle=":",
                linewidth=2.0,
                alpha=0.9,
                label=r"$d\psi/dt$ from heading(x',y')",
            )

        ax2 = ax.twinx()
        ax2.plot(t, v_h, color="k", alpha=0.18, linewidth=1.0, label=r"$v_h$")
        ax2.set_ylabel(r"$v_h$ (m/s)")
        ax.set_title(f"Heading rate consistency: {track}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(r"Heading rate (deg/s)")
        ax.grid(True, alpha=0.3)
        # Merge legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")

    plt.tight_layout()
    plt.show()


def plot_aerodynamic_forces_analysis(tracks_data):
    """Plot aerodynamic forces decomposition and effective coefficients."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    for track_name, data in tracks_data.items():
        # Extract kinematic data
        acc = np.column_stack((data["ax"], data["ay"], data["az"]))
        vel = np.column_stack((data["vx"], data["vy"], data["vz"]))

        # Calculate aerodynamic forces
        F_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
        F_components = decompose_aerodynamic_force(F_aero, vel, speed)

        # Find optimal rotational velocity
        optimal_v_rot, coeffs = find_optimal_v_rot(F_components, speed)

        # Plot 1: Aerodynamic force components
        axes[0].plot(
            data["t"], F_components["F_t"], "--", alpha=0.7, label=f"{track_name} F_t"
        )
        axes[0].plot(
            data["t"], F_components["F_n"], "-", alpha=0.7, label=f"{track_name} F_n"
        )
        axes[0].set_ylabel("Force (N)")
        axes[0].set_title("Aerodynamic Force Components: Tangential vs Normal")
        axes[0].grid(True)
        axes[0].legend(fontsize=8)

        # Plot 2: Effective coefficients without rotation correction
        coeffs_no_rot = calculate_effective_coefficients(F_components, speed, 0.0)
        axes[1].plot(
            data["t"],
            coeffs_no_rot["C_n_eff"],
            "-",
            alpha=0.7,
            label=f"{track_name} C_n",
        )
        axes[1].plot(
            data["t"],
            coeffs_no_rot["C_t_eff"],
            "--",
            alpha=0.7,
            label=f"{track_name} C_t",
        )
        axes[1].set_ylabel("Effective Coefficient")
        axes[1].set_title("Effective Coefficients (No Rotation Correction)")
        axes[1].grid(True)

        # Plot 3: Effective coefficients with optimal rotation correction
        axes[2].plot(
            data["t"],
            coeffs["C_n_eff"],
            "-",
            alpha=0.7,
            label=f"{track_name} C_n (v_rot={optimal_v_rot:.2f})",
        )
        axes[2].plot(
            data["t"],
            coeffs["C_t_eff"],
            "--",
            alpha=0.7,
            label=f"{track_name} C_t (v_rot={optimal_v_rot:.2f})",
        )
        axes[2].set_ylabel("Effective Coefficient")
        axes[2].set_title(
            f"Effective Coefficients (Optimal v_rot={optimal_v_rot:.2f} m/s)"
        )
        axes[2].grid(True)

        # Plot 4: Speed vs time for reference
        axes[3].plot(data["t"], speed, "-", alpha=0.7, label=f"{track_name} speed")
        axes[3].set_ylabel("Speed (m/s)")
        axes[3].set_title("Speed vs Time")
        axes[3].grid(True)
        axes[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def plot_rotational_correction_scan(tracks_data, v_rot_range=np.linspace(0, 5, 20)):
    """Plot how rotational correction affects coefficient flatness."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for track_name, data in tracks_data.items():
        # Extract kinematic data
        acc = np.column_stack((data["ax"], data["ay"], data["az"]))
        vel = np.column_stack((data["vx"], data["vy"], data["vz"]))

        # Calculate aerodynamic forces
        F_aero, speed = calculate_net_aerodynamic_force(acc, vel, MASS)
        F_components = decompose_aerodynamic_force(F_aero, vel, speed)

        # Calculate coefficients for different v_rot values
        v_rot_values = []
        c_n_variances = []
        c_t_variances = []

        for v_rot in v_rot_range:
            coeffs = calculate_effective_coefficients(F_components, speed, v_rot)
            v_rot_values.append(v_rot)
            c_n_variances.append(np.var(coeffs["C_n_eff"]))
            c_t_variances.append(np.var(coeffs["C_t_eff"]))

        # Find optimal v_rot
        total_variances = np.array(c_n_variances) + np.array(c_t_variances)
        optimal_idx = np.argmin(total_variances)
        optimal_v_rot = v_rot_range[optimal_idx]

        # Plot coefficient variances vs v_rot
        axes[0, 0].plot(
            v_rot_values, c_n_variances, "-", alpha=0.7, label=f"{track_name} C_n var"
        )
        axes[0, 1].plot(
            v_rot_values, c_t_variances, "-", alpha=0.7, label=f"{track_name} C_t var"
        )

        # Plot total variance and mark optimal point
        axes[1, 0].plot(
            v_rot_values,
            total_variances,
            "-",
            alpha=0.7,
            label=f"{track_name} total var",
        )
        axes[1, 0].scatter(
            [optimal_v_rot],
            [total_variances[optimal_idx]],
            color="red",
            marker="x",
            s=100,
            label=f"{track_name} optimal",
        )

        # Plot coefficients at optimal v_rot
        coeffs_optimal = calculate_effective_coefficients(
            F_components, speed, optimal_v_rot
        )
        axes[1, 1].plot(
            data["t"],
            coeffs_optimal["C_n_eff"],
            "-",
            alpha=0.7,
            label=f"{track_name} C_n (v_rot={optimal_v_rot:.2f})",
        )
        axes[1, 1].plot(
            data["t"],
            coeffs_optimal["C_t_eff"],
            "--",
            alpha=0.7,
            label=f"{track_name} C_t (v_rot={optimal_v_rot:.2f})",
        )

    # Set titles and labels
    axes[0, 0].set_title("C_n Variance vs Rotational Velocity")
    axes[0, 0].set_ylabel("Variance")
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("C_t Variance vs Rotational Velocity")
    axes[0, 1].set_ylabel("Variance")
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Total Variance vs Rotational Velocity")
    axes[1, 0].set_ylabel("Total Variance")
    axes[1, 0].set_xlabel("v_rot (m/s)")
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Optimal Coefficients")
    axes[1, 1].set_ylabel("Coefficient Value")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def main():
    try:
        df_all = pd.read_csv("data/interm/velocity.csv")
    except FileNotFoundError:
        print("Error: data/velocity.csv not found.")
        return

    tracks = df_all.track.unique()

    # Prepare data for all tracks
    tracks_data = {}
    energy_data = {}

    print(f"Analyzing {len(tracks)} tracks...")

    for track in tracks:
        df_track = df_all[df_all.track == track].sort_values(by="t")  # type: ignore[attr-defined]
        if len(df_track) < 10:
            continue

        # Analyze track for force validation
        track_result = analyze_track(df_track, track)
        tracks_data[track] = track_result

        # Calculate energy for energy conservation validation
        t, total_energy, kinetic_energy, potential_energy = (
            calculate_energy_from_dataframe(df_track)
        )
        energy_data[track] = {
            "t": t,
            "total_energy": total_energy,
            "kinetic_energy": kinetic_energy,
            "potential_energy": potential_energy,
        }

    # Print energy conservation analysis
    print("\n=== Energy Conservation Analysis ===")
    print("Ideal case (no air resistance):")
    print("  - Total mechanical energy E/m should remain constant")
    print("  - Energy change rate dE/dt should be close to 0")
    print("Real case (with air resistance):")
    print("  - Total mechanical energy E/m should gradually decrease over time")
    print("  - Energy change rate dE/dt should be negative (energy dissipation)")
    print("  - There should be clear conversion between kinetic and potential energy")

    # Plot each figure separately
    print("\n=== Generating Plots ===")
    print("Close each window to see the next plot...")

    # Plot 1: Aerodynamic Forces Analysis (NEW)
    print("\n1. Aerodynamic Forces Analysis")
    plot_aerodynamic_forces_analysis(tracks_data)

    # Plot 2: Rotational Correction Scan (NEW)
    print("\n2. Rotational Correction Scan")
    plot_rotational_correction_scan(tracks_data)

    # Plot 3: Vertical Aero Force vs Horizontal Speed Squared
    print("\n3. Vertical Aero Force vs Horizontal Speed Squared")
    plot_vertical_aero_vs_horizontal_speed(tracks_data)

    # Plot 4: Drag Deceleration vs Total Speed Squared
    print("\n4. Drag Deceleration vs Total Speed Squared")
    plot_drag_deceleration_vs_total_speed(tracks_data)

    # Plot 5: Vertical Velocity vs Time
    print("\n5. Vertical Velocity vs Time")
    plot_vertical_velocity_vs_time(tracks_data, df_all)

    # Plot 6: Vertical Aero Force vs Time
    print("\n6. Vertical Aero Force vs Time")
    plot_vertical_aero_force_vs_time(tracks_data)

    # Plot 7: Mechanical Energy  vs Time
    print("\n7. Mechanical Energy vs Time")
    plot_mechanical_energy_vs_time(energy_data)

    # Plot 8: Energy Components vs Time
    print("\n8. Energy Components: Kinetic and Potential vs Time")
    plot_energy_components_vs_time(energy_data)

    # Plot 9: Energy Change Rate vs Time
    print("\n9. Energy Change Rate (dE/dt) vs Time")
    plot_energy_change_rate(energy_data)

    # Plot 10: All Velocity Components (vx, vy, vz) vs Time
    print("\n10. All Velocity Components (vx, vy, vz) vs Time")
    plot_all_velocity_components_vs_time(tracks_data, df_all)

    # Plot 11: All Acceleration Components (ax, ay, az) vs Time
    print("\n11. All Acceleration Components (ax, ay, az) vs Time")
    plot_all_acceleration_components_vs_time(tracks_data, df_all)

    # Plot 12: Pitch Angle (θ) vs Time
    print("\n12. Pitch Angle (θ) vs Time")
    plot_pitch_angle_vs_time(tracks_data, df_all)

    # Plot 13: Perpendicular Acceleration vs Time
    print("\n13. Perpendicular Acceleration vs Time")
    plot_perpendicular_acceleration_vs_time(tracks_data)

    # Plot 14: Power vs Time
    print("\n14. Power vs Time")
    plot_power_vs_time(tracks_data)

    # Plot 15: Heading-rate consistency check (first few tracks)
    print("\n15. Heading-rate consistency (validation)")
    plot_heading_rate_consistency(tracks_data, df_all, max_tracks=4)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
