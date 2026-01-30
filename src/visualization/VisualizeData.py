#!/usr/bin/env python3
"""
Visualize Data
Reads velocity.csv, computes accelerations, and plots forces to validate physical assumptions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Constants
G = 9.793
MASS = 0.00218  # Mass in kg


def analyze_track(df, track_name):
    t = df.t.values
    vx = df.vx.values
    vy = df.vy.values
    vz = df.vz.values

    # calculate smooth acceleration (force/mass)
    # Using savgol filter again for smooth derivatives
    dt = np.mean(np.diff(t))
    window = min(11, len(t))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = 3

    dt_float = float(dt)
    ax = savgol_filter(vx, window, 3, deriv=1, delta=dt_float)
    ay = savgol_filter(vy, window, 3, deriv=1, delta=dt_float)
    az = savgol_filter(vz, window, 3, deriv=1, delta=dt_float)

    # Physics Variables
    v_xy_sq = vx**2 + vy**2
    v_total_sq = vx**2 + vy**2 + vz**2
    v_abs = np.sqrt(v_total_sq)

    # 1. Vertical Force Analysis
    # Measured vertical acceleration = (Lift_z + Drag_z - mg) / m
    # So: Lift_z_accel + Drag_z_accel = az + g
    # Assuming Drag_z is small compared to Lift_z in hover, but let's just look at Net Aerodynamic Vertical Force
    # F_aero_z / m = az + G
    f_z_aero = az + G

    # 2. Drag Analysis
    # Tangential acceleration (along velocity vector)
    # a_tan = dot(a, v) / |v|
    a_tan = (ax * vx + ay * vy + az * vz) / (v_abs + 1e-6)
    # Drag is roughly -a_tan (retarding force), but Gravity also has a component along path.
    # a_tan_measured = a_drag + a_gravity_tangent
    # a_gravity_tangent = -g * (vz / v_abs)  (since g points down -z)
    # So a_drag = a_tan_measured - (-g * vz / v_abs) = a_tan + g * vz / v_abs
    a_drag_est = a_tan + G * (vz / (v_abs + 1e-6))

    return t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est


def calculate_energy(df, track_name):
    """Calculate mechanical energy: E = ½m·(vx² + vy² + vz²) + m·g·z"""
    t = df.t.values
    vx = df.vx.values
    vy = df.vy.values
    vz = df.vz.values
    z = df.z.values

    # Kinetic energy: ½m(vx² + vy² + vz²)
    kinetic_energy = MASS * 0.5 * (vx**2 + vy**2 + vz**2)

    # Potential energy: mg·z
    potential_energy = MASS * G * z

    # Total mechanical energy
    total_energy = kinetic_energy + potential_energy

    return t, total_energy, kinetic_energy, potential_energy


def plot_vertical_aero_vs_horizontal_speed(tracks_data):
    """Plot 1: Vertical Aero Force vs Horizontal Speed Squared"""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10")

    for i, (track, data) in enumerate(tracks_data.items()):
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est = data
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
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est = data
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
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est = data
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
        energy_result = calculate_energy(df_track, track)
        energy_data[track] = energy_result

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

    # Plot 1: Vertical Aero Force vs Horizontal Speed Squared
    print("\n1. Vertical Aero Force vs Horizontal Speed Squared")
    plot_vertical_aero_vs_horizontal_speed(tracks_data)

    # Plot 2: Drag Deceleration vs Total Speed Squared
    print("\n2. Drag Deceleration vs Total Speed Squared")
    plot_drag_deceleration_vs_total_speed(tracks_data)

    # Plot 3: Vertical Velocity vs Time
    print("\n3. Vertical Velocity vs Time")
    plot_vertical_velocity_vs_time(tracks_data, df_all)

    # Plot 4: Vertical Aero Force vs Time
    print("\n4. Vertical Aero Force vs Time")
    plot_vertical_aero_force_vs_time(tracks_data)

    # Plot 5: Mechanical Energy  vs Time
    print("\n5. Mechanical Energy vs Time")
    plot_mechanical_energy_vs_time(energy_data)

    # Plot 6: Energy Components vs Time
    print("\n6. Energy Components: Kinetic and Potential vs Time")
    plot_energy_components_vs_time(energy_data)

    # Plot 7: Energy Change Rate vs Time
    print("\n7. Energy Change Rate (dE/dt) vs Time")
    plot_energy_change_rate(energy_data)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
