#!/usr/bin/env python3
"""
Physics Model Validator
Reads velocity.csv, computes accelerations, and plots forces to validate physical assumptions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

# Constants
G = 9.793

def analyze_track(df, track_name):
    t = df.t.values
    vx = df.vx.values
    vy = df.vy.values
    vz = df.vz.values
    
    # calculate smooth acceleration (force/mass)
    # Using savgol filter again for smooth derivatives
    dt = np.mean(np.diff(t))
    window = min(11, len(t))
    if window % 2 == 0: window -= 1
    if window < 5: window = 3
    
    ax = savgol_filter(vx, window, 3, deriv=1, delta=dt)
    ay = savgol_filter(vy, window, 3, deriv=1, delta=dt)
    az = savgol_filter(vz, window, 3, deriv=1, delta=dt)
    
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
    a_tan = (ax*vx + ay*vy + az*vz) / (v_abs + 1e-6)
    # Drag is roughly -a_tan (retarding force), but Gravity also has a component along path.
    # a_tan_measured = a_drag + a_gravity_tangent
    # a_gravity_tangent = -g * (vz / v_abs)  (since g points down -z)
    # So a_drag = a_tan_measured - (-g * vz / v_abs) = a_tan + g * vz / v_abs
    a_drag_est = a_tan + G * (vz / (v_abs + 1e-6))
    
    return t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est

def main():
    try:
        df_all = pd.read_csv("data/velocity.csv")
    except FileNotFoundError:
        print("Error: data/velocity.csv not found.")
        return

    tracks = df_all.track.unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    
    # Color map
    cmap = plt.get_cmap("tab10")

    print(f"Analyzing {len(tracks)} tracks...")

    for i, track in enumerate(tracks):
        df_track = df_all[df_all.track == track].sort_values("t")
        if len(df_track) < 10: continue
        
        t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est = analyze_track(df_track, track)
        
        # Color by time (fading) or just distinct track color
        color = cmap(i % 10)
        
        # Plot 1: Vertical Aero Force vs Horizontal Speed Squared
        # Hypothesis: If Lift ~ v^2, this should be a straight line.
        # If Banking Effect exists, it should curve down at high v^2.
        ax1.scatter(v_xy_sq, f_z_aero, s=5, alpha=0.5, label=track, color=color)
        
        # Plot 2: Drag Deceleration vs Total Speed Squared
        # Should be linear: a_drag ~ -C * v^2
        ax2.scatter(v_total_sq, -a_drag_est, s=5, alpha=0.5, color=color)

        # Plot 3: Vertical Velocity vs Time (Check Parabolic/Linear limit)
        ax3.plot(t, df_track.vz, label=track, color=color)
        
        # Plot 4: Vertical Aero Force vs Time
        # To see if it settles to a constant (Rover Lift)
        ax4.plot(t, f_z_aero, color=color)

    # Decoration
    ax1.set_title("Vertical Aero Accel (Lift_z) vs $v_{xy}^2$")
    ax1.set_xlabel("Horizontal Speed Squared ($m^2/s^2$)")
    ax1.set_ylabel("Vertical Accel + G ($m/s^2$)")
    ax1.grid(True)
    ax1.axhline(9.8, color='k', linestyle='--', alpha=0.3, label="1G Hover")

    ax2.set_title("Drag Deceleration vs $v_{total}^2$")
    ax2.set_xlabel("Total Speed Squared ($m^2/s^2$)")
    ax2.set_ylabel("Drag Deceleration ($m/s^2$)")
    ax2.grid(True)
    
    ax3.set_title("Vertical Velocity ($v_z$) vs Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("$v_z$ (m/s)")
    ax3.grid(True)
    
    ax4.set_title("Vertical Aero Force (Lift) vs Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Lift Acceleration ($m/s^2$)")
    ax4.grid(True)
    ax4.axhline(9.8, color='k', linestyle='--', label="Gravity Offset")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
