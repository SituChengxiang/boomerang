#!/usr/bin/env python3
"""
Reverse engineer aerodynamic coefficients from BOOMERANG tracking data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
import pandas as pd
import glob
import os

# --- Constants ---
RHO = 1.225          # Air density (kg/m^3)
G = 9.793            # Gravity (m/s^2)
M = 0.002183         # Mass (kg)
A_SPAN = 0.15        # Wingspan (m)
D_WIDTH = 0.028      # Wing width (m)
AREA = 2 * A_SPAN * D_WIDTH # Reference Area (approx)
I_Z = (5/24) * M * (A_SPAN**2) # Moment of Inertia

# Track metadata from README
# (turns, duration) -> omega = turns * 2pi / duration
TRACK_META = {
    'track1': 5.3 / 0.93 * 2 * np.pi,
    'track2': 7.8 / 1.28 * 2 * np.pi,
    'track3': 5.5 / 1.08 * 2 * np.pi, # Corrected based on user table
    'track5': 5.0 / 1.17 * 2 * np.pi,
    'track6': 5.4 / 1.07 * 2 * np.pi,
    'track7': 5.2 / 1.17 * 2 * np.pi,
    'track8': 4.3 / 0.88 * 2 * np.pi,
    'track9': 4.8 / 1.07 * 2 * np.pi,
}

def load_data():
    tracks = {}
    csv_files = sorted(glob.glob("data/*opt.csv"))
    for f in csv_files:
        key = os.path.basename(f).replace("opt.csv", "")
        if key not in TRACK_META: 
            continue
            
        df = pd.read_csv(f)
        # Calculate initial velocity from first few points (simple difference)
        # Using a small fit or mean of first few differences often better than just index 1-0
        dt = df.t.diff().mean()
        vx0 = 5/15 * (df.x.iloc[1] - df.x.iloc[0]) / (df.t.iloc[1] - df.t.iloc[0]) + 4/15 * (df.x.iloc[2] - df.x.iloc[1]) / (df.t.iloc[2] - df.t.iloc[1]) + 3/15 * (df.x.iloc[3] - df.x.iloc[2]) / (df.t.iloc[3] - df.t.iloc[2]) + 2/15 * (df.x.iloc[4] - df.x.iloc[3]) / (df.t.iloc[4] - df.t.iloc[3]) + 1/15 * (df.x.iloc[5] - df.x.iloc[4]) / (df.t.iloc[5] - df.t.iloc[4])
        vy0 = 5/15 * (df.y.iloc[1] - df.y.iloc[0]) / (df.t.iloc[1] - df.t.iloc[0]) + 4/15 * (df.y.iloc[2] - df.y.iloc[1]) / (df.t.iloc[2] - df.t.iloc[1]) + 3/15 * (df.y.iloc[3] - df.y.iloc[2]) / (df.t.iloc[3] - df.t.iloc[2]) + 2/15 * (df.y.iloc[4] - df.y.iloc[3]) / (df.t.iloc[4] - df.t.iloc[3]) + 1/15 * (df.y.iloc[5] - df.y.iloc[4]) / (df.t.iloc[5] - df.t.iloc[4])
        vz0 = 5/15 * (df.z.iloc[1] - df.z.iloc[0]) / (df.t.iloc[1] - df.t.iloc[0]) + 4/15 * (df.z.iloc[2] - df.z.iloc[1]) / (df.t.iloc[2] - df.t.iloc[1]) + 3/15 * (df.z.iloc[3] - df.z.iloc[2]) / (df.t.iloc[3] - df.t.iloc[2]) + 2/15 * (df.z.iloc[4] - df.z.iloc[3]) / (df.t.iloc[4] - df.t.iloc[3]) + 1/15 * (df.z.iloc[5] - df.z.iloc[4]) / (df.t.iloc[5] - df.t.iloc[4])
        
        tracks[key] = {
            't': df.t.values,
            'pos': df[['x', 'y', 'z']].values,
            'v0': [vx0, vy0, vz0],
            'omega': TRACK_META[key]
        }
    return tracks

def boomerang_ode(state, t, C_L, C_D, D_factor, coupling_eff, omega, rotor_lift_ratio):
    x, y, z, vx, vy, vz = state
    
    # Speed variables
    v_sq = vx**2 + vy**2 + vz**2
    v_abs = np.sqrt(v_sq)
    v_xy_sq = vx**2 + vy**2  # Speed contributing to main lift
    
    # Coefficients
    # Lift/Drag factors (F = 0.5 * rho * v^2 * C * A)
    # Accel = F/m = (0.5 * rho * A / m) * C * v^2
    K_aero = (0.5 * RHO * AREA) / M
    
    # Precession logic (Linearized approx from user model, adapted for quadratic nature?)
    # User model: Torque ~ v. Omega ~ Torque. a_c = Omega * v ~ v^2.
    # Current K_gyro constant in code was: (D * rho * CL * w * a^3 * d) / (2 I).
    # This was derived for Torque. 
    # Let's keep the user's Gyro structure but scale it properly or integrate it.
    # Original: dvx = K_gyro * vy. (accel ~ speed).
    # But centripetal force needs accel ~ speed^2 if radius is constant?
    # Actually for boomerang: Radius ~ v / Omega. Omega ~ v. Radius ~ Constant.
    # So accel ~ v^2 / R ~ v^2.
    # So we should multiply by v_abs or similar to upgrade to quadratic if we want to be strict.
    # HOWEVER, let's stick to the "User Form" for turning key, but FIX Z-axis first.
    
    K_gyro_base = (D_factor * RHO * C_L * omega * (A_SPAN**3) * D_WIDTH) / (2 * I_Z)
    
    # --- Equations ---
    
    # 1. Drag (Quadratic is standard for air)
    # a_drag = - K_aero * C_D * v * v_vector
    ax_drag = - K_aero * C_D * v_abs * vx
    ay_drag = - K_aero * C_D * v_abs * vy
    az_drag = - K_aero * C_D * v_abs * vz
    
    # 2. Lift (Vertical) - FIX: Added Rotor Lift Term
    # Translational Lift: ~ v^2 (dominates at high speed)
    # Rotational Lift: ~ (omega * R)^2 (provides "hover" at low speed)
    # rotor_lift_ratio determines how much "helicopter effect" we have relative to wing lift
    # Average blade tip speed approx: V_tip = omega * A_SPAN
    v_tip_sq = (omega * A_SPAN)**2
    az_lift_trans = K_aero * C_L * v_xy_sq
    az_lift_rotor = K_aero * C_L * v_tip_sq * rotor_lift_ratio
    
    az_lift = az_lift_trans + az_lift_rotor
    
    # 3. Precession (Turning)
    # Original: K_gyro * vy. Let's keep structure but maybe scale with v if needed?
    # User's model was Linear. Let's try Linear first for turning, as it produced good shapes.
    # But since we upgraded drag to Quadratic, maybe linear turning is too weak at high speed?
    # Let's try Linear turning (User's) + Quadratic Drag/Lift (New).
    
    ax_gyro = K_gyro_base * vy
    ay_gyro = - (coupling_eff * K_gyro_base) * vx
    
    # Total Accel
    dvx = ax_gyro + ax_drag
    dvy = ay_gyro + ay_drag
    dvz = az_lift + az_drag - G
    
    return [vx, vy, vz, dvx, dvy, dvz]

def simulate_track(params, track_data):
    C_L, C_D, D_factor, coupling_eff, v_scale, rotor_lift = params
    t = track_data['t']
    v0 = track_data['v0']
    r0 = track_data['pos'][0]
    omega = track_data['omega']

    state0 = [
        r0[0], r0[1], r0[2],
        v0[0] * v_scale, v0[1] * v_scale, v0[2] * v_scale,
    ]
 
    sol = odeint(
        boomerang_ode,
        state0,
        t,
        args=(C_L, C_D, D_factor, coupling_eff, omega, rotor_lift),
    )
    return sol[:, 0:3] # Return positions

def loss_function(params, tracks):
    total_error = 0
    
    # Constraint penalties (soft)
    # CL, CD, D should be positive
    for p in params:
        if p < 0: total_error += 1000 + p**2 * 1000
            
    if total_error > 0: return total_error

    for key, data in tracks.items():
        sim_pos = simulate_track(params, data)
        real_pos = data['pos']
        # Mean Squared Error per track
        err = np.mean(np.sum((sim_pos - real_pos)**2, axis=1))
        total_error += err
        
    return total_error

def main():
    tracks = load_data()
    if not tracks:
        print("No input files found! (Looking for data/*opt.csv)")
        return

    print(f"Loaded {len(tracks)} tracks. Fitting parameters...")

    # Initial guess: [C_L, C_D, D_factor, coupling_eff, v_scale, rotor_lift]
    initial_guess = [0.5, 0.5, 0.3, 1.0, 1.0, 0.1]

    # Bounds for realism
    # C_L, C_D: 0-5
    # D: 0-2
    # Coupling: 0-2 (keep close to physical range)
    # v_scale: 0.2-5 (global velocity scale)
    # rotor_lift: 0-0.5 (fraction of lift coming from pure rotation)
    bounds = [
        (0, 5),    # C_L
        (0, 5),    # C_D
        (0, 2),    # D
        (0, 2),    # Coupling
        (0.2, 5),  # v_scale
        (0.0, 1.0) # rotor_lift ratio
    ]
    
    res = minimize(loss_function, initial_guess, args=(tracks,), 
                   method='L-BFGS-B', bounds=bounds)
    
    print("\noptimization Result:")
    p = res.x
    print("params:", p)
    print("loss:", res.fun)
    
    # Unpack params including new rotor lift
    C_L, C_D, D_factor, coupling_eff, v_scale, rotor_lift = p
    
    print(f"\nFitted Parameters:")
    print(f"  C_L (Lift Coeff): {C_L:.4f}")
    print(f"  C_D (Drag Coeff): {C_D:.4f}")
    print(f"  D   (Shape Fac.): {D_factor:.4f}")
    print(f"  Coupling Eff.  : {coupling_eff:.4f}")
    print(f"  Vel Scale      : {v_scale:.4f}")
    print(f"  Rotor Lift %   : {rotor_lift*100:.2f}% (Lift from spin vs speed)")
    
    if coupling_eff < 0.1:
        print("\n[Conclusion] Coupling Efficiency is low -> Original model (no y-coupling) might be closer, OR physics is different.")
    elif coupling_eff > 0.8:
        print("\n[Conclusion] Coupling Efficiency is high -> Symmetric model is required for turning.")
    else:
        print("\n[Conclusion] Coupling Efficiency is intermediate.")

    # Visualization (interactive, one track per window)
    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "SimHei", "DejaVu Sans"]

    valid_keys = sorted(tracks.keys())
    for idx, key in enumerate(valid_keys, start=1):
        data = tracks[key]
        sim_pos = simulate_track(p, data)
        real_pos = data['pos']

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2], 'k--', label='Real', alpha=0.6)
        ax.plot(sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], 'r-', label='Sim', linewidth=2)

        ax.set_title(f"{key} ($\\omega$={data['omega']:.1f})")
        ax.legend(loc="upper left")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        plt.tight_layout()
        print(f"\n[{idx}/{len(valid_keys)}] Close the window to view the next track: {key}")
        plt.show()

if __name__ == "__main__":
    main()
