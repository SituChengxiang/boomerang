#!/usr/bin/env python3
"""
Reverse engineer aerodynamic coefficients from BOOMERANG tracking data.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataIO import load_track_simple, validate_track_data

# --- Constants ---
RHO = 1.225  # Air density (kg/m^3)
G = 9.793  # Gravity (m/s^2)
M = 0.002183  # Mass (kg)
A_SPAN = 0.15  # Wingspan (m)
D_WIDTH = 0.028  # Wing width (m)
AREA = 2 * A_SPAN * D_WIDTH  # Reference Area (approx)
I_Z = (5 / 24) * M * (A_SPAN**2)  # Moment of Inertia
OMEGA_DECAY = 0.1  # Fixed spin decay (1/s)
GROUND_EFFECT = 0.4  # Reduced ground effect strength (40%)
GROUND_HEIGHT = 0.2  # Ground effect scale height (m)

# Track metadata from README
# (turns, duration) -> omega = turns * 2pi / duration
TRACK_META = {
    "track1": 5.3 / 0.93 * 2 * np.pi,
    "track2": 7.8 / 1.28 * 2 * np.pi,
    # 'track3': 5.5 / 1.08 * 2 * np.pi,
    "track5": 5.0 / 1.17 * 2 * np.pi,
    "track6": 5.4 / 1.07 * 2 * np.pi,
    "track7": 5.2 / 1.17 * 2 * np.pi,
    # 'track8': 4.3 / 0.88 * 2 * np.pi,
    "track9": 4.8 / 1.07 * 2 * np.pi,
}


def load_data(data_dir="data/final"):
    """使用统一的dataIO模块加载优化后的轨迹数据"""
    tracks = {}

    # 查找所有优化后的轨迹文件
    csv_files = sorted(glob.glob(f"{data_dir}/*opt.csv"))

    for f in csv_files:
        key = os.path.basename(f).replace("opt.csv", "")

        # 跳过不在TRACK_META中的轨迹
        if key not in TRACK_META:
            continue

        try:
            # 使用dataIO模块加载数据
            t, x, y, z = load_track_simple(f)

            # 验证数据质量
            data_dict = {"t": t, "x": x, "y": y, "z": z}
            is_valid, issues = validate_track_data(data_dict)

            if not is_valid:
                print(f"警告: 轨迹 {key} 数据有问题: {issues}")
                continue

            # 计算初始速度（保持原有逻辑）
            # 使用加权平均减少噪声
            if len(t) >= 6:
                # 计算时间间隔
                dt_vals = np.diff(t[:6])
                avg_dt = np.mean(dt_vals) if len(dt_vals) > 0 else 0.0166667

                # 加权计算初始速度
                weights = [5 / 15, 4 / 15, 3 / 15, 2 / 15, 1 / 15]
                vx0 = 0.0
                vy0 = 0.0
                vz0 = 0.0

                for i, w in enumerate(weights):
                    if i + 1 < len(t):
                        dt = t[i + 1] - t[i]
                        if dt <= 0:
                            dt = avg_dt

                        vx0 += w * (x[i + 1] - x[i]) / dt
                        vy0 += w * (y[i + 1] - y[i]) / dt
                        vz0 += w * (z[i + 1] - z[i]) / dt

            else:
                # 数据点不足，使用简单差分
                if len(t) >= 2:
                    dt = t[1] - t[0]
                    if dt <= 0:
                        dt = 0.0166667
                    vx0 = (x[1] - x[0]) / dt
                    vy0 = (y[1] - y[0]) / dt
                    vz0 = (z[1] - z[0]) / dt
                else:
                    vx0 = vy0 = vz0 = 0.0

            # 存储轨迹数据
            tracks[key] = {
                "t": t,
                "pos": np.column_stack([x, y, z]),
                "v0": [float(vx0), float(vy0), float(vz0)],
                "omega": TRACK_META[key],
            }

            print(
                f"✓ 加载轨迹 {key}: {len(t)} 个点, 初始速度 [{vx0:.2f}, {vy0:.2f}, {vz0:.2f}] m/s"
            )

        except Exception as e:
            print(f"错误: 加载轨迹 {key} 失败: {e}")
            continue

    if not tracks:
        print(f"错误: 在 {data_dir} 中没有找到有效的轨迹数据")
        print(f"找到的文件: {csv_files}")

    return tracks


def boomerang_ode(
    state,
    t,
    CL_trans,
    CL_rotor,
    C_D,
    D_factor,
    coupling_eff,
    dive_steering_loss,
    bank_factor,
    omega0,
):
    x, y, z, vx, vy, vz = state

    # Update Omega (Spin Decay)
    # Omega drops over time due to friction. Let's use simple linear decay with a floor.
    # omega(t) = omega0 * (1 - decay * t)
    omega_t = max(omega0 * 0.1, omega0 * (1.0 - OMEGA_DECAY * t))

    # Speed variables
    v_sq = vx**2 + vy**2 + vz**2
    v_abs = np.sqrt(v_sq) + 1e-6  # Avoid div by zero
    v_xy = np.sqrt(vx**2 + vy**2) + 1e-6

    # Coefficients
    K_aero = (0.5 * RHO * AREA) / M

    # Precession Base
    # Note: Using real-time omega_t here.
    # As omega drops, I*omega drops, making it EASIER to tilt (Acc_precess ~ Torque / (I*omega)).
    # But Torque itself (D_factor * ...) might depend on omega too?
    # Usually Torque ~ Lift_diff ~ v * omega.
    # So Acc_precess ~ (v * omega) / omega ~ v.
    # So Omega cancelling out is a first-order approx.
    # Let's keep the formula but plug in omega_t to be safe.

    K_gyro_val = (D_factor * RHO * CL_trans * omega_t * (A_SPAN**3) * D_WIDTH) / (
        2 * I_Z
    )

    # --- Angle of Attack / Dive Logic ---
    # When diving (vz < 0), we assume the boomerang "flattens" or "planes",
    # causing it to generate Lift but lose some Turning ability (Steering Loss).
    # dive_ratio goes from 0 (level/climb) to 1 (vertical dive)
    dive_ratio = 0.0
    if vz < 0:
        dive_ratio = min(1.0, abs(vz) / v_abs)

    # Modify Turning Efficiency based on Dive
    # If diving, we reduce the gyro effect by dive_steering_loss factor
    # steering_multiplier = 1.0 - (loss_coeff * dive_ratio)
    steering_eff = max(0.0, 1.0 - dive_steering_loss * dive_ratio)

    K_gyro_effective = K_gyro_val * steering_eff

    # --- Equations ---

    # 1. Drag
    ax_drag = -K_aero * C_D * v_abs * vx
    ay_drag = -K_aero * C_D * v_abs * vy
    az_drag = -K_aero * C_D * v_abs * vz

    # 2. Lift (Empirical Power Law)
    # Use v_xy**lift_power instead of v_xy^2, and remove CL_rotor
    # CL_trans: base lift coefficient
    # lift_power: to be optimized (1.2~2.2)
    lift_power = bank_factor  # Reuse bank_factor slot for lift_power (for now)
    az_lift = K_aero * CL_trans * (v_xy**lift_power)
    # Ground effect: mild lift boost near the ground
    ground_factor = 1.0 + GROUND_EFFECT * np.exp(-max(z, 0.0) / GROUND_HEIGHT)
    az_lift *= ground_factor

    # 3. Precession (Turning)
    # Applied with dive correction
    ax_gyro = K_gyro_effective * vy
    ay_gyro = -(coupling_eff * K_gyro_effective) * vx

    # Total Accel
    dvx = ax_gyro + ax_drag
    dvy = ay_gyro + ay_drag
    dvz = az_lift + az_drag - G

    return [vx, vy, vz, dvx, dvy, dvz]


def simulate_track(params, track_data):
    CL_trans, C_D, D_factor, coupling_eff, dive_steering, lift_power = params
    t = track_data["t"]
    v0 = track_data["v0"]
    r0 = track_data["pos"][0]
    omega0 = track_data["omega"]

    state0 = [
        r0[0],
        r0[1],
        r0[2],
        v0[0],
        v0[1],
        v0[2],
    ]

    sol = odeint(
        boomerang_ode,
        state0,
        t,
        args=(
            CL_trans,
            0.0,
            C_D,
            D_factor,
            coupling_eff,
            dive_steering,
            lift_power,
            omega0,
        ),
    )
    return sol[:, 0:3]  # Return positions


def loss_function(params, tracks):
    total_error = 0

    # Constraint penalties (soft)
    # CL, CD, D should be positive
    for p in params:
        if p < 0:
            total_error += 1000 + p**2 * 1000

    if total_error > 0:
        return total_error

    for key, data in tracks.items():
        sim_pos = simulate_track(params, data)
        real_pos = data["pos"]
        # Mean Squared Error per track
        err = np.mean(np.sum((sim_pos - real_pos) ** 2, axis=1))
        total_error += err

    return total_error


def main():
    tracks = load_data()
    if not tracks:
        print("No input files found! (Looking for data/*opt.csv)")
        return

    print(f"Loaded {len(tracks)} tracks. Fitting parameters...")

    # Initial guess: [CL_trans, C_D, D_factor, coupling_eff, dive_steering, lift_power]
    # CL_trans: High speed lift
    # C_D: drag
    # D_factor: precession
    # coupling_eff: y-coupling
    # dive_steering: 0-1. How much turning power we lose when diving.
    # lift_power: exponent for v_xy (1.2~2.2)
    initial_guess = [0.4, 0.5, 0.3, 1.0, 0.5, 1.7]

    bounds = [
        (0, 5),  # CL_trans
        (0, 5),  # C_D
        (0, 2),  # D
        (0, 2),  # Coupling
        (0, 2.0),  # dive_steering_loss
        (1.2, 2.2),  # lift_power
    ]

    res = minimize(
        loss_function, initial_guess, args=(tracks,), method="L-BFGS-B", bounds=bounds
    )

    print("\noptimization Result:")
    p = res.x
    print("params:", p)
    print("loss:", res.fun)

    # Unpack params
    CL_trans, C_D, D_factor, coupling_eff, dive_steering, lift_power = p

    print("\nFitted Parameters:")
    print(f"  CL_trans     : {CL_trans:.4f} (Translational Lift)")
    print(f"  C_D          : {C_D:.4f}")
    print(f"  D_coeff      : {D_factor:.4f}")
    print(f"  Coupling     : {coupling_eff:.4f}")
    print("  Vel Scale    : 1.0000 (fixed)")
    print("  Omega Decay  : 10.00% / sec (fixed)")
    print(f"  Dive SteerLoss: {dive_steering:.2f} (Turning loss when diving)")
    print(f"  Lift Power   : {lift_power:.4f} (v_xy exponent)")

    if coupling_eff < 0.1:
        print(
            "\n[Conclusion] Coupling Efficiency is low -> Original model (no y-coupling) might be closer, OR physics is different."
        )
    elif coupling_eff > 0.8:
        print(
            "\n[Conclusion] Coupling Efficiency is high -> Symmetric model is required for turning."
        )
    else:
        print("\n[Conclusion] Coupling Efficiency is intermediate.")

    # Visualization (interactive, one track per window)
    plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "SimHei", "DejaVu Sans"]

    valid_keys = sorted(tracks.keys())
    for idx, key in enumerate(valid_keys, start=1):
        data = tracks[key]
        sim_pos = simulate_track(p, data)
        real_pos = data["pos"]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            real_pos[:, 0],
            real_pos[:, 1],
            real_pos[:, 2],
            "k--",
            label="Real",
            alpha=0.6,
        )
        ax.plot(
            sim_pos[:, 0], sim_pos[:, 1], sim_pos[:, 2], "r-", label="Sim", linewidth=2
        )

        ax.set_title(f"{key} ($\\omega$={data['omega']:.1f})")
        ax.legend(loc="upper left")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(True)

        plt.tight_layout()
        print(
            f"\n[{idx}/{len(valid_keys)}] Close the window to view the next track: {key}"
        )
        plt.show()


if __name__ == "__main__":
    main()
