#!/usr/bin/env python3
"""
fitBAP.py - Bank Angle Proxy Model Fitting
Fits aerodynamic parameters (CL, CD) using a simplified physics model where
Bank Angle variation is the dominant driver of the trajectory.

Model:
- Lift ~ v^1.5 (Rotary wing approximation)
- Drag ~ v^2 (Standard fluid dynamics)
- Bank Angle (phi): Dynamic proxy variable dependent on speed loss.
  phi(t) = phi_base + k * (v0 - v)
  (Rational: As speed drops, precession accumulates, increasing bank)
"""

import argparse
import glob
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.dataIO import load_track_simple, validate_track_data

try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:

    def njit(*_args, **_kwargs):  # type: ignore
        def _wrap(func):
            return func

        return _wrap

    _HAVE_NUMBA = False

# --- Simulation engine (set via CLI) ---
# ivp: scipy solve_ivp (accurate, slower)
# rk4: fixed-step RK4 (fast, deterministic)
# rk4-numba: fixed-step RK4 with Numba JIT (fastest)
SIM_ENGINE = "rk4-numba"
RK_SUBSTEPS = 4


def set_engine(engine: str, substeps: int) -> None:
    global SIM_ENGINE, RK_SUBSTEPS
    SIM_ENGINE = str(engine)
    RK_SUBSTEPS = int(max(1, substeps))


# --- Model knobs (fixed, not optimized) ---
# Add a time-accumulated banking term to make turning start earlier even when speed
# hasn't dropped much yet (proxy for precession accumulation over time).
PHI_TIME_GAIN = 0.20  # rad per (omega_scale * second)

# Self-leveling strength in dive; too strong -> turn weakens and descent slows.
SELF_LEVEL_GAIN = 0.20

# Bank-angle efficiency effects: banked turning tends to cost energy and can shed lift.
BANK_LIFT_LOSS = 0.20  # fraction of lift reduced at |phi|=90deg
BANK_DRAG_GAIN = 0.40  # fraction of drag increased at |phi|=90deg

# --- Constants ---
RHO = 1.225  # Air density (kg/m^3)
G = 9.793  # Gravity (m/s^2)
M = 0.002183  # Mass (kg)
A_SPAN = 0.15  # Wingspan (m)
D_WIDTH = 0.028  # Wing width (m)
AREA = 2 * A_SPAN * D_WIDTH  # Reference Area
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

OMEGA_REF = float(np.median(list(TRACK_META.values())))


def load_data(data_dir="data/interm"):
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


def boomerang_rhs(t, state, CL, CD, phi_base, k_bank, v0_scalar, omega_scale):
    """RHS for solve_ivp.

    State: [x, y, z, vx, vy, vz]
    Coordinate convention (right-handed): x-right, y-forward, z-up.
    """
    x, y, z, vx, vy, vz = state

    # Velocity magnitudes
    v_sq = vx**2 + vy**2 + vz**2
    v = np.sqrt(v_sq) + 1e-9
    v_xy = np.sqrt(vx**2 + vy**2) + 1e-9

    # --- 1. Dynamic Bank Angle Proxy ---
    # phi increases as speed drops (accumulation of precession).
    # Add spin-rate scaling: higher omega accumulates bank faster -> tighter turns.
    speed_loss = v0_scalar - v_xy
    phi = (
        phi_base
        + (k_bank * omega_scale) * speed_loss
        + (PHI_TIME_GAIN * omega_scale) * float(t)
    )

    # Correction: Dive Self-Leveling (smooth)
    # When diving (vz < 0), wings tend to level out; use a smooth factor to avoid
    # discontinuities that can confuse the optimizer.
    dive_ratio = float(np.clip((-vz) / v, 0.0, 1.0))
    phi *= 1.0 - SELF_LEVEL_GAIN * dive_ratio

    # Clamp physically
    phi = np.clip(phi, -1.5, 1.5)  # +/- ~85 degrees

    # Directions (3D)
    # Drag direction: opposite to velocity
    v_hat = np.array([vx, vy, vz]) / v

    # Lift direction: perpendicular to velocity, banked by phi around v_hat.
    # Right-handed coordinates: x-right, y-forward, z-up.
    # We define the unbanked lift direction using world-up, then rotate by phi.
    up = np.array([0.0, 0.0, 1.0])
    # "Right" lateral axis relative to motion
    lateral = np.cross(v_hat, up)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-9:
        # If moving nearly vertical, choose an arbitrary lateral axis
        lateral = np.array([1.0, 0.0, 0.0])
    else:
        lateral /= lateral_norm

    # Base lift direction (roughly up) perpendicular to v_hat
    lift0 = np.cross(lateral, v_hat)
    lift0_norm = float(np.linalg.norm(lift0))
    if lift0_norm < 1e-9:
        lift0 = up
    else:
        lift0 /= lift0_norm

    # Rodrigues rotation of lift0 about v_hat by bank angle phi
    # Positive phi -> bank to the right -> turn towards +x when vy>0.
    lift_dir = (
        lift0 * np.cos(phi)
        + np.cross(v_hat, lift0) * np.sin(phi)
        + v_hat * (np.dot(v_hat, lift0)) * (1.0 - np.cos(phi))
    )

    # --- 2. Aerodynamic Forces ---
    q = 0.5 * RHO * AREA

    # Lift Force Magnitude (v^1.5 model)
    # Use full airspeed magnitude; direction is handled by lift_dir.
    F_lift_mag = q * CL * (v**1.5)

    # Bank efficiency: at high bank, some lift is lost (stall/induced effects).
    # Keep a floor to avoid non-physical negative lift.
    lift_eff = 1.0 - BANK_LIFT_LOSS * (np.sin(phi) ** 2)
    F_lift_mag *= float(np.clip(lift_eff, 0.25, 1.0))

    # Ground Effect (Z < 0.2m correction)
    # Updated: Only apply positive lift boost, no crazy exponential growth
    if z < 0.2:
        # Cap the multiplier to avoid explosion at z < 0
        h_eff = max(z, 0.05)
        F_lift_mag *= 1.0 + 0.2 * 0.2 / h_eff  # mild ground cushion

    # Lift Vector (3D)
    lift_vec = F_lift_mag * lift_dir
    ax_lift, ay_lift, az_lift = (
        float(lift_vec[0]),
        float(lift_vec[1]),
        float(lift_vec[2]),
    )

    # Drag Force Magnitude (v^2 standard model)
    # Mild dive drag reduction to help recover horizontal speed during return.
    # Banked turn costs energy -> add bank-dependent drag.
    cd_reduction = 0.25
    CD_eff = (
        CD
        * (1.0 - cd_reduction * dive_ratio)
        * (1.0 + BANK_DRAG_GAIN * (np.sin(phi) ** 2))
    )
    F_drag_mag = q * CD_eff * v_sq

    # Drag Vector
    ax_drag = -F_drag_mag * v_hat[0]
    ay_drag = -F_drag_mag * v_hat[1]
    az_drag = -F_drag_mag * v_hat[2]

    # Sum Accels (F/m)
    ax = (ax_lift + ax_drag) / M
    ay = (ay_lift + ay_drag) / M
    az = (az_lift + az_drag) / M - G

    return [vx, vy, vz, ax, ay, az]


def _rk4_step(rhs_func, t, y, h, args):
    k1 = np.asarray(rhs_func(t, y, *args), dtype=np.float64)
    k2 = np.asarray(rhs_func(t + 0.5 * h, y + 0.5 * h * k1, *args), dtype=np.float64)
    k3 = np.asarray(rhs_func(t + 0.5 * h, y + 0.5 * h * k2, *args), dtype=np.float64)
    k4 = np.asarray(rhs_func(t + h, y + h * k3, *args), dtype=np.float64)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_rk4(
    t_eval: np.ndarray, state0: np.ndarray, args, substeps: int
) -> np.ndarray:
    n = len(t_eval)
    states = np.empty((n, 6), dtype=np.float64)
    states[0] = state0
    y = state0

    for i in range(n - 1):
        t0 = float(t_eval[i])
        t1 = float(t_eval[i + 1])
        dt = t1 - t0
        h = dt / float(substeps)
        tt = t0
        for _ in range(substeps):
            y = _rk4_step(boomerang_rhs, tt, y, h, args)
            tt += h
        states[i + 1] = y

    return states


if _HAVE_NUMBA:

    @njit(cache=True, fastmath=True)
    def _cross_nb(ax, ay, az, bx, by, bz):
        return (
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        )

    @njit(cache=True, fastmath=True)
    def boomerang_rhs_nb(t, state, CL, CD, phi_base, k_bank, v0_scalar, omega_scale):
        x, y, z, vx, vy, vz = state

        v_sq = vx * vx + vy * vy + vz * vz
        v = np.sqrt(v_sq) + 1e-9
        v_xy = np.sqrt(vx * vx + vy * vy) + 1e-9

        speed_loss = v0_scalar - v_xy
        phi = (
            phi_base
            + (k_bank * omega_scale) * speed_loss
            + (PHI_TIME_GAIN * omega_scale) * t
        )

        dive_ratio = (-vz) / v
        if dive_ratio < 0.0:
            dive_ratio = 0.0
        elif dive_ratio > 1.0:
            dive_ratio = 1.0
        phi = phi * (1.0 - SELF_LEVEL_GAIN * dive_ratio)

        if phi < -1.5:
            phi = -1.5
        elif phi > 1.5:
            phi = 1.5

        v_hat_x = vx / v
        v_hat_y = vy / v
        v_hat_z = vz / v

        # up = (0,0,1)
        lat_x, lat_y, lat_z = _cross_nb(v_hat_x, v_hat_y, v_hat_z, 0.0, 0.0, 1.0)
        lat_norm = np.sqrt(lat_x * lat_x + lat_y * lat_y + lat_z * lat_z)
        if lat_norm < 1e-9:
            lat_x, lat_y, lat_z = 1.0, 0.0, 0.0
        else:
            inv = 1.0 / lat_norm
            lat_x *= inv
            lat_y *= inv
            lat_z *= inv

        lift0_x, lift0_y, lift0_z = _cross_nb(
            lat_x, lat_y, lat_z, v_hat_x, v_hat_y, v_hat_z
        )
        lift0_norm = np.sqrt(lift0_x * lift0_x + lift0_y * lift0_y + lift0_z * lift0_z)
        if lift0_norm < 1e-9:
            lift0_x, lift0_y, lift0_z = 0.0, 0.0, 1.0
        else:
            inv = 1.0 / lift0_norm
            lift0_x *= inv
            lift0_y *= inv
            lift0_z *= inv

        cv = np.cos(phi)
        sv = np.sin(phi)
        cx, cy, cz = _cross_nb(v_hat_x, v_hat_y, v_hat_z, lift0_x, lift0_y, lift0_z)
        dot = v_hat_x * lift0_x + v_hat_y * lift0_y + v_hat_z * lift0_z
        lift_dir_x = lift0_x * cv + cx * sv + v_hat_x * dot * (1.0 - cv)
        lift_dir_y = lift0_y * cv + cy * sv + v_hat_y * dot * (1.0 - cv)
        lift_dir_z = lift0_z * cv + cz * sv + v_hat_z * dot * (1.0 - cv)

        q = 0.5 * RHO * AREA
        F_lift_mag = q * CL * (v**1.5)
        lift_eff = 1.0 - BANK_LIFT_LOSS * (np.sin(phi) ** 2)
        if lift_eff < 0.25:
            lift_eff = 0.25
        F_lift_mag *= lift_eff

        if z < 0.2:
            h_eff = z
            if h_eff < 0.05:
                h_eff = 0.05
            F_lift_mag *= 1.0 + 0.2 * 0.2 / h_eff

        lift_x = F_lift_mag * lift_dir_x
        lift_y = F_lift_mag * lift_dir_y
        lift_z = F_lift_mag * lift_dir_z

        cd_reduction = 0.25
        CD_eff = (
            CD
            * (1.0 - cd_reduction * dive_ratio)
            * (1.0 + BANK_DRAG_GAIN * (np.sin(phi) ** 2))
        )
        F_drag_mag = q * CD_eff * v_sq
        drag_x = -F_drag_mag * v_hat_x
        drag_y = -F_drag_mag * v_hat_y
        drag_z = -F_drag_mag * v_hat_z

        ax = (lift_x + drag_x) / M
        ay = (lift_y + drag_y) / M
        az = (lift_z + drag_z) / M - G

        return np.array([vx, vy, vz, ax, ay, az], dtype=np.float64)

    @njit(cache=True, fastmath=True)
    def integrate_rk4_nb(
        t_eval, state0, CL, CD, phi_base, k_bank, v0_scalar, omega_scale, substeps
    ):
        n = t_eval.shape[0]
        states = np.empty((n, 6), dtype=np.float64)
        states[0, :] = state0
        y = state0.copy()

        for i in range(n - 1):
            t0 = float(t_eval[i])
            t1 = float(t_eval[i + 1])
            dt = t1 - t0
            h = dt / float(substeps)
            tt = t0
            for _ in range(substeps):
                k1 = boomerang_rhs_nb(
                    tt, y, CL, CD, phi_base, k_bank, v0_scalar, omega_scale
                )
                k2 = boomerang_rhs_nb(
                    tt + 0.5 * h,
                    y + 0.5 * h * k1,
                    CL,
                    CD,
                    phi_base,
                    k_bank,
                    v0_scalar,
                    omega_scale,
                )
                k3 = boomerang_rhs_nb(
                    tt + 0.5 * h,
                    y + 0.5 * h * k2,
                    CL,
                    CD,
                    phi_base,
                    k_bank,
                    v0_scalar,
                    omega_scale,
                )
                k4 = boomerang_rhs_nb(
                    tt + h, y + h * k3, CL, CD, phi_base, k_bank, v0_scalar, omega_scale
                )
                y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                tt += h
            states[i + 1, :] = y

        return states


def simulate_track(params, track_data, return_full=False):
    CL, CD, phi_base, k_bank = params
    t = track_data["t"]
    v0 = track_data["v0"]
    r0 = track_data["pos"][0]
    omega = track_data["omega"]

    # Scalar initial speed for bank proxy reference
    v0_xy_scalar = np.sqrt(v0[0] ** 2 + v0[1] ** 2)

    omega_scale = float(omega / OMEGA_REF)

    state0 = [*r0, *v0]

    engine = SIM_ENGINE
    substeps = int(RK_SUBSTEPS)
    t_eval = np.asarray(t, dtype=np.float64)
    y0 = np.asarray(state0, dtype=np.float64)

    try:
        if engine == "ivp":
            dt = float(np.median(np.diff(t_eval))) if len(t_eval) > 1 else 0.01
            sol = solve_ivp(
                lambda tt, yy: boomerang_rhs(
                    tt, yy, CL, CD, phi_base, k_bank, v0_xy_scalar, omega_scale
                ),
                (float(t_eval[0]), float(t_eval[-1])),
                y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-6,
                max_step=max(dt, 1e-4),
            )
            if (not sol.success) or sol.y.shape[1] != len(t_eval):
                return (
                    np.zeros((len(t_eval), 6))
                    if return_full
                    else np.zeros((len(t_eval), 3))
                )
            states = sol.y.T
        elif engine == "rk4":
            args = (CL, CD, phi_base, k_bank, v0_xy_scalar, omega_scale)
            states = integrate_rk4(t_eval, y0, args=args, substeps=substeps)
        elif engine == "rk4-numba":
            if not _HAVE_NUMBA:
                # Fallback when numba isn't available
                args = (CL, CD, phi_base, k_bank, v0_xy_scalar, omega_scale)
                states = integrate_rk4(t_eval, y0, args=args, substeps=substeps)
            else:
                states = integrate_rk4_nb(
                    t_eval,
                    y0,
                    float(CL),
                    float(CD),
                    float(phi_base),
                    float(k_bank),
                    float(v0_xy_scalar),
                    float(omega_scale),
                    int(substeps),
                )
        else:
            raise ValueError(f"Unknown SIM_ENGINE: {engine}")

        if states.shape[0] != len(t_eval):
            return (
                np.zeros((len(t_eval), 6))
                if return_full
                else np.zeros((len(t_eval), 3))
            )
        return states if return_full else states[:, 0:3]
    except Exception:
        return np.zeros((len(t_eval), 6)) if return_full else np.zeros((len(t_eval), 3))


def loss_function(params, tracks):
    CL, CD, phi_base, k_bank = params

    total_mse = 0
    # Do NOT bias the bank sign here. Direction (left/right) depends on the coordinate
    # convention and on how the experimental data was recorded.

    for key, data in tracks.items():
        sim_state = simulate_track(params, data, return_full=True)
        sim_pos = sim_state[:, 0:3]
        sim_vel = sim_state[:, 3:6]
        real_pos = data["pos"]
        real_vel = data.get("vel_meas", np.zeros_like(real_pos))

        # Check for NaN divergence or empty result
        if np.isnan(sim_pos).any() or np.all(sim_pos == 0):
            return 1e9 + np.random.rand()  # Return high loss

        # Bail out early on blow-ups to avoid overflow in mse.
        if np.max(np.abs(sim_pos - real_pos)) > 5.0:
            return 1e8

        # Position loss: emphasize XY (turning radius) slightly more than Z.
        dpos = sim_pos - real_pos
        w_t = np.linspace(1.0, 3.0, len(real_pos))
        w_xy = 1.0
        w_z = 0.6
        pos_err = w_xy * (dpos[:, 0] ** 2 + dpos[:, 1] ** 2) + w_z * (dpos[:, 2] ** 2)
        mse_pos = float(np.mean(pos_err * w_t))

        # Heading/turning loss: compare heading angle in XY plane.
        # This directly penalizes wrong curvature / too-large turn radius.
        theta_sim = np.unwrap(np.arctan2(sim_vel[:, 1], sim_vel[:, 0]))
        theta_real = np.unwrap(np.arctan2(real_vel[:, 1], real_vel[:, 0]))
        mse_theta = float(np.mean((theta_sim - theta_real) ** 2 * w_t))

        # Turn-rate loss: compare dtheta/dt (more directly tied to curvature and turn radius)
        dtheta_sim = np.gradient(theta_sim, data["t"])
        dtheta_real = np.gradient(theta_real, data["t"])
        mse_dtheta = float(np.mean((dtheta_sim - dtheta_real) ** 2 * w_t))

        # End-segment speed loss: penalize |vy| being too small near the end.
        n = len(real_pos)
        tail = slice(max(0, int(0.8 * n)), n)
        vy_sim = sim_vel[tail, 1]
        vy_real = real_vel[tail, 1]
        mse_vy_tail = float(np.mean((np.abs(vy_sim) - np.abs(vy_real)) ** 2))

        # Weights:
        # - mse_pos keeps overall alignment
        # - mse_theta/mse_dtheta tighten turn timing/radius
        # - mse_vy_tail forces end-segment speed along y not to collapse
        total_mse += mse_pos + 0.35 * mse_theta + 0.80 * mse_dtheta + 1.20 * mse_vy_tail

    return total_mse


def fit_multistart(tracks, bounds, x0, starts, seed):
    rng = np.random.default_rng(seed)

    best_res = None
    best_x = None
    best_fun = np.inf

    def sample_within_bounds():
        sampled = []
        for lo, hi in bounds:
            sampled.append(float(rng.uniform(lo, hi)))
        return sampled

    initial_points = [x0] + [sample_within_bounds() for _ in range(max(0, starts - 1))]

    for i, guess in enumerate(initial_points, start=1):
        res = minimize(
            loss_function, guess, args=(tracks,), method="L-BFGS-B", bounds=bounds
        )
        if res.success and res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x
            best_res = res
        print(
            f"[start {i}/{len(initial_points)}] loss={res.fun:.4f} success={res.success}"
        )

    return best_res, best_x


def _optimize_one_start(
    guess, tracks, bounds, start_timeout_s, maxiter, engine, substeps
):
    """Run one local optimization.

    Uses SIGALRM to time-limit pathological starts so they don't block the whole run.
    Note: SIGALRM works on Unix/Linux (your environment).
    """

    def _alarm_handler(signum, frame):
        raise TimeoutError("start timed out")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    try:
        # Ensure worker process uses the requested integrator.
        set_engine(engine, substeps)

        if start_timeout_s and start_timeout_s > 0:
            signal.alarm(int(start_timeout_s))

        res = minimize(
            loss_function,
            guess,
            args=(tracks,),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter)},
        )

        return {
            "success": bool(res.success),
            "fun": float(res.fun),
            "x": np.array(res.x, dtype=float),
            "message": str(getattr(res, "message", "")),
        }
    except TimeoutError as e:
        return {
            "success": False,
            "fun": float("inf"),
            "x": np.array(guess, dtype=float),
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "fun": float("inf"),
            "x": np.array(guess, dtype=float),
            "message": f"error: {e}",
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def fit_multistart_parallel(
    tracks, bounds, x0, starts, seed, workers, start_timeout_s, maxiter
):
    rng = np.random.default_rng(seed)

    def sample_within_bounds():
        sampled = []
        for lo, hi in bounds:
            sampled.append(float(rng.uniform(lo, hi)))
        return sampled

    initial_points = [x0] + [sample_within_bounds() for _ in range(max(0, starts - 1))]
    n_workers = int(workers)
    if n_workers <= 0:
        # Default: use all cores. This is CPU-heavy and the user explicitly asked for speed.
        n_workers = max(1, (os.cpu_count() or 1))
    n_workers = min(n_workers, len(initial_points))

    best = None
    best_x = None
    best_fun = np.inf

    # On Linux the default multiprocessing start method is typically 'fork',
    # which is suitable for this script-style usage.
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {}
        for idx, guess in enumerate(initial_points, start=1):
            fut = ex.submit(
                _optimize_one_start,
                guess,
                tracks,
                bounds,
                start_timeout_s,
                maxiter,
                SIM_ENGINE,
                RK_SUBSTEPS,
            )
            futures[fut] = idx

        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"[start {idx}/{len(initial_points)}] crashed: {e}")
                continue

            msg = out.get("message", "")
            msg = (" " + msg) if msg else ""
            if np.isfinite(out["fun"]):
                print(
                    f"[start {idx}/{len(initial_points)}] loss={out['fun']:.4f} success={out['success']}{msg}"
                )
            else:
                print(
                    f"[start {idx}/{len(initial_points)}] loss=inf success={out['success']}{msg}"
                )
            if out["success"] and out["fun"] < best_fun:
                best_fun = out["fun"]
                best_x = out["x"]
                best = out

    return best, best_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Fit only, do not show matplotlib windows",
    )
    parser.add_argument(
        "--starts",
        type=int,
        default=max(12, (os.cpu_count() or 12)),
        help="Number of multistart initializations",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for multistart")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel worker processes for multistart (0=auto)",
    )
    parser.add_argument(
        "--start-timeout",
        type=int,
        default=90,
        help="Per-start timeout seconds (0=disable)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=120, help="Max L-BFGS-B iterations per start"
    )
    parser.add_argument(
        "--track",
        type=str,
        default="",
        help="Only report/plot a single track key (e.g. track1)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print per-track error summary and tail metrics",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default="",
        help="Directory to save plots as PNG instead of only showing",
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        default="",
        help="Export per-time-step diagnostics CSV (requires --track)",
    )
    parser.add_argument(
        "--turn",
        choices=["right", "left", "free"],
        default="right",
        help="Constrain turning direction via phi_base bounds",
    )
    parser.add_argument(
        "--engine",
        choices=["rk4-numba", "rk4", "ivp"],
        default=SIM_ENGINE,
        help="Simulation engine: rk4-numba (fast, default), rk4 (pure python), ivp (scipy solve_ivp)",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=RK_SUBSTEPS,
        help="RK4 substeps per sample interval (used by rk4/rk4-numba)",
    )
    args = parser.parse_args()

    set_engine(args.engine, args.substeps)
    if SIM_ENGINE == "rk4-numba" and not _HAVE_NUMBA:
        print(
            "[warn] --engine=rk4-numba requested but numba not available; falling back to rk4"
        )
        set_engine("rk4", args.substeps)
    if SIM_ENGINE == "rk4-numba" and _HAVE_NUMBA:
        # Warm up JIT compilation before forking worker processes.
        _ = integrate_rk4_nb(
            np.array([0.0, 0.01], dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            1.0,
            0.2,
            0.2,
            0.1,
            1.0,
            1.0,
            int(RK_SUBSTEPS),
        )

    tracks = load_data()
    print(f"Loaded {len(tracks)} tracks.")

    # --- Optimization ---
    # Params: [CL, CD, phi_base, k_bank]
    # Updated Guesses based on physics intuition:
    # Need High CL to counter High Drag and sustain lift.
    # Need significant phi to turn.
    x0 = [1.2, 0.35, 0.3, 0.15]

    # Bounds - Widen them!
    phi_bounds = (-1.5, 1.5)
    if args.turn == "right":
        phi_bounds = (0.0, 1.5)
    elif args.turn == "left":
        phi_bounds = (-1.5, 0.0)

    bounds = [
        (1.0, 5.0),  # CL: Force High Lift to counteract gravity with high bank
        (0.15, 0.45),  # CD: Allow slightly lower drag to match end-segment speeds
        phi_bounds,  # phi_base
        (-2.0, 2.0),  # k_bank
    ]

    print(
        f"Starting optimization (L-BFGS-B), multistart={args.starts}, workers={args.workers or 'auto'}, "
        f"timeout={args.start_timeout}s, maxiter={args.maxiter}..."
    )
    # Parallel multistart: each start runs in its own process.
    # This is where most wall-clock time goes.
    best, best_x = fit_multistart_parallel(
        tracks,
        bounds,
        x0,
        starts=args.starts,
        seed=args.seed,
        workers=args.workers,
        start_timeout_s=args.start_timeout,
        maxiter=args.maxiter,
    )
    if best_x is None:
        print("Optimization failed for all starts.")
        return

    # Synthesize a minimal result-like object for downstream printing.
    class _Res:
        def __init__(self, success, fun, x):
            self.success = success
            self.fun = fun
            self.x = x

    if best is None:
        print("Optimization finished but no successful result was selected.")
        return

    res = _Res(best["success"], best["fun"], best_x)

    print("\noptimization Success:", res.success)
    print("Final Loss:", res.fun)
    print("Params:", best_x)

    CL, CD, phi_base, k_bank = best_x
    print("\n--- Fit Results ---")
    print(f"CL (Lift Coeff v^1.5): {CL:.4f}")
    print(f"CD (Drag Coeff v^2)  : {CD:.4f}")
    print(f"phi_base (Initial)   : {phi_base:.4f} rad ({np.degrees(phi_base):.1f} deg)")
    print(f"k_bank   (Dynamic)   : {k_bank:.4f} rad/(m/s)")

    # --- Visualization ---
    def compute_track_metrics(track_key, best_params):
        data = tracks[track_key]
        sim_state = simulate_track(best_params, data, return_full=True)
        sim_pos = sim_state[:, 0:3]
        sim_vel = sim_state[:, 3:6]
        real_pos = data["pos"]
        real_vel = data.get("vel_meas", np.zeros_like(real_pos))

        dpos = sim_pos - real_pos
        pos_rmse = float(np.sqrt(np.mean(np.sum(dpos**2, axis=1))))

        n = len(real_pos)
        tail = slice(max(0, int(0.8 * n)), n)
        pos_tail_rmse = float(np.sqrt(np.mean(np.sum(dpos[tail] ** 2, axis=1))))

        theta_sim = np.unwrap(np.arctan2(sim_vel[:, 1], sim_vel[:, 0]))
        theta_real = np.unwrap(np.arctan2(real_vel[:, 1], real_vel[:, 0]))
        theta_rmse = float(np.sqrt(np.mean((theta_sim - theta_real) ** 2)))

        vy_tail_rmse = float(
            np.sqrt(
                np.mean((np.abs(sim_vel[tail, 1]) - np.abs(real_vel[tail, 1])) ** 2)
            )
        )

        final_err = float(np.linalg.norm(sim_pos[-1] - real_pos[-1]))
        final_vy_abs_sim = float(abs(sim_vel[-1, 1]))
        final_vy_abs_real = float(abs(real_vel[-1, 1]))

        return {
            "key": track_key,
            "pos_rmse": pos_rmse,
            "pos_tail_rmse": pos_tail_rmse,
            "theta_rmse": theta_rmse,
            "vy_tail_rmse": vy_tail_rmse,
            "final_err": final_err,
            "final_vy_abs_sim": final_vy_abs_sim,
            "final_vy_abs_real": final_vy_abs_real,
        }

    if args.report:
        keys = sorted(tracks.keys())
        if args.track:
            if args.track not in tracks:
                print(f"Unknown track: {args.track}. Available: {keys}")
            else:
                keys = [args.track]

        rows = [compute_track_metrics(k, best_x) for k in keys]
        rows = sorted(rows, key=lambda r: r["pos_tail_rmse"], reverse=True)
        print("\nPer-track summary (sorted by tail RMSE):")
        for r in rows:
            print(
                f"  {r['key']}: pos_rmse={r['pos_rmse']:.3f}  tail_rmse={r['pos_tail_rmse']:.3f}  "
                f"theta_rmse={r['theta_rmse']:.3f}  vy_tail_rmse={r['vy_tail_rmse']:.3f}  "
                f"final_err={r['final_err']:.3f}  |vy|_end(sim/real)={r['final_vy_abs_sim']:.3f}/{r['final_vy_abs_real']:.3f}"
            )

    if args.export_csv:
        if not args.track:
            print("--export-csv requires --track <trackKey> (e.g. --track track1)")
        elif args.track not in tracks:
            print(f"Unknown track: {args.track}. Available: {sorted(tracks.keys())}")
        else:
            k = args.track
            data = tracks[k]
            sim_state = simulate_track(best_x, data, return_full=True)
            sim_pos = sim_state[:, 0:3]
            sim_vel = sim_state[:, 3:6]
            real_pos = data["pos"]
            real_vel = data.get("vel_meas", np.zeros_like(real_pos))
            t = data["t"]
            v_xy_hist = np.sqrt(sim_vel[:, 0] ** 2 + sim_vel[:, 1] ** 2) + 1e-9
            v0_scalar = float(np.sqrt(data["v0"][0] ** 2 + data["v0"][1] ** 2))
            omega_scale = float(data["omega"] / OMEGA_REF)
            phi_hist = np.clip(
                phi_base
                + (k_bank * omega_scale) * (v0_scalar - v_xy_hist)
                + (PHI_TIME_GAIN * omega_scale) * t,
                -1.5,
                1.5,
            )

            out_df = pd.DataFrame(
                {
                    "t": t,
                    "x_real": real_pos[:, 0],
                    "y_real": real_pos[:, 1],
                    "z_real": real_pos[:, 2],
                    "x_sim": sim_pos[:, 0],
                    "y_sim": sim_pos[:, 1],
                    "z_sim": sim_pos[:, 2],
                    "vx_real": real_vel[:, 0],
                    "vy_real": real_vel[:, 1],
                    "vz_real": real_vel[:, 2],
                    "vx_sim": sim_vel[:, 0],
                    "vy_sim": sim_vel[:, 1],
                    "vz_sim": sim_vel[:, 2],
                    "phi_deg": np.degrees(phi_hist),
                }
            )
            out_df.to_csv(args.export_csv, index=False)
            print(f"Exported diagnostics CSV -> {args.export_csv}")

    if args.no_plot and not args.save_plots:
        return

    if args.save_plots:
        os.makedirs(args.save_plots, exist_ok=True)

    valid_keys = sorted(tracks.keys())
    if args.track:
        if args.track not in tracks:
            print(f"Unknown track: {args.track}. Available: {valid_keys}")
            return
        valid_keys = [args.track]

    for idx, key in enumerate(valid_keys, start=1):
        data = tracks[key]
        sim_state = simulate_track(best_x, data, return_full=True)
        sim_pos = sim_state[:, 0:3]
        real_pos = data["pos"]
        real_vel = data.get("vel_meas", np.zeros_like(real_pos))
        sim_vel = sim_state[:, 3:6]

        fig = plt.figure(figsize=(10, 6))

        # 3D Plot
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        ax.plot(
            real_pos[:, 0],
            real_pos[:, 1],
            real_pos[:, 2],
            "k--",
            label="Tracker Data",
            alpha=0.5,
        )
        ax.plot(
            sim_pos[:, 0],
            sim_pos[:, 1],
            sim_pos[:, 2],
            "r-",
            label="Fit Model",
            linewidth=2,
        )
        ax.set_title(f"{key} Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # 2D Height Profile + vy(t)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(data["t"], real_pos[:, 2], "k--", label="Real Z")
        ax2.plot(data["t"], sim_pos[:, 2], "r-", label="Sim Z")
        ax2.set_ylabel("Height (m)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="upper left")
        ax2.grid(True)

        ax4 = ax2.twinx()
        ax4.plot(data["t"], np.abs(real_vel[:, 1]), "k:", alpha=0.6, label="Real |vy|")
        ax4.plot(data["t"], np.abs(sim_vel[:, 1]), "r:", alpha=0.8, label="Sim |vy|")
        ax4.set_ylabel("|vy| (m/s)")

        # Twin axis for Bank Angle
        # Reconstruct bank angle from simulated velocities (no rough proxy)
        v_xy_hist = np.sqrt(sim_state[:, 3] ** 2 + sim_state[:, 4] ** 2) + 1e-9
        v0_scalar = float(
            np.sqrt(tracks[key]["v0"][0] ** 2 + tracks[key]["v0"][1] ** 2)
        )
        omega_scale = float(tracks[key]["omega"] / OMEGA_REF)
        phi_hist = np.degrees(
            np.clip(
                phi_base + (k_bank * omega_scale) * (v0_scalar - v_xy_hist), -1.5, 1.5
            )
        )

        # ax3 = ax.plot(
        #     [], [], [], alpha=0.0
        # )  # keep layout stable; phi plotted on same right axis as |vy|
        ax4.plot(data["t"], phi_hist, "b-.", alpha=0.5, label="Bank Angle (deg)")

        # Merge legends for right axis
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="lower left")

        plt.tight_layout()
        if args.save_plots:
            out_path = os.path.join(args.save_plots, f"{key}.png")
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            print(f"Saved {key} plot -> {out_path}")
        else:
            print(f"Displaying {key}...")
            plt.show()


if __name__ == "__main__":
    main()
