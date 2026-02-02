import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Add project root to path
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.Coefficient import (
    _coeff_summary,
    _eval_coeff,
    estimate_cl_cd,
)
from src.utils.dataIO import load_track

# --- Physical Constants (matching inverseSolve.py) ---
MASS = 2.183e-3
G = 9.793
RHO = 1.225
A = 0.15  # radius
CHOR = 0.028  # average chord
S = 2 * A * CHOR  # Area
I_z = (
    MASS / 24 * (5 * A**2 + 2 * CHOR**2)
)  # Rotational inertia (spin axis) approximation.

SIGMA = 0.4  # Spin lift correction factor, guess
OMEGA_DECAY = 0.1  # Angular velocity attenuation amplitude
OMEGA0 = -80.0  # Initial angular velocity (- implies clock wise)

TILT0 = 45  # Initial Facial angles, the boomerang and the ground

# Debug/verbosity control
DEBUG = False

# ODE & LOSS
R_TOLERANCE = 1e-4
A_TOLERANCE = 1e-5
F_TOLERANCE = 1e-4
G_TOLERANCE = 1e-5


# --- Physics Model ---


def _col_as_array(data, col_name: str) -> np.ndarray:
    """Return a column as a 1D numpy array from load_track output.

    `load_track` in this repo may return a dict-like of numpy arrays or a pandas DataFrame.
    This helper normalizes both.
    """
    if hasattr(data, "to_dict") and hasattr(data, "__getitem__"):
        # likely a pandas DataFrame
        try:
            return np.asarray(data[col_name].to_numpy())
        except Exception:
            pass
    return np.asarray(data[col_name])


def _safe_unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if not np.isfinite(nrm) or nrm < 1e-12:
        return fallback
    return vec / nrm


def _lift_dir_from_spin_axis(s_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """Lift direction from a body/spin axis.

    For a CLOCKWISE spinning boomerang (viewed from above), the angular velocity
    points downward (-Z by right-hand rule). The lift direction is:
        lift_dir ∝ v × s (velocity cross spin axis)
    This gives rightward turn for clockwise rotation with positive tilt.
    """
    lift_raw = np.cross(v_hat, s_hat)  # v × s for clockwise rotation
    lift_dir = _safe_unit(lift_raw, np.array([0.0, 0.0, 1.0]))
    return lift_dir


def _horizontal_centripetal_accel(
    t: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Horizontal-plane centripetal acceleration magnitude a_perp_h.

    a_perp_h = |v_h x a_h| / |v_h| = |vx*ay - vy*ax| / sqrt(vx^2+vy^2)
    """
    t = np.asarray(t, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    if t.size < 3:
        ax = np.gradient(vx)
        ay = np.gradient(vy)
    else:
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
    v_h = np.sqrt(vx * vx + vy * vy + 1e-12)
    v_cross_a_h = vx * ay - vy * ax
    return np.abs(v_cross_a_h) / (v_h + 1e-6)


def _horizontal_centripetal_accel_signed(
    t: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Signed horizontal-plane centripetal acceleration.

    a_perp_h_signed = (v_h x a_h) / |v_h| = (vx*ay - vy*ax) / sqrt(vx^2+vy^2)
    Positive/negative indicates turn direction in the horizontal plane.
    """
    t = np.asarray(t, dtype=float)
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    if t.size < 3:
        ax = np.gradient(vx)
        ay = np.gradient(vy)
    else:
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
    v_h = np.sqrt(vx * vx + vy * vy + 1e-12)
    v_cross_a_h = vx * ay - vy * ax
    return v_cross_a_h / (v_h + 1e-6)


def _calculate_lift(
    cl_model: dict | float, speed: float, q: float, lift_dir: np.ndarray
) -> np.ndarray:
    """Calculate lift force.

    Args:
        cl_model: Lift coefficient model (dict or float).
        speed: Current speed.
        q: Dynamic pressure.
        lift_dir: Lift direction (unit vector).

    Returns:
        Lift force vector.
    """
    Cl_0 = _eval_coeff(cl_model, speed)
    return Cl_0 * q * lift_dir


def _calculate_drag(
    cd_model: dict | float, speed: float, q: float, v_hat: np.ndarray
) -> np.ndarray:
    """Calculate drag force.

    Args:
        cd_model: Drag coefficient model (dict or float).
        speed: Current speed.
        q: Dynamic pressure.
        v_hat: Velocity unit vector.

    Returns:
        Drag force vector.
    """
    Cd_0 = _eval_coeff(cd_model, speed)
    return -Cd_0 * q * v_hat


def _calculate_torque(
    D_scale: float,
    Cl_0: float,
    omega: float,
    speed: float,
    v_hat: np.ndarray,
    s_hat: np.ndarray,
) -> np.ndarray:
    """Calculate torque vector.

    Args:
        D_scale: Torque scale factor.
        Cl_0: Lift coefficient.
        omega: Angular velocity.
        speed: Current speed.
        v_hat: Velocity unit vector.
        s_hat: Spin axis unit vector.

    Returns:
        Torque vector.
    """
    tau_mag = 0.5 * D_scale * RHO * Cl_0 * omega * speed * (A**3) * CHOR
    tau_dir_raw = np.cross(v_hat, s_hat)
    tau_dir_nrm = float(np.linalg.norm(tau_dir_raw))
    if (not np.isfinite(tau_dir_nrm)) or tau_dir_nrm < 1e-12:
        return np.zeros(3, dtype=float)
    return (tau_mag / tau_dir_nrm) * tau_dir_raw


def _calculate_spin_axis_precession(
    tau_vec: np.ndarray,
    s_hat: np.ndarray,
    omega: float,
) -> np.ndarray:
    """Calculate spin axis precession.

    Args:
        tau_vec: Torque vector.
        s_hat: Spin axis unit vector.
        omega: Angular velocity.

    Returns:
        Spin axis precession rate (ds/dt).
    """
    if omega < 1e-3:
        return np.zeros(3, dtype=float)
    return (tau_vec - float(np.dot(tau_vec, s_hat)) * s_hat) / (I_z * omega)


def boomerang_ode(t, state, params):
    """Forward dynamics using angular-momentum precession.

    State:
        [x, y, z, vx, vy, vz, sx, sy, sz]
        where s is the (approx.) spin axis / body normal direction.
    Params:
        [Cl_0, Cd_0, D, initial_tilt_deg, omega0, omega_decay]

    First-principles core:
        L = I_z * omega * s
        dL/dt = tau
        ds/dt = (tau - (tau·s)s) / (I_z * omega)

    Aerodynamics:
        Lift direction is enforced perpendicular to velocity:
            lift_dir = normalize(s - (s·v_hat)v_hat)
    """
    x, y, z, vx, vy, vz, sx, sy, sz = state

    # Unpack params
    cl_model = params[0]
    cd_model = params[1]
    D_scale = float(params[2])
    omega0 = float(params[4])
    omega_decay = float(params[5])

    # 1. State normalization
    v = np.array([vx, vy, vz], dtype=float)
    speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return np.zeros(9)
    v_hat = v / speed

    s = np.array([sx, sy, sz], dtype=float)
    s_hat = _safe_unit(s, np.array([0.0, 0.0, 1.0]))

    # 2. Angular Velocity (Spin)
    # Use signed omega for rotation direction; decay acts on magnitude.
    omega_mag = max(abs(omega0) - omega_decay * float(t), 0.0)
    omega = np.sign(omega0) * omega_mag

    # 3. Dynamic Pressure (with Rotation Correction)
    sigma = SIGMA
    v_rot = sigma * abs(omega) * A
    v_eff_sq = speed * speed + v_rot * v_rot
    q = 0.5 * RHO * S * v_eff_sq

    # 4. Aerodynamic Forces
    lift_dir = _lift_dir_from_spin_axis(s_hat, v_hat)
    f_lift = _calculate_lift(cl_model, speed, q, lift_dir)
    f_drag = _calculate_drag(cd_model, speed, q, v_hat)
    f_aero = f_lift + f_drag
    f_gravity = np.array([0.0, 0.0, -MASS * G])
    acc = (f_aero + f_gravity) / MASS

    # 5. Torque and precession
    Cl_0 = _eval_coeff(cl_model, speed)
    tau_vec = _calculate_torque(D_scale, Cl_0, omega, speed, v_hat, s_hat)
    ds_dt = _calculate_spin_axis_precession(tau_vec, s_hat, omega)

    return [
        vx,
        vy,
        vz,
        float(acc[0]),
        float(acc[1]),
        float(acc[2]),
        float(ds_dt[0]),
        float(ds_dt[1]),
        float(ds_dt[2]),
    ]


def simulate(params, initial_state, t_eval):
    t_eval = np.asarray(t_eval, dtype=float)
    if t_eval.ndim != 1 or t_eval.size < 2:
        raise ValueError("t_eval must be a 1D array with at least 2 points")

    # params = [Cl_0, Cd_0, D, initial_tilt_deg, omega0, omega_decay]
    initial_tilt_deg = float(params[3])

    # Unpack initial state (only kinematic)
    x0, y0, z0, vx0, vy0, vz0 = initial_state

    # Initialize spin axis/body normal s.
    # Interpret tilt as the angle between s and +Z (a frisbee-like face angle).
    # For a right-hand spinning boomerang thrown with tilt, the spin axis
    # tilts toward the thrower's left (perpendicular to velocity in XY plane).
    v0 = np.array([vx0, vy0, vz0], dtype=float)
    v0_hat = _safe_unit(v0, np.array([1.0, 0.0, 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    side = _safe_unit(np.cross(up, v0_hat), np.array([0.0, 1.0, 0.0]))
    tilt = float(np.radians(initial_tilt_deg))
    s0 = np.cos(tilt) * up + np.sin(tilt) * side
    s0 = _safe_unit(s0, up)
    # Do NOT flip s0 - initial direction matters for lift direction (s × v)
    state0 = [x0, y0, z0, vx0, vy0, vz0, float(s0[0]), float(s0[1]), float(s0[2])]

    sol = solve_ivp(
        boomerang_ode,
        [t_eval[0], t_eval[-1]],
        state0,
        t_eval=t_eval,
        args=(params,),
        rtol=R_TOLERANCE,
        atol=A_TOLERANCE,
        dense_output=True,
        max_step=float(np.max(np.diff(t_eval))) if t_eval.size > 1 else np.inf,
    )

    if DEBUG:
        print(
            f"      ODE solver: success={sol.success}, nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}"
        )
        if not sol.success:
            print(f"      WARNING: ODE solver failed with message: {sol.message}")
    # If integration fails early, return NaNs so caller can penalize safely.
    if (not sol.success) or (sol.y.shape[1] != t_eval.size):
        if DEBUG:
            print(
                f"      WARNING: Integration failed or incomplete. sol.y shape: {sol.y.shape}, expected: (9, {t_eval.size})"
            )
        y = np.full((9, t_eval.size), np.nan, dtype=float)
        if sol.sol is not None and sol.t.size >= 2:
            t0 = float(sol.t[0])
            t1 = float(sol.t[-1])
            ok = (t_eval >= t0) & (t_eval <= t1)
            if DEBUG:
                print(
                    f"      Partial integration from t={t0:.3f} to t={t1:.3f}, covering {np.sum(ok)}/{len(t_eval)} points"
                )
            try:
                y[:, ok] = sol.sol(t_eval[ok])
            except Exception as e:
                if DEBUG:
                    print(f"      Error during interpolation: {e}")
                pass
        return y

    if DEBUG:
        final_v = np.sqrt(sol.y[3, -1] ** 2 + sol.y[4, -1] ** 2 + sol.y[5, -1] ** 2)
        final_s = np.array([sol.y[6, -1], sol.y[7, -1], sol.y[8, -1]])
        final_tilt = np.degrees(np.arccos(final_s[2]))
        print(
            f"      Final state: v={final_v:.3f} m/s, s=({final_s[0]:.3f}, {final_s[1]:.3f}, {final_s[2]:.3f}), tilt={final_tilt:.1f}°"
        )

    return sol.y


def calculate_heading(vx, vy):
    """Calculate unwrapped heading angle in degrees."""
    heading = np.degrees(np.arctan2(vy, vx))
    # Unwrap to avoid jumps from 180 to -180
    return np.degrees(np.unwrap(np.radians(heading)))


def calculate_heading_from_xy(
    x: np.ndarray, y: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Heading computed from position derivatives (more robust than noisy vx/vy)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    if x.size < 3:
        return calculate_heading(np.gradient(x), np.gradient(y))
    dt = np.gradient(t)
    dt = np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    return calculate_heading(vx, vy)


def calculate_tilt_deg_from_s(
    sx: np.ndarray, sy: np.ndarray, sz: np.ndarray
) -> np.ndarray:
    """Return tilt angle of spin axis relative to +Z in degrees.

    Returns angle in [0, 180] where:
    - 0° = spin axis pointing up (+Z)
    - 90° = spin axis horizontal (knife-edge)
    - 180° = spin axis pointing down (-Z)
    """
    s = np.vstack([sx, sy, sz]).T.astype(float)
    nrm = np.linalg.norm(s, axis=1)
    nrm = np.where(nrm < 1e-12, 1e-12, nrm)
    s_hat = s / nrm[:, None]
    # Do NOT flip s_hat - we want to see the full [0, 180] range
    cos_tilt = np.clip(s_hat[:, 2], -1.0, 1.0)
    return np.degrees(np.arccos(cos_tilt))


def visualize_fit(t_true, state_true, t_sim, state_sim, params, title="Trajectory Fit"):
    """
    Detailed visualization of the fit.
    state_true/sim shape: [x, y, z, vx, vy, vz, ...] at columns
    """
    x_true, y_true, z_true = state_true[:, 0], state_true[:, 1], state_true[:, 2]
    vx_true, vy_true = state_true[:, 3], state_true[:, 4]

    x_sim, y_sim, z_sim = state_sim[0], state_sim[1], state_sim[2]
    vx_sim, vy_sim = state_sim[3], state_sim[4]

    # Calculate heading
    heading_true = calculate_heading(vx_true, vy_true)
    heading_sim = calculate_heading(vx_sim, vy_sim)
    heading_true_xy = calculate_heading_from_xy(x_true, y_true, t_true)
    heading_sim_xy = calculate_heading_from_xy(x_sim, y_sim, t_sim)

    # Setup Figure
    fig = plt.figure(figsize=(14, 6))

    # --- Left Plot: 3D Trajectory ---
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    # True data (Scatter/Faint line)
    ax3d.plot(
        x_true, y_true, z_true, "k.", markersize=4, alpha=0.3, label="Measured Data"
    )
    # ax3d.plot(x_true, y_true, z_true, 'k-', alpha=0.1)

    # Measured Start/End
    ax3d.scatter(
        x_true[0], y_true[0], z_true[0], c="g", marker="^", s=50, label="Start"
    )
    ax3d.scatter(
        x_true[-1], y_true[-1], z_true[-1], c="r", marker="x", s=50, label="End"
    )

    # Sim data (Solid Color)
    ax3d.plot(x_sim, y_sim, z_sim, color="#0D47A1", linewidth=2, label="Simulation")

    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"3D Trajectory: {title}")
    ax3d.legend()

    # Equal aspect ratio hack for 3D
    # (Matplotlib 3D doesn't support 'equal' nicely, simple bounding box)
    max_range = (
        np.array(
            [
                x_true.max() - x_true.min(),
                y_true.max() - y_true.min(),
                z_true.max() - z_true.min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (x_true.max() + x_true.min()) * 0.5
    mid_y = (y_true.max() + y_true.min()) * 0.5
    mid_z = (z_true.max() + z_true.min()) * 0.5
    ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

    # --- Right Plot: Z & Heading ---
    ax_z = fig.add_subplot(1, 2, 2)

    # Z-axis (Left Y)
    (line_z_true,) = ax_z.plot(
        t_true, z_true, markersize=4, alpha=0.4, label="Z Measured", color="#004884"
    )
    (line_z_sim,) = ax_z.plot(t_sim, z_sim, color="#0077D9", label="Z Sim")
    ax_z.set_xlabel("Time (s)")
    ax_z.set_ylabel("Height Z (m)", color="#005091")
    ax_z.tick_params(axis="y", labelcolor="#005091")
    ax_z.grid(True, alpha=0.3)

    # Heading (Right Y)
    ax_h = ax_z.twinx()
    (line_h_true,) = ax_h.plot(
        t_true,
        heading_true,
        markersize=4,
        alpha=0.3,
        label="Heading Measured",
        color="#B71C1C",
    )
    (line_h_sim,) = ax_h.plot(
        t_sim, heading_sim, linewidth=2, label="Heading Sim", color="#D32F2F"
    )
    ax_h.set_ylabel("XY Heading (deg)", color="m")
    ax_h.tick_params(axis="y", labelcolor="m")

    # Unified Legend
    lines = [line_z_true, line_z_sim, line_h_true, line_h_sim]
    labels = [str(l.get_label()) for l in lines]
    ax_z.legend(lines, labels, loc="upper left")

    # params: [Cl_model, Cd_model, D]
    cl_disp = _coeff_summary(params[0])
    cd_disp = _coeff_summary(params[1])
    res_str = f"Cl={cl_disp:.3f}, Cd={cd_disp:.3f}, D={params[2]:.3f}"
    ax_z.set_title(f"Z-Height & Heading Analysis\n[{res_str}]")

    plt.tight_layout()
    plt.show()

    # --- Diagnostics Figure (attitude + speed + heading robustness) ---
    # Simulated spin axis and derived tilt
    sx_sim, sy_sim, sz_sim = state_sim[6], state_sim[7], state_sim[8]
    tilt_sim = calculate_tilt_deg_from_s(sx_sim, sy_sim, sz_sim)

    vxy_true = np.sqrt(vx_true * vx_true + vy_true * vy_true)
    vxy_sim = np.sqrt(vx_sim * vx_sim + vy_sim * vy_sim)

    fig2, (ax_tilt, ax_vxy, ax_head) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax_tilt.plot(t_sim, tilt_sim, "c-", linewidth=2, label="Tilt Sim (deg)")
    ax_tilt.set_ylabel("Tilt (deg)")
    ax_tilt.grid(True, alpha=0.3)
    ax_tilt.legend(loc="best")

    ax_vxy.plot(
        t_true, vxy_true, "k.", markersize=3, alpha=0.35, label="|v_xy| Measured"
    )
    ax_vxy.plot(t_sim, vxy_sim, "b-", linewidth=2, label="|v_xy| Sim")
    ax_vxy.set_ylabel("|v_xy| (m/s)")
    ax_vxy.grid(True, alpha=0.3)
    ax_vxy.legend(loc="best")

    ax_head.plot(
        t_true,
        heading_true,
        "m.",
        markersize=5,
        alpha=0.25,
        label="Heading Measured (vx,vy)",
    )
    ax_head.plot(
        t_true,
        heading_true_xy,
        "r.",
        markersize=5,
        alpha=0.25,
        label="Heading Measured (dx/dt,dy/dt)",
    )
    ax_head.plot(t_sim, heading_sim, "m--", linewidth=2, label="Heading Sim (vx,vy)")
    ax_head.plot(
        t_sim,
        heading_sim_xy,
        "r-",
        linewidth=1.8,
        alpha=0.8,
        label="Heading Sim (dx/dt,dy/dt)",
    )
    ax_head.set_ylabel("Heading (deg)")
    ax_head.set_xlabel("Time (s)")
    ax_head.grid(True, alpha=0.3)
    ax_head.legend(loc="best", ncols=2)

    fig2.suptitle(
        f"Diagnostics: {title}\n(tilt, |v_xy|, and heading computed two ways)"
    )
    fig2.tight_layout()
    plt.show()


def loss_function(params, t_target, pos_target, cl_fixed, cd_fixed, tilt_deg, omega0):
    """Phase space MSE loss for one track (positions + velocities).

    We only optimize D (torque scale), while Cl/Cd are fixed from inverse solve.
    """

    try:
        D = float(np.atleast_1d(params)[0])
    except Exception:
        return 1e9

    omega_decay = 1.5
    full_params = [
        cl_fixed,
        cd_fixed,
        D,
        float(tilt_deg),
        float(omega0),
        omega_decay,
    ]

    initial_state = pos_target[0]  # x,y,z, vx,vy,vz
    sim_res = simulate(full_params, initial_state, t_target)

    # Calculate error in phase space (positions + velocities)
    # sim_res has shape (9, N) = [x, y, z, vx, vy, vz, sx, sy, sz]
    sim_phase = sim_res[0:6, :]  # x,y,z, vx,vy,vz

    # PHASE MSE
    # pos_target is shape (N, 6) = [x, y, z, vx, vy, vz]
    true_phase = pos_target[:, 0:6].T

    # If simulation failed at any time step, heavily penalize.
    if not np.all(np.isfinite(sim_phase)):
        return 1e9

    diff = sim_phase - true_phase
    mse = float(np.mean(diff**2))
    if not np.isfinite(mse):
        return 1e9

    # Horizontal centripetal acceleration alignment (evidence-driven)
    vx_t, vy_t = pos_target[:, 3], pos_target[:, 4]
    vx_s, vy_s = sim_phase[3], sim_phase[4]
    a_perp_t = _horizontal_centripetal_accel(t_target, vx_t, vy_t)
    a_perp_s = _horizontal_centripetal_accel(t_target, vx_s, vy_s)
    a_perp_t_signed = _horizontal_centripetal_accel_signed(t_target, vx_t, vy_t)
    a_perp_s_signed = _horizontal_centripetal_accel_signed(t_target, vx_s, vy_s)
    v_h = np.sqrt(vx_t * vx_t + vy_t * vy_t + 1e-12)
    weight = v_h / (v_h + 0.5)  # downweight low v_h segments
    scale = (
        float(np.nanmedian(a_perp_t)) if np.isfinite(np.nanmedian(a_perp_t)) else 1.0
    )
    scale = max(scale, 1e-3)
    mse_perp = float(np.mean(((a_perp_s - a_perp_t) * weight / scale) ** 2))
    scale_signed = float(np.nanmedian(np.abs(a_perp_t_signed)))
    scale_signed = max(scale_signed, 1e-3)
    mse_perp_signed = float(
        np.mean(((a_perp_s_signed - a_perp_t_signed) * weight / scale_signed) ** 2)
    )
    if not np.isfinite(mse_perp):
        return 1e9

    mse = mse + 0.2 * mse_perp + 0.2 * mse_perp_signed

    if DEBUG and mse < 1e6:
        print(
            f"    Loss evaluation: D={D:.4f}, total={mse:.6f}, "
            f"phase={mse:.6f}, perp={mse_perp:.6f}, perp_signed={mse_perp_signed:.6f}"
        )

    return mse


def optimize_track(file_path):
    print(f"Optimizing for {file_path.name}...")
    # Load data
    data = load_track(
        file_path, required_columns=["x", "y", "z", "vx", "vy", "vz", "t"]
    )
    t = _col_as_array(data, "t")

    # Limit to 1 second as requested or full duration
    mask = t <= 1.0

    # Sort by time to satisfy solve_ivp expectations and keep arrays aligned.
    t_masked = t[mask]
    order = np.argsort(t_masked)
    t = t_masked[order]

    pos_target = np.column_stack(
        [
            _col_as_array(data, "x")[mask][order],
            _col_as_array(data, "y")[mask][order],
            _col_as_array(data, "z")[mask][order],
            _col_as_array(data, "vx")[mask][order],
            _col_as_array(data, "vy")[mask][order],
            _col_as_array(data, "vz")[mask][order],
        ]
    )

    if DEBUG:
        print(
            f"  Initial state: x={pos_target[0, 0]:.3f}, y={pos_target[0, 1]:.3f}, z={pos_target[0, 2]:.3f}"
        )
        print(
            f"  Initial velocity: vx={pos_target[0, 3]:.3f}, vy={pos_target[0, 4]:.3f}, vz={pos_target[0, 5]:.3f}"
        )

    # Fixed parameters from inverse solve (shrink search space)
    cl_fixed, cd_fixed = estimate_cl_cd(file_path)
    tilt_deg = TILT0
    omega0 = OMEGA0

    # Optimize only D (torque scale factor). Allow sign to choose turn direction.
    x0 = [1.0]
    bounds = [(-20.0, 20.0)]

    if DEBUG:
        print("  Starting optimization for D parameter...")
        print(f"    Initial guess: D = {x0[0]}")
        print(f"    Bounds: [{bounds[0][0]}, {bounds[0][1]}]")
        print("    Optimization method: L-BFGS-B")

    res = minimize(
        loss_function,
        x0,
        args=(t, pos_target, cl_fixed, cd_fixed, tilt_deg, omega0),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 50,
            "ftol": F_TOLERANCE,
            "gtol": G_TOLERANCE,
        },
        callback=(
            (lambda xk: print(f"    Iteration: D = {xk[0]:.4f}")) if DEBUG else None
        ),
    )

    D_best = float(np.atleast_1d(res.x)[0])
    print(f"Result: {res.x}")
    print(f"  Cl (model median): {_coeff_summary(cl_fixed):.3f}")
    print(f"  Cd (model median): {_coeff_summary(cd_fixed):.3f}")
    print(f"  D: {D_best:.3f}")
    print(f"  Tilt (fixed): {tilt_deg:.1f} deg")
    print(f"  Omega0 (fixed): {omega0:.1f} rad/s")

    # Visualize final result
    # Re-run simulation with valid range
    sim_t = np.linspace(t[0], t[-1], 200)  # Higher res for smooth plot

    # full params reconstruction
    best_params = [cl_fixed, cd_fixed, D_best]
    best_full = [
        cl_fixed,
        cd_fixed,
        D_best,
        float(tilt_deg),
        float(omega0),
        1.5,
    ]
    initial_state = pos_target[0]

    if DEBUG:
        print("  Running final simulation...")
    sim_res = simulate(best_full, initial_state, sim_t)

    # Check simulation quality
    if DEBUG:
        if np.any(np.isnan(sim_res)):
            print("  WARNING: Simulation contains NaN values!")
        else:
            print(f"  Simulation successful, shape: {sim_res.shape}")

    # Export simulated spin axis for inspection
    out_dir = PROJECT_ROOT / "out" / "optimizeParams"
    out_dir.mkdir(parents=True, exist_ok=True)
    spin_csv = out_dir / f"{file_path.stem}_spin.csv"
    if sim_res.shape[0] >= 9:
        tilt_deg = calculate_tilt_deg_from_s(sim_res[6], sim_res[7], sim_res[8])
        arr = np.column_stack([sim_t, sim_res[6], sim_res[7], sim_res[8], tilt_deg])
        np.savetxt(
            spin_csv,
            arr,
            delimiter=",",
            header="t,sx,sy,sz,tilt_deg",
            comments="",
        )
        if DEBUG:
            print(f"  Spin axis data saved to: {spin_csv}")

    # Call visualization
    if DEBUG:
        print("  Generating visualization...")
    visualize_fit(t, pos_target, sim_t, sim_res, best_params, title=file_path.name)

    return res.x


if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = Path(sys.argv[1])
        optimize_track(f)
    else:
        # Default to first track file
        data_dir = PROJECT_ROOT / "data" / "final"
        files = sorted(data_dir.glob("*opt.csv"))
        if files:
            optimize_track(files[0])
