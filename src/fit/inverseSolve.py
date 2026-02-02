import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to Python path for absolute imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataIO import load_track

# Parameters from user
m = 2.183e-3  # kg
a = 0.15  # m (arm length)
d = 0.028  # m (arm width)
S = 2 * a * d  # m^2, roughtly planform area. 2 wings * length * width
rho = 1.225  # kg/m^3
g = 9.793  # m/s^2

# Rotation parameters
SIGMA_ROTATION = 0.4  # Rotary lift contribution factor
OMEGA0 = 85.0  # rad/s, initial rotation
OMEGA_DECAY = 0.15  # rad/s^2, realistic decay rate (τ ≈ 20s for omega0=85)

# Batch configuration
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "final"
OUTPUT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "interm"
OUTPUT_PLOT_DIR = Path(__file__).resolve().parents[2] / "out"

# Manually skip problematic trajectories by filename
SKIP_FILES = {
    # "trackXopt.csv",
}

ANGLE_COLUMN_CANDIDATES = ("tilt_deg", "bank_deg", "phi_deg", "plane_deg")


def solve_coefficients(file_path):
    print(f"Reading file: {file_path}")
    required_cols = ["ax", "ay", "az", "vx", "vy", "vz", "speed", "t"]
    data = load_track(file_path, required_columns=required_cols)

    # Extract vectors
    a_vec = np.column_stack((data["ax"], data["ay"], data["az"]))
    v_vec = np.column_stack((data["vx"], data["vy"], data["vz"]))
    speed = data["speed"]
    t = data["t"]

    # 1. Net Force (Newtons)
    # F_net = m * a
    F_net = m * a_vec

    # 2. Gravity Vector (Newtons)
    # gravity is [0, 0, -g]
    F_g = np.zeros_like(F_net)
    F_g[:, 2] = -g * m

    # 3. Aerodynamic Force Vector
    # F_net = F_aero + F_g  =>  F_aero = F_net - F_g
    F_aero = F_net - F_g

    # 4. Decompose F_aero into Lift and Drag
    # Drag is parallel to velocity (opposite direction)
    # Lift is perpendicular to velocity

    valid_mask = speed > 0.05
    v_hat = np.zeros_like(v_vec)
    v_hat[valid_mask] = v_vec[valid_mask] / speed[valid_mask, np.newaxis]

    # Decompose aerodynamic force into components parallel/perpendicular to velocity.
    # F_parallel = (F_aero · v_hat) v_hat
    # F_perp     = F_aero - F_parallel
    #
    # NOTE: If (F_aero · v_hat) > 0, the force has a component *along* the velocity
    # direction (a "thrust proxy"). For a passive boomerang this is typically caused by
    # measurement noise, smoothing artifacts, wind, or axis/sign inconsistencies.
    f_parallel = np.einsum("ij,ij->i", F_aero, v_hat)
    F_parallel_vec = f_parallel[:, np.newaxis] * v_hat
    F_lift_vec = F_aero - F_parallel_vec  # strictly perpendicular to v_hat

    # Magnitudes
    L_mag = np.linalg.norm(F_lift_vec, axis=1)

    # Drag magnitude is the *opposing* (negative) parallel component.
    # Keep a signed version for diagnostics; use a physical (non-negative) version for Cd.
    D_mag_signed = -f_parallel
    D_mag = np.maximum(D_mag_signed, 0.0)
    thrust_mag = np.maximum(f_parallel, 0.0)

    # 5. Calculate Coefficients with Rotational Correction
    # Omega Model: w(t) = w0 - k*t
    omega = np.maximum(OMEGA0 - OMEGA_DECAY * t, 0)

    # Rotational Dynamic Pressure
    # v_eff^2 = v^2 + (sigma * omega * a)^2
    # This prevents singularity at low speed (Apex)
    v_rot = SIGMA_ROTATION * omega * a
    v_eff_sq = speed**2 + v_rot**2

    # q = 0.5 * rho * S * v_eff^2
    # Note: We use constant Planform Area S now, effectively treating the
    # rotary disc area interaction via 'sigma'.
    dynamic_pressure = 0.5 * rho * S * v_eff_sq

    Cl = np.zeros_like(L_mag)
    Cd = np.full_like(D_mag, np.nan, dtype=float)
    Cd_signed = np.zeros_like(D_mag_signed, dtype=float)

    # Avoid division by zero (though v_eff_sq shouldn't be zero with rotation)
    q_mask = dynamic_pressure > 1e-6
    Cl[q_mask] = L_mag[q_mask] / dynamic_pressure[q_mask]
    Cd_signed[q_mask] = D_mag_signed[q_mask] / dynamic_pressure[q_mask]
    # Only define physical Cd where drag opposes motion (no thrust proxy)
    cd_ok = q_mask & (f_parallel <= 0)
    Cd[cd_ok] = D_mag[cd_ok] / dynamic_pressure[cd_ok]

    # Store results
    results = pd.DataFrame(
        {
            "t": t,
            "speed": speed,
            "omega": omega,
            "q_dynamic": dynamic_pressure,
            "f_parallel": f_parallel,
            "thrust_mag": thrust_mag,
            "D_mag": D_mag,
            "D_mag_signed": D_mag_signed,
            "L_mag": L_mag,
            "Cl": Cl,
            "Cd": Cd,
            "Cd_signed": Cd_signed,
        }
    )

    return results


def plot_and_save(results, output_prefix, output_dir):
    # Filter for plotting (remove t=0 or end where speed is low)
    plot_data = results[results["speed"] > 0.5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Cl and Cd
    ax1.plot(plot_data["t"], plot_data["Cl"], label="$C_L$", color="blue")
    ax1.plot(plot_data["t"], plot_data["Cd"], label="$C_D$", color="red")
    ax1.set_ylabel("Coefficient Value")
    ax1.set_title("Aerodynamic Coefficients ($C_L, C_D$) vs Time")
    ax1.grid(True)
    ax1.legend()

    # Plot 2: L/D Ratio and Speed
    color = "tab:green"
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)", color=color)
    ax2.plot(plot_data["t"], plot_data["speed"], color=color, label="Speed")
    ax2.tick_params(axis="y", labelcolor=color)

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = "tab:purple"
    ax3.set_ylabel("L/D Ratio", color=color)  # we already handled the x-label with ax2

    # Avoid division by zero for ratio
    cd_vals = plot_data["Cd"].to_numpy()
    cl_vals = plot_data["Cl"].to_numpy()
    valid_ld = np.isfinite(cd_vals) & (cd_vals != 0)
    ld_ratio = np.divide(
        cl_vals, cd_vals, out=np.full_like(cl_vals, np.nan, dtype=float), where=valid_ld
    )
    ax3.plot(plot_data["t"], ld_ratio, color=color, linestyle="--", label="L/D")
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_ylim(-5, 15)  # Limit to reasonable range

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    output_file = output_dir / f"{output_prefix}_analysis.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")
    plt.close(fig)


def plot_all_overlay(all_results, output_dir):
    fig_cl, ax_cl = plt.subplots(1, 1, figsize=(10, 5))
    fig_cd, ax_cd = plt.subplots(1, 1, figsize=(10, 5))

    for name, res in all_results.items():
        plot_data = res[res["speed"] > 0.5]
        ax_cl.plot(plot_data["t"], plot_data["Cl"], alpha=0.8, label=name)
        ax_cd.plot(plot_data["t"], plot_data["Cd"], alpha=0.8, label=name)

    ax_cl.set_title("All Tracks: $C_L$ vs Time")
    ax_cl.set_xlabel("Time (s)")
    ax_cl.set_ylabel("$C_L$")
    ax_cl.grid(True)
    ax_cl.legend(fontsize=8)

    ax_cd.set_title("All Tracks: $C_D$ vs Time")
    ax_cd.set_xlabel("Time (s)")
    ax_cd.set_ylabel("$C_D$")
    ax_cd.grid(True)
    ax_cd.legend(fontsize=8)

    cl_path = output_dir / "coeffs_all_cl.png"
    cd_path = output_dir / "coeffs_all_cd.png"
    fig_cl.tight_layout()
    fig_cd.tight_layout()
    fig_cl.savefig(cl_path, dpi=150)
    fig_cd.savefig(cd_path, dpi=150)
    print(f"Overlay Cl plot saved to: {cl_path}")
    print(f"Overlay Cd plot saved to: {cd_path}")
    plt.close(fig_cl)
    plt.close(fig_cd)


if __name__ == "__main__":
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1]).resolve()
        if not file_path.exists():
            print(f"Error: File not found {file_path}")
            sys.exit(1)
        files = [file_path]
    else:
        files = sorted(DATA_DIR.glob("*opt.csv"))

    if not files:
        print(f"No *opt.csv files found in {DATA_DIR}")
        sys.exit(1)

    all_results = {}
    for file_path in files:
        if file_path.name in SKIP_FILES:
            print(f"[skip] {file_path.name}")
            continue

        try:
            res = solve_coefficients(file_path)
        except Exception as e:
            print(f"[error] {file_path.name}: {e}")
            continue

        # Calculate Statistics for Cl and Cd (filtering out extremes)
        mask = (res["speed"] > 1.0) & np.isfinite(res["Cd"])
        stats = res[mask][["Cl", "Cd", "Cd_signed"]].describe()  # type: ignore
        print(f"\n--- {file_path.name} Statistics (Speed > 1.0 m/s) ---")
        print(stats)

        # Export result to CSV (data/interm)
        base_name = file_path.name.replace(".csv", "")
        output_csv = OUTPUT_DATA_DIR / f"{base_name}_coeffs.csv"
        res.to_csv(output_csv, index=False)
        print(f"Derived coefficients saved to: {output_csv}")

        # Per-track plot (out/)
        plot_and_save(res, base_name, OUTPUT_PLOT_DIR)

        all_results[base_name] = res

    if all_results:
        plot_all_overlay(all_results, OUTPUT_PLOT_DIR)
