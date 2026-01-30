#!/usr/bin/env python3
"""Manual per-track preprocessing pipeline.

Design goal:
- One CLI argument: the input CSV path.
- Conservative, inspectable workflow (no over-aggressive auto truncation).

Pipeline:
1) Load track from CSV (t,x,y,z), sort & clean duplicate timestamps
2) RTS smoothing -> write data/interm/*SMR.csv
3) Physics self-consistency check (energy / dE/dt) -> print diagnostics + suggestion
4) Optional interactive trimming -> write data/final/*opt.csv
5) Save debug plots for manual review
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.utils.dataIO import load_track, save_track  # noqa: E402
from src.utils.visualize import (  # noqa: E402
    plot_3d_trajectory_compare,
    # plot_time_series_multiple,
    setup_debug_style,
)

from .derivatives import compute_derivatives  # noqa: E402
from .physicsCal import calculate_energy_per_unit_mass  # noqa: E402
from .smoother import smooth_trajectory  # noqa: E402

TOLERANCE_UP = 10.0  # W/kg threshold for increasing dE/dt energy growth detection
TOLERANCE_DOWN = 30  # W/kg threshold for decreasing dE/dt energy loss detection


def _default_paths(
    input_csv: pathlib.Path,
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    stem = input_csv.stem
    interm = REPO_ROOT / "data" / "interm" / f"{stem}SMR.csv"
    final = REPO_ROOT / "data" / "final" / f"{stem}opt.csv"
    plot = REPO_ROOT / "out" / f"{stem}_preprocess.png"
    return interm, final, plot


def _energy_diagnostics(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    warmup: int = 3,
    tolerance: float = TOLERANCE_UP,
) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
    """Compute energy & suggest a truncation index (optional).

    Suggestion rule (intentionally conservative): after a warm-up region,
    find the first index where dE/dt is above tolerance for 2 consecutive
    samples. Returns None if no such point.
    """
    derivs = compute_derivatives(t, x, y, z, method="auto")
    energy, dE_dt = calculate_energy_per_unit_mass(
        t, x, y, z, vx=derivs.vx, vy=derivs.vy, vz=derivs.vz
    )

    if len(t) < warmup + 3:
        return energy, dE_dt, None

    bad = np.isfinite(dE_dt) & (dE_dt > tolerance)
    bad[:warmup] = False

    idxs = np.where(bad)[0]
    for i in idxs:
        if i + 1 < len(bad) and bad[i + 1]:
            return energy, dE_dt, int(i)
    return energy, dE_dt, None


def _prompt_trim_index(n: int, suggestion: Optional[int]) -> Optional[int]:
    """Prompt user for trim index. Returns None to keep all points."""
    hint = ""
    if suggestion is not None:
        hint = f" (suggest {suggestion})"
    raw = input(
        f"Trim? Enter end index (exclusive) 1..{n}{hint}, or press Enter to keep all: "
    ).strip()
    if raw == "":
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("Invalid integer; keep all.")
        return None

    idx = max(1, min(idx, n))
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual per-track preprocessing: RTS smoothing -> energy check -> optional trim -> export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=pathlib.Path, help="Input track CSV (t,x,y,z)")
    args = parser.parse_args()

    input_csv = args.csv_path
    if not input_csv.is_absolute():
        input_csv = (pathlib.Path.cwd() / input_csv).resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input not found: {input_csv}")

    interm_csv, final_csv, plot_png = _default_paths(input_csv)
    plot_png.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_csv}")
    print(f"Interm (SMR): {interm_csv}")
    print(f"Final (opt): {final_csv}")
    print(f"Plot: {plot_png}")

    track = load_track(input_csv)
    t = track["t"]
    x = track["x"]
    y = track["y"]
    z = track["z"]
    print(
        f"Loaded {len(t)} points. t=[{t[0]:.3f}, {t[-1]:.3f}] duration={t[-1] - t[0]:.3f}s"
    )

    # Step 2: Kalman -> time normalization -> startup smoothing
    smooth = smooth_trajectory(t, x, y, z)
    t_s = smooth["t"]
    x_s = smooth["x"]
    y_s = smooth["y"]
    z_s = smooth["z"]

    # Save SMR (keep original t for traceability)
    derivs_smr = compute_derivatives(t_s, x_s, y_s, z_s, method="auto")
    save_track(
        interm_csv,
        {
            "t": t_s,
            "x": x_s,
            "y": y_s,
            "z": z_s,
            "vx": derivs_smr.vx,
            "vy": derivs_smr.vy,
            "vz": derivs_smr.vz,
            "speed": derivs_smr.speed,
        },
        columns=["t", "x", "y", "z", "vx", "vy", "vz", "speed"],
    )
    print("Saved SMR.")

    # Step 3: Energy self-consistency check
    # Note: Physics dictates energy should dissipate (dE/dt < 0).
    # We only flag large POSITIVE dE/dt (non-physical energy gain) as errors.
    energy, dE_dt, suggestion = _energy_diagnostics(
        t_s, x_s, y_s, z_s, tolerance=TOLERANCE_UP
    )
    finite_ratio = float(np.mean(np.isfinite(energy))) if len(energy) else 0.0
    print(
        "Energy check: "
        f"finite={finite_ratio * 100:.1f}% "
        f"dE/dt median={np.nanmedian(dE_dt):.3g} "
        f"p95={np.nanpercentile(dE_dt, 95):.3g}"
    )
    if suggestion is not None:
        print(
            f"Suggested trim start around index {suggestion} (t={t_s[suggestion]:.3f}s)"
        )
    else:
        print("No obvious energy-growth violation found (conservative rule).")

    # Step 4: Visualization
    setup_debug_style()

    fig1 = plot_3d_trajectory_compare(
        t_s,
        x_raw=x,
        y_raw=y,
        z_raw=z,
        x_smooth=x_s,
        y_smooth=y_s,
        z_smooth=z_s,
        labels=("Raw", "SMR", ""),
    )

    # Energy diagnostics with dual y-axes
    fig2, ax_left = plt.subplots(figsize=(12, 7))
    ax_left.plot(t_s, energy, color="#1f77b4", linewidth=2, label="Energy (J/kg)")
    ax_left.scatter(t_s, energy, s=18, color="#1f77b4", alpha=0.7)
    ax_left.set_xlabel("Time (s)")
    ax_left.set_ylabel("Energy (J/kg)", color="#1f77b4")
    ax_left.tick_params(axis="y", labelcolor="#1f77b4")
    ax_left.grid(True)

    ax_right = ax_left.twinx()
    ax_right.plot(t_s, dE_dt, color="#d62728", linewidth=2, label="dE/dt (W/kg)")
    ax_right.scatter(t_s, dE_dt, s=18, color="#d62728", alpha=0.7)
    ax_right.axhline(y=0.0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_right.set_ylabel("dE/dt (W/kg)", color="#d62728")
    ax_right.tick_params(axis="y", labelcolor="#d62728")

    fig2.suptitle(f"Energy diagnostics: {input_csv.name}")
    fig2.tight_layout()

    # Save as a single multi-page style: write 3D plot and time series separately
    out1 = plot_png.with_name(plot_png.stem + "_3d.png")
    out2 = plot_png.with_name(plot_png.stem + "_energy.png")
    fig1.savefig(out1, dpi=200)
    fig2.savefig(out2, dpi=200)
    print(f"Saved plots: {out1} , {out2}")

    # Show interactive GUI for manual inspection
    plt.show()

    # Step 5: Optional manual trimming
    end_idx = _prompt_trim_index(len(t_s), suggestion)
    if end_idx is None:
        end_idx = len(t_s)

    t2 = t_s[:end_idx] - float(t_s[0])  # type: ignore normalize time to start at 0
    x2 = x_s[:end_idx]
    y2 = y_s[:end_idx]
    z2 = z_s[:end_idx]
    derivs_final = compute_derivatives(t2, x2, y2, z2, method="auto")
    save_track(
        final_csv,
        {
            "t": t2,
            "x": x2,
            "y": y2,
            "z": z2,
            "vx": derivs_final.vx,
            "vy": derivs_final.vy,
            "vz": derivs_final.vz,
            "speed": derivs_final.speed,
        },
        columns=["t", "x", "y", "z", "vx", "vy", "vz", "speed"],
    )
    print(f"Saved opt ({end_idx} points).")


if __name__ == "__main__":
    main()
