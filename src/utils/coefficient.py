"""Coeffient utilities for Cl (lift coefficient) and Cd (drag coefficient) modeling."""

from pathlib import Path

import numpy as np

# Project root path
PROJECT_ROOT = Path(__file__).parents[3]


def _build_coeff_model(coeff_path: Path, key: str) -> dict | float:
    """Build a speed-dependent coefficient model from inverse-solved data."""
    try:
        arr = np.genfromtxt(coeff_path, delimiter=",", names=True, encoding="utf-8")
        names = set(arr.dtype.names or [])
        if (key not in names) or ("speed" not in names):
            return float("nan")
        speed = np.asarray(arr["speed"], dtype=float)
        coeff = np.asarray(arr[key], dtype=float)
        mask = np.isfinite(speed) & np.isfinite(coeff) & (speed > 0.5)
        if int(mask.sum()) < 5:
            return float("nan")

        speed = speed[mask]
        coeff = coeff[mask]
        # Bin speeds and take median in each bin for robustness
        nbins = 8
        bins = np.linspace(float(np.min(speed)), float(np.max(speed)), nbins + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        medians = []
        centers_keep = []
        for lo, hi, c in zip(bins[:-1], bins[1:], centers):
            m = (speed >= lo) & (speed < hi)
            if int(m.sum()) >= 3:
                centers_keep.append(c)
                medians.append(float(np.nanmedian(coeff[m])))
        if len(medians) < 3:
            return float(np.nanmedian(coeff))

        return {
            "type": "interp",
            "speed": np.asarray(centers_keep, dtype=float),
            "values": np.asarray(medians, dtype=float),
            "median": float(np.nanmedian(coeff)),
        }
    except Exception:
        return float("nan")


def estimate_cl_cd(track_path: Path) -> tuple[dict | float, dict | float]:
    """Estimate speed-dependent Cl/Cd for forward simulation.

    Use inverse-solved coefficients from data/interm/*_coeffs.csv to build a
    speed->Cl/Cd model. Fallback to conservative constants if data is insufficient.
    """
    coeff_path = PROJECT_ROOT / "data" / "interm" / f"{track_path.stem}_coeffs.csv"
    if coeff_path.exists():
        cl_model = _build_coeff_model(coeff_path, "Cl")
        cd_model = _build_coeff_model(coeff_path, "Cd")
        if np.isfinite(float(cl_model)) if isinstance(cl_model, float) else True:
            if np.isfinite(float(cd_model)) if isinstance(cd_model, float) else True:
                return cl_model, cd_model
    # Fallback: close to what inverseSolve.py reports for stable segments
    return 0.10, 0.05


def _eval_coeff(model: dict | float, speed: float) -> float:
    """Evaluate a coefficient model at a given speed."""
    if isinstance(model, dict):
        sp = np.asarray(model.get("speed", []), dtype=float)
        vals = np.asarray(model.get("values", []), dtype=float)
        if sp.size >= 2 and vals.size == sp.size:
            return float(np.interp(speed, sp, vals, left=vals[0], right=vals[-1]))
        if "median" in model:
            return float(model["median"])
        return float("nan")
    return float(model)


def _coeff_summary(model: dict | float) -> float:
    """Return a summary value for a coefficient model."""
    if isinstance(model, dict) and "median" in model:
        return float(model["median"])
    if isinstance(model, dict):
        return float("nan")
    return float(model)
