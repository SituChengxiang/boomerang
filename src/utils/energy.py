"""
Energy analysis utilities for boomerang flight data.

This module provides functions for calculating energy-related quantities
such as kinetic energy, potential energy, power, and energy conservation metrics.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.utils.mathUtils import derivatives_smooth
from src.utils.physicsCons import MASS, G


def calculate_energy_from_dataframe(df: pd.DataFrame) -> tuple:
    """Calculate mechanical energy from DataFrame.

    This function computes kinetic, potential, and total mechanical energy
    from track data contained in a DataFrame.

    Args:
        df: DataFrame containing track data with columns: t, vx, vy, vz, z

    Returns:
        Tuple containing: (t, total_energy, kinetic_energy, potential_energy)
    """
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


def calculate_kinetic_energy(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mass: float = MASS
) -> np.ndarray:
    """Calculate kinetic energy.

    E_kin = 0.5 * m * v^2

    Args:
        vx, vy, vz: Velocity components
        mass: Mass of the boomerang

    Returns:
        E_kin: Kinetic energy array
    """
    speed_sq = vx**2 + vy**2 + vz**2
    return 0.5 * mass * speed_sq


def calculate_potential_energy(
    z: np.ndarray, mass: float = MASS, g: float = G
) -> np.ndarray:
    """Calculate potential energy.

    E_pot = m * g * z

    Args:
        z: Height array
        mass: Mass of the boomerang
        g: Gravitational acceleration

    Returns:
        E_pot: Potential energy array
    """
    return mass * g * z


def calculate_total_energy(E_kin: np.ndarray, E_pot: np.ndarray) -> np.ndarray:
    """Calculate total mechanical energy.

    E_total = E_kin + E_pot

    Args:
        E_kin: Kinetic energy array
        E_pot: Potential energy array

    Returns:
        E_total: Total mechanical energy array
    """
    return E_kin + E_pot


def calculate_energy_components(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    t: np.ndarray,
    mass: float = MASS,
    g: float = G,
) -> Dict[str, np.ndarray]:
    """Calculate all energy components.

    Args:
        x, y, z: Position arrays
        vx, vy, vz: Velocity arrays
        t: Time array
        mass: Mass of the boomerang
        g: Gravitational acceleration

    Returns:
        Dictionary containing energy components and derivatives
    """
    # Calculate energy components
    E_kin = calculate_kinetic_energy(vx, vy, vz, mass)
    E_pot = calculate_potential_energy(z, mass, g)
    E_total = calculate_total_energy(E_kin, E_pot)

    # Calculate energy rates
    dE_kin_dt, _ = derivatives_smooth(t, E_kin)
    dE_pot_dt, _ = derivatives_smooth(t, E_pot)
    dE_total_dt, _ = derivatives_smooth(t, E_total)

    return {
        "E_kin": E_kin,
        "E_pot": E_pot,
        "E_total": E_total,
        "dE_kin_dt": dE_kin_dt,
        "dE_pot_dt": dE_pot_dt,
        "dE_total_dt": dE_total_dt,
    }


def calculate_power(
    Fx: np.ndarray,
    Fy: np.ndarray,
    Fz: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> np.ndarray:
    """Calculate mechanical power.

    P = F · v

    Args:
        Fx, Fy, Fz: Force components
        vx, vy, vz: Velocity components

    Returns:
        P: Power array
    """
    return Fx * vx + Fy * vy + Fz * vz


def calculate_energy_conservation_metrics(
    E_total: np.ndarray, t: np.ndarray
) -> Dict[str, float]:
    """Calculate energy conservation metrics.

    Args:
        E_total: Total energy array
        t: Time array

    Returns:
        Dictionary containing conservation metrics
    """
    # Calculate energy change
    delta_E = E_total[-1] - E_total[0]
    relative_change = delta_E / (np.abs(E_total[0]) + 1e-12)

    # Calculate energy fluctuation
    energy_std = np.std(E_total)
    energy_mean = np.mean(E_total)
    relative_fluctuation = energy_std / (energy_mean + 1e-12)

    return {
        "initial_energy": float(E_total[0]),
        "final_energy": float(E_total[-1]),
        "delta_E": float(delta_E),
        "relative_change": float(relative_change),
        "energy_std": float(energy_std),
        "relative_fluctuation": float(relative_fluctuation),
    }


def calculate_energy_dissipation_rate(E_total: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate energy dissipation rate.

    Args:
        E_total: Total energy array
        t: Time array

    Returns:
        dE_dt: Energy dissipation rate array
    """
    dE_dt, _ = derivatives_smooth(t, E_total)
    return -dE_dt  # Negative because dissipation is loss of energy
