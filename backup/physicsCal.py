#!/usr/bin/env python3
"""Physics module for boomerang trajectory analysis.

Contains:
- Physical constants (gravity, air density)
- Boomerang parameters (mass, dimensions, moment of inertia)
- Energy calculations (kinetic, potential, total, dE/dt)
- Force calculations (drag, lift, aerodynamic forces)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# ==========================================
# Physical Constants
# ==========================================


G = 9.793  # Gravitational acceleration (m/s^2)
RHO_AIR = 1.204  # Standard air density at 20°C, 1 atm (kg/m^3)
MASS = 0.00218  # Boomerang mass (kg)


@dataclass
class BoomerangParams:
    """Physical parameters of the paper boomerang."""

    mass: float = MASS  # Mass (kg)
    rho: float = RHO_AIR  # Air density (kg/m^3)
    g: float = G  # Gravity (m/s^2)
    wing_area: float = 1.8e-3  # Wing area (m^2) - approximate
    c_lift: float = 0.8  # Lift coefficient (dimensionless) For a flat plate at optimal angle of attack
    c_drag: float = 0.12  # Drag coefficient (dimensionless) Guess value for boomerang
    arm_length: float = 0.25  # Arm length (m)
    i_total: float = 6.8e-5  # Moment of inertia (kg·m^2) Approximate as two point masses at arm length


# Global instance with default parameters
DEFAULT_BOOMERANG = BoomerangParams()


# ==========================================
# Energy Calculations
# ==========================================


def calculate_kinetic_energy(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, mass: float = MASS
) -> np.ndarray:
    """Calculate kinetic energy per unit mass or with mass.

    Args:
        vx: x-velocity (m/s)
        vy: y-velocity (m/s)
        vz: z-velocity (m/s)
        mass: Mass (kg) - if 1.0, returns per unit mass; otherwise returns energy

    Returns:
        Kinetic energy (J)
    """
    return 0.5 * mass * (vx**2 + vy**2 + vz**2)


def calculate_potential_energy(
    z: np.ndarray, mass: float = MASS, g: float = G
) -> np.ndarray:
    """Calculate gravitational potential energy per unit mass or with mass.

    Args:
        z: Height (m)
        mass: Mass (kg)
        g: Gravitational acceleration (m/s^2)

    Returns:
        Potential energy (J)
    """
    return mass * g * z


def calculate_total_energy(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: Optional[np.ndarray] = None,
    vy: Optional[np.ndarray] = None,
    vz: Optional[np.ndarray] = None,
    mass: float = MASS,
    g: float = G,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate total mechanical energy and its time derivative.

    For trajectories with velocity arrays provided:
        E/m = 0.5*(vx² + vy² + vz²) + g*z
        dE/dt = v·a + g*vz

    For trajectories with only position (no velocity):
        Uses numerical derivatives via derivatives module

    Args:
        t: Time array (s)
        x, y, z: Position arrays (m)
        vx, vy, vz: Velocity arrays (m/s) - optionally provided
        mass: Mass (kg)
        g: Gravitational acceleration (m/s^2)

    Returns:
        Tuple of (total_energy, dE_dt)
            - total_energy: Total mechanical energy per unit mass (J/kg)
            - dE_dt: Time derivative of energy (W/kg)

    Example:
        >>> t = np.array([0, 1, 2])
        >>> x, y, z = np.array([0, 1, 2]), np.array([0, 0, 0]), np.array([0, -5, -10])
        >>> vx, vy, vz = np.array([0, 1, 1]), np.array([0, 0, 0]), np.array([0, -5, -5])
        >>> energy, dE_dt = calculate_total_energy(t, x, y, z, vx, vy, vz)
    """
    from derivatives import compute_derivatives

    # If velocities not provided, compute derivatives
    if vx is None or vy is None or vz is None:
        derives = compute_derivatives(t, x, y, z, method="auto")
        vx, vy, vz = derives.vx, derives.vy, derives.vz
        ax, ay, az = derives.ax, derives.ay, derives.az
    else:
        # Compute accelerations from velocities.
        # NOTE: compute_derivatives returns:
        # - .vx as first derivative of the provided x input
        # - .ax as second derivative of the provided x input
        # So when we pass velocity arrays as inputs, acceleration is in .vx/.vy/.vz.
        derives_acc = compute_derivatives(t, vx, vy, vz, method="auto", edge_order=2)
        ax, ay, az = derives_acc.vx, derives_acc.vy, derives_acc.vz

    # Kinetic energy per unit mass: 0.5 * v^2
    kinetic_energy = 0.5 * (vx**2 + vy**2 + vz**2)

    # Potential energy per unit mass: g * z
    potential_energy = g * z

    # Total mechanical energy per unit mass
    total_energy = kinetic_energy + potential_energy

    # Time derivative: dE/dt = v·a + g*vz
    # (from E = 0.5*v^2 + g*z => dE/dt = v·a + g*vz)
    dE_dt = vx * ax + vy * ay + vz * az + g * vz

    return total_energy, dE_dt


def calculate_energy_per_unit_mass(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: Optional[np.ndarray] = None,
    vy: Optional[np.ndarray] = None,
    vz: Optional[np.ndarray] = None,
    g: float = G,
) -> Tuple[np.ndarray, np.ndarray]:
    """Alias for calculate_total_energy with mass=1 (per unit mass).

    Returns:
        Tuple of (energy_per_mass, dE_dt)
    """
    return calculate_total_energy(t, x, y, z, vx, vy, vz, mass=1.0, g=g)


# ==========================================
# Force Calculations
# ==========================================


def calculate_drag_force(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho: float = RHO_AIR,
    area: float = 1.8e-3,
    c_drag: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate drag force components (simplified model).

    Uses drag equation: F_drag = 0.5 * ρ * v^2 * C_d * A
    Orientation is assumed opposite to velocity direction.

    Args:
        vx, vy, vz: Velocity components (m/s)
        rho: Air density (kg/m^3)
        area: Reference area (m^2)
        c_drag: Drag coefficient

    Returns:
        Tuple of (F_drag_x, F_drag_y, F_drag_z)
    """
    speed_sq = vx**2 + vy**2 + vz**2
    speed = np.sqrt(speed_sq)
    speed = np.where(speed > 0, speed, 1e-6)  # Avoid division by zero

    # Drag magnitude
    F_drag_mag = 0.5 * rho * speed_sq * c_drag * area

    # Drag direction is opposite to velocity
    F_drag_x = -F_drag_mag * (vx / speed)
    F_drag_y = -F_drag_mag * (vy / speed)
    F_drag_z = -F_drag_mag * (vz / speed)

    return F_drag_x, F_drag_y, F_drag_z


def calculate_lift_force(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho: float = RHO_AIR,
    area: float = 1.8e-3,
    c_lift: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate lift force components (simplified model).

    Uses lift equation: F_lift = 0.5 * ρ * v^2 * C_l * A
    Lift is perpendicular to velocity direction (vertical lift).

    Args:
        vx, vy, vz: Velocity components (m/s)
        rho: Air density (kg/m^3)
        area: Reference area (m^2)
        c_lift: Lift coefficient

    Returns:
        Tuple of (F_lift_x, F_lift_y, F_lift_z)
        For a simplified vertical lift model:
        - F_lift_z = 0.5 * ρ * (vx² + vy²) * C_l * A  (vertical lift)
        - F_lift_x = F_lift_y = 0
    """
    # Horizontal speed squared for lift calculation
    v_horizontal = vx**2 + vy**2

    # Lift magnitude (vertical)
    F_lift_z = 0.5 * rho * v_horizontal * c_lift * area

    # Assume lift is purely vertical for this simplified model
    F_lift_x = np.zeros_like(vx)
    F_lift_y = np.zeros_like(vy)

    return F_lift_x, F_lift_y, F_lift_z


def calculate_aerodynamic_forces(
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    rho: float = RHO_AIR,
    area: float = 1.8e-3,
    c_lift: float = 0.8,
    c_drag: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate both lift and drag forces.

    Args:
        vx, vy, vz: Velocity components (m/s)
        rho: Air density (kg/m^3)
        area: Reference area (m^2)
        c_lift: Lift coefficient
        c_drag: Drag coefficient

    Returns:
        Tuple of (F_lift_x, F_lift_y, F_lift_z, F_drag_x, F_drag_y, F_drag_z)
    """
    F_lift_x, F_lift_y, F_lift_z = calculate_lift_force(vx, vy, vz, rho, area, c_lift)
    F_drag_x, F_drag_y, F_drag_z = calculate_drag_force(vx, vy, vz, rho, area, c_drag)
    return F_lift_x, F_lift_y, F_lift_z, F_drag_x, F_drag_y, F_drag_z


# ==========================================
# Heat/Energy Loss Calculations
# ==========================================


def calculate_dissipative_energy_rate(
    t: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    ax: Optional[np.ndarray] = None,
    ay: Optional[np.ndarray] = None,
    az: Optional[np.ndarray] = None,
    mass: float = MASS,
    g: float = G,
) -> np.ndarray:
    """Calculate rate of energy dissipation (power loss).

    For energy conservation:
        dE/dt = F_aero · v (aerodynamic work)

    With air resistance, dE/dt = -P_drag

    Args:
        t: Time array
        vx, vy, vz: Velocity (m/s)
        ax, ay, az: Acceleration (m/s²) - computed if None
        mass: Mass (kg)
        g: Gravitational acceleration (m/s²)

    Returns:
        Power dissipation (W) = -dE/dt
    """
    from derivatives import compute_derivatives

    # Compute accelerations if not provided
    if ax is None or ay is None or az is None:
        # Pass velocities as "positions" to get their derivatives (accelerations)
        derives = compute_derivatives(t, vx, vy, vz, method="gradient", edge_order=2)
        ax, ay, az = derives.ax, derives.ay, derives.az

    # From energy equation: dE/dt = v·a + g*vz
    # If we know actual dE/dt < 0, then power lost:
    dE_dt = vx * ax + vy * ay + vz * az + g * vz

    # Negative value indicates energy loss (dissipation)
    return -dE_dt


# ==========================================
# Validation Functions
# ==========================================


def validate_energy_conservation(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: Optional[np.ndarray] = None,
    vy: Optional[np.ndarray] = None,
    vz: Optional[np.ndarray] = None,
    tolerance: float = 0.1,
) -> Tuple[bool, np.ndarray, float, str]:
    """Validate energy conservation for a trajectory.

    Checks if energy changes are reasonable (close to 0 for ideal flight,
    or negative for energy dissipation).

    Args:
        t, x, y, z: Trajectory data
        vx, vy, vz: Velocity data (computed if None)
        tolerance: Maximum allowed |dE/dt| for validation

    Returns:
        Tuple of:
        - is_valid (bool): Whether trajectory passes validation
        - dE_dt (np.ndarray): Energy change rate
        - mean_dE_dt (float): Mean dE/dt
        - message (str): Validation message
    """
    total_energy, dE_dt = calculate_total_energy(t, x, y, z, vx, vy, vz)

    mean_dE_dt = float(np.mean(dE_dt))
    max_abs_dE_dt = float(np.max(np.abs(dE_dt)))

    if max_abs_dE_dt <= tolerance:
        is_valid = True
        message = f"✓ Energy conservation OK (|dE/dt| ≤ {tolerance:.3f}, max={max_abs_dE_dt:.3f})"
    elif mean_dE_dt > tolerance:
        is_valid = False
        message = (
            f"✗ Energy increasing (mean dE/dt={mean_dE_dt:.3f}) - check for errors"
        )
    else:
        is_valid = False
        message = f"⚠ Energy dissipation high (max |dE/dt|={max_abs_dE_dt:.3f})"

    return is_valid, dE_dt, mean_dE_dt, message


# ==========================================
# Summary/Stats Functions
# ==========================================


def get_energy_stats(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    vx: Optional[np.ndarray] = None,
    vy: Optional[np.ndarray] = None,
    vz: Optional[np.ndarray] = None,
) -> dict:
    """Calculate and return energy statistics.

    Args:
        t, x, y, z: Trajectory data
        vx, vy, vz: Velocity data

    Returns:
        Dictionary with energy statistics
    """
    energy, dE_dt = calculate_total_energy(t, x, y, z, vx, vy, vz)

    return {
        "energy_mean": float(np.mean(energy)),
        "energy_std": float(np.std(energy)),
        "energy_min": float(np.min(energy)),
        "energy_max": float(np.max(energy)),
        "dE_dt_mean": float(np.mean(dE_dt)),
        "dE_dt_std": float(np.std(dE_dt)),
        "dE_dt_abs_mean": float(np.mean(np.abs(dE_dt))),
        "dE_dt_max": float(np.max(dE_dt)),
        "dE_dt_min": float(np.min(dE_dt)),
    }


# ==========================================
# Utility Functions
# ==========================================


def get_default_params(override_mass: Optional[float] = None) -> BoomerangParams:
    """Get default boomerang parameters with optional override.

    Args:
        override_mass: If provided, use this mass instead of default

    Returns:
        BoomerangParams instance
    """
    if override_mass is not None:
        return BoomerangParams(mass=override_mass)
    return DEFAULT_BOOMERANG


def get_constants_summary() -> dict:
    """Get summary of physical constants.

    Returns:
        Dictionary with constants
    """
    return {
        "G": G,
        "RHO_AIR": RHO_AIR,
        "MASS": MASS,
        "description": "Gravitational acceleration (m/s²), air density (kg/m³), boomerang mass (kg)",
    }


# ==========================================
# Module-level constants (for backward compatibility)
# ==========================================

# Re-export some common constants for ease of use
GRAVITY_G = G
AIR_DENSITY = RHO_AIR
BOOMERANG_MASS = MASS

__all__ = [
    # Constants
    "G",
    "GRAVITY_G",
    "RHO_AIR",
    "AIR_DENSITY",
    "MASS",
    "BOOMERANG_MASS",
    # Classes
    "BoomerangParams",
    # Energy functions
    "calculate_kinetic_energy",
    "calculate_potential_energy",
    "calculate_total_energy",
    "calculate_energy_per_unit_mass",
    # Force functions
    "calculate_drag_force",
    "calculate_lift_force",
    "calculate_aerodynamic_forces",
    # Utility functions
    "validate_energy_conservation",
    "get_energy_stats",
    "get_default_params",
    "get_constants_summary",
]

if __name__ == "__main__":
    # Demo
    print("Physics module loaded")
    print(f"Constants: G={G}, RHO_AIR={RHO_AIR}, MASS={MASS}")

    # Simple test
    t = np.array([0, 1, 2], dtype=float)
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([0, 0, 0], dtype=float)
    z = np.array([0, -5, -10], dtype=float)

    # Add explicit velocities for test
    vx = np.array([0.5, 0.5, 0.5], dtype=float)
    vy = np.array([0.0, 0.0, 0.0], dtype=float)
    vz = np.array([-5.0, -5.0, -5.0], dtype=float)

    energy, dE_dt = calculate_total_energy(t, x, y, z, vx, vy, vz)
    print("\nDemo trajectory energy stats:")
    print(f"  Energy (per unit mass): {float(np.mean(energy)):.3f} J/kg")
    print(f"  Energy change rate: {float(np.mean(dE_dt)):.3f} W/kg")

    # Test energy validation
    is_valid, dE_dt, mean_dE_dt, message = validate_energy_conservation(
        t, x, y, z, vx, vy, vz
    )
    print(f"\nEnergy validation: {message}")
