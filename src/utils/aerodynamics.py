"""
Aerodynamics analysis utilities for boomerang flight data.

This module provides functions for calculating aerodynamic forces,
decomposing them into meaningful components, and analyzing their behavior.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from src.utils.mathUtils import safe_unit_vector
from src.utils.physicsCons import MASS, RHO, SIGMA_ROTATION, A, G, S


def calculate_net_aerodynamic_force(
    acceleration: np.ndarray, velocity: np.ndarray, mass: float = MASS
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate net aerodynamic force from acceleration data.

    F_aero = m*a - m*g

    Args:
        acceleration: Nx3 array of [ax, ay, az]
        velocity: Nx3 array of [vx, vy, vz]
        mass: mass of the boomerang

    Returns:
        F_aero: Nx3 array of aerodynamic force
        speed: N array of speed magnitudes
    """
    # Net force: F_net = m * a
    f_net = mass * acceleration

    # Gravity vector: [0, 0, -m*g]
    f_g = np.zeros_like(f_net)
    f_g[:, 2] = -mass * G

    # Aerodynamic force: F_aero = F_net - F_g
    f_aero = f_net - f_g

    # Calculate speed
    speed = np.linalg.norm(velocity, axis=1)

    return f_aero, speed


def decompose_aerodynamic_force(
    f_aero: np.ndarray, velocity: np.ndarray, speed: np.ndarray
) -> Dict[str, np.ndarray]:
    """Decompose aerodynamic force into tangential and normal components.

    Args:
        f_aero: Nx3 array of aerodynamic force
        velocity: Nx3 array of velocity
        speed: N array of speed magnitudes

    Returns:
        Dictionary containing:
        - F_t: tangential component (parallel to velocity)
        - F_n: normal component (perpendicular to velocity, horizontal)
        - F_z: vertical component
        - t_hat: unit tangent vector
        - n_hat: unit normal vector (horizontal)
    """
    # Unit tangent vector (velocity direction)
    t_hat = np.zeros_like(velocity)
    valid_mask = speed > 0.05
    t_hat[valid_mask] = velocity[valid_mask] / speed[valid_mask, np.newaxis]

    # Horizontal velocity components
    v_h = np.sqrt(velocity[:, 0] ** 2 + velocity[:, 1] ** 2)
    v_h = np.maximum(v_h, 1e-6)  # Avoid division by zero

    # Unit horizontal tangent vector
    t_h_hat = np.zeros_like(velocity)
    t_h_hat[:, 0] = velocity[:, 0] / v_h
    t_h_hat[:, 1] = velocity[:, 1] / v_h

    # Unit normal vector (z × t_h)
    z_hat = np.array([0, 0, 1])
    n_hat = np.cross(z_hat, t_h_hat)

    # Project aerodynamic force
    f_t = np.einsum("ij,ij->i", f_aero, t_hat)
    f_n = np.einsum("ij,ij->i", f_aero, n_hat)
    f_z = f_aero[:, 2]

    return {"F_t": f_t, "F_n": f_n, "F_z": f_z, "t_hat": t_hat, "n_hat": n_hat}


def calculate_effective_coefficients(
    F_components: Dict[str, np.ndarray],
    speed: np.ndarray,
    v_rot: float | np.ndarray = 0.0,
) -> Dict[str, np.ndarray]:
    """Calculate effective coefficients with optional rotational correction.

    Args:
        F_components: Dictionary from decompose_aerodynamic_force
        speed: N array of speed magnitudes
        v_rot: Rotational velocity for correction (default 0.0)

    Returns:
        Dictionary containing:
        - C_n_eff: Normal force coefficient
        - C_z_eff: Vertical force coefficient
        - C_t_eff: Tangential force coefficient
        - q: Dynamic pressure (with rotational correction)
    """
    # Calculate dynamic pressure with rotational correction
    # v_rot can be a scalar or an array aligned with speed.
    v_rot_arr = np.asarray(v_rot, dtype=float)
    v_eff_sq = speed**2 + v_rot_arr**2
    q = 0.5 * RHO * S * v_eff_sq

    # Calculate effective coefficients (use NaN outside valid q range)
    C_n_eff = np.full_like(F_components["F_n"], np.nan, dtype=float)
    C_z_eff = np.full_like(F_components["F_z"], np.nan, dtype=float)
    C_t_eff = np.full_like(F_components["F_t"], np.nan, dtype=float)

    valid_mask = q > 1e-6
    C_n_eff[valid_mask] = F_components["F_n"][valid_mask] / q[valid_mask]
    C_z_eff[valid_mask] = F_components["F_z"][valid_mask] / q[valid_mask]
    C_t_eff[valid_mask] = F_components["F_t"][valid_mask] / q[valid_mask]

    return {"C_n_eff": C_n_eff, "C_z_eff": C_z_eff, "C_t_eff": C_t_eff, "q": q}


def find_optimal_v_rot(
    F_components: Dict[str, np.ndarray],
    speed: np.ndarray,
    v_rot_range: np.ndarray = np.linspace(0, 5, 50),
) -> Tuple[float, Dict[str, np.ndarray]]:
    """Find optimal v_rot that flattens coefficient curves.

    Args:
        F_components: Dictionary from decompose_aerodynamic_force
        speed: N array of speed magnitudes
        v_rot_range: Range of v_rot values to test

    Returns:
        Tuple of (optimal_v_rot, coefficients_at_optimal_v_rot)
    """
    best_v_rot = 0.0
    best_score = float("inf")
    best_coeffs: Dict[str, np.ndarray] = {}

    for v_rot in v_rot_range:
        coeffs = calculate_effective_coefficients(F_components, speed, v_rot)

        # Calculate "flatness" score (variance of coefficients) on valid samples only.
        # This avoids low-speed/low-q regions dominating the score.
        q = coeffs.get("q")
        if q is None:
            continue
        valid = (q > 1e-6) & (speed > 0.5)
        if int(np.sum(valid)) < 8:
            continue

        score = (
            float(np.nanvar(coeffs["C_n_eff"][valid]))
            + float(np.nanvar(coeffs["C_z_eff"][valid]))
            + float(np.nanvar(coeffs["C_t_eff"][valid]))
        )

        if score < best_score:
            best_score = score
            best_v_rot = v_rot
            best_coeffs = coeffs

    return best_v_rot, best_coeffs


def calculate_dynamic_pressure(speed: np.ndarray, v_rot: float = 0.0) -> np.ndarray:
    """Calculate dynamic pressure with optional rotational correction.

    q = 0.5 * rho * S * (v^2 + v_rot^2)

    Args:
        speed: N array of speed magnitudes
        v_rot: Rotational velocity for correction

    Returns:
        Dynamic pressure array
    """
    v_eff_sq = speed**2 + v_rot**2
    return 0.5 * RHO * S * v_eff_sq


def lift_direction_from_spin_axis(s_hat: np.ndarray, v_hat: np.ndarray) -> np.ndarray:
    """Calculate lift direction from spin axis and velocity.

    For a CLOCKWISE spinning boomerang (viewed from above), the angular velocity
    points downward (-Z by right-hand rule). The lift direction is:
        lift_dir ∝ v × s (velocity cross spin axis)

    Args:
        s_hat: Unit spin axis vector
        v_hat: Unit velocity vector

    Returns:
        Unit lift direction vector
    """
    lift_raw = np.cross(v_hat, s_hat)  # v × s for clockwise rotation
    return safe_unit_vector(lift_raw, np.array([0.0, 0.0, 1.0]))


def calculate_vertical_aero_force(
    az: np.ndarray, mass: float = MASS, g: float = G
) -> np.ndarray:
    """Calculate vertical aerodynamic force.

    F_z_aero = m*az - m*g (vertical component of F_aero)

    Args:
        az: Vertical acceleration array
        mass: Mass of the boomerang
        g: Gravitational acceleration

    Returns:
        F_z_aero: Vertical aerodynamic force array
    """
    return mass * az - mass * g


def calculate_drag_force_magnitude(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    mass: float = MASS,
) -> np.ndarray:
    """Calculate drag force magnitude.

    Drag is the component of aerodynamic force opposite to velocity direction.

    Args:
        ax, ay, az: Acceleration components
        vx, vy, vz: Velocity components
        mass: Mass of the boomerang

    Returns:
        F_drag_mag: Drag force magnitude array
        :param ax:
    """
    # Net force
    F_net = mass * np.column_stack((ax, ay, az))

    # Gravity
    F_g = np.zeros_like(F_net)
    F_g[:, 2] = -mass * G

    # Aerodynamic force
    F_aero = F_net - F_g

    # Velocity unit vector
    speed = np.linalg.norm(np.column_stack((vx, vy, vz)), axis=1)
    v_hat = np.zeros_like(F_aero)
    valid_mask = speed > 0.05
    v_hat[valid_mask] = (
        np.column_stack((vx, vy, vz))[valid_mask] / speed[valid_mask, np.newaxis]
    )

    # Drag is the component opposite to velocity
    F_parallel = np.einsum("ij,ij->i", F_aero, v_hat)

    # Drag magnitude (opposing component)
    return -np.minimum(F_parallel, 0.0)


def calculate_aerodynamic_coefficients_from_data(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    speed: np.ndarray,
    omega: Optional[np.ndarray] = None,
    mass: float = MASS,
    sigma: float = SIGMA_ROTATION,
) -> Dict[str, np.ndarray]:
    """Calculate aerodynamic coefficients from flight data.

    Args:
        ax, ay, az: Acceleration components
        vx, vy, vz: Velocity components
        speed: Speed magnitude array
        omega: Angular velocity array (optional)
        mass: Mass of the boomerang
        sigma: Rotational correction factor

    Returns:
        Dictionary containing aerodynamic coefficients
    """
    # Calculate net aerodynamic force
    f_aero, _ = calculate_net_aerodynamic_force(
        np.column_stack((ax, ay, az)), np.column_stack((vx, vy, vz)), mass
    )

    # Decompose into components
    f_components = decompose_aerodynamic_force(
        f_aero, np.column_stack((vx, vy, vz)), speed
    )

    # Calculate rotational velocity if provided
    v_rot: float | np.ndarray = 0.0
    if omega is not None:
        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim == 0:
            omega_use = float(omega_arr)
        elif omega_arr.ndim == 1:
            omega_use = omega_arr
        elif omega_arr.ndim == 2 and omega_arr.shape[1] == 3:
            # If provided as Nx3 angular velocity vectors, use z-component by convention.
            omega_use = omega_arr[:, 2]
        else:
            # Fallback: try to squeeze to 1D
            omega_use = np.squeeze(omega_arr)
        v_rot = sigma * omega_use * A  # A is arm length from physicsCons

    # Calculate coefficients
    coeffs = calculate_effective_coefficients(f_components, speed, v_rot)

    return {
        "Cl": coeffs["C_n_eff"],  # Normal coefficient as lift coefficient
        "Cd": coeffs["C_t_eff"],  # Tangential coefficient as drag coefficient
        "Cz": coeffs["C_z_eff"],  # Vertical coefficient
        "q": coeffs["q"],  # Dynamic pressure
        "F_lift": f_components["F_n"],  # Lift force magnitude
        "F_drag": f_components["F_t"],  # Drag force magnitude
        "F_vertical": f_components["F_z"],  # Vertical force
    }
