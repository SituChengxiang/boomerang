"""Kinematics analysis utilities for boomerang flight data.

This module provides functions for calculating kinematic quantities
such as velocity, acceleration, heading, and other motion-related properties.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from src.utils.mathUtils import derivatives_smooth, unwrap_angle_rad
from src.utils.physicsCons import G


def analyze_track(df: pd.DataFrame, track_name: str = "") -> tuple:
    """Analyze track data and compute various kinematic quantities.

    This function computes a comprehensive set of kinematic quantities from track data,
    including accelerations, forces, heading rates, and power.

    Args:
        df: DataFrame containing track data with columns: t, vx, vy, vz, (x, y, z optional)
        track_name: Optional track name for identification

    Returns:
        Tuple containing:
        (t, v_xy_sq, f_z_aero, v_total_sq, a_drag_est, a_perp_h, power_per_mass,
         v_h, a_perp_h_signed, a_perp_h_curv, a_perp_h_curv_signed, v_h_xy,
         heading_deg, heading_rate_deg, heading_rate_from_cross_deg, heading_rate_xy_deg)
    """
    t = np.asarray(df.t.values)
    vx = np.asarray(df.vx.values)
    vy = np.asarray(df.vy.values)
    vz = np.asarray(df.vz.values)
    x = np.asarray(df.x.values) if "x" in df.columns else None
    y = np.asarray(df.y.values) if "y" in df.columns else None

    # 1. Compute acceleration from DataFrame
    ax_val, ay_val, az_val = compute_acceleration_from_dataframe(df, t, vx, vy, vz)

    # 2. Physics Variables
    v_xy_sq = vx**2 + vy**2
    v_total_sq = vx**2 + vy**2 + vz**2

    # 3. Vertical Force Analysis
    f_z_aero = compute_vertical_force_analysis(az_val)

    # 4. Drag Analysis
    a_tan, a_drag_est = compute_drag_analysis(ax_val, ay_val, az_val, vx, vy, vz)

    # 5. Horizontal perpendicular acceleration (centripetal acceleration in horizontal plane)
    v_h, a_perp_h, a_perp_h_signed = compute_horizontal_centripetal_from_velocity(
        vx, vy, ax_val, ay_val
    )

    # Cross product for heading rate calculation
    v_cross_a_h = vx * ay_val - vy * ax_val

    # 6. Curvature-based horizontal centripetal acceleration from x(t), y(t)
    a_perp_h_curv, a_perp_h_curv_signed, v_h_xy = compute_curvature_based_acceleration(
        t, x, y
    )

    # 7. Heading rate consistency checks
    heading_deg, heading_rate_deg, heading_rate_from_cross_deg = (
        compute_heading_analysis(vx, vy, t, v_cross_a_h, v_xy_sq)
    )

    # 8. Heading rate from position (if available)
    heading_rate_xy_deg = None
    if x is not None and y is not None and len(t) >= 5:
        dx, _ = derivatives_smooth(t, x)
        dy, _ = derivatives_smooth(t, y)
        heading_xy = unwrap_angle_rad(np.arctan2(dy, dx))
        heading_rate_xy_deg = np.degrees(np.gradient(heading_xy, t))

    # 9. Power: P = F·v = m·a·v + m·g·v_z
    power_per_mass = compute_power_analysis(ax_val, ay_val, az_val, vx, vy, vz)

    return (
        t,
        v_xy_sq,
        f_z_aero,
        v_total_sq,
        a_drag_est,
        a_perp_h,
        power_per_mass,
        # --- extra diagnostics (appended for backwards compatibility) ---
        v_h,
        a_perp_h_signed,
        a_perp_h_curv,
        a_perp_h_curv_signed,
        v_h_xy,
        heading_deg,
        heading_rate_deg,
        heading_rate_from_cross_deg,
        heading_rate_xy_deg,
    )


def compute_centripetal_force_demands(
    analysis_result: tuple,
    mass: float,
) -> dict[str, np.ndarray]:
    """Build centripetal-force demand series from analyze_track output.

    Returns three force-demand series in horizontal plane:
    - vector: from cross-product acceleration
    - heading: from heading-rate acceleration
    - curvature: from curvature-based acceleration
    """
    v_h = np.asarray(analysis_result[7], dtype=float)
    a_perp_vec_signed = np.asarray(analysis_result[8], dtype=float)

    if analysis_result[10] is None:
        a_perp_curv_signed = np.full_like(v_h, np.nan)
    else:
        a_perp_curv_signed = np.asarray(analysis_result[10], dtype=float)

    heading_rate_deg = analysis_result[13]
    if heading_rate_deg is None:
        heading_rate_deg = analysis_result[14]

    if heading_rate_deg is None:
        a_perp_heading_signed = np.full_like(v_h, np.nan)
    else:
        psi_dot = np.deg2rad(np.asarray(heading_rate_deg, dtype=float))
        a_perp_heading_signed = v_h * psi_dot

    return {
        "vector": float(mass) * a_perp_vec_signed,
        "heading": float(mass) * a_perp_heading_signed,
        "curvature": float(mass) * a_perp_curv_signed,
    }


def calculate_velocity_from_position(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate velocity components from position data.

    Args:
        x, y, z: Position arrays
        t: Time array

    Returns:
        vx, vy, vz: Velocity components
    """
    vx, _ = derivatives_smooth(t, x)
    vy, _ = derivatives_smooth(t, y)
    vz, _ = derivatives_smooth(t, z)
    return vx, vy, vz


def calculate_acceleration_from_velocity(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate acceleration components from velocity data.

    Args:
        vx, vy, vz: Velocity arrays
        t: Time array

    Returns:
        ax, ay, az: Acceleration components
    """
    ax, _ = derivatives_smooth(t, vx)
    ay, _ = derivatives_smooth(t, vy)
    az, _ = derivatives_smooth(t, vz)
    return ax, ay, az


def calculate_speed(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Calculate speed magnitude from velocity components.

    Args:
        vx, vy, vz: Velocity components

    Returns:
        speed: Speed magnitude array
    """
    return np.sqrt(vx**2 + vy**2 + vz**2)


def calculate_horizontal_speed(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Calculate horizontal speed magnitude.

    Args:
        vx, vy: Horizontal velocity components

    Returns:
        v_h: Horizontal speed magnitude
    """
    return np.sqrt(vx**2 + vy**2)


def calculate_heading(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Calculate heading angle from velocity components.

    Args:
        vx, vy: Horizontal velocity components

    Returns:
        heading: Heading angle in radians
    """
    return np.arctan2(vy, vx)


def calculate_heading_rate(vx: np.ndarray, vy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Calculate heading rate from velocity components.

    Args:
        vx, vy: Horizontal velocity components
        t: Time array

    Returns:
        heading_rate: Heading rate in rad/s
    """
    heading = calculate_heading(vx, vy)
    heading_unwrapped = unwrap_angle_rad(heading)
    heading_rate, _ = derivatives_smooth(t, heading_unwrapped)
    return heading_rate


def calculate_perpendicular_acceleration(
    vx: np.ndarray, vy: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Calculate perpendicular acceleration in horizontal plane.

    a_perp_h = |v_h x a_h| / |v_h| = |vx*ay - vy*ax| / sqrt(vx^2+vy^2)

    Args:
        vx, vy: Horizontal velocity components
        t: Time array

    Returns:
        a_perp: Perpendicular acceleration magnitude
    """
    ax, _ = derivatives_smooth(t, vx)
    ay, _ = derivatives_smooth(t, vy)

    v_h = calculate_horizontal_speed(vx, vy)
    v_cross_a_h = vx * ay - vy * ax

    # Avoid division by zero
    v_h = np.maximum(v_h, 1e-6)
    return np.abs(v_cross_a_h) / v_h


def calculate_pitch_angle(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray) -> np.ndarray:
    """Calculate pitch angle from velocity components.

    Args:
        vx, vy, vz: Velocity components

    Returns:
        pitch: Pitch angle in radians
    """
    v_h = calculate_horizontal_speed(vx, vy)
    return np.arctan2(vz, v_h)


def calculate_angular_velocity_from_heading(
    heading: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Calculate angular velocity from heading angle.

    Args:
        heading: Heading angle array
        t: Time array

    Returns:
        omega: Angular velocity in rad/s
    """
    heading_unwrapped = unwrap_angle_rad(heading)
    omega, _ = derivatives_smooth(t, heading_unwrapped)
    return omega


def horizontal_centripetal_accel(
    t: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Calculate horizontal-plane centripetal acceleration magnitude.

    a_perp_h = |v_h x a_h| / |v_h| = |vx*ay - vy*ax| / sqrt(vx^2+vy^2)

    Args:
        t: Time array
        vx: X-velocity array
        vy: Y-velocity array

    Returns:
        Centripetal acceleration magnitude
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


def horizontal_centripetal_accel_signed(
    t: np.ndarray, vx: np.ndarray, vy: np.ndarray
) -> np.ndarray:
    """Calculate signed horizontal-plane centripetal acceleration.

    Positive/negative indicates turn direction in the horizontal plane.

    Args:
        t: Time array
        vx: X-velocity array
        vy: Y-velocity array

    Returns:
        Signed centripetal acceleration
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


def compute_acceleration_from_dataframe(
    df: pd.DataFrame, t: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute acceleration from DataFrame, using existing columns if available or calculating from velocity.

    Args:
        df: DataFrame containing track data
        t: Time array
        vx: X-velocity array
        vy: Y-velocity array
        vz: Z-velocity array

    Returns:
        ax, ay, az: Acceleration components
    """
    # Use ax, ay, az from CSV if available, otherwise compute them
    if "ax" in df.columns and "ay" in df.columns and "az" in df.columns:
        ax_val = np.asarray(df.ax.values)
        ay_val = np.asarray(df.ay.values)
        az_val = np.asarray(df.az.values)
    else:
        # Fallback to computing smooth acceleration if not present
        # Using savgol filter again for smooth derivatives
        dt = np.mean(np.diff(t))
        window = min(11, len(t))
        if window % 2 == 0:
            window -= 1
        if window < 5:
            window = 3

        dt_float = float(dt)
        ax_val = savgol_filter(vx, window, 3, deriv=1, delta=dt_float)
        ay_val = savgol_filter(vy, window, 3, deriv=1, delta=dt_float)
        az_val = savgol_filter(vz, window, 3, deriv=1, delta=dt_float)

    return ax_val, ay_val, az_val


def compute_vertical_force_analysis(az_val: np.ndarray) -> np.ndarray:
    """Compute vertical aerodynamic force analysis.

    Args:
        az_val: Vertical acceleration array

    Returns:
        f_z_aero: Vertical aerodynamic force per unit mass
    """
    # F_aero_z / m = az + G
    return az_val + G


def compute_drag_analysis(
    ax_val: np.ndarray,
    ay_val: np.ndarray,
    az_val: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute drag analysis including tangential acceleration and drag estimate.

    Args:
        ax_val, ay_val, az_val: Acceleration components
        vx, vy, vz: Velocity components

    Returns:
        a_tan: Tangential acceleration
        a_drag_est: Drag acceleration estimate
    """
    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)

    # Tangential acceleration (along velocity vector)
    # a_tan = dot(a, v) / |v|
    a_tan = (ax_val * vx + ay_val * vy + az_val * vz) / (v_abs + 1e-6)

    # Drag is roughly -a_tan (retarding force), but Gravity also has a component along path.
    # a_tan_measured = a_drag + a_gravity_tangent
    # a_gravity_tangent = -g * (vz / v_abs)  (since g points down -z)
    # So a_drag = a_tan_measured - (-g * vz / v_abs) = a_tan + g * vz / v_abs
    a_drag_est = a_tan + G * (vz / (v_abs + 1e-6))

    return a_tan, a_drag_est


def compute_horizontal_centripetal_from_velocity(
    vx: np.ndarray, vy: np.ndarray, ax_val: np.ndarray, ay_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute horizontal centripetal acceleration from velocity and acceleration.

    Args:
        vx, vy: Horizontal velocity components
        ax_val, ay_val: Horizontal acceleration components

    Returns:
        v_h: Horizontal speed
        a_perp_h: Horizontal centripetal acceleration magnitude
        a_perp_h_signed: Signed horizontal centripetal acceleration
    """
    v_h = calculate_horizontal_speed(vx, vy)

    # Cross product magnitude in horizontal plane (z-component only)
    v_cross_a_h = vx * ay_val - vy * ax_val
    a_perp_h = np.abs(v_cross_a_h) / (v_h + 1e-6)
    a_perp_h_signed = v_cross_a_h / (v_h + 1e-6)

    return v_h, a_perp_h, a_perp_h_signed


def compute_curvature_based_acceleration(
    t: np.ndarray, x: Optional[np.ndarray], y: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute curvature-based horizontal centripetal acceleration from position data.

    Args:
        t: Time array
        x: X-position array (can be None)
        y: Y-position array (can be None)

    Returns:
        a_perp_h_curv: Curvature-based centripetal acceleration magnitude (None if input is None)
        a_perp_h_curv_signed: Signed curvature-based centripetal acceleration (None if input is None)
        v_h_xy: Horizontal speed from position derivatives (None if input is None)
    """
    if x is None or y is None or len(t) != len(x) or len(t) != len(y) or len(t) < 5:
        return None, None, None

    dx, d2x = derivatives_smooth(t, x)
    dy, d2y = derivatives_smooth(t, y)
    v_h_xy = np.sqrt(dx * dx + dy * dy + 1e-12)
    num = dx * d2y - dy * d2x
    a_perp_h_curv_signed = num / (v_h_xy + 1e-6)
    a_perp_h_curv = np.abs(a_perp_h_curv_signed)

    return a_perp_h_curv, a_perp_h_curv_signed, v_h_xy


def compute_heading_analysis(
    vx: np.ndarray,
    vy: np.ndarray,
    t: np.ndarray,
    v_cross_a_h: np.ndarray,
    v_h_sq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute heading and heading rate analysis.

    Args:
        vx, vy: Horizontal velocity components
        t: Time array
        v_cross_a_h: Cross product of velocity and acceleration in horizontal plane
        v_h_sq: Horizontal velocity squared

    Returns:
        heading_deg: Heading angle in degrees
        heading_rate_deg: Heading rate in degrees per second
        heading_rate_from_cross_deg: Heading rate from cross product
    """
    # Heading from velocity using existing function
    heading_rad = unwrap_angle_rad(calculate_heading(vx, vy))
    heading_deg = np.degrees(heading_rad)
    heading_rate_rad = calculate_heading_rate(vx, vy, t)
    heading_rate_deg = np.degrees(heading_rate_rad)
    heading_rate_from_cross_deg = np.degrees(v_cross_a_h / (v_h_sq + 1e-6))

    return heading_deg, heading_rate_deg, heading_rate_from_cross_deg


def compute_power_analysis(
    ax_val: np.ndarray,
    ay_val: np.ndarray,
    az_val: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
) -> np.ndarray:
    """Compute power per unit mass.

    Args:
        ax_val, ay_val, az_val: Acceleration components
        vx, vy, vz: Velocity components

    Returns:
        power_per_mass: Power per unit mass
    """
    # Dot product: a·v
    a_dot_v = ax_val * vx + ay_val * vy + az_val * vz
    # Power per unit mass: P/m = a·v + g·v_z
    return a_dot_v + G * vz
