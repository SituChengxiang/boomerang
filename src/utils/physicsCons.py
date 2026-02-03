"""
Shared physical constants for boomerang aerodynamics modeling.

This module centralizes all physical parameters to ensure consistency
across different analysis modules.
"""

# Boomerang physical properties
MASS = 2.183e-3  # kg
A = 0.15  # m (arm length)
CHOR = 0.028  # m (arm width)
S = 2 * A * CHOR  # m^2, planform area

# Environmental constants
RHO = 1.225  # kg/m^3 (air density)
G = 9.793  # m/s^2 (gravitational acceleration)

# Rotation parameters
SIGMA_ROTATION = 0.4  # Rotary lift contribution factor
OMEGA0 = 85.0  # rad/s, initial rotation speed
OMEGA_DECAY = 0.15  # rad/s^2, rotation decay rate

# Derived properties
I_z = MASS / 24 * (5 * A**2 + 2 * CHOR**2)  # Rotational inertia approximation
