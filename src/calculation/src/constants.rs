/// Physical constants and model parameters for aerodynamic calculations.
///
/// This module contains all physical constants and model-specific parameters
/// that are used throughout the simulation. It's designed to be easily
/// configurable and consistent with both the Python and Julia implementations.

use serde::{Deserialize, Serialize};

/// Global physical constants for the boomerang system.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Constants {
    /// Gravitational acceleration (m/s²)
    /// Standard value: 9.793 m/s² (often used for experimental data)
    pub g: f64,

    /// Air density (kg/m³)
    /// Standard value: 1.225 kg/m³ at 20°C, 1 atm
    pub rho_air: f64,

    /// Boomerang mass (kg)
    /// Typical value: 2.18 grams = 0.00218 kg
    pub mass: f64,

    /// Reference wing area (m²)
    /// Approximation: 2 * span * width = 2 * 0.15 * 0.028 = 0.0084 m²
    pub area: f64,

    /// Wing span (m)
    pub span: f64,

    /// Wing width (m)
    pub width: f64,

    /// Moment of inertia (kg·m²)
    /// For a boomernag with arms: I = m * L² / 2
    /// L ≈ 0.25 m, m ≈ 0.00218 kg → I ≈ 6.8e-5 kg·m²
    pub i_moment: f64,

    /// Arm length (m)
    pub arm_length: f64,

    /// Default time span for spin decay simulation (seconds)
    pub default_time_span: f64,

    /// Default number of time points for trajectory simulation
    pub default_time_points: usize,

    /// Default relative tolerance for numerical integration
    pub default_rel_tol: f64,

    /// Default absolute tolerance for numerical integration
    pub default_abs_tol: f64,
}

impl Constants {
    /// Create a new Constants instance with default values.
    pub const fn new() -> Self {
        Self {
            g: 9.793,
            rho_air: 1.225,
            mass: 0.002183,
            area: 0.0084,
            span: 0.15,
            width: 0.028,
            i_moment: 6.8e-5,
            arm_length: 0.25,
            default_time_span: 5.0,
            default_time_points: 500,
            default_rel_tol: 1e-6,
            default_abs_tol: 1e-8,
        }
    }

    /// Create constants with precision engineering values.
    /// These match the exact values used in the experiments.
    pub fn experimental() -> Self {
        Self {
            g: 9.793,
            rho_air: 1.225,
            mass: 0.002183,
            area: 2.0 * 0.15 * 0.028, // 0.0084 m² exactly
            span: 0.15,
            width: 0.028,
            i_moment: 5.0 / 24.0 * 0.002183 * (0.15_f64).powi(2), // 6.8e-5 derived
            arm_length: 0.25,
            default_time_span: 5.0,
            default_time_points: 500,
            default_rel_tol: 1e-6,
            default_abs_tol: 1e-8,
        }
    }

    /// Create constants for potential field simulations.
    /// Slightly different density and gravity values.
    pub fn field_simulation() -> Self {
        Self {
            g: 9.806,
            rho_air: 1.184, // At higher altitude/temperature
            mass: 0.002183,
            area: 0.0084,
            span: 0.15,
            width: 0.028,
            i_moment: 6.8e-5,
            arm_length: 0.25,
            default_time_span: 5.0,
            default_time_points: 500,
            default_rel_tol: 1e-6,
            default_abs_tol: 1e-8,
        }
    }

    /// SST (Stratospheric) conditions.
    /// Low temperature, lower air density.
    pub fn stratospheric() -> Self {
        Self {
            g: 9.793,
            rho_air: 0.9, // Approx at high altitude
            mass: 0.002183,
            area: 0.0084,
            span: 0.15,
            width: 0.028,
            i_moment: 6.8e-5,
            arm_length: 0.25,
            default_time_span: 5.0,
            default_time_points: 500,
            default_rel_tol: 1e-6,
            default_abs_tol: 1e-8,
        }
    }

    /// Create custom constants from provided values.
    pub fn custom(
        g: f64,
        rho_air: f64,
        mass: f64,
        area: Option<f64>,
        span: Option<f64>,
        width: Option<f64>,
    ) -> Self {
        let span = span.unwrap_or(0.15);
        let width = width.unwrap_or(0.028);
        let area = area.unwrap_or(2.0 * span * width);

        Self {
            g,
            rho_air,
            mass,
            area,
            span,
            width,
            i_moment: 5.0 / 24.0 * mass * span.powi(2),
            arm_length: 0.25,
            default_time_span: 5.0,
            default_time_points: 500,
            default_rel_tol: 1e-6,
            default_abs_tol: 1e-8,
        }
    }

    /// Calculate q = 0.5 * ρ * area (common factor for aerodynamic forces)
    #[inline(always)]
    pub fn q_factor(&self) -> f64 {
        0.5 * self.rho_air * self.area
    }

    /// Calculate moment of inertia for a spanned rod.
    /// I = (1/12) * m * L² for uniform rod
    /// I = (1/3) * m * L² for point masses at ends
    pub fn moment_of_inertia(&self) -> f64 {
        // Using the same calculation as in the experiment note
        5.0 / 24.0 * self.mass * self.span.powi(2)
    }

    /// Simulate the effective moment of inertia with a near-zero stomach/core.
    /// Used for BAP model adjustments.
    pub fn moment_of_inertia_eff(&self, core_ratio: f64) -> f64 {
        let core_ratio = core_ratio.clamp(0.0, 0.8);
        // Core mass is neglected, only wings contribute
        self.moment_of_inertia() * (1.0 - core_ratio)
    }

    /// Get a summary of constants as a hash map for reporting.
    pub fn summary(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("g (m/s²)".to_string(), format!("{:.4}", self.g));
        map.insert("ρ_air (kg/m³)".to_string(), format!("{:.4}", self.rho_air));
        map.insert("mass (kg)".to_string(), format!("{:.6}", self.mass));
        map.insert("area (m²)".to_string(), format!("{:.6}", self.area));
        map.insert("span (m)".to_string(), format!("{:.3}", self.span));
        map.insert("width (m)".to_string(), format!("{:.3}", self.width));
        map.insert("I_total (kg·m²)".to_string(), format!("{:.3e}", self.i_moment));
        map.insert("Arm length (m)".to_string(), format!("{:.3}", self.arm_length));
        map
    }

    /// Convert to Python-compatible dictionary format.
    #[cfg(feature = "python-bindings")]
    pub fn to_pydict(&self) -> std::collections::HashMap<String, f64> {
        let mut dict = std::collections::HashMap::new();
        dict.insert("g".to_string(), self.g);
        dict.insert("rho_air".to_string(), self.rho_air);
        dict.insert("mass".to_string(), self.mass);
        dict.insert("area".to_string(), self.area);
        dict.insert("span".to_string(), self.span);
        dict.insert("width".to_string(), self.width);
        dict.insert("i_moment".to_string(), self.i_moment);
        dict.insert("arm_length".to_string(), self.arm_length);
        dict
    }
}

impl Default for Constants {
    fn default() -> Self {
        Self::new()
    }
}

/// Model-specific parameters and tuning constants.
pub mod model_params {
    /// Parameters for SPE (Simplified Physics Equation) model.
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct SpeParams {
        pub cl_trans: f64,         // Translational lift coefficient
        pub cl_rotor: f64,        // Rotor lift coefficient (usually 0.0)
        pub cd: f64,              // Drag coefficient
        pub d_factor: f64,        // D factor for torque
        pub coupling_eff: f64,   // Coupling efficiency
        pub dive_steering: f64,   // Dive steering loss
        pub bank_factor: f64,     // Bank factor (lift power exponent)
        pub omega_decay: f64,     // Spin decay rate (1/s)
        pub ground_effect: f64,   // Ground effect strength
        pub ground_height: f64,   // Ground effect scale height (m)
    }

    impl SpeParams {
        pub fn default_params() -> Self {
            Self {
                cl_trans: 0.4,
                cl_rotor: 0.0,
                cd: 0.5,
                d_factor: 0.3,
                coupling_eff: 1.0,
                dive_steering: 0.5,
                bank_factor: 1.7,
                omega_decay: 0.1,
                ground_effect: 0.4,
                ground_height: 0.2,
            }
        }

        pub fn from_slice(p: &[f64]) -> Result<Self, &'static str> {
            if p.len() < 6 {
                return Err("SPE params require at least 6 elements");
            }
            Ok(Self {
                cl_trans: p[0],
                cl_rotor: 0.0,
                cd: p[1],
                d_factor: p[2],
                coupling_eff: p[3],
                dive_steering: p[4],
                bank_factor: p[5],
                omega_decay: 0.1,
                ground_effect: 0.4,
                ground_height: 0.2,
            })
        }

        pub fn to_slice(&self) -> [f64; 6] {
            [self.cl_trans, self.cd, self.d_factor, self.coupling_eff, self.dive_steering, self.bank_factor]
        }
    }

    /// Parameters for BAP (Bank Angle Proxy) model.
    #[derive(Debug, Clone, Copy, PartialEq, Default)]
    pub struct BapParams {
        pub cl: f64,               // Lift coefficient
        pub cd: f64,               // Drag coefficient
        pub phi_base: f64,         // Base bank angle (rad)
        pub k_bank: f64,          // Bank coefficient (speed→bank angle)
        pub v0_scalar: f64,        // Initial velocity reference
        pub omega_scale: f64,      // Spin rate scaling (dimensionless)
        pub phi_time_gain: f64,    // Time-dependent bank angle gain (rad/s)
        pub self_level_gain: f64,  // Self-leveling gain (0-1)
        pub bank_lift_loss: f64,   // Lift loss at high bank (0-1)
        pub bank_drag_gain: f64,   // Drag increase at high bank (0-1)
    }

    impl BapParams {
        pub fn default_params() -> Self {
            Self {
                cl: 0.8,
                cd: 0.5,
                phi_base: 0.0,
                k_bank: 0.3,
                v0_scalar: 10.0,
                omega_scale: 1.0,
                phi_time_gain: 0.2,
                self_level_gain: 0.2,
                bank_lift_loss: 0.2,
                bank_drag_gain: 0.4,
            }
        }

        pub fn from_slice(p: &[f64]) -> Result<Self, &'static str> {
            if p.len() < 6 {
                return Err("BAP params require at least 6 elements");
            }
            Ok(Self {
                cl: p[0],
                cd: p[1],
                phi_base: 0.0,
                k_bank: 0.0,        // Not optimized in current model
                v0_scalar: 0.0,     // Not optimized
                omega_scale: 1.0,   // Not optimized
                phi_time_gain: 0.2, // Fixed
                self_level_gain: 0.2, // Fixed
                bank_lift_loss: 0.2, // Fixed
                bank_drag_gain: 0.4, // Fixed
            })
        }

        pub fn to_slice(&self) -> [f64; 6] {
            [self.cl, self.cd, self.phi_base, self.k_bank, self.v0_scalar, self.omega_scale]
        }
    }

    /// Runtime parameters for the simulation.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct RuntimeParams {
        pub max_steps: usize,
        pub dt_min: f64,
        pub dt_max: f64,
        pub safety_factor: f64,
        pub ground_height: f64,
    }

    impl Default for RuntimeParams {
        fn default() -> Self {
            Self {
                max_steps: 10000,
                dt_min: 1e-6,
                dt_max: 0.01,
                safety_factor: 0.9,
                ground_height: 0.0, // Ground at z=0
            }
        }
    }
}

/// Track metadata from experimental data.
pub mod track_meta {
    use std::collections::HashMap;

    /// Experimental data for each track.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct TrackInfo {
        pub track_id: char,
        pub turns: f64,
        pub duration: f64,
        pub omega: f64, // rad/s (calculated)
    }

    impl TrackInfo {
        pub fn new(track_id: char, turns: f64, duration: f64) -> Self {
            let omega = turns * 2.0 * std::f64::consts::PI / duration;
            Self { track_id, turns, duration, omega }
        }
    }

    /// Build a map of track information from README data.
    pub fn build_track_meta() -> HashMap<char, TrackInfo> {
        let mut map = HashMap::new();
        map.insert('1', TrackInfo::new('1', 5.3, 0.93));
        map.insert('2', TrackInfo::new('2', 7.8, 1.28));
        // '3' and '8' are skipped in experiments
        // map.insert('3', TrackInfo::new('3', 5.5, 1.08));
        map.insert('5', TrackInfo::new('5', 5.0, 1.17));
        map.insert('6', TrackInfo::new('6', 5.4, 1.07));
        map.insert('7', TrackInfo::new('7', 5.2, 1.17));
        // map.insert('8', TrackInfo::new('8', 4.3, 0.88));
        map.insert('9', TrackInfo::new('9', 4.8, 1.07));
        map
    }

    /// Get the reference omega (median of all tracks).
    pub fn omega_ref() -> f64 {
        let meta = build_track_meta();
        let mut omegas: Vec<f64> = meta.values().map(|info| info.omega).collect();
        omegas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = omegas.len() / 2;
        if omegas.len() % 2 == 0 {
            (omegas[mid - 1] + omegas[mid]) / 2.0
        } else {
            omegas[mid]
        }
    }

    /// Get omega for a specific track.
    pub fn get_omega(track_id: char) -> Option<f64> {
        build_track_meta().get(&track_id).map(|info| info.omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_default_constants() {
        let c = Constants::new();
        assert_eq!(c.g, 9.793);
        assert_eq!(c.rho_air, 1.225);
        assert_eq!(c.mass, 0.002183);
    }

    #[test]
    fn test_q_factor() {
        let c = Constants::experimental();
        let q = c.q_factor();
        // 0.5 * 1.225 * 0.0084 = 0.005145
        assert_relative_eq!(q, 0.5 * 1.225 * 8.4e-3, epsilon = 1e-10);
    }

    #[test]
    fn test_track_meta() {
        use track_meta::{get_omega, omega_ref};
        assert_relative_eq!(get_omega('1').unwrap(), 5.3 / 0.93 * 2.0 * std::f64::consts::PI);
        assert_relative_eq!(get_omega('2').unwrap(), 7.8 / 1.28 * 2.0 * std::f64::consts::PI);
        // Omega ref should be around 30 rad/s
        let ref_omega = omega_ref();
        assert!(ref_omega > 25.0 && ref_omega < 35.0);
    }

    #[test]
    fn test_spe_params() {
        let p = model_params::SpeParams::default_params();
        assert_eq!(p.cl_trans, 0.4);
        assert_eq!(p.cd, 0.5);
        assert_eq!(p.omega_decay, 0.1);
    }

    #[test]
    fn test_spe_params_slice() {
        let p = model_params::SpeParams::default_params();
        let slice = p.to_slice();
        assert_eq!(slice.len(), 6);
        let p2 = model_params::SpeParams::from_slice(&slice).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_bap_params_slice() {
        let p = model_params::BapParams::default_params();
        let slice = p.to_slice();
        assert_eq!(slice.len(), 6);
        let p2 = model_params::BapParams::from_slice(&slice).unwrap();
        assert_eq!(p, p2);
    }
}
