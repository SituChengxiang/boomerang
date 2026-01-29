//! High-performance aerodynamic calculation library for boomerang trajectory simulation.
//!
//! This library provides optimized numerical methods for:
//! - ODE integration (RK4, RK4 with adaptive stepping)
//! - 3D vector operations and geometric calculations
//! - Aerodynamic force calculations (SPE and BAP models)
//! - Energy and trajectory metric computations
//! - Parameter fitting and optimization utilities
//!
//! # Features
//!
//! - **High-performance**: Pure Rust with SIMD-ready math operations
//! - **Flexible**: Support for multiple aerodynamic models
//! - **Accurate**: Numerically stable integration methods
//! - **Python bindings**: Optional PyO3 bindings for seamless Python integration

#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
#![warn(clippy::doc_markdown)]
#![allow(clippy::tabs_in_doc_comments)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::useless_format)]

// Re-export core modules
pub mod vector;
pub mod ode;
pub mod aerodynamics;
pub mod constants;
pub mod loss;
pub mod metrics;

// Optional Python bindings
#[cfg(feature = "python-bindings")]
pub mod python;

// Re-export key types and functions for easy use
pub use aerodynamics::{BapModel, SpeModel, AerodynamicModel};
pub use constants::Constants;
pub use loss::LossFn;
pub use metrics::{TrajectoryMetrics, compute_trajectory_metrics};
pub use ode::{RK4Integrator, Integrator, OdeSystem};
pub use vector::{Vec3, Quaternion};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Honeycomb structure for BAP model parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BapParams {
    pub cl: f64,                    // Lift coefficient
    pub cd: f64,                    // Drag coefficient
    pub phi_base: f64,              // Base bank angle
    pub k_bank: f64,               // Bank coefficient
    pub v0_scalar: f64,            // Initial velocity scale
    pub omega_scale: f64,          // Spin rate scaling
}

impl Default for BapParams {
    fn default() -> Self {
        Self {
            cl: 0.8,
            cd: 0.5,
            phi_base: 0.0,
            k_bank: 0.3,
            v0_scalar: 10.0,
            omega_scale: 1.0,
        }
    }
}

/// SPE model parameters
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeParams {
    pub cl_trans: f64,             // Translational lift coefficient
    pub cl_rotor: f64,            // Rotor lift coefficient
    pub cd: f64,                  // Drag coefficient
    pub d_factor: f64,            // D factor
    pub coupling_eff: f64,        // Coupling efficiency
    pub dive_steering: f64,       // Dive steering loss
    pub bank_factor: f64,         // Bank factor (also used as lift power)
    pub omega_decay: f64,         // Omega decay rate (1/s)
}

impl Default for SpeParams {
    fn default() -> Self {
        Self {
            cl_trans: 0.4,
            cl_rotor: 0.0,
            cd: 0.5,
            d_factor: 0.3,
            coupling_eff: 1.0,
            dive_steering: 0.5,
            bank_factor: 1.7,
            omega_decay: 0.1,
        }
    }
}
