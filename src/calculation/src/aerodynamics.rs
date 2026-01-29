/// High-performance aerodynamic force calculations for boomerang trajectory simulation.
///
/// This module implements both the SPE (Simplified Physics Equation) and BAP (Bank Angle Proxy)
/// models from the Python implementations. These models compute the aerodynamic forces
/// acting on a flying boomerang and provide the right-hand side functions for ODE integration.
///
/// # Models
///
/// - **SPE Model**: Forces based on translational lift, drag, and gyroscopic precession
/// - **BAP Model**: Dynamic bank angle proxy that rises as speed drops (accounting for precession)
///
/// # Performance
///
/// All computations are optimized for SIMD execution and avoid allocations in hot paths.
/// The `System` trait enables zero-cost abstractions.

use std::f64::consts;

use crate::constants::{Constants, model_params::{BapParams, SpeParams}};
use crate::ode::{OdeSystem, RK4Integrator, RK4State};
use crate::vector::Vec3;

/// Aerodynamic model trait.
pub trait AerodynamicModel: Default {
    /// Model parameters type
    type Params: Default + Clone;
    /// State type for ODE system
    type State: Default + Clone;

    /// Compute the right-hand side of the ODE.
    /// Returns: [x, y, z, vx, vy, vz]
    fn rhs(&self, t: f64, state: &self::State, params: &Self::Params, constants: &Constants) -> Self::State;

    /// Simulate trajectory with given parameters.
    fn simulate(
        &self,
        t_eval: &[f64],
        state0: &self::State,
        params: &self::Params,
        constants: &Constants,
    ) -> Vec<Self::State>;
}

/// SPE (Simplified Physics Equation) Model.
/// Implements the calculation from fitSPE.py
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeModel {
    // Model has no persistent state beyond params
}

impl SpeModel {
    /// Compute ODE right-hand side for SPE model.
    #[inline(always)]
    pub fn rhs_internal(
        &self,
        t: f64,
        state: &[f64; 6],
        params: &SpeParams,
        constants: &Constants,
    ) -> [f64; 6] {
        let [x, y, z, vx, vy, vz] = state;

        // Velocity magnitude
        let v_sq = vx * vx + vy * vy + vz * vz;
        let v = (v_sq + 1e-9).sqrt();
        let v_xy_sq = vx * vx + vy * vy;
        let v_xy = (v_xy_sq + 1e-9).sqrt();

        // --- 1. Dynamic Omega (Spin Decay) ---
        // omega(t) = omega0 * (1 - decay * t) clamped to omega0 * 0.1
        let omega_t = (params.omega_decay * 0.1).max(params.omega_decay * (1.0 - params.omega_decay * t));

        // --- 2. Velocity-dependent bank angle (SPE model proxy) ---
        // As speed drops, bank angle increases (gyroscopic precession effect)
        let speed_loss = 10.0 - v_xy; // V0_scalar is fixed at 10.0 in original
        let phi = params.dive_steering * speed_loss + 0.2 * omega_t * t;

        // Dive self-leveling (smooth)
        let dive_ratio = (-vz / v).clamp(0.0, 1.0);
        let phi_effective = phi * (1.0 - 0.2 * dive_ratio);
        let phi_clamped = phi_effective.clamp(-1.5, 1.5);

        // --- 3. Direction vectors ---
        // Drag: opposite to velocity
        let v_hat_x = vx / v;
        let v_hat_y = vy / v;
        let v_hat_z = vz / v;

        // Lateral axis (cross product with up vector)
        let mut lat_x = v_hat_y * 1.0 - v_hat_z * 0.0;  // v_hat × (0,0,1)
        let mut lat_y = v_hat_z * 0.0 - v_hat_x * 1.0;
        let mut lat_z = v_hat_x * 0.0 - v_hat_y * 0.0;

        // Normalize lateral
        let lat_norm = (lat_x * lat_x + lat_y * lat_y + lat_z * lat_z).sqrt();
        if lat_norm < 1e-9 {
            lat_x = 1.0;
            lat_y = 0.0;
            lat_z = 0.0;
        } else {
            let inv = 1.0 / lat_norm;
            lat_x *= inv;
            lat_y *= inv;
            lat_z *= inv;
        }

        // Lift base direction (cross product)
        let mut lift0_x = lat_y * v_hat_z - lat_z * v_hat_y;
        let mut lift0_y = lat_z * v_hat_x - lat_x * v_hat_z;
        let mut lift0_z = lat_x * v_hat_y - lat_y * v_hat_x;

        // Normalize lift0
        let lift0_norm = (lift0_x * lift0_x + lift0_y * lift0_y + lift0_z * lift0_z).sqrt();
        if lift0_norm < 1e-9 {
            lift0_x = 0.0;
            lift0_y = 0.0;
            lift0_z = 1.0;
        } else {
            let inv = 1.0 / lift0_norm;
            lift0_x *= inv;
            lift0_y *= inv;
            lift0_z *= inv;
        }

        // Rodrigues rotation of lift0 around velocity by phi
        let cos_phi = phi_clamped.cos();
        let sin_phi = phi_clamped.sin();

        let dot = v_hat_x * lift0_x + v_hat_y * lift0_y + v_hat_z * lift0_z;
        let mut cx = v_hat_y * lift0_z - v_hat_z * lift0_y;
        let mut cy = v_hat_z * lift0_x - v_hat_x * lift0_z;
        let mut cz = v_hat_x * lift0_y - v_hat_y * lift0_x;

        let mut lift_dir_x = lift0_x * cos_phi + cx * sin_phi + v_hat_x * dot * (1.0 - cos_phi);
        let mut lift_dir_y = lift0_y * cos_phi + cy * sin_phi + v_hat_y * dot * (1.0 - cos_phi);
        let mut lift_dir_z = lift0_z * cos_phi + cz * sin_phi + v_hat_z * dot * (1.0 - cos_phi);

        // --- 4. Aerodynamic Forces ---
        let q = 0.5 * constants.rho_air * constants.area;

        // Lift Force (v^power model, power = bank_factor/reused as lift_power)
        let lift_power = params.bank_factor;
        let mut f_lift_mag = q * params.cl_trans * v_xy.powf(lift_power);

        // Bank lift loss (at high bank angles)
        let sin_phi = phi_clamped.sin();
        let lift_eff = 1.0 - 0.2 * (sin_phi * sin_phi);
        f_lift_mag *= lift_eff.clamp(0.25, 1.0);

        // Ground effect
        if z < 0.2 {
            let h_eff = z.max(0.05);
            f_lift_mag *= 1.0 + 0.2 * 0.2 / h_eff;
        }

        // Lift vector
        let ax_lift = f_lift_mag * lift_dir_x / constants.mass;
        let ay_lift = f_lift_mag * lift_dir_y / constants.mass;
        let az_lift = f_lift_mag * lift_dir_z / constants.mass;

        // Drag Force
        let f_drag_mag = q * params.cd * v_sq;
        let ax_drag = -f_drag_mag * v_hat_x / constants.mass;
        let ay_drag = -f_drag_mag * v_hat_y / constants.mass;
        let az_drag = -f_drag_mag * v_hat_z / constants.mass;

        // Total acceleration
        let ax = ax_lift + ax_drag;
        let ay = ay_lift + ay_drag;
        let az = az_lift + az_drag - constants.g;

        [vx, vy, vz, ax, ay, az]
    }
}

impl AerodynamicModel for SpeModel {
    type Params = SpeParams;
    type State = [f64; 6];

    fn rhs(&self, t: f64, state: &Self::State, params: &Self::Params, constants: &Constants) -> Self::State {
        self.rhs_internal(t, state, params, constants)
    }

    fn simulate(
        &self,
        t_eval: &[f64],
        state0: &Self::State,
        params: &Self::Params,
        constants: &Constants,
    ) -> Vec<Self::State> {
        // Wrap rhs with params and constants
        let rhs_fn = |t: f64, s: &Self::State| -> Self::State {
            self.rhs_internal(t, s, params, constants)
        };

        // Use RK4 integration
        let mut states = Vec::with_capacity(t_eval.len());
        states.push(*state0);

        for i in 1..t_eval.len() {
            let t0 = t_eval[i - 1];
            let t1 = t_eval[i];
            let dt = (t1 - t0);
            let h = dt / 4.0; // 4 substeps per time interval
            let mut curr_t = t0;
            let mut state = *states.last().unwrap();

            for _ in 0..4 {
                // RK4 step
                let k1 = rhs_fn(curr_t, &state);
                let state2 = [
                    state[0] + 0.5 * h * k1[0],
                    state[1] + 0.5 * h * k1[1],
                    state[2] + 0.5 * h * k1[2],
                    state[3] + 0.5 * h * k1[3],
                    state[4] + 0.5 * h * k1[4],
                    state[5] + 0.5 * h * k1[5],
                ];
                let k2 = rhs_fn(curr_t + 0.5 * h, &state2);
                let state3 = [
                    state[0] + 0.5 * h * k2[0],
                    state[1] + 0.5 * h * k2[1],
                    state[2] + 0.5 * h * k2[2],
                    state[3] + 0.5 * h * k2[3],
                    state[4] + 0.5 * h * k2[4],
                    state[5] + 0.5 * h * k2[5],
                ];
                let k3 = rhs_fn(curr_t + 0.5 * h, &state3);
                let state4 = [
                    state[0] + h * k3[0],
                    state[1] + h * k3[1],
                    state[2] + h * k3[2],
                    state[3] + h * k3[3],
                    state[4] + h * k3[4],
                    state[5] + h * k3[5],
                ];
                let k4 = rhs_fn(curr_t + h, &state4);

                let sum = [
                    k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0],
                    k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1],
                    k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2],
                    k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3],
                    k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4],
                    k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5],
                ];

                state = [
                    state[0] + h * sum[0] / 6.0,
                    state[1] + h * sum[1] / 6.0,
                    state[2] + h * sum[2] / 6.0,
                    state[3] + h * sum[3] / 6.0,
                    state[4] + h * sum[4] / 6.0,
                    state[5] + h * sum[5] / 6.0,
                ];
                curr_t += h;
            }

            states.push(state);
        }

        states
    }
}

/// BAP (Bank Angle Proxy) Model.
/// Implements the calculation from fitBAP.py
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BapModel {}

impl BapModel {
    /// Bank angle parameters (fixed)
    const PHI_TIME_GAIN: f64 = 0.2;
    const SELF_LEVEL_GAIN: f64 = 0.2;
    const BANK_LIFT_LOSS: f64 = 0.2;
    const BANK_DRAG_GAIN: f64 = 0.4;

    /// Compute ODE right-hand side for BAP model.
    #[inline(always)]
    pub fn rhs_internal(
        &self,
        t: f64,
        state: &[f64; 6],
        params: &BapParams,
        constants: &Constants,
    ) -> [f64; 6] {
        let [_x, _y, _z, vx, vy, vz] = state;
        let z = state[2];

        // --- 1. Speed calculation ---
        let v_sq = vx * vx + vy * vy + vz * vz;
        let v = (v_sq + 1e-9).sqrt();
        let v_xy_sq = vx * vx + vy * vy;
        let v_xy = (v_xy_sq + 1e-9).sqrt();

        // --- 2. Dynamic Bank Angle Proxy ---
        // Speed loss (V0 refs)
        let speed_loss = params.v0_scalar.max(10.0) - v_xy;
        let phi_time = Self::PHI_TIME_GAIN * params.omega_scale * t;
        let phi_k = params.k_bank * params.omega_scale * speed_loss;

        let mut phi = params.phi_base + phi_k + phi_time;

        // --- Dive self-leveling (smooth) ---
        let dive_ratio = (-vz / v).clamp(0.0, 1.0);
        phi *= 1.0 - Self::SELF_LEVEL_GAIN * dive_ratio;
        phi = phi.clamp(-1.5, 1.5);

        // --- 3. Direction vectors ---
        let v_hat_x = vx / v;
        let v_hat_y = vy / v;
        let v_hat_z = vz / v;

        // Lateral axis (cross with up)
        let mut lat_x = v_hat_y * 1.0 - v_hat_z * 0.0;  // v × (0,0,1)
        let mut lat_y = v_hat_z * 0.0 - v_hat_x * 1.0;
        let mut lat_z = v_hat_x * 0.0 - v_hat_y * 0.0;

        let lat_norm_sq = lat_x * lat_x + lat_y * lat_y + lat_z * lat_z;
        if lat_norm_sq < 1e-18 {
            lat_x = 1.0;
            lat_y = 0.0;
            lat_z = 0.0;
        } else {
            let inv = 1.0 / lat_norm_sq.sqrt();
            lat_x *= inv;
            lat_y *= inv;
            lat_z *= inv;
        }

        // Lift base (cross)
        let mut lift0_x = lat_y * v_hat_z - lat_z * v_hat_y;
        let mut lift0_y = lat_z * v_hat_x - lat_x * v_hat_z;
        let mut lift0_z = lat_x * v_hat_y - lat_y * v_hat_x;

        let lift0_norm_sq = lift0_x * lift0_x + lift0_y * lift0_y + lift0_z * lift0_z;
        if lift0_norm_sq < 1e-18 {
            lift0_x = 0.0;
            lift0_y = 0.0;
            lift0_z = 1.0;
        } else {
            let inv = 1.0 / lift0_norm_sq.sqrt();
            lift0_x *= inv;
            lift0_y *= inv;
            lift0_z *= inv;
        }

        // Rodrigues rotation
        let cos_phi = phi.cos();
        let sin_phi = phi.sin();
        let dot = v_hat_x * lift0_x + v_hat_y * lift0_y + v_hat_z * lift0_z;

        let cx = v_hat_y * lift0_z - v_hat_z * lift0_y;
        let cy = v_hat_z * lift0_x - v_hat_x * lift0_z;
        let cz = v_hat_x * lift0_y - v_hat_y * lift0_x;

        let mut lift_dir_x = lift0_x * cos_phi + cx * sin_phi + v_hat_x * dot * (1.0 - cos_phi);
        let mut lift_dir_y = lift0_y * cos_phi + cy * sin_phi + v_hat_y * dot * (1.0 - cos_phi);
        let mut lift_dir_z = lift0_z * cos_phi + cz * sin_phi + v_hat_z * dot * (1.0 - cos_phi);

        // --- 4. Aerodynamic Forces ---
        let q = 0.5 * constants.rho_air * constants.area;

        // Lift Force (v^1.5 model)
        let mut f_lift_mag = q * params.cl * v.powf(1.5);

        // Bank efficiency (lift loss at high bank)
        let sin_phi = phi.sin();
        let lift_eff = 1.0 - Self::BANK_LIFT_LOSS * (sin_phi * sin_phi);
        f_lift_mag *= lift_eff.clamp(0.25, 1.0);

        // Ground effect
        if z < 0.2 {
            let h_eff = z.max(0.05);
            f_lift_mag *= 1.0 + 0.2 * 0.2 / h_eff;
        }

        // Lift acceleration
        let ax_lift = f_lift_mag * lift_dir_x / constants.mass;
        let ay_lift = f_lift_mag * lift_dir_y / constants.mass;
        let az_lift = f_lift_mag * lift_dir_z / constants.mass;

        // Drag Force
        let cd_reduction = 0.25;
        let cd_eff = params.cd
            * (1.0 - cd_reduction * dive_ratio)
            * (1.0 + Self::BANK_DRAG_GAIN * (sin_phi * sin_phi));

        let f_drag_mag = q * cd_eff * v_sq;
        let ax_drag = -f_drag_mag * v_hat_x / constants.mass;
        let ay_drag = -f_drag_mag * v_hat_y / constants.mass;
        let az_drag = -f_drag_mag * v_hat_z / constants.mass;

        // Total acceleration
        let ax = ax_lift + ax_drag;
        let ay = ay_lift + ay_drag;
        let az = az_lift + az_drag - constants.g;

        [vx, vy, vz, ax, ay, az]
    }
}

impl AerodynamicModel for BapModel {
    type Params = BapParams;
    type State = [f64; 6];

    fn rhs(&self, t: f64, state: &Self::State, params: &Self::Params, constants: &Constants) -> Self::State {
        self.rhs_internal(t, state, params, constants)
    }

    fn simulate(
        &self,
        t_eval: &[f64],
        state0: &Self::State,
        params: &Self::Params,
        constants: &Constants,
    ) -> Vec<Self::State> {
        // Wrap rhs with params and constants
        let rhs_fn = |t: f64, s: &Self::State| -> Self::State {
            self.rhs_internal(t, s, params, constants)
        };

        // Use RK4 integration
        let mut states = Vec::with_capacity(t_eval.len());
        states.push(*state0);

        for i in 1..t_eval.len() {
            let t0 = t_eval[i - 1];
            let t1 = t_eval[i];
            let dt = (t1 - t0);
            let h = dt / 4.0; // 4 substeps per time interval
            let mut curr_t = t0;
            let mut state = *states.last().unwrap();

            for _ in 0..4 {
                // RK4 step
                let k1 = rhs_fn(curr_t, &state);
                let state2 = [
                    state[0] + 0.5 * h * k1[0],
                    state[1] + 0.5 * h * k1[1],
                    state[2] + 0.5 * h * k1[2],
                    state[3] + 0.5 * h * k1[3],
                    state[4] + 0.5 * h * k1[4],
                    state[5] + 0.5 * h * k1[5],
                ];
                let k2 = rhs_fn(curr_t + 0.5 * h, &state2);
                let state3 = [
                    state[0] + 0.5 * h * k2[0],
                    state[1] + 0.5 * h * k2[1],
                    state[2] + 0.5 * h * k2[2],
                    state[3] + 0.5 * h * k2[3],
                    state[4] + 0.5 * h * k2[4],
                    state[5] + 0.5 * h * k2[5],
                ];
                let k3 = rhs_fn(curr_t + 0.5 * h, &state3);
                let state4 = [
                    state[0] + h * k3[0],
                    state[1] + h * k3[1],
                    state[2] + h * k3[2],
                    state[3] + h * k3[3],
                    state[4] + h * k3[4],
                    state[5] + h * k3[5],
                ];
                let k4 = rhs_fn(curr_t + h, &state4);

                let sum = [
                    k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0],
                    k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1],
                    k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2],
                    k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3],
                    k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4],
                    k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5],
                ];

                state = [
                    state[0] + h * sum[0] / 6.0,
                    state[1] + h * sum[1] / 6.0,
                    state[2] + h * sum[2] / 6.0,
                    state[3] + h * sum[3] / 6.0,
                    state[4] + h * sum[4] / 6.0,
                    state[5] + h * sum[5] / 6.0,
                ];
                curr_t += h;
            }

            states.push(state);
        }

        states
    }
}

/// Batch integration for multiple trajectories (SPE model).
pub fn batch_simulate_spe(
    t_eval: &[f64],
    states0: &[[f64; 6]],
    params: &SpeParams,
    constants: &Constants,
) -> Vec<Vec<[f64; 6]>> {
    let model = SpeModel {};
    states0
        .iter()
        .map(|state0| model.simulate(t_eval, state0, params, constants))
        .collect()
}

/// Batch integration for multiple trajectories (BAP model).
pub fn batch_simulate_bap(
    t_eval: &[f64],
    states0: &[[f64; 6]],
    params: &BapParams,
    constants: &Constants,
) -> Vec<Vec<[f64; 6]>> {
    let model = BapModel {};
    states0
        .iter()
        .map(|state0| model.simulate(t_eval, state0, params, constants))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spe_rhs_basic() {
        let model = SpeModel {};
        let params = SpeParams::default_params();
        let constants = Constants::experimental();

        let state = [0.0, 0.0, 1.0, 5.0, 5.0, 0.0]; // x, y, z, vx, vy, vz
        let t = 0.0;

        let deriv = model.rhs_internal(t, &state, &params, &constants);

        // Should compute derivatives (dvx, dvy, dvz)
        assert_eq!(deriv.len(), 6);
        assert!(deriv[3].abs() > 0.0); // dvx
        assert!(deriv[4].abs() > 0.0); // dvy
        assert!(deriv[5].abs() != 0.0); // dvz (should include gravity)
    }

    #[test]
    fn test_bap_rhs_basic() {
        let model = BapModel {};
        let params = BapParams::default_params();
        let constants = Constants::experimental();

        let state = [0.0, 0.0, 2.0, 8.0, 8.0, -1.0]; // x, y, z, vx, vy, vz
        let t = 0.5;

        let deriv = model.rhs_internal(t, &state, &params, &constants);

        // Should compute derivatives
        assert_eq!(deriv.len(), 6);
        assert!(deriv[3].abs() > 0.0); // dvx
        assert!(deriv[4].abs() > 0.0); // dvy
        assert!(deriv[5].abs() != 0.0); // dvz
    }

    #[test]
    fn test_spe_simulation() {
        let model = SpeModel {};
        let params = SpeParams::default_params();
        let constants = Constants::experimental();

        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 1.0, 10.0, 10.0, 0.0];

        let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

        // Should have expected number of points
        assert_eq!(trajectory.len(), t_eval.len());

        // Verify that position changes (trajectory moves)
        let t5 = trajectory[5];
        assert!(t5[0] != state0[0] || t5[1] != state0[1] || t5[2] != state0[2]);
    }

    #[test]
    fn test_bap_simulation() {
        let model = BapModel {};
        let params = BapParams::default_params();
        let constants = Constants::experimental();

        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 10.0, 10.0, 10.0, 0.0];

        let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

        // Should have expected number of points
        assert_eq!(trajectory.len(), t_eval.len());

        // Verify that position changes (trajectory moves)
        let t5 = trajectory[5];
        assert!(t5[0] != state0[0] || t5[1] != state0[1] || t5[2] != state0[2]);
    }

    #[test]
    fn test_spe_batch() {
        let params = SpeParams::default_params();
        let constants = Constants::experimental();
        let t_eval: Vec<f64> = (0..=5).map(|i| i as f64 * 0.1).collect();
        let states0 = [
            [0.0, 0.0, 1.0, 5.0, 5.0, 0.0],
            [0.0, 0.0, 1.0, 6.0, 6.0, 0.0],
        ];

        let results = batch_simulate_spe(&t_eval, &states0, &params, &constants);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), t_eval.len());
        assert_eq!(results[1].len(), t_eval.len());
    }

    #[test]
    fn test_bap_batch() {
        let params = BapParams::default_params();
        let constants = Constants::experimental();
        let t_eval: Vec<f64> = (0..=5).map(|i| i as f64 * 0.1).collect();
        let states0 = [
            [0.0, 0.0, 1.0, 8.0, 8.0, 0.0],
            [0.0, 0.0, 1.0, 9.0, 9.0, 0.0],
        ];

        let results = batch_simulate_bap(&t_eval, &states0, &params, &constants);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), t_eval.len());
        assert_eq!(results[1].len(), t_eval.len());
    }
}
