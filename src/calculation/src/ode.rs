/// High-performance ODE (Ordinary Differential Equation) solvers.
///
/// This module provides numerical integration methods specifically optimized
/// for the fast-changing dynamics of aerodynamic simulations.
/// It uses RK4 (Runge-Kutta 4th order) as the primary method due to its
/// excellent balance of accuracy and computational efficiency.

use std::f64::consts;

use crate::vector::Vec3;

/// Trait representing an ODE system: dy/dt = f(t, y)
pub trait OdeSystem {
    /// The state type (usually [f64; 6] for 3D position + velocity)
    type State: Clone + Copy;

    /// The time type
    type Time: Clone + Copy + std::ops::Add<Output = Self::Time> + std::ops::Mul<f64, Output = Self::Time>;

    /// Compute the right-hand side of the ODE system
    fn rhs(&self, t: Self::Time, state: &Self::State) -> Self::State;
}

/// RK4 (Runge-Kutta 4th order) integrator.
///
/// The classic RK4 method provides O(h⁴) local error and O(h⁴) global error,
/// making it an excellent choice for snake trajectory simulations where
/// spacetime variation can be rapid but smooth.
pub struct RK4Integrator;

impl RK4Integrator {
    /// Integrate using RK4 over a time interval.
    ///
    /// # Arguments
    /// * `system` - The ODE system to integrate
    /// * `t_start` - Start time
    /// * `state0` - Initial state
    /// * `t_eval` - Time points where the solution should be evaluated
    /// * `args` - Additional arguments for the system (e.g., aerodynamic parameters)
    ///
    /// # Returns
    /// A vector of states at each time point in `t_eval`
    pub fn integrate_f64<S, T>(
        system: &S,
        t_start: f64,
        state0: S::State,
        t_eval: &[f64],
    ) -> Vec<S::State>
    where
        S: OdeSystem<Time = f64>,
    {
        let mut states = Vec::with_capacity(t_eval.len());
        states.push(state0.clone());

        for i in 1..t_eval.len() {
            let t0 = t_eval[i - 1];
            let t1 = t_eval[i];
            let dt = t1 - t0;

            // RK4 integration step
            let current_state = states.last().unwrap().clone();
            let next_state = rk4_step(system, current_state, t0, dt);
            states.push(next_state);
        }

        states
    }

    /// Integrate an adaptive step using RK4 with error checking.
    ///
    /// # Returns
    /// `(final_state, step_count, max_step_error)`
    pub fn integrate_with_error<S>(
        system: &S,
        t_start: f64,
        state0: S::State,
        dt: f64,
        error_tol: f64,
    ) -> (S::State, usize, f64)
    where
        S: OdeSystem<Time = f64>,
    {
        let mut t = t_start;
        let mut state = state0;
        let mut step_count = 0;
        let mut max_error = 0.0;

        // Simple constant step RK4 for now
        // (Adaptive step sizing would require nested integration and interpolation)
        let next_state = rk4_step(system, state.clone(), t, dt);
        let error = estimate_error(&state, &next_state);
        max_error = max_error.max(error);

        let final_state = next_state;
        step_count += 1;

        (final_state, step_count, max_error)
    }
}

/// Single RK4 step.
#[inline(always)]
fn rk4_step<S: OdeSystem<Time = f64>>(
    system: &S,
    state: S::State,
    t: f64,
    h: f64,
) -> S::State {
    let h2 = 0.5 * h;

    // k1 = f(t, y)
    let k1 = system.rhs(t, &state);

    // k2 = f(t + h/2, y + h/2 * k1)
    let state2 = state_add_mul(&state, &k1, h2);
    let k2 = system.rhs(t + h2, &state2);

    // k3 = f(t + h/2, y + h/2 * k2)
    let state3 = state_add_mul(&state, &k2, h2);
    let k3 = system.rhs(t + h2, &state3);

    // k4 = f(t + h, y + h * k3)
    let state4 = state_add_mul(&state, &k3, h);
    let k4 = system.rhs(t + h, &state4);

    // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    let sum = state_add_mul(&k1, &k2, 2.0)
        .then_add_mul(&k3, 2.0)
        .then_add_mul(&k4, 1.0);
    state_add_mul(&state, &sum, h / 6.0)
}

/// Helper function for state addition with multiplication: state + scalar * other
#[inline(always)]
fn state_add_mul<State: Clone>(state: &State, other: &State, scalar: f64) -> State {
    // This is a generic implementation. Specialized implementations for arrays
    // would be more efficient. For now, we use a trait-based approach.
    unimplemented!("Specialized for specific state types")
}

/// Estimate error using difference between steps (simple heuristic).
#[inline(always)]
fn estimate_error<State>(_state1: &State, _state2: &State) -> f64 {
    // TODO: Implement proper error estimation
    0.0
}

/// Trait for adding states with different weights.
pub trait StateAddMul {
    fn add_mul(&self, other: &Self, scalar: f64) -> Self;
}

/// Specialized implementation for [f64; 6] (position + velocity).
impl StateAddMul for [f64; 6] {
    #[inline(always)]
    fn add_mul(&self, other: &Self, scalar: f64) -> Self {
        [
            self[0] + other[0] * scalar,
            self[1] + other[1] * scalar,
            self[2] + other[2] * scalar,
            self[3] + other[3] * scalar,
            self[4] + other[4] * scalar,
            self[5] + other[5] * scalar,
        ]
    }
}

/// Trait for chaining StateAddMul operations.
pub trait StateAddMulChain: StateAddMul {
    fn then_add_mul(&self, other: &Self, scalar: f64) -> Self;
}

impl StateAddMulChain for [f64; 6] {
    #[inline(always)]
    fn then_add_mul(&self, other: &Self, scalar: f64) -> Self {
        self.add_mul(other, scalar)
    }
}

/// Fixed-step RK4 integrator for 6DOF state [x, y, z, vx, vy, vz].
pub struct RK4Integrator6DOF {
    pub dt: f64,
    pub substeps: usize,
}

impl RK4Integrator6DOF {
    pub fn new(dt: f64, substeps: usize) -> Self {
        Self { dt, substeps }
    }

    /// Integrate multiple trajectories from initial conditions.
    pub fn integrate_batch<S>(
        &self,
        system: &S,
        t_start: f64,
        states0: &[[f64; 6]],
        t_eval: &[f64],
    ) -> Vec<Vec<[f64; 6]>>
    where
        S: OdeSystem<Time = f64, State = [f64; 6]>,
    {
        states0
            .iter()
            .map(|state0| {
                let mut states = Vec::with_capacity(t_eval.len());
                states.push(*state0);

                for i in 1..t_eval.len() {
                    let t0 = t_eval[i - 1];
                    let t1 = t_eval[i];
                    let dt = (t1 - t0) / (self.substeps as f64);

                    let mut state = *states.last().unwrap();
                    let mut curr_t = t0;

                    for _ in 0..self.substeps {
                        // RK4 step
                        let k1 = system.rhs(curr_t, &state);
                        let state2 = state.add_mul(&k1, dt * 0.5);
                        let k2 = system.rhs(curr_t + dt * 0.5, &state2);
                        let state3 = state.add_mul(&k2, dt * 0.5);
                        let k3 = system.rhs(curr_t + dt * 0.5, &state3);
                        let state4 = state.add_mul(&k3, dt);
                        let k4 = system.rhs(curr_t + dt, &state4);

                        let sum = k1
                            .add_mul(&k2, 2.0)
                            .add_mul(&k3, 2.0)
                            .add_mul(&k4, 1.0);
                        state = state.add_mul(&sum, dt / 6.0);
                        curr_t += dt;
                    }

                    states.push(state);
                }

                states
            })
            .collect()
    }
}

/// Trait for converting to/from Vec3 for convenience.
pub trait StateConversion {
    fn to_vec3(&self, index: usize) -> Vec3;
    fn from_vec3(&mut self, index: usize, vec: &Vec3);
}

impl StateConversion for [f64; 6] {
    fn to_vec3(&self, index: usize) -> Vec3 {
        match index {
            0 => Vec3::new(self[0], self[1], self[2]), // position
            1 => Vec3::new(self[3], self[4], self[5]), // velocity
            _ => Vec3::zero(),
        }
    }

    fn from_vec3(&mut self, index: usize, vec: &Vec3) {
        match index {
            0 => {
                self[0] = vec.x;
                self[1] = vec.y;
                self[2] = vec.z;
            }
            1 => {
                self[3] = vec.x;
                self[4] = vec.y;
                self[5] = vec.z;
            }
            _ => {}
        }
    }
}

/// State trait for common operations.
pub trait StateOps: Clone {
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, scalar: f64) -> Self;
    fn to_array(&self) -> Vec<f64>;
}

impl StateOps for [f64; 6] {
    fn add(&self, other: &Self) -> Self {
        [
            self[0] + other[0],
            self[1] + other[1],
            self[2] + other[2],
            self[3] + other[3],
            self[4] + other[4],
            self[5] + other[5],
        ]
    }

    fn sub(&self, other: &Self) -> Self {
        [
            self[0] - other[0],
            self[1] - other[1],
            self[2] - other[2],
            self[3] - other[3],
            self[4] - other[4],
            self[5] - other[5],
        ]
    }

    fn mul(&self, scalar: f64) -> Self {
        [
            self[0] * scalar,
            self[1] * scalar,
            self[2] * scalar,
            self[3] * scalar,
            self[4] * scalar,
            self[5] * scalar,
        ]
    }

    fn to_array(&self) -> Vec<f64> {
        self.to_vec()
    }
}

/// State for RK4 integration: x, y, z, vx, vy, vz
pub type RK4State = [f64; 6];

/// Helper function for RK4 integration step.
pub fn rk4_step_6dof(
    rhs: &dyn Fn(f64, &[f64; 6]) -> [f64; 6],
    t: f64,
    state: &[f64; 6],
    h: f64,
) -> [f64; 6] {
    let h2 = 0.5 * h;

    // k1 = f(t, y)
    let k1 = rhs(t, state);

    // k2 = f(t + h/2, y + h/2 * k1)
    let state2 = [
        state[0] + 0.5 * h * k1[0],
        state[1] + 0.5 * h * k1[1],
        state[2] + 0.5 * h * k1[2],
        state[3] + 0.5 * h * k1[3],
        state[4] + 0.5 * h * k1[4],
        state[5] + 0.5 * h * k1[5],
    ];
    let k2 = rhs(t + h2, &state2);

    // k3 = f(t + h/2, y + h/2 * k2)
    let state3 = [
        state[0] + 0.5 * h * k2[0],
        state[1] + 0.5 * h * k2[1],
        state[2] + 0.5 * h * k2[2],
        state[3] + 0.5 * h * k2[3],
        state[4] + 0.5 * h * k2[4],
        state[5] + 0.5 * h * k2[5],
    ];
    let k3 = rhs(t + h2, &state3);

    // k4 = f(t + h, y + h * k3)
    let state4 = [
        state[0] + h * k3[0],
        state[1] + h * k3[1],
        state[2] + h * k3[2],
        state[3] + h * k3[3],
        state[4] + h * k3[4],
        state[5] + h * k3[5],
    ];
    let k4 = rhs(t + h, &state4);

    // y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    [
        state[0] + h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        state[1] + h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        state[2] + h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        state[3] + h * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        state[4] + h * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
        state[5] + h * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5]) / 6.0,
    ]
}

/// Integration function for a single trajectory.
pub fn integrate_trajectory(
    rhs: &dyn Fn(f64, &[f64; 6]) -> [f64; 6],
    t_eval: &[f64],
    state0: &[f64; 6],
    substeps: usize,
) -> Vec<[f64; 6]> {
    let n = t_eval.len();
    let mut result = Vec::with_capacity(n);
    result.push(*state0);

    for i in 1..n {
        let t0 = t_eval[i - 1];
        let t1 = t_eval[i];
        let dt = (t1 - t0) / (substeps as f64);

        let mut state = *result.last().unwrap();
        let mut curr_t = t0;

        for _ in 0..substeps {
            state = rk4_step_6dof(rhs, curr_t, &state, dt);
            curr_t += dt;
        }

        result.push(state);
    }

    result
}

/// Optimized integration for multiple trajectories (SIMD-friendly layout).
pub fn integrate_trajectories_batch_slice(
    rhs: &dyn Fn(f64, &[f64; 6]) -> [f64; 6],
    t_eval: &[f64],
    states0: &[[f64; 6]],
    substeps: usize,
) -> Vec<[f64; 6]> {
    // Flatten output: [state1_t0, state1_t1, ..., state1_tN, state2_t0, ...]
    let n_states = states0.len();
    let n_times = t_eval.len();
    let total_len = n_states * n_times;

    let mut result = Vec::with_capacity(total_len);

    // Resize and fill
    for _ in 0..total_len {
        result.push([0.0; 6]);
    }

    // For parallelism, we could use rayon here
    for (idx, state0) in states0.iter().enumerate() {
        let trajectory = integrate_trajectory(rhs, t_eval, state0, substeps);
        for (t_idx, state) in trajectory.iter().enumerate() {
            let flat_idx = idx * n_times + t_idx;
            result[flat_idx] = *state;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Simple harmonic oscillator for testing
    struct HarmonicOscillator {
        k: f64,  // spring constant
        m: f64,  // mass
    }

    impl OdeSystem for HarmonicOscillator {
        type State = [f64; 6];
        type Time = f64;

        fn rhs(&self, _t: f64, state: &[f64; 6]) -> [f64; 6] {
            let x = state[0];
            let v = state[3]; // use x-velocity slot for 1D test

            // a = -k/m * x
            let a = -self.k / self.m * x;

            [x, 0.0, 0.0, v, 0.0, 0.0]
        }
    }

    #[test]
    fn test_rk4_step() {
        let system = HarmonicOscillator { k: 1.0, m: 1.0 };
        let state0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let t = 0.0;
        let h = 0.1;

        let state1 = rk4_step_6dof(&|t, s| system.rhs(t, s), t, &state0, h);

        // Should have moved from equilibrium
        assert!(state1[0] != state0[0]);
    }

    #[test]
    fn test_integrate_trajectory() {
        let system = HarmonicOscillator { k: 1.0, m: 1.0 };
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let trajectory =
            integrate_trajectory(&|t, s| system.rhs(t, s), &t_eval, &state0, 1);

        // Should have moved and returned expected number of points
        assert_eq!(trajectory.len(), t_eval.len());
        assert!(trajectory[0][0] != trajectory[5][0]); // Different states at different times
    }

    #[test]
    fn test_batch_integration() {
        let system = HarmonicOscillator { k: 1.0, m: 1.0 };
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let states0 = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  // Different initial displacements
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let result = integrate_trajectories_batch_slice(
            &|t, s| system.rhs(t, s),
            &t_eval,
            &states0,
            1,
        );

        // We should have 2 trajectories × 11 time points = 22 states
        assert_eq!(result.len(), 22);
        // First trajectory initial state
        assert_eq!(result[0][0], 1.0);
        // Second trajectory initial state
        assert_eq!(result[11][0], 2.0);
    }
}
