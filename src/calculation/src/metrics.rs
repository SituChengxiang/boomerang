/// Trajectory metrics and statistics calculation.
///
/// This module provides tools to compute various metrics from simulated trajectories,
/// including energy, velocity, acceleration, and other physical quantities.
/// These are used for fitting and validation of aerodynamic models.

use std::collections::HashMap;

use crate::aerodynamics::{AerodynamicModel, SpeModel, BapModel};
use crate::constants::{Constants, model_params::{BapParams, SpeParams}};
use crate::ode::RK4Integrator;
use crate::vector::Vec3;

/// Container for trajectory metrics.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryMetrics {
    /// Time array
    pub t: Vec<f64>,
    /// Position arrays
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    /// Velocity arrays
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    pub vz: Vec<f64>,
    /// Acceleration arrays
    pub ax: Vec<f64>,
    pub ay: Vec<f64>,
    pub az: Vec<f64>,
    /// Kinetic energy
    pub kinetic_energy: Vec<f64>,
    /// Potential energy
    pub potential_energy: Vec<f64>,
    /// Total energy
    pub total_energy: Vec<f64>,
    /// Energy change rate
    pub dE_dt: Vec<f64>,
    /// Speed magnitude
    pub speed: Vec<f64>,
    /// Horizontal speed
    pub speed_xy: Vec<f64>,
}

impl Default for TrajectoryMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajectoryMetrics {
    /// Create an empty TrajectoryMetrics instance.
    pub fn new() -> Self {
        Self {
            t: Vec::new(),
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            vx: Vec::new(),
            vy: Vec::new(),
            vz: Vec::new(),
            ax: Vec::new(),
            ay: Vec::new(),
            az: Vec::new(),
            kinetic_energy: Vec::new(),
            potential_energy: Vec::new(),
            total_energy: Vec::new(),
            dE_dt: Vec::new(),
            speed: Vec::new(),
            speed_xy: Vec::new(),
        }
    }

    /// Create metrics from trajectory data (t, x, y, z).
    pub fn from_trajectory(
        t: &[f64],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        constants: &Constants,
    ) -> Result<Self, &'static str> {
        if t.len() != x.len() || t.len() != y.len() || t.len() != z.len() {
            return Err("All arrays must have the same length");
        }

        if t.len() < 2 {
            return Err("Need at least 2 points to compute derivatives");
        }

        let n = t.len();
        let mut metrics = Self::new();
        metrics.t = t.to_vec();
        metrics.x = x.to_vec();
        metrics.y = y.to_vec();
        metrics.z = z.to_vec();

        // Compute velocities using numerical derivatives
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        let mut vz = vec![0.0; n];

        for i in 0..n {
            if i == 0 {
                // Forward difference
                let dt = t[i + 1] - t[i];
                if dt <= 0.0 {
                    return Err("Time must be strictly increasing");
                }
                vx[i] = (x[i + 1] - x[i]) / dt;
                vy[i] = (y[i + 1] - y[i]) / dt;
                vz[i] = (z[i + 1] - z[i]) / dt;
            } else if i == n - 1 {
                // Backward difference
                let dt = t[i] - t[i - 1];
                if dt <= 0.0 {
                    return Err("Time must be strictly increasing");
                }
                vx[i] = (x[i] - x[i - 1]) / dt;
                vy[i] = (y[i] - y[i - 1]) / dt;
                vz[i] = (z[i] - z[i - 1]) / dt;
            } else {
                // Central difference
                let dt1 = t[i] - t[i - 1];
                let dt2 = t[i + 1] - t[i];
                if dt1 <= 0.0 || dt2 <= 0.0 {
                    return Err("Time must be strictly increasing");
                }
                vx[i] = 0.5 * ((x[i + 1] - x[i]) / dt2 + (x[i] - x[i - 1]) / dt1);
                vy[i] = 0.5 * ((y[i + 1] - y[i]) / dt2 + (y[i] - y[i - 1]) / dt1);
                vz[i] = 0.5 * ((z[i + 1] - z[i]) / dt2 + (z[i] - z[i - 1]) / dt1);
            }
        }

        // Compute accelerations
        let mut ax = vec![0.0; n];
        let mut ay = vec![0.0; n];
        let mut az = vec![0.0; n];

        for i in 0..n {
            if i == 0 {
                // Forward difference
                let dt = t[i + 1] - t[i];
                ax[i] = (vx[i + 1] - vx[i]) / dt;
                ay[i] = (vy[i + 1] - vy[i]) / dt;
                az[i] = (vz[i + 1] - vz[i]) / dt;
            } else if i == n - 1 {
                // Backward difference
                let dt = t[i] - t[i - 1];
                ax[i] = (vx[i] - vx[i - 1]) / dt;
                ay[i] = (vy[i] - vy[i - 1]) / dt;
                az[i] = (vz[i] - vz[i - 1]) / dt;
            } else {
                // Central difference
                let dt1 = t[i] - t[i - 1];
                let dt2 = t[i + 1] - t[i];
                ax[i] = 0.5 * ((vx[i + 1] - vx[i]) / dt2 + (vx[i] - vx[i - 1]) / dt1);
                ay[i] = 0.5 * ((vy[i + 1] - vy[i]) / dt2 + (vy[i] - vy[i - 1]) / dt1);
                az[i] = 0.5 * ((vz[i + 1] - vz[i]) / dt2 + (vz[i] - vz[i - 1]) / dt1);
            }
        }

        // Compute energies and derived metrics
        metrics.vx = vx;
        metrics.vy = vy;
        metrics.vz = vz;
        metrics.ax = ax;
        metrics.ay = ay;
        metrics.az = az;

        metrics.kinetic_energy.resize(n, 0.0);
        metrics.potential_energy.resize(n, 0.0);
        metrics.total_energy.resize(n, 0.0);
        metrics.dE_dt.resize(n, 0.0);
        metrics.speed.resize(n, 0.0);
        metrics.speed_xy.resize(n, 0.0);

        for i in 0..n {
            let v_sq = metrics.vx[i].powi(2) + metrics.vy[i].powi(2) + metrics.vz[i].powi(2);
            let speed = v_sq.sqrt();

            // v and v_xy for potential calculations
            let v_xy = (metrics.vx[i].powi(2) + metrics.vy[i].powi(2)).sqrt();

            // Kinetic energy (per unit mass): 0.5 * v^2
            let ke = 0.5 * v_sq;
            // Potential energy (per unit mass): g * z
            let pe = constants.g * z[i];  // Use original z

            metrics.kinetic_energy[i] = ke;
            metrics.potential_energy[i] = pe;
            metrics.total_energy[i] = ke + pe;
            metrics.speed[i] = speed;
            metrics.speed_xy[i] = v_xy;

            // Energy change rate: dE/dt = v·a + g*vz
            let dE_dt = metrics.vx[i] * metrics.ax[i]
                + metrics.vy[i] * metrics.ay[i]
                + metrics.vz[i] * metrics.az[i]
                + constants.g * metrics.vz[i];
            metrics.dE_dt[i] = dE_dt;
        }

        Ok(metrics)
    }

    /// Simulate trajectory using SPE model and compute metrics.
    pub fn simulate_spe(
        t_eval: &[f64],
        state0: &[f64; 6],
        params: &SpeParams,
        constants: &Constants,
    ) -> Result<Self, &'static str> {
        let model = SpeModel {};
        let states = model.simulate(t_eval, state0, params, constants);

        // Extract x, y, z
        let n = states.len();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut z = vec![0.0; n];

        for (i, state) in states.iter().enumerate() {
            x[i] = state[0];
            y[i] = state[1];
            z[i] = state[2];
        }

        Self::from_trajectory(&t_eval, &x, &y, &z, constants)
    }

    /// Simulate trajectory using BAP model and compute metrics.
    pub fn simulate_bap(
        t_eval: &[f64],
        state0: &[f64; 6],
        params: &BapParams,
        constants: &Constants,
    ) -> Result<Self, &'static str> {
        let model = BapModel {};
        let states = model.simulate(t_eval, state0, params, constants);

        // Extract x, y, z
        let n = states.len();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut z = vec![0.0; n];

        for (i, state) in states.iter().enumerate() {
            x[i] = state[0];
            y[i] = state[1];
            z[i] = state[2];
        }

        Self::from_trajectory(&t_eval, &x, &y, &z, constants)
    }

    /// Compute the mean energy dissipation rate (W/kg).
    pub fn mean_energy_dissipation(&self) -> f64 {
        if self.dE_dt.is_empty() {
            return 0.0;
        }
        // Average dE/dt (negative indicates dissipation)
        self.dE_dt.iter().sum::<f64>() / (self.dE_dt.len() as f64)
    }

    /// Compute the peak kinetic energy.
    pub fn peak_kinetic_energy(&self) -> f64 {
        self.kinetic_energy
            .iter()
            .cloned()
            .fold(0.0, f64::max)
    }

    /// Compute total flight time.
    pub fn flight_time(&self) -> f64 {
        if self.t.is_empty() {
            return 0.0;
        }
        self.t[self.t.len() - 1] - self.t[0]
    }

    /// Compute horizontal travel distance.
    pub fn horizontal_distance(&self) -> f64 {
        if self.x.is_empty() || self.y.is_empty() {
            return 0.0;
        }
        let dx = self.x[self.x.len() - 1] - self.x[0];
        let dy = self.y[self.y.len() - 1] - self.y[0];
        (dx.powi(2) + dy.powi(2)).sqrt()
    }

    /// Compute vertical displacement.
    pub fn vertical_displacement(&self) -> f64 {
        if self.z.is_empty() {
            return 0.0;
        }
        self.z[self.z.len() - 1] - self.z[0]
    }

    /// Compute mean speed.
    pub fn mean_speed(&self) -> f64 {
        if self.speed.is_empty() {
            return 0.0;
        }
        self.speed.iter().sum::<f64>() / (self.speed.len() as f64)
    }

    /// Compute mean horizontal speed.
    pub fn mean_speed_xy(&self) -> f64 {
        if self.speed_xy.is_empty() {
            return 0.0;
        }
        self.speed_xy.iter().sum::<f64>() / (self.speed_xy.len() as f64)
    }

    /// Get summary statistics as a dictionary.
    pub fn get_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        if !self.t.is_empty() {
            summary.insert("n_points".to_string(), self.t.len() as f64);
            summary.insert("flight_time".to_string(), self.flight_time());
            summary.insert("horizontal_distance".to_string(), self.horizontal_distance());
            summary.insert("vertical_displacement".to_string(), self.vertical_displacement());
            summary.insert("mean_speed".to_string(), self.mean_speed());
            summary.insert("mean_speed_xy".to_string(), self.mean_speed_xy());
            summary.insert("mean_energy_dissipation".to_string(), self.mean_energy_dissipation());
            summary.insert("peak_kinetic_energy".to_string(), self.peak_kinetic_energy());
            summary.insert("min_altitude".to_string(), self.z.iter().cloned().fold(f64::INFINITY, f64::min));
            summary.insert("max_altitude".to_string(), self.z.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
            summary.insert("max_speed".to_string(), self.speed.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        }

        summary
    }

    /// Check if trajectory is physically valid.
    pub fn is_valid(&self, constants: &Constants) -> bool {
        if self.t.len() < 2 {
            return false;
        }

        // Time must be strictly increasing
        for i in 1..self.t.len() {
            if self.t[i] <= self.t[i - 1] {
                return false;
            }
        }

        // Validate energy dissipation (should be mostly negative)
        let mean_dE = self.mean_energy_dissipation();
        if mean_dE > 0.1 {
            // Too much energy gain
            return false;
        }

        // Check for NaN values
        for val in self.ax.iter().chain(&self.ay).chain(&self.az) {
            if val.is_nan() || val.is_infinite() {
                return false;
            }
        }

        // Check altitude bounds
        for z in &self.z {
            if z < &-50.0 || z > &500.0 {
                // Unreasonable bounds
                return false;
            }
        }

        true
    }
}

/// Compute trajectory metrics from raw data.
pub fn compute_trajectory_metrics(
    t: &[f64],
    x: &[f64],
    y: &[f64],
    z: &[f64],
    constants: &Constants,
) -> Result<TrajectoryMetrics, &'static str> {
    TrajectoryMetrics::from_trajectory(t, x, y, z, constants)
}

/// Simulate and compute metrics for SPE model.
pub fn simulate_and_compute_spe(
    t_eval: &[f64],
    state0: &[f64; 6],
    params: &SpeParams,
    constants: &Constants,
) -> Result<TrajectoryMetrics, &'static str> {
    TrajectoryMetrics::simulate_spe(t_eval, state0, params, constants)
}

/// Simulate and compute metrics for BAP model.
pub fn simulate_and_compute_bap(
    t_eval: &[f64],
    state0: &[f64; 6],
    params: &BapParams,
    constants: &Constants,
) -> Result<TrajectoryMetrics, &'static str> {
    TrajectoryMetrics::simulate_bap(t_eval, state0, params, constants)
}

/// Batch compute metrics for multiple trajectories.
pub fn batch_compute_metrics(
    trajectories: &[(&[f64], &[f64], &[f64], &[f64])],
    constants: &Constants,
) -> Vec<TrajectoryMetrics> {
    trajectories
        .iter()
        .filter_map(|(t, x, y, z)| {
            TrajectoryMetrics::from_trajectory(t, x, y, z, constants).ok()
        })
        .collect()
}

/// Compute mean squared error between two trajectories.
pub fn trajectory_mse(
    traj1: &TrajectoryMetrics,
    traj2: &TrajectoryMetrics,
) -> Result<f64, &'static str> {
    if traj1.t.len() != traj2.t.len() {
        return Err("Trajectories must have same length for MSE");
    }

    if traj1.t.is_empty() {
        return Ok(0.0);
    }

    let n = traj1.t.len();
    let mut total_error = 0.0;

    for i in 0..n {
        let dx = traj1.x[i] - traj2.x[i];
        let dy = traj1.y[i] - traj2.y[i];
        let dz = traj1.z[i] - traj2.z[i];
        total_error += dx * dx + dy * dy + dz * dz;
    }

    Ok(total_error / (n as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trajectory_metrics_from_data() {
        let t = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let z = vec![0.0, -0.1, -0.2, -0.3, -0.4, -0.5];

        let constants = Constants::experimental();
        let metrics = TrajectoryMetrics::from_trajectory(&t, &x, &y, &z, &constants);

        assert!(metrics.is_ok());
        let metrics = metrics.unwrap();

        assert_eq!(metrics.t.len(), 6);
        assert!(!metrics.vx.is_empty());
        assert!(!metrics.ax.is_empty());

        // Check velocities (should be ~10 m/s in x, 0 in y, -1 in z)
        assert_relative_eq!(metrics.vx[0], 10.0, epsilon = 0.1);
        assert_relative_eq!(metrics.vz[0], -1.0, epsilon = 0.1);

        // Check energies
        assert!(metrics.kinetic_energy[0] > 0.0);
        assert!(metrics.potential_energy[0] >= 0.0);
    }

    #[test]
    fn test_spe_simulation_metrics() {
        let params = SpeParams::default_params();
        let constants = Constants::experimental();
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 1.0, 10.0, 10.0, 0.0];

        let result = TrajectoryMetrics::simulate_spe(&t_eval, &state0, &params, &constants);

        assert!(result.is_ok());
        let metrics = result.unwrap();

        assert_eq!(metrics.t.len(), t_eval.len());
        assert!(metrics.is_valid(&constants));

        // Check summary statistics
        let summary = metrics.get_summary();
        assert!(summary.contains_key("flight_time"));
        assert!(summary.contains_key("horizontal_distance"));
        assert!(summary.contains_key("mean_energy_dissipation"));
    }

    #[test]
    fn test_bap_simulation_metrics() {
        let params = BapParams::default_params();
        let constants = Constants::experimental();
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 1.0, 10.0, 10.0, 0.0];

        let result = TrajectoryMetrics::simulate_bap(&t_eval, &state0, &params, &constants);

        assert!(result.is_ok());
        let metrics = result.unwrap();

        assert_eq!(metrics.t.len(), t_eval.len());
        assert!(metrics.is_valid(&constants));
    }

    #[test]
    fn test_energy_dissipation() {
        let t = vec![0.0, 0.1, 0.2];
        let x = vec![0.0, 1.0, 1.8]; // Actually decelerating slightly
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, -0.5, -1.0]; // Falling

        let constants = Constants::experimental();
        let metrics = TrajectoryMetrics::from_trajectory(&t, &x, &y, &z, &constants).unwrap();

        // Energy dissipation should be negative (energy is being lost)
        let mean_dE = metrics.mean_energy_dissipation();
        assert!(mean_dE < 0.0);
    }

    #[test]
    fn test_mse_calculation() {
        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 1.0, 2.1]; // Slightly different
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let constants = Constants::experimental();
        let metrics1 = TrajectoryMetrics::from_trajectory(&t, &x1, &y, &z, &constants).unwrap();
        let metrics2 = TrajectoryMetrics::from_trajectory(&t, &x2, &y, &z, &constants).unwrap();

        let mse = trajectory_mse(&metrics1, &metrics2);
        assert!(mse.is_ok());
        let mse_val = mse.unwrap();

        // MSE should be small and positive
        assert!(mse_val > 0.0);
        assert!(mse_val < 1.0);
    }

    #[test]
    fn test_batch_compute() {
        let t1 = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let y1 = vec![0.0, 0.0, 0.0];
        let z1 = vec![0.0, -0.1, -0.2];

        let t2 = vec![0.0, 0.15, 0.3];
        let x2 = vec![0.0, 1.5, 3.0];
        let y2 = vec![0.0, 0.0, 0.0];
        let z2 = vec![0.0, -0.15, -0.3];

        let constants = Constants::experimental();
        let trajectories = vec![
            (&t1[..], &x1[..], &y1[..], &z1[..]),
            (&t2[..], &x2[..], &y2[..], &z2[..]),
        ];

        let metrics = batch_compute_metrics(&trajectories, &constants);

        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].t.len(), 3);
        assert_eq!(metrics[1].t.len(), 3);
    }
}
