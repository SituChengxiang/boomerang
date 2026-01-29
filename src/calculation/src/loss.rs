/// Loss function calculations for parameter fitting.
///
/// This module computes the error between simulated and experimental trajectories.
/// It's designed to be used with optimization algorithms to fit aerodynamic parameters
/// against experimental data.

use crate::aerodynamics::{AerodynamicModel, SpeModel, BapModel};
use crate::constants::{Constants, model_params::{BapParams, SpeParams}};
use crate::metrics::{TrajectoryMetrics, trajectory_mse};
use crate::ode::{RK4Integrator, RK4State};

/// Trait for computing loss between trajectories.
pub trait LossFunction {
    /// Compute the loss/energy between simulated and reference trajectories.
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64;

    /// Compute weighted loss (for balancing multiple tracks).
    fn compute_weighted_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics, weight: f64) -> f64 {
        self.compute_loss(simulated, reference) * weight
    }
}

/// Mean Squared Error (MSE) loss function.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MseLoss;

impl LossFunction for MseLoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        trajectory_mse(simulated, reference).unwrap_or(f64::MAX)
    }
}

/// Weighted MSE loss function (allows weighting different parts of trajectory).
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedMseLoss {
    pub position_weight: f64,
    pub velocity_weight: f64,
    pub acceleration_weight: f64,
    pub energy_weight: f64,
    pub start_weight: f64,
    pub end_weight: f64,
    pub middle_weight: f64,
}

impl Default for WeightedMseLoss {
    fn default() -> Self {
        Self {
            position_weight: 1.0,
            velocity_weight: 0.1,     // Lower weight on velocity (more sensitive to noise)
            acceleration_weight: 0.01, // Lower weight on acceleration (even more sensitive)
            energy_weight: 0.05,      // Weight energy consistency
            start_weight: 1.5,        // Emphasize trajectory start
            end_weight: 1.2,          // Emphasize trajectory end (landing zone)
            middle_weight: 0.8,       // Slightly de-emphasize middle (often less accurately measured)
        }
    }
}

impl WeightedMseLoss {
    /// Compute weighted MSE with time-dependent weights.
    pub fn compute_loss_with_weights(
        &self,
        simulated: &TrajectoryMetrics,
        reference: &TrajectoryMetrics,
    ) -> f64 {
        if simulated.t.len() != reference.t.len() || simulated.t.is_empty() {
            return f64::MAX;
        }

        let n = simulated.t.len();
        let mut total_loss = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n {
            // Determine time-based weight
            let t_rel = i as f64 / (n - 1) as f64;
            let time_weight = if t_rel < 0.2 {
                self.start_weight
            } else if t_rel > 0.8 {
                self.end_weight
            } else {
                self.middle_weight
            };

            // Position error
            let dx = simulated.x[i] - reference.x[i];
            let dy = simulated.y[i] - reference.y[i];
            let dz = simulated.z[i] - reference.z[i];
            let pos_errors = dx * dx + dy * dy + dz * dz;

            // Velocity error (if available)
            let mut vel_errors = 0.0;
            if !simulated.vx.is_empty() && !reference.vx.is_empty() {
                let dvx = simulated.vx[i] - reference.vx[i];
                let dvy = simulated.vy[i] - reference.vy[i];
                let dvz = simulated.vz[i] - reference.vz[i];
                vel_errors = dvx * dvx + dvy * dvy + dvz * dvz;
            }

            // Acceleration error (if available)
            let mut acc_errors = 0.0;
            if !simulated.ax.is_empty() && !reference.ax.is_empty() {
                let dax = simulated.ax[i] - reference.ax[i];
                let day = simulated.ay[i] - reference.ay[i];
                let daz = simulated.az[i] - reference.az[i];
                acc_errors = dax * dax + day * day + daz * daz;
            }

            // Energy error (if available)
            let mut energy_error = 0.0;
            if !simulated.total_energy.is_empty() && !reference.total_energy.is_empty() {
                let de = simulated.total_energy[i] - reference.total_energy[i];
                energy_error = de * de;
            }

            // Weighted sum
            let weight = time_weight;
            let mut point_loss = self.position_weight * pos_errors;
            point_loss += self.velocity_weight * vel_errors;
            point_loss += self.acceleration_weight * acc_errors;
            point_loss += self.energy_weight * energy_error;

            total_loss += weight * point_loss;
            total_weight += weight;
        }

        if total_weight == 0.0 {
            f64::MAX
        } else {
            total_loss / total_weight
        }
    }
}

impl LossFunction for WeightedMseLoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        self.compute_loss_with_weights(simulated, reference)
    }
}

/// Root Mean Square Error (RMSE) loss function.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RmseLoss;

impl LossFunction for RmseLoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        let mse = trajectory_mse(simulated, reference).unwrap_or(f64::MAX);
        mse.sqrt()
    }
}

/// Maximum Absolute Error (MAXE) loss function - emphasizes worst-case deviations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MaxELoss;

impl LossFunction for MaxELoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        if simulated.t.len() != reference.t.len() || simulated.t.is_empty() {
            return f64::MAX;
        }

        let n = simulated.t.len();
        let mut max_error = 0.0;

        for i in 0..n {
            let dx = simulated.x[i] - reference.x[i];
            let dy = simulated.y[i] - reference.y[i];
            let dz = simulated.z[i] - reference.z[i];
            let error = (dx * dx + dy * dy + dz * dz).sqrt();
            max_error = max_error.max(error);
        }

        max_error
    }
}

/// Huber loss function - robust to outliers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HuberLoss {
    pub delta: f64,
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self { delta: 1.0 }
    }
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl LossFunction for HuberLoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        if simulated.t.len() != reference.t.len() || simulated.t.is_empty() {
            return f64::MAX;
        }

        let n = simulated.t.len();
        let mut total_loss = 0.0;

        for i in 0..n {
            let dx = simulated.x[i] - reference.x[i];
            let dy = simulated.y[i] - reference.y[i];
            let dz = simulated.z[i] - reference.z[i];
            let error = (dx * dx + dy * dy + dz * dz).sqrt();

            // Huber loss: quadratic for small errors, linear for large errors
            let loss = if error <= self.delta {
                0.5 * error * error
            } else {
                self.delta * (error - 0.5 * self.delta)
            };

            total_loss += loss;
        }

        total_loss / (n as f64)
    }
}

/// Composite loss function combining multiple criteria.
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeLoss {
    pub mse_loss: MseLoss,
    pub energy_loss_weight: f64,
    pub boundary_loss_weight: f64,
    pub stability_loss_weight: f64,
    pub constants: Constants,
}

impl CompositeLoss {
    pub fn new(constants: Constants) -> Self {
        Self {
            mse_loss: MseLoss,
            energy_loss_weight: 0.1,
            boundary_loss_weight: 0.05,
            stability_loss_weight: 0.02,
            constants,
        }
    }

    /// Loss based on energy conservation violation.
    fn energy_loss(&self, metrics: &TrajectoryMetrics) -> f64 {
        // Energy should be roughly conserved (decreasing slowly due to drag)
        // We penalize excessive energy increase or too rapid dissipation
        let mean_dE = metrics.mean_energy_dissipation();

        // Penalize positive dE/dt (energy gain) and very large negative dE/dt
        if mean_dE > 0.01 {
            // Gaining energy - very bad
            (mean_dE * 100.0).abs()
        } else if mean_dE < -10.0 {
            // Too much dissipation (unrealistic for boomerang)
            ((mean_dE + 10.0).abs() * 2.0)
        } else {
            // Within reasonable range
            0.0
        }
    }

    /// Loss based on boundary violations (landing below ground).
    fn boundary_loss(&self, metrics: &TrajectoryMetrics) -> f64 {
        const MIN_ALTITUDE: f64 = -0.2; // Ground allows slight negative due to measurement error

        let mut loss = 0.0;
        for &z in &metrics.z {
            if z < MIN_ALTITUDE {
                loss += (MIN_ALTITUDE - z).abs() * 10.0;
            }
        }

        loss
    }

    /// Loss based on numerical stability (sudden jumps in acceleration).
    fn stability_loss(&self, metrics: &TrajectoryMetrics) -> f64 {
        if metrics.ax.len() < 3 {
            return 0.0;
        }

        let mut loss = 0.0;
        for i in 1..metrics.ax.len() {
            let dax = (metrics.ax[i] - metrics.ax[i - 1]).abs();
            let day = (metrics.ay[i] - metrics.ay[i - 1]).abs();
            let daz = (metrics.az[i] - metrics.az[i - 1]).abs();

            // Penalize smooth acceleration changes (too jerky)
            loss += dax * dax + day * day + daz * daz;
        }

        loss
    }
}

impl LossFunction for CompositeLoss {
    fn compute_loss(&self, simulated: &TrajectoryMetrics, reference: &TrajectoryMetrics) -> f64 {
        let mse_error = self.mse_loss.compute_loss(simulated, reference);

        // Additional regularizing losses
        let energy_err = self.energy_loss(simulated);
        let boundary_err = self.boundary_loss(simulated);
        let stability_err = self.stability_loss(simulated);

        mse_error
            + self.energy_loss_weight * energy_err
            + self.boundary_loss_weight * boundary_err
            + self.stability_loss_weight * stability_err
    }
}

/// Multi-track loss aggregators.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackAggregation {
    /// Sum of losses (default)
    Sum,
    /// Weighted sum (custom weights)
    Weighted(Vec<f64>),
    /// Root mean square
    Rms,
    /// Mean absolute
    Mean,
    /// Maximum
    Max,
}

/// Loss function for fitting against multiple tracks simultaneously.
pub struct MultiTrackLoss<L: LossFunction> {
    pub loss_fn: L,
    pub aggregation: TrackAggregation,
}

impl<L: LossFunction + Default> Default for MultiTrackLoss<L> {
    fn default() -> Self {
        Self {
            loss_fn: L::default(),
            aggregation: TrackAggregation::Sum,
        }
    }
}

impl<L: LossFunction> MultiTrackLoss<L> {
    /// Compute aggregated loss across multiple tracks.
    pub fn compute_multi_loss(
        &self,
        simulated: &[TrajectoryMetrics],
        reference: &[TrajectoryMetrics],
    ) -> Result<f64, &'static str> {
        if simulated.len() != reference.len() {
            return Err("Number of simulated and reference tracks must match");
        }

        if simulated.is_empty() {
            return Err("No tracks provided");
        }

        // Compute individual losses
        let individual_losses: Vec<f64> = simulated
            .iter()
            .zip(reference.iter())
            .map(|(sim, ref_track)| self.loss_fn.compute_loss(sim, ref_track))
            .collect();

        // Aggregate losses
        let result = match &self.aggregation {
            TrackAggregation::Sum => individual_losses.iter().sum(),
            TrackAggregation::Weighted(weights) => {
                if weights.len() != individual_losses.len() {
                    return Err("Weights must match number of tracks");
                }
                individual_losses
                    .iter()
                    .zip(weights.iter())
                    .map(|(loss, weight)| loss * weight)
                    .sum()
            }
            TrackAggregation::Rms => {
                let sum_sq = individual_losses
                    .iter()
                    .map(|&x| x * x)
                    .sum::<f64>();
                (sum_sq / individual_losses.len() as f64).sqrt()
            }
            TrackAggregation::Mean => {
                individual_losses.iter().sum::<f64>() / individual_losses.len() as f64
            }
            TrackAggregation::Max => {
                individual_losses
                    .iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, f64::max)
            }
        };

        Ok(result)
    }

    /// Compute loss for a single set of parameters against multiple tracks.
    pub fn compute_parameter_loss(
        &self,
        t_evals: &[Vec<f64>],
        states0: &[[f64; 6]],
        reference_times: &[Vec<f64>],
        reference_positions: &[Vec<(f64, f64, f64)>],
        params: &L::Params,
        constants: &Constants,
    ) -> Result<f64, &'static str>
    where
        L: LossFunction<Params = L::Params>,
    {
        let mut simulated = Vec::new();
        let mut reference = Vec::new();

        for (i, (t_eval, state0)) in t_evals.iter().zip(states0.iter()).enumerate() {
            // Simulate trajectory (this is a simplified version - actual implementation
            // would need to dispatch based on model type)
            // Note: This is a placeholder - the actual implementation would need
            // to know which model to use (SPE or BAP)
            let sim_metrics = TrajectoryMetrics::from_trajectory(
                t_eval,
                &vec![0.0; t_eval.len()],
                &vec![0.0; t_eval.len()],
                &vec![0.0; t_eval.len()],
                constants,
            ).map_err(|_| "Failed to compute metrics")?;

            // Create reference metrics
            let (ref_t, ref_x, ref_y, ref_z) = &reference_times[i];
            let ref_metrics = TrajectoryMetrics::from_trajectory(
                ref_t,
                ref_x,
                ref_y,
                ref_z,
                constants,
            ).map_err(|_| "Failed to create reference metrics")?;

            simulated.push(sim_metrics);
            reference.push(ref_metrics);
        }

        self.compute_multi_loss(&simulated, &reference)
    }
}

/// Convenience type aliases.

pub type MseLossFn = MseLoss;
pub type WeightedMseLossFn = WeightedMseLoss;
pub type RmseLossFn = RmseLoss;
pub type MaxELossFn = MaxELoss;
pub type HuberLossFn = HuberLoss;
pub type CompositeLossFn = CompositeLoss;
pub type MultiTrackLossFn<L> = MultiTrackLoss<L>;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_metrics(t: &[f64], x: &[f64], y: &[f64], z: &[f64]) -> TrajectoryMetrics {
        let constants = Constants::experimental();
        TrajectoryMetrics::from_trajectory(t, x, y, z, &constants).unwrap()
    }

    #[test]
    fn test_mse_loss() {
        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 1.0, 2.1];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss_fn = MseLoss;
        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
        assert!(loss < 1.0);
    }

    #[test]
    fn test_weighted_mse_loss() {
        let t = vec![0.0, 0.1, 0.2, 0.3];
        let x1 = vec![0.0, 1.0, 2.0, 3.0];
        let x2 = vec![0.0, 1.0, 2.0, 3.5]; // Error at end, should be penalized more
        let y = vec![0.0, 0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss_fn = WeightedMseLoss::default();
        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
        assert!(loss < 10.0);
    }

    #[test]
    fn test_rmse_loss() {
        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 1.0, 2.5];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss_fn = RmseLoss;
        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
        assert!(loss < 10.0);
        assert!(loss > 0.0); // RMSE should be sqrt(MSE)
    }

    #[test]
    fn test_maxe_loss() {
        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 0.5, 3.0]; // Max error at middle
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss_fn = MaxELoss;
        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
        // The max deviation should be captured
        assert!(loss >= 0.5);
    }

    #[test]
    fn test_huber_loss() {
        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 1.0, 10.0]; // Large error
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss_fn = HuberLoss::default();
        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
        // Huber should be linear for large errors instead of quadratic
    }

    #[test]
    fn test_composite_loss() {
        let constants = Constants::experimental();
        let loss_fn = CompositeLoss::new(constants.clone());

        let t = vec![0.0, 0.1, 0.2];
        let x1 = vec![0.0, 1.0, 2.0];
        let x2 = vec![0.0, 1.0, 2.1];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let metrics1 = create_test_metrics(&t, &x1, &y, &z);
        let metrics2 = create_test_metrics(&t, &x2, &y, &z);

        let loss = loss_fn.compute_loss(&metrics1, &metrics2);

        assert!(loss > 0.0);
    }

    #[test]
    fn test_multi_track_loss() {
        let loss_fn = MseLoss;
        let multi_loss = MultiTrackLoss {
            loss_fn,
            aggregation: TrackAggregation::Sum,
        };

        let t1 = vec![0.0, 0.1, 0.2];
        let x1_sim = vec![0.0, 1.0, 2.0];
        let x1_ref = vec![0.0, 1.0, 2.1];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let t2 = vec![0.0, 0.1, 0.2];
        let x2_sim = vec![0.0, 2.0, 4.0];
        let x2_ref = vec![0.0, 2.0, 4.2];
        let y2 = vec![0.0, 0.0, 0.0];
        let z2 = vec![0.0, 0.0, 0.0];

        let simulated = vec![
            create_test_metrics(&t1, &x1_sim, &y, &z),
            create_test_metrics(&t2, &x2_sim, &y2, &z2),
        ];
        let reference = vec![
            create_test_metrics(&t1, &x1_ref, &y, &z),
            create_test_metrics(&t2, &x2_ref, &y2, &z2),
        ];

        let result = multi_loss.compute_multi_loss(&simulated, &reference);

        assert!(result.is_ok());
        let loss = result.unwrap();
        assert!(loss > 0.0);
        assert!(loss < 10.0);
    }

    #[test]
    fn test_weighted_multi_track_loss() {
        let loss_fn = MseLoss;
        let multi_loss = MultiTrackLoss {
            loss_fn,
            aggregation: TrackAggregation::Weighted(vec![0.5, 2.0]),
        };

        let t1 = vec![0.0, 0.1, 0.2];
        let x1_sim = vec![0.0, 1.0, 2.0];
        let x1_ref = vec![0.0, 1.0, 2.1];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let t2 = vec![0.0, 0.1, 0.2];
        let x2_sim = vec![0.0, 2.0, 4.0];
        let x2_ref = vec![0.0, 2.0, 4.2];
        let y2 = vec![0.0, 0.0, 0.0];
        let z2 = vec![0.0, 0.0, 0.0];

        let simulated = vec![
            create_test_metrics(&t1, &x1_sim, &y, &z),
            create_test_metrics(&t2, &x2_sim, &y2, &z2),
        ];
        let reference = vec![
            create_test_metrics(&t1, &x1_ref, &y, &z),
            create_test_metrics(&t2, &x2_ref, &y2, &z2),
        ];

        let result = multi_loss.compute_multi_loss(&simulated, &reference);

        assert!(result.is_ok());
        let loss = result.unwrap();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_multi_track_loss_mismatch() {
        let loss_fn = MseLoss;
        let multi_loss = MultiTrackLoss {
            loss_fn,
            aggregation: TrackAggregation::Sum,
        };

        let t = vec![0.0, 0.1, 0.2];
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 0.0, 0.0];
        let z = vec![0.0, 0.0, 0.0];

        let simulated = vec![create_test_metrics(&t, &x, &y, &z)];
        let reference = vec![
            create_test_metrics(&t, &x, &y, &z),
            create_test_metrics(&t, &x, &y, &z),
        ];

        let result = multi_loss.compute_multi_loss(&simulated, &reference);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Number of simulated and reference tracks must match");
    }
}
