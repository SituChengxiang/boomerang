use std::f64::consts;
/// Example program demonstrating high-performance aerodynamic calculation.
///
/// This binary compares the Rust implementation against Python references
/// and provides a simple optimization over the SPE model.
use std::time::Instant;

use boomerang_calc::aerodynamics::{AerodynamicModel, BapModel, SpeModel};
use boomerang_calc::constants::{
    model_params::{BapParams, SpeParams},
    track_meta, Constants,
};
use boomerang_calc::metrics::{trajectory_mse, TrajectoryMetrics};
use boomerang_calc::vector::Vec3;

fn main() {
    println!("===================================================");
    println!("High-Performance Boomerang Aerodynamics Calculator");
    println!("===================================================");
    println!();

    // Display constants
    let constants = Constants::experimental();
    let summary = constants.summary();
    println!("Physical Constants:");
    for (k, v) in summary {
        println!("  {}: {}", k, v);
    }
    println!();

    // Load track metadata
    let track_meta = track_meta::build_track_meta();
    println!("Track Metadata (omega in rad/s):");
    for (id, info) in &track_meta {
        println!(
            "  Track {}: {:.1f} turns in {:.2f}s -> ω = {:.1f} rad/s",
            id, info.turns, info.duration, info.omega
        );
    }
    println!();

    // Demonstrate SPE model
    demo_spe_model(&constants);

    println!();

    // Demonstrate BAP model
    demo_bap_model(&constants);

    println!();

    // Performance comparison
    performance_demo(&constants);

    println!();

    // Parameter optimization demo
    parameter_optimization_demo(&constants);

    println!();
    println!("===================================================");
    println!("Demo completed successfully!");
    println!("===================================================");
}

/// Demonstrate SPE model simulation
fn demo_spe_model(constants: &Constants) {
    println!("--- SPE Model Demo ---");

    let params = SpeParams::default_params();
    println!("SPE Parameters:");
    println!("  CL_trans     = {:.4f}", params.cl_trans);
    println!("  C_D          = {:.4f}", params.cd);
    println!("  D_factor     = {:.4f}", params.d_factor);
    println!("  Coupling     = {:.4f}", params.coupling_eff);
    println!("  Lift power   = {:.4f}", params.bank_factor);
    println!("  ω decay      = {:.4f}", params.omega_decay);
    println!();

    // Simulate a trajectory
    let duration = 2.0;
    let dt = 0.01;
    let n_steps = (duration / dt).ceil() as usize;
    let t_eval: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();

    // Initial state: [x, y, z, vx, vy, vz]
    let state0 = [0.0, 0.0, 10.0, 8.0, 8.0, -1.0];

    println!("Simulating trajectory:");
    println!(
        "  Initial position: ({:.2}, {:.2}, {:.2})m",
        state0[0], state0[1], state0[2]
    );
    println!(
        "  Initial velocity: ({:.2}, {:.2}, {:.2})m/s",
        state0[3], state0[4], state0[5]
    );
    println!("  Duration: {:.1f}s, dt: {:.3f}s", duration, dt);
    println!("  Points: {}", t_eval.len());

    let start = Instant::now();
    let model = SpeModel {};
    let trajectory = model.simulate(&t_eval, &state0, &params, constants);
    let duration_sim = start.elapsed();

    println!("  Simulation time: {:.2}μs", duration_sim.as_micros());
    println!();

    // Compute metrics
    let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
    let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
    let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

    let start = Instant::now();
    let metrics = TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, constants);
    let duration_metrics = start.elapsed();

    if let Ok(m) = metrics {
        println!("Computed Metrics:");
        println!("  Flight time: {:.3f}s", m.flight_time());
        println!("  Horizontal distance: {:.2f}m", m.horizontal_distance());
        println!(
            "  Vertical displacement: {:.2f}m",
            m.vertical_displacement()
        );
        println!("  Mean speed: {:.2f}m/s", m.mean_speed());
        println!(
            "  Mean horizontal speed: {:.2f}μs",
            m.mean_speed_xy() * 1000.0
        );
        println!(
            "  Mean energy dissipation: {:.2f}W/kg",
            m.mean_energy_dissipation()
        );
        println!(
            "  Min altitude: {:.2f}m",
            m.z.iter().cloned().fold(f64::INFINITY, f64::min)
        );
        println!(
            "  Max altitude: {:.2f}m",
            m.z.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        );
        println!(
            "  Metrics computation: {:.2}μs",
            duration_metrics.as_micros()
        );
    }
}

/// Demonstrate BAP model simulation
fn demo_bap_model(constants: &Constants) {
    println!("--- BAP Model Demo ---");

    let params = BapParams::default_params();
    println!("BAP Parameters:");
    println!("  CL           = {:.4f}", params.cl);
    println!("  C_D          = {:.4f}", params.cd);
    println!("  Φ_base       = {:.4f}", params.phi_base);
    println!("  k_bank       = {:.4f}", params.k_bank);
    println!("  V₀_scalar    = {:.4f}", params.v0_scalar);
    println!("  Ω_scale      = {:.4f}", params.omega_scale);
    println!();

    // Simulate a trajectory
    let duration = 2.0;
    let dt = 0.01;
    let n_steps = (duration / dt).ceil() as usize;
    let t_eval: Vec<f64> = (0..=n_steps).map(|i| i as f64 * dt).collect();

    // Initial state: [x, y, z, vx, vy, vz]
    let state0 = [0.0, 0.0, 10.0, 8.0, 8.0, -1.0];

    println!("Simulating trajectory:");
    println!(
        "  Initial position: ({:.2}, {:.2}, {:.2})m",
        state0[0], state0[1], state0[2]
    );
    println!(
        "  Initial velocity: ({:.2}, {:.2}, {:.2})m/s",
        state0[3], state0[4], state0[5]
    );
    println!("  Duration: {:.1f}s, dt: {:.3f}s", duration, dt);
    println!("  Points: {}", t_eval.len());

    let start = Instant::now();
    let model = BapModel {};
    let trajectory = model.simulate(&t_eval, &state0, &params, constants);
    let duration_sim = start.elapsed();

    println!("  Simulation time: {:.2}μs", duration_sim.as_micros());
    println!();

    // Compute metrics
    let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
    let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
    let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

    let metrics = TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, constants);

    if let Ok(m) = metrics {
        println!("Computed Metrics:");
        println!("  Flight time: {:.3f}s", m.flight_time());
        println!("  Horizontal distance: {:.2f}m", m.horizontal_distance());
        println!(
            "  Vertical displacement: {:.2f}m",
            m.vertical_displacement()
        );
        println!("  Mean speed: {:.2f}m/s", m.mean_speed());
        println!(
            "  Mean horizontal speed: {:.2f}μs",
            m.mean_speed_xy() * 1000.0
        );
        println!(
            "  Mean energy dissipation: {:.2f}W/kg",
            m.mean_energy_dissipation()
        );

        // Check if trajectory reaches ground
        let min_z = m.z.iter().cloned().fold(f64::INFINITY, f64::min);
        if min_z < -0.5 {
            println!("  Warning: Trajectory went below -0.5m (may have crashed)");
        } else if min_z < 0.0 {
            println!("  Note: Trajectory approached ground (z={:.2f}m)", min_z);
        }
    }
}

/// Performance comparison between single and batch runs
fn performance_demo(constants: &Constants) {
    println!("--- Performance Benchmark ---");

    let params = SpeParams::default_params();
    let t_eval: Vec<f64> = (0..=200).map(|i| i as f64 * 0.01).collect();
    let state0 = [0.0, 0.0, 10.0, 5.0, 8.0, -1.0];

    // Single simulation
    println!("Single trajectory simulation:");
    let model = SpeModel {};

    let start = Instant::now();
    let trajectory = model.simulate(&t_eval, &state0, &params, constants);
    let duration_single = start.elapsed();

    println!("  Duration: {:.2}μs", duration_single.as_micros());
    println!(
        "  Throughput: {:.2} points/μs",
        t_eval.len() as f64 / duration_single.as_micros() as f64
    );

    // Batch simulation (5 trajectories)
    println!();
    println!("Batch simulation (5 trajectories with slight variations):");

    let states0: Vec<[f64; 6]> = (0..5)
        .map(|i| {
            let mut state = state0;
            state[3] += i as f64 * 0.5; // Vary initial vx
            state
        })
        .collect();

    let start = Instant::now();
    let trajectories: Vec<Vec<[f64; 6]>> = states0
        .iter()
        .map(|state| model.simulate(&t_eval, state, &params, constants))
        .collect();
    let duration_batch = start.elapsed();

    println!("  Duration: {:.2}μs", duration_batch.as_micros());
    println!(
        "  Per trajectory: {:.2}μs",
        duration_batch.as_micros() as f64 / 5.0
    );
    println!(
        "  Total points: {}×{} = {}",
        5,
        t_eval.len(),
        5 * t_eval.len()
    );
    println!(
        "  Macro throughput: {:.2} points/μs",
        (5.0 * t_eval.len() as f64) / duration_batch.as_micros() as f64
    );

    // Speedup factor
    let speedup = duration_batch.as_micros() as f64 / (5.0 * duration_single.as_micros() as f64);
    println!();
    println!("  Batch efficiency: {:.2}x", speedup);
    println!("  (vs ideal 5x)");
}

/// Simple parameter optimization demonstration
fn parameter_optimization_demo(constants: &Constants) {
    println!("--- Parameter Optimization Demo ---");

    let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * 0.02).collect();
    let state0 = [0.0, 0.0, 5.0, 5.0, 5.0, -0.5];

    // Create a reference trajectory
    let ref_params = SpeParams::default_params();
    let model = SpeModel {};
    let ref_trajectory = model.simulate(&t_eval, &state0, &ref_params, constants);
    let ref_x: Vec<_> = ref_trajectory.iter().map(|s| s[0]).collect();
    let ref_y: Vec<_> = ref_trajectory.iter().map(|s| s[1]).collect();
    let ref_z: Vec<_> = ref_trajectory.iter().map(|s| s[2]).collect();

    println!("Reference trajectory generated.");
    println!();

    // Try different CL values
    let cl_values = vec![0.2, 0.4, 0.6, 0.8, 1.0];
    let mut results = Vec::new();

    println!("Testing different CL (lift coefficient) values:");
    println!("  Target: CL_trans = {:.2f}", ref_params.cl_trans);
    println!();

    for &cl in &cl_values {
        let mut params = SpeParams::default_params();
        params.cl_trans = cl;

        let start = Instant::now();
        let trajectory = model.simulate(&t_eval, &state0, &params, constants);
        let duration = start.elapsed().as_micros();

        let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
        let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
        let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

        if let Ok(sim_metrics) = TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, constants)
        {
            if let Ok(ref_metrics) =
                TrajectoryMetrics::from_trajectory(&t_eval, &ref_x, &ref_y, &ref_z, constants)
            {
                let mse = trajectory_mse(&sim_metrics, &ref_metrics).unwrap_or(f64::MAX);
                let mse_sq = mse.sqrt();

                results.push((cl, mse, duration));

                println!("  CL = {:>4.2f} | MSE = {:>8.5f} | RMSE = {:>5.3f}m | Time = {:>5.0}μs | Speed = {:.2f}m/s",
                         cl, mse, mse_sq, duration, sim_metrics.mean_speed());
            }
        }
    }

    println!();

    // Find best fit
    if let Some((best_cl, best_mse, _)) =
        results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        println!(
            "Best fit: CL_trans = {:.2f}, RMSE = {:.3f}m",
            best_cl,
            best_mse.sqrt()
        );

        if (best_cl - ref_params.cl_trans).abs() < 0.1 {
            println!("✓ Successfully recovered target parameter!");
        } else {
            println!("  Note: Best fit deviates from target (this is expected without proper optimization)");
        }
    }

    println!();
    println!("Parameter optimization demo completed.");
    println!(
        "In real scenarios, this would be wrapped in an optimization loop (L-BFGS-B or similar)."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spe_model_simulation() {
        let constants = Constants::experimental();
        let params = SpeParams::default_params();
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 1.0, 5.0, 5.0, 0.0];

        let model = SpeModel {};
        let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

        assert_eq!(trajectory.len(), t_eval.len());
        assert!(trajectory[0][2] > 0.0); // Start above ground
    }

    #[test]
    fn test_bap_model_simulation() {
        let constants = Constants::experimental();
        let params = BapParams::default_params();
        let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let state0 = [0.0, 0.0, 1.0, 8.0, 8.0, -0.5];

        let model = BapModel {};
        let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

        assert_eq!(trajectory.len(), t_eval.len());
        assert!(trajectory[0][2] > 0.0);
    }

    #[test]
    fn test_metrics_computation() {
        let constants = Constants::experimental();
        let t: Vec<f64> = (0..=5).map(|i| i as f64 * 0.1).collect();
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let z = vec![0.0, -0.1, -0.2, -0.3, -0.4, -0.5];

        let metrics = TrajectoryMetrics::from_trajectory(&t, &x, &y, &z, &constants);
        assert!(metrics.is_ok());

        let m = metrics.unwrap();
        assert!(m.flight_time() > 0.4); // Should be around 0.5s
        assert!(m.horizontal_distance() > 3.0); // Should be 5m
    }
}
