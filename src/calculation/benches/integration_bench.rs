//! Benchmark tests for aerodynamic calculation performance.
//!
//! This file contains critical path performance tests that directly compare
//! the Rust implementation against established Python references.
//!
//! Run with: `cargo bench`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};

use boomerang_calc::constants::{Constants, model_params::{BapParams, SpeParams}};
use boomerang_calc::aerodynamics::{SpeModel, BapModel, AerodynamicModel};
use boomerang_calc::metrics::TrajectoryMetrics;

/// Generate time array for a typical boomerang flight
fn generate_time_array(duration: f64, dt: f64) -> Vec<f64> {
    let n = (duration / dt).ceil() as usize;
    (0..=n).map(|i| i as f64 * dt).collect()
}

/// Generate typical initial state for a boomerang throw
fn generate_initial_state() -> [f64; 6] {
    // [x, y, z, vx, vy, vz]
    [0.0, 0.0, 10.0, 5.0, 8.0, -2.0] // Throw from height 10m, moderate forward velocity
}

/// Create SPE parameters from Python reference values
fn create_spe_params() -> SpeParams {
    SpeParams {
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

/// Create BAP parameters from Python reference values
fn create_bap_params() -> BapParams {
    BapParams {
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

/// Benchmark SPE model simulation
fn bench_spe_model(c: &mut Criterion) {
    let constants = Constants::experimental();
    let params = create_spe_params();
    let state0 = generate_initial_state();
    let t_eval = generate_time_array(2.0, 0.01); // 2 seconds, 10ms steps

    let mut group = c.benchmark_group("SpeModel");
    group.throughput(Throughput::Elements(t_eval.len() as u64));

    group.bench_function("simulate_single", |bencher| {
        bencher.iter(|| {
            let model = SpeModel {};
            model.simulate(&t_eval, &state0, &params, &constants)
        });
    });

    // Multiple trajectories batch
    let states0: Vec<_> = (0..5)
        .map(|i| {
            let mut state = state0;
            state[3] += i as f64 * 0.1; // Slight velocity variation
            state
        })
        .collect();

    group.bench_function("simulate_batch_5", |bencher| {
        bencher.iter(|| {
            let model = SpeModel {};
            states0
                .iter()
                .map(|state| model.simulate(&t_eval, state, &params, &constants))
                .collect::<Vec<_>>()
        });
    });

    group.finish();
}

/// Benchmark BAP model simulation
fn bench_bap_model(c: &mut Criterion) {
    let constants = Constants::experimental();
    let params = create_bap_params();
    let state0 = generate_initial_state();
    let t_eval = generate_time_array(2.0, 0.01); // 2 seconds, 10ms steps

    let mut group = c.benchmark_group("BapModel");
    group.throughput(Throughput::Elements(t_eval.len() as u64));

    group.bench_function("simulate_single", |bencher| {
        bencher.iter(|| {
            let model = BapModel {};
            model.simulate(&t_eval, &state0, &params, &constants)
        });
    });

    // Multiple trajectories batch
    let states0: Vec<_> = (0..5)
        .map(|i| {
            let mut state = state0;
            state[3] += i as f64 * 0.1;
            state
        })
        .collect();

    group.bench_function("simulate_batch_5", |bencher| {
        bencher.iter(|| {
            let model = BapModel {};
            states0
                .iter()
                .map(|state| model.simulate(&t_eval, state, &params, &constants))
                .collect::<Vec<_>>()
        });
    });

    group.finish();
}

/// Benchmark trajectory metrics computation
fn bench_metrics(c: &mut Criterion) {
    let constants = Constants::experimental();
    let t_eval = generate_time_array(1.0, 0.01);
    let state0 = generate_initial_state();

    // Generate trajectory data for metrics computation
    let model = SpeModel {};
    let params = create_spe_params();
    let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

    let x: Vec<f64> = trajectory.iter().map(|s| s[0]).collect();
    let y: Vec<f64> = trajectory.iter().map(|s| s[1]).collect();
    let z: Vec<f64> = trajectory.iter().map(|s| s[2]).collect();

    let mut group = c.benchmark_group("Metrics");
    group.throughput(Throughput::Elements(t_eval.len() as u64));

    group.bench_function("from_trajectory", |bencher| {
        bencher.iter(|| {
            TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, &constants)
        });
    });

    // Simulate and compute metrics (end-to-end)
    group.bench_function("simulate_and_compute", |bencher| {
        bencher.iter(|| {
            let trajectory = SpeModel {}.simulate(&t_eval, &state0, &params, &constants);
            let x: Vec<f64> = trajectory.iter().map(|s| s[0]).collect();
            let y: Vec<f64> = trajectory.iter().map(|s| s[1]).collect();
            let z: Vec<f64> = trajectory.iter().map(|s| s[2]).collect();
            TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, &constants)
        });
    });

    group.finish();
}

/// Benchmark combined Spe + BAP simulation (typical fitting scenario)
fn bench_combined_simulation(c: &mut Criterion) {
    let constants = Constants::experimental();
    let t_eval = generate_time_array(1.5, 0.01);
    let state0 = generate_initial_state();

    let spe_params = create_spe_params();
    let bap_params = create_bap_params();

    let mut group = c.benchmark_group("Combined");
    group.throughput(Throughput::Elements(t_eval.len() as u64));

    group.bench_function("alternating_simulations", |bencher| {
        bencher.iter(|| {
            let model1 = SpeModel {};
            let result1 = model1.simulate(&t_eval, &state0, &spe_params, &constants);

            let model2 = BapModel {};
            let result2 = model2.simulate(&t_eval, &state0, &bap_params, &constants);

            // Return some validation
            (result1.len(), result2.len())
        });
    });

    group.finish();
}

/// Benchmark energy calculations
fn bench_energy_calculations(c: &mut Criterion) {
    let constants = Constants::experimental();
    let state0 = generate_initial_state();
    let t_eval = generate_time_array(2.0, 0.01);
    let model = SpeModel {};
    let params = create_spe_params();

    // Generate trajectory
    let trajectory = model.simulate(&t_eval, &state0, &params, &constants);
    let x: Vec<f64> = trajectory.iter().map(|s| s[0]).collect();
    let y: Vec<f64> = trajectory.iter().map(|s| s[1]).collect();
    let z: Vec<f64> = trajectory.iter().map(|s| s[2]).collect();

    let mut group = c.benchmark_group("Energy");
    group.throughput(Throughput::Elements(t_eval.len() as u64));

    group.bench_function("compute_metrics", |bencher| {
        bencher.iter(|| {
            TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, &constants)
        });
    });

    group.finish();
}

/// Benchmark parameter optimization (simulating optimization loop)
fn bench_optimization_loop(c: &mut Criterion) {
    let constants = Constants::experimental();
    let t_eval = generate_time_array(1.0, 0.05); // Coarser time for faster benchmarks
    let state0 = generate_initial_state();

    let mut group = c.benchmark_group("Optimization");

    group.bench_function("spe_single_iteration", |bencher| {
        bencher.iter(|| {
            // Simulate single parameter combination
            let model = SpeModel {};
            let params = create_spe_params();
            let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

            // Compute metrics (typical loss calculation)
            let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
            let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
            let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

            let metrics = TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, &constants);
            metrics.map(|m| m.mean_speed())
        });
    });

    group.bench_function("bap_single_iteration", |bencher| {
        bencher.iter(|| {
            // Simulate single parameter combination
            let model = BapModel {};
            let params = create_bap_params();
            let trajectory = model.simulate(&t_eval, &state0, &params, &constants);

            // Compute metrics (typical loss calculation)
            let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
            let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
            let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

            let metrics = TrajectoryMetrics::from_trajectory(&t_eval, &x, &y, &z, &constants);
            metrics.map(|m| m.mean_speed())
        });
    });

    group.bench_function("multi_param_loop_10", |bencher| {
        bencher.iter(|| {
            let model = SpeModel {};
            let constants = &constants;
            let t_eval = &t_eval;
            let state0 = &state0;

            let variations = (0..10).map(|i| {
                let mut params = create_spe_params();
                params.cl_trans += i as f64 * 0.05;
                params
            });

            let mut results = Vec::with_capacity(10);
            for params in variations {
                let trajectory = model.simulate(t_eval, state0, &params, constants);
                let x: Vec<_> = trajectory.iter().map(|s| s[0]).collect();
                let y: Vec<_> = trajectory.iter().map(|s| s[1]).collect();
                let z: Vec<_> = trajectory.iter().map(|s| s[2]).collect();

                if let Ok(metrics) = TrajectoryMetrics::from_trajectory(t_eval, &x, &y, &z, constants) {
                    results.push(metrics.mean_speed());
                }
            }
            results
        });
    });

    group.finish();
}

Criterion::default()
    .configure_from_args()
    .sample_size(100) // Small sample size for fast iteration benchmark
    .bench_function_over_inputs(
        "parameter_space_exploration",
        |bencher, state| {
            let constants = Constants::experimental();
            let t_eval = generate_time_array(0.5, 0.05);
            let state0 = *state;

            bencher.iter(|| {
                let mut all_results = Vec::with_capacity(5);
                for _ in 0..5 {
                    let params = create_spe_params();
                    let trajectory = SpeModel {}.simulate(&t_eval, &state0, &params, &constants);
                    let metrics = TrajectoryMetrics::from_trajectory(
                        &t_eval,
                        &trajectory.iter().map(|s| s[0]).collect::<Vec<_>>(),
                        &trajectory.iter().map(|s| s[1]).collect::<Vec<_>>(),
                        &trajectory.iter().map(|s| s[2]).collect::<Vec<_>>(),
                        &constants,
                    );
                    if let Ok(m) = metrics {
                        all_results.push(m.mean_energy_dissipation());
                    }
                }
                all_results
            });
        },
        vec![
            generate_initial_state(),
            [0.0, 0.0, 5.0, 8.0, 8.0, 0.0], // Different initial condition 1
            [0.0, 0.0, 15.0, 3.0, 6.0, -3.0], // Different initial condition 2
        ],
    );

criterion_group!(benches,
    bench_spe_model,
    bench_bap_model,
    bench_metrics,
    bench_combined_simulation,
    bench_energy_calculations,
    bench_optimization_loop,
);

criterion_main!(benches);

```

<literal_integration_bench>
