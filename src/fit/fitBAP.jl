#!/usr/bin/env julia
"""
fitBAP.jl - Bank Angle Proxy Model Fitting (Julia Version)
Fits aerodynamic parameters (CL, CD) using a simplified physics model where
Bank Angle variation is the dominant driver of the trajectory.

Julia version provides significant performance improvements over Python.
"""

using CSV
using DataFrames
using DifferentialEquations
using Optim
using LinearAlgebra
using Statistics
using Random
using Glob
using Printf
using Dates
using ArgParse
using Base.Filesystem

# --- Model knobs (fixed, not optimized) ---
const PHI_TIME_GAIN = 0.20  # rad per (omega_scale * second)
const SELF_LEVEL_GAIN = 0.20
const BANK_LIFT_LOSS = 0.20  # fraction of lift reduced at |phi|=90deg
const BANK_DRAG_GAIN = 0.40  # fraction of drag increased at |phi|=90deg

# --- Constants ---
const RHO = 1.225          # Air density (kg/m^3)
const G = 9.793            # Gravity (m/s^2)
const M = 0.002183         # Mass (kg)
const A_SPAN = 0.15        # Wingspan (m)
const D_WIDTH = 0.028      # Wing width (m)
const AREA = 2 * A_SPAN * D_WIDTH # Reference Area

const TRACK_META = Dict(
    "track1" => 5.3 / 0.93 * 2 * π,
    "track2" => 7.8 / 1.28 * 2 * π,
    "track5" => 5.0 / 1.17 * 2 * π,
    "track6" => 5.4 / 1.07 * 2 * π,
    "track7" => 5.2 / 1.17 * 2 * π,
    "track9" => 4.8 / 1.07 * 2 * π,
)

const OMEGA_REF = Statistics.median(collect(values(TRACK_META)))

# --- Data Structures ---
struct TrackData
    t::Vector{Float64}
    pos::Matrix{Float64}  # n×3 matrix
    v0::Vector{Float64}   # initial velocity [vx, vy, vz]
    omega::Float64
    vel_meas::Matrix{Float64}  # measured velocity
    acc_meas::Matrix{Float64}  # measured acceleration
end

# --- Helper Functions ---
function gradient(y::Vector{Float64}, t::Vector{Float64})
    # Simple gradient calculation
    n = length(y)
    dy = similar(y)
    if n >= 2
        dy[1] = (y[2] - y[1]) / (t[2] - t[1])
        dy[end] = (y[end] - y[end-1]) / (t[end] - t[end-1])
        for i in 2:n-1
            dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
        end
    else
        fill!(dy, 0.0)
    end
    return dy
end

function unwrap(angles::Vector{Float64})
    # Unwrap phase angles
    result = copy(angles)
    for i in 2:length(angles)
        delta = angles[i] - angles[i-1]
        if delta > π
            result[i] -= 2π
        elseif delta < -π
            result[i] += 2π
        end
    end
    return result
end

# --- Data Loading ---
function load_data()
    """
    Loads optimized smoothed track data from data/*opt.csv
    Calculates initial velocities to seed the integration.
    """
    tracks = Dict{String, TrackData}()
    csv_files = glob("data/*opt.csv")
    sort!(csv_files)

    if isempty(csv_files)
        println("No data/*opt.csv files found.")
        return tracks
    end

    for f in csv_files
        key = basename(f)[1:end-8]  # remove "opt.csv"
        if !haskey(TRACK_META, key)
            continue
        end

        df = CSV.read(f, DataFrame)

        # Calculate initial velocity (Average of first few finite differences for stability)
        dt_start = df.t[2] - df.t[1]  # Assume constant roughly
        if dt_start == 0
            continue
        end

        vx0 = 0.0; vy0 = 0.0; vz0 = 0.0
        weights = [5, 4, 3, 2, 1]
        w_sum = sum(weights)

        for i in 1:min(5, size(df, 1)-1)
            dt = df.t[i+1] - df.t[i]
            if dt == 0
                continue
            end
            wx = (df.x[i+1] - df.x[i]) / dt
            wy = (df.y[i+1] - df.y[i]) / dt
            wz = (df.z[i+1] - df.z[i]) / dt
            vx0 += wx * weights[i]
            vy0 += wy * weights[i]
            vz0 += wz * weights[i]
        end

        vx0 /= w_sum
        vy0 /= w_sum
        vz0 /= w_sum

        t = Vector{Float64}(df.t)
        pos = Matrix{Float64}(hcat(df.x, df.y, df.z))
        v0 = [vx0, vy0, vz0]
        omega = TRACK_META[key]

        # Precompute measured velocity and acceleration
        if length(t) >= 5
            vx_m = gradient(pos[:, 1], t)
            vy_m = gradient(pos[:, 2], t)
            vz_m = gradient(pos[:, 3], t)
            vel_meas = hcat(vx_m, vy_m, vz_m)

            ax_m = gradient(vx_m, t)
            ay_m = gradient(vy_m, t)
            az_m = gradient(vz_m, t)
            acc_meas = hcat(ax_m, ay_m, az_m)
        else
            vel_meas = zeros(size(pos))
            acc_meas = zeros(size(pos))
        end

        tracks[key] = TrackData(t, pos, v0, omega, vel_meas, acc_meas)
    end

    return tracks
end

# --- ODE Right-Hand Side ---
function boomerang_rhs(du, u, p, t)
    """
    RHS for DifferentialEquations.jl

    State u: [x, y, z, vx, vy, vz]
    Parameters p: (CL, CD, phi_base, k_bank, v0_scalar, omega_scale)
    Coordinate convention (right-handed): x-right, y-forward, z-up.
    """
    x, y, z, vx, vy, vz = u
    CL, CD, phi_base, k_bank, v0_scalar, omega_scale = p

    # Velocity magnitudes
    v_sq = vx^2 + vy^2 + vz^2
    v = sqrt(v_sq) + 1e-9
    v_xy = sqrt(vx^2 + vy^2) + 1e-9

    # --- 1. Dynamic Bank Angle Proxy ---
    speed_loss = v0_scalar - v_xy
    phi = phi_base + (k_bank * omega_scale) * speed_loss + (PHI_TIME_GAIN * omega_scale) * t

    # Correction: Dive Self-Leveling (smooth)
    dive_ratio = clamp(-vz / v, 0.0, 1.0)
    phi *= (1.0 - SELF_LEVEL_GAIN * dive_ratio)

    # Clamp physically
    phi = clamp(phi, -1.5, 1.5)  # +/- ~85 degrees

    # Directions (3D)
    # Drag direction: opposite to velocity
    v_hat = [vx, vy, vz] / v

    # Lift direction: perpendicular to velocity, banked by phi around v_hat.
    up = [0.0, 0.0, 1.0]

    # "Right" lateral axis relative to motion
    lateral = cross(v_hat, up)
    lateral_norm = norm(lateral)
    if lateral_norm < 1e-9
        # If moving nearly vertical, choose an arbitrary lateral axis
        lateral = [1.0, 0.0, 0.0]
    else
        lateral /= lateral_norm
    end

    # Base lift direction (roughly up) perpendicular to v_hat
    lift0 = cross(lateral, v_hat)
    lift0_norm = norm(lift0)
    if lift0_norm < 1e-9
        lift0 = up
    else
        lift0 /= lift0_norm
    end

    # Rodrigues rotation of lift0 about v_hat by bank angle phi
    lift_dir = (
        lift0 * cos(phi)
        + cross(v_hat, lift0) * sin(phi)
        + v_hat * (dot(v_hat, lift0)) * (1.0 - cos(phi))
    )

    # --- 2. Aerodynamic Forces ---
    q = 0.5 * RHO * AREA

    # Lift Force Magnitude (v^1.5 model)
    F_lift_mag = q * CL * (v^1.5)

    # Bank efficiency
    lift_eff = 1.0 - BANK_LIFT_LOSS * (sin(phi)^2)
    F_lift_mag *= clamp(lift_eff, 0.25, 1.0)

    # Ground Effect (Z < 0.2m correction)
    if z < 0.2
        h_eff = max(z, 0.05)
        F_lift_mag *= (1.0 + 0.2 * 0.2 / h_eff)  # mild ground cushion
    end

    # Lift Vector (3D)
    lift_vec = F_lift_mag * lift_dir

    # Drag Force Magnitude (v^2 standard model)
    cd_reduction = 0.25
    CD_eff = CD * (1.0 - cd_reduction * dive_ratio) * (1.0 + BANK_DRAG_GAIN * (sin(phi)^2))
    F_drag_mag = q * CD_eff * v_sq

    # Drag Vector
    drag_vec = -F_drag_mag * v_hat

    # Sum Accels (F/m)
    total_acc = (lift_vec + drag_vec) / M
    total_acc[3] -= G  # gravity

    du[1] = vx
    du[2] = vy
    du[3] = vz
    du[4] = total_acc[1]
    du[5] = total_acc[2]
    du[6] = total_acc[3]

    return nothing
end

# --- Simulation ---
function simulate_track(params, track_data::TrackData; return_full=false)
    CL, CD, phi_base, k_bank = params
    t = track_data.t
    v0 = track_data.v0
    r0 = track_data.pos[1, :]
    omega = track_data.omega

    # Scalar initial speed for bank proxy reference
    v0_xy_scalar = sqrt(v0[1]^2 + v0[2]^2)
    omega_scale = omega / OMEGA_REF

    state0 = [r0[1], r0[2], r0[3], v0[1], v0[2], v0[3]]
    p = (CL, CD, phi_base, k_bank, v0_xy_scalar, omega_scale)

    try
        tspan = (t[1], t[end])
        prob = ODEProblem(boomerang_rhs, state0, tspan, p)

        # Adaptive time stepping with dense output
        sol = solve(prob, Tsit5(), saveat=t, reltol=1e-6, abstol=1e-6)

        if !sol.retcode.success || size(sol, 2) != length(t)
            return zeros(length(t), 6) if return_full else zeros(length(t), 3)
        end

        states = transpose(hcat(sol.u...))
        return states if return_full else states[:, 1:3]
    catch e
        # Return zeros if failed
        return zeros(length(t), 6) if return_full else zeros(length(t), 3)
    end
end

# --- Loss Function ---
function loss_function(params, tracks::Dict{String, TrackData})
    CL, CD, phi_base, k_bank = params

    total_mse = 0.0

    for (key, data) in tracks
        sim_state = simulate_track(params, data, return_full=true)
        sim_pos = sim_state[:, 1:3]
        sim_vel = sim_state[:, 4:6]
        real_pos = data.pos
        real_vel = data.vel_meas
        real_acc = data.acc_meas

        # Check for NaN divergence or empty result
        if any(isnan.(sim_pos)) || all(sim_pos .== 0)
            return 1e9 + rand()  # Return high loss
        end

        # Bail out early on blow-ups
        if maximum(abs.(sim_pos - real_pos)) > 5.0
            return 1e8
        end

        # Position loss
        dpos = sim_pos - real_pos
        w_t = LinRange(1.0, 3.0, size(real_pos, 1))
        w_xy = 1.0
        w_z = 0.6
        pos_err = w_xy .* (dpos[:, 1].^2 .+ dpos[:, 2].^2) .+ w_z .* (dpos[:, 3].^2)
        mse_pos = mean(pos_err .* w_t)

        # Heading/turning loss
        theta_sim = unwrap(atan.(sim_vel[:, 2], sim_vel[:, 1]))
        theta_real = unwrap(atan.(real_vel[:, 2], real_vel[:, 1]))
        mse_theta = mean((theta_sim - theta_real).^2 .* w_t)

        # Turn-rate loss
        dtheta_sim = gradient(theta_sim, data.t)
        dtheta_real = gradient(theta_real, data.t)
        mse_dtheta = mean((dtheta_sim - dtheta_real).^2 .* w_t)

        # End-segment speed loss
        n = size(real_pos, 1)
        tail_start = max(1, Int(floor(0.8 * n)))
        tail = tail_start:n
        vy_sim = sim_vel[tail, 2]
        vy_real = real_vel[tail, 2]
        mse_vy_tail = mean((abs.(vy_sim) - abs.(vy_real)).^2)

        # Weighted total
        total_mse += mse_pos + 0.35 * mse_theta + 0.80 * mse_dtheta + 1.20 * mse_vy_tail
    end

    return total_mse
end

# --- Optimization ---
function sample_within_bounds(bounds, rng)
    sampled = Float64[]
    for (lo, hi) in bounds
        push!(sampled, rand(rng) * (hi - lo) + lo)
    end
    return sampled
end

function _optimize_one_start(guess, tracks, bounds)
    """Run one local optimization."""
    # Define the objective function
    function objective(params)
        return loss_function(params, tracks)
    end

    # Set up bounds for Optim
    lower = [b[1] for b in bounds]
    upper = [b[2] for b in bounds]

    # Use L-BFGS with bounds
    result = optimize(objective, lower, upper, guess, Fminbox(LBFGS()))

    return Dict(
        "success" => Optim.converged(result),
        "fun" => Optim.minimum(result),
        "x" => Optim.minimizer(result),
        "message" => string(Optim.summary(result)),
    )
end

function fit_multistart_parallel(tracks, bounds, x0, starts, seed, workers)
    rng = MersenneTwister(seed)

    initial_points = [x0]
    for _ in 1:max(0, starts - 1)
        push!(initial_points, sample_within_bounds(bounds, rng))
    end

    n_workers = workers <= 0 ? max(1, Threads.nthreads()) : workers
    n_workers = min(n_workers, length(initial_points))

    best = nothing
    best_x = nothing
    best_fun = Inf

    # Use Threads for parallel optimization in Julia
    results = Vector{Dict}(undef, length(initial_points))

    Threads.@threads for idx in 1:length(initial_points)
        guess = initial_points[idx]
        results[idx] = _optimize_one_start(guess, tracks, bounds)
    end

    for (idx, out) in enumerate(results)
        println("[start $(idx)/$(length(initial_points))] loss=$(out["fun"]:.4f) success=$(out["success"])")
        if out["success"] && out["fun"] < best_fun
            best_fun = out["fun"]
            best_x = out["x"]
            best = out
        end
    end

    return best, best_x
end

function compute_track_metrics(track_key, best_params, tracks)
    data = tracks[track_key]
    sim_state = simulate_track(best_params, data, return_full=true)
    sim_pos = sim_state[:, 1:3]
    sim_vel = sim_state[:, 4:6]
    real_pos = data.pos
    real_vel = data.vel_meas

    dpos = sim_pos - real_pos
    pos_rmse = sqrt(mean(sum(dpos .^ 2, dims=2)))

    n = size(real_pos, 1)
    tail_start = max(1, Int(floor(0.8 * n)))
    tail = tail_start:n
    pos_tail_rmse = sqrt(mean(sum(dpos[tail, :] .^ 2, dims=2)))

    theta_sim = unwrap(atan.(sim_vel[:, 2], sim_vel[:, 1]))
    theta_real = unwrap(atan.(real_vel[:, 2], real_vel[:, 1]))
    theta_rmse = sqrt(mean((theta_sim - theta_real) .^ 2))

    vy_tail_rmse = sqrt(mean((abs.(sim_vel[tail, 2]) - abs.(real_vel[tail, 2])) .^ 2))

    final_err = norm(sim_pos[end, :] - real_pos[end, :])
    final_vy_abs_sim = abs(sim_vel[end, 2])
    final_vy_abs_real = abs(real_vel[end, 2])

    return Dict(
        "key" => track_key,
        "pos_rmse" => pos_rmse,
        "pos_tail_rmse" => pos_tail_rmse,
        "theta_rmse" => theta_rmse,
        "vy_tail_rmse" => vy_tail_rmse,
        "final_err" => final_err,
        "final_vy_abs_sim" => final_vy_abs_sim,
        "final_vy_abs_real" => final_vy_abs_real,
    )
end

# --- Main Function ---
function main()
    # Parse command line arguments
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--no-plot"
            help = "Fit only, do not show matplotlib windows"
            action = :store_true
        "--starts"
            help = "Number of multistart initializations"
            arg_type = Int
            default = max(12, Threads.nthreads())
        "--seed"
            help = "RNG seed for multistart"
            arg_type = Int
            default = 0
        "--workers"
            help = "Parallel worker processes for multistart (0=auto)"
            arg_type = Int
            default = 0
        "--track"
            help = "Only report/plot a single track key (e.g. track1)"
            arg_type = String
            default = ""
        "--report"
            help = "Print per-track error summary and tail metrics"
            action = :store_true
        "--save-plots"
            help = "Directory to save plots as PNG instead of only showing"
            arg_type = String
            default = ""
        "--export-csv"
            help = "Export per-time-step diagnostics CSV (requires --track)"
            arg_type = String
            default = ""
        "--turn"
            help = "Constrain turning direction via phi_base bounds"
            arg_type = String
            default = "right"
            range_tester = x -> x in ["right", "left", "free"]
    end

    args = parse_args(s)

    tracks = load_data()
    println("Loaded $(length(tracks)) tracks.")

    # --- Optimization ---
    # Params: [CL, CD, phi_base, k_bank]
    x0 = [1.2, 0.35, 0.3, 0.15]

    # Bounds
    phi_bounds = (-1.5, 1.5)
    if args["turn"] == "right"
        phi_bounds = (0.0, 1.5)
    elseif args["turn"] == "left"
        phi_bounds = (-1.5, 0.0)
    end

    bounds = [
        (1.0, 5.0),    # CL: Force High Lift to counteract gravity with high bank
        (0.15, 0.45),  # CD: Allow slightly lower drag to match end-segment speeds
        phi_bounds,    # phi_base
        (-2.0, 2.0)    # k_bank
    ]

    println("Starting optimization (L-BFGS-B), multistart=$(args["starts"]), workers=$(args["workers"] > 0 ? args["workers"] : "auto")...")

    best, best_x = fit_multistart_parallel(tracks, bounds, x0,
                                          starts=args["starts"],
                                          seed=args["seed"],
                                          workers=args["workers"])

    if best_x === nothing
        println("Optimization failed for all starts.")
        return
    end

    println("\noptimization Success: $(best["success"])")
    println("Final Loss: $(best["fun"])")
    println("Params: $best_x")

    CL, CD, phi_base, k_bank = best_x
    println("\n--- Fit Results ---")
    println("CL (Lift Coeff v^1.5): $(@sprintf("%.4f", CL))")
    println("CD (Drag Coeff v^2)  : $(@sprintf("%.4f", CD))")
    println("phi_base (Initial)   : $(@sprintf("%.4f", phi_base)) rad ($(@sprintf("%.1f", rad2deg(phi_base))) deg)")
    println("k_bank   (Dynamic)   : $(@sprintf("%.4f", k_bank)) rad/(m/s)")

    # --- Report ---
    if args["report"]
        keys = sort(collect(keys(tracks)))
        if !isempty(args["track"])
            if !haskey(tracks, args["track"])
                println("Unknown track: $(args["track"]). Available: $keys")
            else
                keys = [args["track"]]
            end
        end

        rows = [compute_track_metrics(k, best_x, tracks) for k in keys]
        sort!(rows, by=r -> r["pos_tail_rmse"], rev=true)

        println("\nPer-track summary (sorted by tail RMSE):")
        for r in rows
            println(
                "  $(r["key"]): pos_rmse=$(@sprintf("%.3f", r["pos_rmse"]))  " *
                "tail_rmse=$(@sprintf("%.3f", r["pos_tail_rmse"]))  " *
                "theta_rmse=$(@sprintf("%.3f", r["theta_rmse"]))  " *
                "vy_tail_rmse=$(@sprintf("%.3f", r["vy_tail_rmse"]))  " *
                "final_err=$(@sprintf("%.3f", r["final_err"]))  " *
                "|vy|_end(sim/real)=$(@sprintf("%.3f", r["final_vy_abs_sim"]))/$(@sprintf("%.3f", r["final_vy_abs_real"]))"
            )
        end
    end

    # --- Export CSV ---
    if !isempty(args["export-csv"])
        if isempty(args["track"])
            println("--export-csv requires --track <trackKey> (e.g. --track track1)")
        elseif !haskey(tracks, args["track"])
            println("Unknown track: $(args["track"]). Available: $(sort(collect(keys(tracks))))")
        else
            k = args["track"]
            data = tracks[k]
            sim_state = simulate_track(best_x, data, return_full=true)
            sim_pos = sim_state[:, 1:3]
            sim_vel = sim_state[:, 4:6]
            real_pos = data.pos
            real_vel = data.vel_meas
            t = data.t

            v_xy_hist = sqrt.(sim_vel[:, 1].^2 .+ sim_vel[:, 2].^2) .+ 1e-9
            v0_scalar = sqrt(data.v0[1]^2 + data.v0[2]^2)
            omega_scale = data.omega / OMEGA_REF
            phi_hist = clamp.(phi_base .+ (k_bank * omega_scale) .* (v0_scalar .- v_xy_hist) .+ (PHI_TIME_GAIN * omega_scale) .* t, -1.5, 1.5)

            out_df = DataFrame(
                t = t,
                x_real = real_pos[:, 1], y_real = real_pos[:, 2], z_real = real_pos[:, 3],
                x_sim = sim_pos[:, 1], y_sim = sim_pos[:, 2], z_sim = sim_pos[:, 3],
                vx_real = real_vel[:, 1], vy_real = real_vel[:, 2], vz_real = real_vel[:, 3],
                vx_sim = sim_vel[:, 1], vy_sim = sim_vel[:, 2], vz_sim = sim_vel[:, 3],
                phi_deg = rad2deg.(phi_hist),
            )

            CSV.write(args["export-csv"], out_df)
            println("Exported diagnostics CSV -> $(args["export-csv"])")
        end
    end

    # Note: Plotting functionality would require Plots.jl and PyPlot backend
    # For now, we'll skip the plotting part in this initial version
    if !args["no-plot"] || !isempty(args["save-plots"])
        println("\nPlotting functionality requires Plots.jl with PyPlot backend.")
        println("To enable plotting, install: using Pkg; Pkg.add(\"Plots\"); Pkg.add(\"PyPlot\")")
        println("Then add: using Plots; pyplot()")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
