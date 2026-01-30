#!/usr/bin/env python3
"""Manual per-track preprocessing preProcess.

Design goal:
- One CLI argument: the input CSV path.
- Conservative, inspectable workflow (no over-aggressive auto truncation).
- Automatic: load -> estimate -> physics verdict -> save intermediate.
- NO INTERACTIVE UI (remove all interactive plots and text UI).

Dependencies: L1 (mathUtils), L2 (estimators), L3 (physicsVerdict)
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path for absolute imports
FILE_PATH = pathlib.Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.dataIO import load_track, save_track
from src.utils.estimators import EstimatorOutput, KalmanEstimator
from src.utils.physicsVerdict import PhysicsVerdict, VerdictReport
from src.utils.visualize import plot_3d_trajectory_compare, setup_debug_style


class preProcessController:
    """Three-layer preProcess controller for trajectory preprocessing.

    L1: mathUtils - Pure numerical operations (imported in estimator)
    L2: estimators - State estimation (p, v, a)
    L3: physicsVerdict - Physical validation and trimming
    """

    def __init__(
        self,
        csv_path: str | pathlib.Path,
        mass: float = 0.005,
        energy_tol: float = 0.05,
        sigma_threshold: float = 1.0,
        dt: float = 0.0166667,
    ) -> None:
        """Initialize the preProcess controller.

        Args:
            csv_path: Path to raw CSV file with columns [t, x, y, z]
            mass: Mass of the boomerang (kg)
            energy_tol: Energy tolerance for validation
            sigma_threshold: Uncertainty threshold for rejection
            dt: Target time step for standardization (for interpolation if needed)
        """
        self.csv_path = pathlib.Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Initialize modules
        self.estimator = KalmanEstimator(
            em_iters=10, process_noise=1e-4, measurement_noise=1e-2
        )
        self.verdict = PhysicsVerdict(
            mass=mass,
            energy_tol=energy_tol,
            sigma_threshold=sigma_threshold,
        )

        # Storage for intermediate results
        self.raw_data: Optional[Dict[str, np.ndarray]] = None
        self.estimated_state: Optional[EstimatorOutput] = None
        self.verdict_report: Optional[VerdictReport] = None
        self.cleaned_trajectory: Optional[Dict[str, np.ndarray]] = None

        self.dt = dt

        print(f"[preProcess] Initialized with estimator='kalman', mass={mass}kg")

    def load_and_preprocess(self) -> Dict[str, np.ndarray]:
        """Load CSV file and perform initial smoothing/preprocessing.

        Returns:
            Dictionary with raw data (t, x, y, z)

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If data validation fails
        """
        print(f"[preProcess] Loading raw data from {self.csv_path}")

        try:
            # Load raw data using dataIO module
            self.raw_data = load_track(
                filepath=self.csv_path,
                required_columns=["t", "x", "y", "z"],
                sort_by_time=True,
                clean_time=True,
                min_points=50,
            )

            n_points = len(self.raw_data["t"])
            print(f"[preProcess] Loaded {n_points} data points")
            print(
                f"[preProcess] Time range: [{self.raw_data['t'][0]:.3f}s, {self.raw_data['t'][-1]:.3f}s]"
            )

            return self.raw_data

        except Exception as e:
            print(f"[ERROR] Failed to load/validate data: {e}")
            raise

    def estimate_state(self) -> EstimatorOutput:
        """Estimate state (position, velocity, acceleration) from raw trajectory.

        Uses L2 (estimators) to perform state estimation.

        Returns:
            EstimatorOutput with standardized time grid and [p, v, a]
        """
        if self.raw_data is None:
            raise RuntimeError(
                "Must call load_and_preprocess() before estimate_state()"
            )

        print(f"[preProcess] Estimating state with Kalman estimator")

        try:
            # Extract raw arrays
            t_raw = self.raw_data["t"]
            pos_raw = np.column_stack(
                [
                    self.raw_data["x"],
                    self.raw_data["y"],
                    self.raw_data["z"],
                ]
            )

            # Call L2 estimator
            self.estimated_state = self.estimator.estimate(t_raw, pos_raw)

            print(
                f"[preProcess] Estimated state on uniform grid with {len(self.estimated_state.t_std)} points"
            )
            print(
                f"[preProcess] Time range (normalized): [{self.estimated_state.t_std[0]:.3f}s, {self.estimated_state.t_std[-1]:.3f}s]"
            )

            return self.estimated_state

        except Exception as e:
            print(f"[ERROR] State estimation failed: {e}")
            raise

    def evaluate_physics(self) -> VerdictReport:
        """Evaluate physical validity using L3 (physicsVerdict).

        Returns:
            VerdictReport with analysis results
        """
        if self.estimated_state is None:
            raise RuntimeError("Must call estimate_state() before evaluate_physics()")

        print("[preProcess] Evaluating physical validity")

        try:
            # Call L3 physics verdict
            self.verdict_report = self.verdict.evaluate(self.estimated_state)
            self.energies = self.verdict.calculate_energies(
                self.estimated_state.t_std,
                self.estimated_state.pos,
                self.estimated_state.vel,
            )
            print("[preProcess] Physics verdict completed")
            print(
                f"[preProcess]   - Valid duration: {self.verdict_report.valid_duration:.3f}s"
            )
            print(
                f"[preProcess]   - Termination reason: {self.verdict_report.reason_for_termination}"
            )
            print(f"[preProcess]   - Start index: {self.verdict_report.start_idx}")
            print(f"[preProcess]   - End index: {self.verdict_report.end_idx}")

            return self.verdict_report

        except Exception as e:
            print(f"[ERROR] Physics evaluation failed: {e}")
            raise

    def visualize_energy(self) -> None:
        """Step 2: Create dual-axis energy plot."""
        if self.energies is None or self.estimated_state is None:
            raise RuntimeError("Must call evaluate_physics() before visualize_energy()")

        print("[preProcess] Rendering energy plot...")

        # Setup Matplotlib style
        setup_debug_style()

        fig, ax = plt.subplots(figsize=(11, 6))

        t = self.estimated_state.t_std
        energy = self.energies["E"]
        dE_dt = self.energies["dE_dt"]

        # Left axis - Energy
        color_left = "#4285f4"
        ax.plot(t, energy, color=color_left, linewidth=2, label="Total Energy E(t)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Energy (J/kg)", color=color_left)
        ax.tick_params(axis="y", labelcolor=color_left)
        ax.grid(True, alpha=0.3)

        # Right axis - dE/dt (power)
        ax_right = ax.twinx()
        color_right = "#ea4335"
        ax_right.plot(
            t,
            dE_dt,
            color=color_right,
            linestyle="--",
            alpha=0.8,
            label="Power (dE/dt)",
        )
        ax_right.set_ylabel("Power (dE/dt)", color=color_right)
        ax_right.tick_params(axis="y", labelcolor=color_right)

        # Mark algorithm suggestion (vertical lines)
        if self.verdict_report is not None:
            start_idx = self.verdict_report.start_idx
            end_idx = self.verdict_report.end_idx
            start_t = t[start_idx] if start_idx < len(t) else t[0]
            end_t = t[end_idx - 1] if end_idx <= len(t) else t[-1]

            # Orange line for algorithm-suggested start
            ax.axvline(
                start_t,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Suggested Start ≈ {start_t:.3f}s",
            )

            # Green line for algorithm-suggested end
            ax.axvline(
                end_t,
                color="green",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Suggested End ≈ {end_t:.3f}s",
            )

        # Legend (combine left + right labels)
        lines_left, labels_left = ax.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()
        ax.legend(
            lines_left + lines_right, labels_left + labels_right, loc="upper left"
        )

        ax.set_title("Total Energy & dE/dt (Dual Axis)")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)

        return fig

    def visualize_3d_trajectory(self) -> None:
        """Step 2: Compare raw data vs filtered/estimated trajectory in 3D."""
        if self.raw_data is None or self.estimated_state is None:
            raise RuntimeError(
                "Must call load_and_preprocess() and estimate_state() before 3D visualization"
            )

        print("[preProcess] Rendering 3D trajectory comparison...")

        # Setup style
        setup_debug_style()

        # Extract raw data (scatter)
        t_raw = self.raw_data["t"]
        x_raw = self.raw_data["x"]
        y_raw = self.raw_data["y"]
        z_raw = self.raw_data["z"]

        # Extract estimated trajectory (smoothed line)
        t_est = self.estimated_state.t_std
        pos_est = self.estimated_state.pos

        # Create the 3D comparison plot
        fig = plot_3d_trajectory_compare(
            t=t_raw,
            x_raw=x_raw,
            y_raw=y_raw,
            z_raw=z_raw,
            x_smooth=pos_est[:, 0],  # ← 滤波后 X
            y_smooth=pos_est[:, 1],  # ← 滤波后 Y
            z_smooth=pos_est[:, 2],  # ← 滤波后 Z
            labels=("Raw Data (Scatter)", "Filtered Trajectory", ""),
        )

        # Show the plot
        plt.show(block=False)
        plt.pause

        return fig

    def get_user_trim_range(
        self, start_hint: float, end_hint: float
    ) -> Tuple[float, float]:
        """Step 3: Interactive CLI for selecting trim range.

        Returns:
            (start_time, end_time) based on user input or algorithm suggestion.
        """
        if self.estimated_state is None:
            raise RuntimeError(
                "Must call estimate_state() before get_user_trim_range()"
            )

        t_last = float(self.estimated_state.t_std[-1])
        prompt = (
            "\n[preProcess] 请输入裁剪范围（开始 结束），空格分隔。"
            "\n  - 直接回车：使用建议范围"
            f"\n  - 建议开始: {start_hint:.3f}s"
            f"\n  - 建议结束: {end_hint:.3f}s"
            f"\n  - 最后时间点: {t_last:.3f}s"
            "\n> "
        )

        user_input = input(prompt).strip()
        if user_input == "":
            return float(start_hint), float(end_hint)

        parts = user_input.split()
        if len(parts) != 2:
            print("[preProcess] 输入格式无效，使用建议范围。")
            return float(start_hint), float(end_hint)

        try:
            start_val = float(parts[0])
            end_val = float(parts[1])
        except ValueError:
            print("[preProcess] 输入无法解析为数字，使用建议范围。")
            return float(start_hint), float(end_hint)

        return start_val, end_val

    def save_intermediate(
        self, output_dir: str | pathlib.Path = "data/interm"
    ) -> pathlib.Path:
        """Save intermediate state data (with full [p, v, a] state vector) to interm folder.

        Args:
            output_dir: Output directory path

        Returns:
            Path to saved file
        """
        if self.estimated_state is None:
            raise RuntimeError("Must call estimate_state() before save_intermediate()")

        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        output_file = output_path / f"{self.csv_path.stem}SMR.csv"

        # Prepare data for saving
        t = self.estimated_state.t_std
        pos = self.estimated_state.pos
        vel = self.estimated_state.vel
        acc = self.estimated_state.acc

        data_dict = {
            "t": t,
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "vx": vel[:, 0],
            "vy": vel[:, 1],
            "vz": vel[:, 2],
            "ax": acc[:, 0],
            "ay": acc[:, 1],
            "az": acc[:, 2],
        }

        # Add velocity magnitude (speed)
        speed = np.linalg.norm(vel, axis=1)
        data_dict["speed"] = speed

        # Save
        save_track(output_file, data_dict)

        print(f"[preProcess] Saved intermediate data to: {output_file}")
        return output_file

    def generate_verdict_report(self) -> Dict[str, Any]:
        """Generate comprehensive verdict report for terminal output."""
        if self.verdict_report is None:
            raise RuntimeError(
                "Must call evaluate_physics() before generate_verdict_report()"
            )

        report = {
            "valid_duration": self.verdict_report.valid_duration,
            "termination_reason": self.verdict_report.reason_for_termination,
            "initial_energy": self.verdict_report.initial_energy,
            "start_idx": self.verdict_report.start_idx,
            "end_idx": self.verdict_report.end_idx,
        }

        # Add trajectory statistics
        if self.estimated_state is not None:
            t = self.estimated_state.t_std
            pos = self.estimated_state.pos
            vel = self.estimated_state.vel

            # Calculate statistics
            total_duration = float(t[-1] - t[0])
            avg_speed = float(np.mean(np.linalg.norm(vel, axis=1)))
            max_speed = float(np.max(np.linalg.norm(vel, axis=1)))
            distance_traveled = float(
                np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
            )

            report["total_duration"] = total_duration
            report["avg_speed"] = avg_speed
            report["max_speed"] = max_speed
            report["distance_traveled"] = distance_traveled

        return report


def main() -> None:
    """Main entry point for the preprocessing preProcess."""
    if len(sys.argv) < 2:
        print("Usage: python preProcess.py <csv_path>")
        print("  csv_path: Path to raw CSV file with [t, x, y, z] columns")
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        # Initialize preProcess
        controller = preProcessController(csv_path)

        # Step 1: Load and preprocess
        raw_data = controller.load_and_preprocess()

        # Step 2: Estimate state using L2 estimators
        estimated_state = controller.estimate_state()

        # Step 3: Evaluate physics using L3
        verdict_report = controller.evaluate_physics()

        # Step 3.5: Visualize energy (dual Y-axis), 3D trajectory comparison
        fig1 = controller.visualize_energy()
        fig2 = controller.visualize_3d_trajectory()
        print("\n[preProcess] Showing plots (close windows to continue)...")
        plt.show()

        # Step 4: Mannual Intervention and Save intermediate data to interm folder
        interm_path = controller.save_intermediate()
        start_hint = controller.estimated_state.t_std[
            controller.verdict_report.start_idx
        ]
        end_idx = min(
            controller.verdict_report.end_idx,
            len(controller.estimated_state.t_std) - 1,
        )
        end_hint = controller.estimated_state.t_std[end_idx]
        start_select, end_select = controller.get_user_trim_range(start_hint, end_hint)
        t_std = controller.estimated_state.t_std
        pos = controller.estimated_state.pos
        vel = controller.estimated_state.vel
        acc = controller.estimated_state.acc

        # 找索引
        start_idx = int(np.argmin(np.abs(t_std - start_select)))
        end_idx = int(np.argmin(np.abs(t_std - end_select)))

        # 截切
        t_slice = t_std[start_idx : end_idx + 1]
        pos_slice = pos[start_idx : end_idx + 1]
        vel_slice = vel[start_idx : end_idx + 1]
        acc_slice = acc[start_idx : end_idx + 1]

        # 时间归零：t[0] = 0
        t0 = t_slice[0]
        t_slice = t_slice - t0

        # 速度大小
        speed_slic = np.linalg.norm(vel_slice, axis=1)

        # 存 data/final/trackXopt.csv
        output_file = pathlib.Path("data/final") / f"{controller.csv_path.stem}opt.csv"
        save_track(
            output_file,
            {
                "t": t_slice,
                "x": pos_slice[:, 0],
                "y": pos_slice[:, 1],
                "z": pos_slice[:, 2],
                "vx": vel_slice[:, 0],
                "vy": vel_slice[:, 1],
                "vz": vel_slice[:, 2],
                "speed": speed_slic,
                "ax": acc_slice[:, 0],
                "ay": acc_slice[:, 1],
                "az": acc_slice[:, 2],
            },
        )
        print(f"\n[STEP 4] trim Finished: {output_file}")

        # Print verdict report
        report = controller.generate_verdict_report()
        print("\n" + "=" * 60)
        print("TRAJECTORY VERDICT REPORT")
        print("=" * 60)
        print(f"Valid flight duration: {report['valid_duration']:.3f} s")
        print(f"Termination reason: {report['termination_reason']}")
        print(f"Initial energy: {report['initial_energy']:.6f} J/kg")
        print(f"Start index: {report['start_idx']}")
        print(f"End index: {report['end_idx']}")
        if "total_duration" in report:
            print(f"Total duration: {report['total_duration']:.3f} s")
            print(f"Average speed: {report['avg_speed']:.3f} m/s")
            print(f"Max speed: {report['max_speed']:.3f} m/s")
            print(f"Distance traveled: {report['distance_traveled']:.3f} m")
        print("=" * 60)
        print(f"\nIntermediate data saved to: {interm_path}")
        print("\n[SUCCESS] preProcess completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] preProcess failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
