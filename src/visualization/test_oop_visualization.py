#!/usr/bin/env python3
"""
Integration test for the OOP visualization system using real boomerang data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to Python path for absolute imports
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.oopVisualization import (
    CompositeVisualizer,
    ForceAnalysisVisualizer,
    TimeSeriesVisualizer,
    TrackDataWrapper,
    TrajectoryVisualizer,
    create_standard_visualization_suite,
)


def test_with_real_data():
    """Test the OOP visualization system with real boomerang data."""
    try:
        # Load real data
        df_all = pd.read_csv("data/interm/velocity.csv")
        print(f"Loaded {len(df_all)} data points from real boomerang tracks")

        # Get unique tracks
        tracks = df_all.track.unique()
        print(f"Found {len(tracks)} unique tracks: {list(tracks)}")

        # Create track data wrappers
        track_data = {}
        for track in tracks:
            df_track = df_all[df_all.track == track].sort_values(by="t")
            if len(df_track) >= 10:  # Only use tracks with enough data
                track_data[track] = TrackDataWrapper(df_track, str(track))

        print(f"Created {len(track_data)} track data wrappers")

        # Test individual visualizers
        print("\n=== Testing Individual Visualizers ===")

        # 1. Trajectory Visualizer
        print("1. Testing TrajectoryVisualizer...")
        traj_vis = TrajectoryVisualizer()
        fig1 = traj_vis.plot_3d_trajectory_compare(track_data)
        fig1.savefig("out/oop_test_trajectory_compare.png", dpi=150)
        print("   ✓ Trajectory comparison plot saved")

        # 2. Time Series Visualizer
        print("2. Testing TimeSeriesVisualizer...")
        time_vis = TimeSeriesVisualizer()
        fig2 = time_vis.plot_velocity_components(track_data)
        fig2.savefig("out/oop_test_velocity_components.png", dpi=150)
        print("   ✓ Velocity components plot saved")

        fig3 = time_vis.plot_energy_components(track_data)
        fig3.savefig("out/oop_test_energy_components.png", dpi=150)
        print("   ✓ Energy components plot saved")

        # 3. Force Analysis Visualizer
        print("3. Testing ForceAnalysisVisualizer...")
        force_vis = ForceAnalysisVisualizer()
        fig4 = force_vis.plot_vertical_aero_vs_horizontal_speed(track_data)
        fig4.savefig("out/oop_test_vertical_aero.png", dpi=150)
        print("   ✓ Vertical aero plot saved")

        fig5 = force_vis.plot_aerodynamic_forces_decomposition(track_data)
        fig5.savefig("out/oop_test_force_decomposition.png", dpi=150)
        print("   ✓ Force decomposition plot saved")

        # Test Composite Visualizer
        print("\n=== Testing Composite Visualizer ===")
        composite = create_standard_visualization_suite()
        composite.generate_all_plots(track_data, "out/oop_composite_")
        print("   ✓ Composite visualization completed")

        print(f"\n All OOP visualization tests completed successfully!")
        print(f"Generated {len(track_data)} track visualizations")
        print("Plots saved to out/ directory")

        return True

    except FileNotFoundError:
        print("❌ Error: data/interm/velocity.csv not found.")
        print("Please run the data preprocessing pipeline first.")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False


if __name__ == "__main__":
    print(" Starting OOP Visualization Integration Test")
    print("=" * 50)

    success = test_with_real_data()

    if success:
        print("\n Integration test PASSED!")
        print("The OOP visualization system is working correctly with real data.")
    else:
        print("\n Integration test FAILED!")
        print("Please check the error messages above.")
