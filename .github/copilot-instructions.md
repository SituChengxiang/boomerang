# Copilot Instructions for Boomerang Trajectory Analysis

## Project Overview
This is a physics competition project analyzing paper boomerang trajectories. It combines Python for data analysis and Rust for high-performance numerical fitting of 3D motion models.

## Architecture
- **Python (archive/)**: Interactive analysis, curve fitting, plotting. Use Jupyter notebooks for exploratory work.
- **Rust (archive/src/)**: Performance-critical least squares fitting using ndarray and polars for data handling.
- **Data (data/)**: CSV files with columns [t, x, y, z] from manual tracking using Tracker software.

**currently we are working on data processing and analysis with python in the root directory**.

<parameter name="filePath">/home/kuan/Git/boomerang/.github/copilot-instructions.md