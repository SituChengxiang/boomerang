# Copilot Instructions for Boomerang Trajectory Analysis

## Project Overview
This is a physics competition project analyzing paper boomerang trajectories. It combines Python for data analysis and Rust for high-performance numerical fitting of 3D motion models.

## Architecture
- **Data (data/)**: CSV files with columns `[t, x, y, z]` exported from Tracker.
	- `data/raw/`: raw tracker exports
	- `data/interm/`: intermediate processed files (e.g. `*SMR.csv`, `velocity.csv`)
	- `data/final/`: manually accepted/trimmed files used for fitting(e.g. `*opt.csv`)
- **Python (src/)**: primary analysis + preprocessing code.
	- `src/preprocess/`: preprocessing helpers (smoothing, velocity/derivatives, sanity checks)
	- `src/utils/`: shared utilities (I/O, filters, physics, plotting helpers)
	- `src/visualization/`: visualization entry points
	- `src/fit/`: fitting and model code
- **Command-style scripts (cmd/)**:
	- `cmd/preprocess/trackCheck.py`: CLI preprocessing pipeline

## Current Working Style (Important)
- We do NOT aim for fully automatic “perfect” cleaning.
- Preferred workflow is conservative and inspectable:
	1) run RTS smoothing and export `*SMR.csv`
	2) run velocity calculation + visualization
	3) manually trim suspicious tail / outliers

## Dependencies & Constraints
- Assume the environment have heavy scientific deps preinstalled (e.g. `scipy`).
- Prefer solutions based on `numpy` + `matplotlib` (and `pandas` only if it’s already present/needed).
- If a proposed change requires new dependencies, explain why and keep the code path optional.

## Coding Guidelines for This Repo
- Keep changes minimal and aligned with the existing code style.
- Do not refactor entire modules unless requested; focus on the data-processing pipeline in the repository root.
- When adding/adjusting preprocessing outputs, keep headers consistent with existing CSV conventions (`t,x,y,z,...`).

<parameter name="filePath">/home/kuan/Git/boomerang/.github/copilot-instructions.md