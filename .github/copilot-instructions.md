# Copilot Instructions for Boomerang Trajectory Analysis

## Project Overview
This is a physics competition project analyzing paper boomerang trajectories. It combines Python for data analysis and Rust for high-performance numerical fitting of 3D motion models.

## Architecture
- **Python (archive/)**: Interactive analysis, curve fitting, plotting. Use Jupyter notebooks for exploratory work.
- **Rust (archive/src/)**: Performance-critical least squares fitting using ndarray and polars for data handling.
- **Data (data/)**: CSV files with columns [t, x, y, z] from manual tracking using Tracker software.

## Key Workflows
- **Data Processing**: Load CSV with `np.loadtxt('ps.csv', delimiter=',', skiprows=1, dtype=[('t', float), ('x', float), ('y', float), ('z', float)])`
- **Fitting**: Use scipy.optimize.curve_fit or custom least squares for damped oscillatory models in x/y and gravitational in z.
- **Visualization**: 3D plots with matplotlib, set Chinese fonts: `plt.rcParams['font.sans-serif'] = ['Source Han Sans CN']`
- **Build Rust**: `cd archive && cargo build --release` for optimized fitting.

## Coding Patterns
- **Physics Models**: x/y coordinates fit damped harmonic motion: `A*(1-cos(ωt))*exp(-αt) + B*cos(ωt)*exp(-αt) + C*exp(-αt) + linear terms`
- **Z Coordinate**: Gravity model: `v0*t - 0.5*g*t² - β*v0*t² + constant`
- **Error Calculation**: RMS error between fitted and measured data points.
- **Parameter Estimation**: Solve overdetermined systems using SVD (Rust) or numpy.linalg.lstsq (Python).

## Constants and Dependencies
- Fixed params: wingspan a=0.15m, mass m=0.002183kg, air density ρ=1.225, gravity g=9.793
- Inertia: `I = (5/24)*m*a²`
- Use scipy.signal.savgol_filter for smoothing noisy data.

## Examples
- Fit x-coordinate: `coeffs = np.linalg.lstsq(basis_matrix, x_data)[0]`
- Plot 3D trajectory: `ax.plot(x, y, z, label='trajectory')`
- Rust data loading: `CsvReader::from_path("track1.csv")?.infer_schema(None).has_header(true).finish()?`</content>
<parameter name="filePath">/home/kuan/Git/boomerang/.github/copilot-instructions.md