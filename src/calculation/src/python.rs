//! Python bindings for the Boomerang Calculation library.
//!
//! This module provides PyO3-based Python bindings for the high-performance
//! Rust library. It allows Python code to use the optimized Rust implementations
//! for trajectory simulation and parameter fitting.
//!
//! NOTE: This module requires the `python-bindings` feature to be enabled.
//! Compile with: `cargo build --features python-bindings --release`
//!
//! Usage in Python:
//! ```python
//! import boomerang_calc as bc
//! constants = bc.Constants.experimental()
//! params = bc.SpeParams.default_params()
//! [...]
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::{exceptions, PyErr};

use crate::aerodynamics::{AerodynamicModel, BapModel, SpeModel, batch_simulate_spe, batch_simulate_bap};
use crate::constants::{Constants, model_params};
use crate::constants::track_meta::build_track_meta;
use crate::loss::{MseLoss, WeightedMseLoss, LossFunction, CompositeLoss, MultiTrackLoss, TrackAggregation};
use crate::metrics::{TrajectoryMetrics, trajectory_mse};
use crate::ode::RK4Integrator;
use crate::vector::{Vec3, Quaternion};

// ============================================================================
// Python Type Wrappers
// ============================================================================

/// 3D vector with useful operations for trajectory calculations.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVec3 {
    vec: Vec3,
}

#[pymethods]
impl PyVec3 {
    #[new]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { vec: Vec3::new(x, y, z) }
    }

    #[getter]
    pub fn x(&self) -> f64 { self.vec.x }

    #[getter]
    pub fn y(&self) -> f64 { self.vec.y }

    #[getter]
    pub fn z(&self) -> f64 { self.vec.z }

    #[getter]
    pub fn magnitude(&self) -> f64 { self.vec.magnitude() }

    #[getter]
    pub fn magnitude_sq(&self) -> f64 { self.vec.magnitude_sq() }

    pub fn normalized(&self) -> Option<Self> {
        self.vec.normalized().map(|v| Self { vec: v })
    }

    pub fn dot(&self, other: &PyVec3) -> f64 {
        self.vec.dot(&other.vec)
    }

    pub fn cross(&self, other: &PyVec3) -> PyVec3 {
        PyVec3 { vec: self.vec.cross(&other.vec) }
    }

    pub fn add(&self, other: &PyVec3) -> PyVec3 {
        PyVec3 { vec: self.vec.add(&other.vec) }
    }

    pub fn sub(&self, other: &PyVec3) -> PyVec3 {
        PyVec3 { vec: self.vec.sub(&other.vec) }
    }

    pub fn mul(&self, scalar: f64) -> PyVec3 {
        PyVec3 { vec: self.vec.mul(scalar) }
    }

    pub fn rotate_around(&self, axis: &PyVec3, theta: f64) -> PyResult<PyObject> {
        match self.vec.rotate_around(&axis.vec, theta) {
            Some(v) => Ok(PyVec3 { vec: v }.into()),
            None => Err(PyErr::new::<exceptions::ValueError, _>(
                "Cannot rotate around zero-vector axis"
            )),
        }
    }

    pub fn to_array(&self) -> [f64; 3] {
        self.vec.to_array()
    }

    pub fn __repr__(&self) -> String {
        format!("Vec3(x={:.6}, y={:.6}, z={:.6})", self.vec.x, self.vec.y, self.vec.z)
    }

    pub fn __str__(&self) -> String {
        format!("({:.6}, {:.6}, {:.6})", self.vec.x, self.vec.y, self.vec.z)
    }

    // Python magic methods
    pub fn __add__(&self, other: &PyVec3) -> PyVec3 {
        self.add(other)
    }

    pub fn __sub__(&self, other: &PyVec3) -> PyVec3 {
        self.sub(other)
    }

    pub fn __mul__(&self, scalar: f64) -> PyVec3 {
        self.mul(scalar)
    }

    pub fn __truediv__(&self, scalar: f64) -> PyResult<PyVec3> {
        if scalar.abs() < 1e-12 {
            return Err(PyErr::new::<exceptions::ZeroDivisionError, _>(
                "Division by zero",
            ));
        }
        Ok(PyVec3 { vec: self.vec.div(scalar).unwrap() })
    }

    pub fn __eq__(&self, other: PyRef<Self>) -> bool {
        self.vec.x == other.vec.x && self.vec.y == other.vec.y && self.vec.z == other.vec.z
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyQuaternion {
    quat: Quaternion,
}

#[pymethods]
impl PyQuaternion {
    #[new]
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { quat: Quaternion::new(w, x, y, z) }
    }

    #[staticmethod]
    pub fn identity() -> Self {
        Self { quat: Quaternion::identity() }
    }

    #[staticmethod]
    pub fn from_axis_angle(axis: &PyVec3, theta: f64) -> PyResult<Self> {
        match Quaternion::from_axis_angle(&axis.vec, theta) {
            Ok(q) => Ok(Self { quat: q }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    #[staticmethod]
    pub fn from_rpy(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self { quat: Quaternion::from_rpy(roll, pitch, yaw) }
    }

    #[getter]
    pub fn w(&self) -> f64 { self.quat.w }

    #[getter]
    pub fn x(&self) -> f64 { self.quat.x }

    #[getter]
    pub fn y(&self) -> f64 { self.quat.y }

    #[getter]
    pub fn z(&self) -> f64 { self.quat.z }

    #[getter]
    pub fn magnitude(&self) -> f64 { self.quat.magnitude() }

    #[getter]
    pub fn magnitude_sq(&self) -> f64 { self.quat.magnitude_sq() }

    pub fn normalized(&self) -> PyResult<Self> {
        match self.quat.normalized() {
            Some(q) => Ok(Self { quat: q }),
            None => Err(PyErr::new::<exceptions::ValueError, _>(
                "Quaternion has zero magnitude"
            )),
        }
    }

    pub fn conjugate(&self) -> PyResult<Self> {
        match self.quat.conjugate() {
            Some(q) => Ok(Self { quat: q }),
            None => Err(PyErr::new::<exceptions::ValueError, _>(
                "Cannot conjugate zero-quaternion"
            )),
        }
    }

    pub fn inverse(&self) -> PyResult<Self> {
        match self.quat.inverse() {
            Some(q) => Ok(Self { quat: q }),
            None => Err(PyErr::new::<exceptions::ValueError, _>(
                "Cannot invert zero-quaternion"
            )),
        }
    }

    pub fn rotate(&self, vec: &PyVec3) -> PyResult<PyVec3> {
        match self.quat.rotate(&vec.vec) {
            Ok(v) => Ok(PyVec3 { vec: v }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    pub fn mul(&self, other: &PyQuaternion) -> PyQuaternion {
        PyQuaternion { quat: self.quat.mul(&other.quat) }
    }

    pub fn slerp(&self, other: &PyQuaternion, t: f64) -> PyResult<Self> {
        match self.quat.slerp(&other.quat, t) {
            Ok(q) => Ok(Self { quat: q }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    pub fn to_array(&self) -> [f64; 4] {
        [self.quat.w, self.quat.x, self.quat.y, self.quat.z]
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Quaternion(w={:.6}, x={:.6}, y={:.6}, z={:.6})",
            self.quat.w, self.quat.x, self.quat.y, self.quat.z
        )
    }

    pub fn __mul__(&self, other: &PyQuaternion) -> PyQuaternion {
        self.mul(other)
    }
}

// ============================================================================
// Physical Constants
// ============================================================================

/// Physical constants and parameters for the boomerang system.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyConstants {
    constants: Constants,
}

#[pymethods]
impl PyConstants {
    /// Standard experimental values (9.793 m/s², 1.225 kg/m³, 0.002183 kg).
    #[staticmethod]
    pub fn experimental() -> PyConstants {
        PyConstants { constants: Constants::experimental() }
    }

    /// Default values (new()).
    #[staticmethod]
    pub fn new() -> PyConstants {
        PyConstants { constants: Constants::new() }
    }

    /// Field simulation conditions (different density/temperature).
    #[staticmethod]
    pub fn field_simulation() -> PyConstants {
        PyConstants { constants: Constants::field_simulation() }
    }

    /// Stratospheric conditions (low density).
    #[staticmethod]
    pub fn stratospheric() -> PyConstants {
        PyConstants { constants: Constants::stratospheric() }
    }

    #[new]
    #[pyo3(signature = (g=9.793, rho=1.225, mass=0.002183, area=0.0084, span=0.15, width=0.028))]
    pub fn custom(g: f64, rho: f64, mass: f64, area: f64, span: f64, width: f64) -> PyConstants {
        PyConstants {
            constants: Constants::custom(g, rho, mass, Some(area), Some(span), Some(width)),
        }
    }

    #[getter]
    pub fn g(&self) -> f64 { self.constants.g }

    #[getter]
    pub fn rho_air(&self) -> f64 { self.constants.rho_air }

    #[getter]
    pub fn mass(&self) -> f64 { self.constants.mass }

    #[getter]
    pub fn area(&self) -> f64 { self.constants.area }

    #[getter]
    pub fn q_factor(&self) -> f64 { self.constants.q_factor() }

    #[getter]
    pub fn i_moment(&self) -> f64 { self.constants.i_moment }

    pub fn summary(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            for (key, value) in self.constants.summary().iter() {
                dict.set_item(key, value)?;
            }
            Ok(dict.into())
        })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "Constants(g={:.3}, ρ={:.4}, mass={:.6})",
            self.constants.g, self.constants.rho_air, self.constants.mass
        )
    }
}

// ============================================================================
// Model Parameters
// ============================================================================

/// SPE (Simplified Physics Equation) model parameters.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PySpeParams {
    params: model_params::SpeParams,
}

#[pymethods]
impl PySpeParams {
    #[staticmethod]
    pub fn default_params() -> PySpeParams {
        PySpeParams { params: model_params::SpeParams::default_params() }
    }

    #[new]
    pub fn new(cl_trans: f64) -> PySpeParams {
        PySpeParams {
            params: model_params::SpeParams { cl_trans, ..model_params::SpeParams::default_params() }
        }
    }

    #[getter]
    pub fn cl_trans(&self) -> f64 { self.params.cl_trans }

    #[getter]
    pub fn cl_rotor(&self) -> f64 { self.params.cl_rotor }

    #[getter]
    pub fn cd(&self) -> f64 { self.params.cd }

    #[getter]
    pub fn d_factor(&self) -> f64 { self.params.d_factor }

    #[getter]
    pub fn coupling_eff(&self) -> f64 { self.params.coupling_eff }

    #[getter]
    pub fn dive_steering(&self) -> f64 { self.params.dive_steering }

    #[getter]
    pub fn bank_factor(&self) -> f64 { self.params.bank_factor }

    #[getter]
    pub fn omega_decay(&self) -> f64 { self.params.omega_decay }

    #[cl_trans.setter]
    pub fn set_cl_trans(&mut self, value: f64) {
        self.params.cl_trans = value;
    }

    #[cd.setter]
    pub fn set_cd(&mut self, value: f64) {
        self.params.cd = value;
    }

    #[d_factor.setter]
    pub fn set_d_factor(&mut self, value: f64) {
        self.params.d_factor = value;
    }

    #[coupling_eff.setter]
    pub fn set_coupling_eff(&mut self, value: f64) {
        self.params.coupling_eff = value;
    }

    #[dive_steering.setter]
    pub fn set_dive_steering(&mut self, value: f64) {
        self.params.dive_steering = value;
    }

    #[bank_factor.setter]
    pub fn set_bank_factor(&mut self, value: f64) {
        self.params.bank_factor = value;
    }

    pub fn to_array(&self) -> [f64; 6] {
        self.params.to_slice()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SpeParams(CL_trans={:.3}, C_D={:.3}, D={:.3}, Coupling={:.3}, DiveSteer={:.3}, Bank= {:.3})",
            self.params.cl_trans, self.params.cd, self.params.d_factor,
            self.params.coupling_eff, self.params.dive_steering, self.params.bank_factor
        )
    }
}

/// BAP (Bank Angle Proxy) model parameters.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyBapParams {
    params: model_params::BapParams,
}

#[pymethods]
impl PyBapParams {
    #[staticmethod]
    pub fn default_params() -> PyBapParams {
        PyBapParams { params: model_params::BapParams::default_params() }
    }

    #[new]
    pub fn new(cl: f64, cd: f64) -> PyBapParams {
        PyBapParams {
            params: model_params::BapParams {
                cl, cd,
                ..model_params::BapParams::default_params()
            },
        }
    }

    #[getter]
    pub fn cl(&self) -> f64 { self.params.cl }

    #[getter]
    pub fn cd(&self) -> f64 { self.params.cd }

    #[getter]
    pub fn phi_base(&self) -> f64 { self.params.phi_base }

    #[getter]
    pub fn k_bank(&self) -> f64 { self.params.k_bank }

    #[getter]
    pub fn v0_scalar(&self) -> f64 { self.params.v0_scalar }

    #[getter]
    pub fn omega_scale(&self) -> f64 { self.params.omega_scale }

    #[cl.setter]
    pub fn set_cl(&mut self, value: f64) {
        self.params.cl = value;
    }

    #[cd.setter]
    pub fn set_cd(&mut self, value: f64) {
        self.params.cd = value;
    }

    #[phi_base.setter]
    pub fn set_phi_base(&mut self, value: f64) {
        self.params.phi_base = value;
    }

    #[k_bank.setter]
    pub fn set_k_bank(&mut self, value: f64) {
        self.params.k_bank = value;
    }

    #[v0_scalar.setter]
    pub fn set_v0_scalar(&mut self, value: f64) {
        self.params.v0_scalar = value;
    }

    #[omega_scale.setter]
    pub fn set_omega_scale(&mut self, value: f64) {
        self.params.omega_scale = value;
    }

    pub fn __repr__(&self) -> String {
        format!(
            "BapParams(CL={:.3}, C_D={:.3}, Φ₀={:.3}, kbank={:.3}, V₀={:.3}, Ω={:.3})",
            self.params.cl, self.params.cd, self.params.phi_base,
            self.params.k_bank, self.params.v0_scalar, self.params.omega_scale
        )
    }
}

// ============================================================================
// Aerodynamic Models
// ============================================================================

/// SPE (Simplified Physics Equation) model.
#[pyclass]
pub struct PySpeModel {
    model: SpeModel,
}

impl PySpeModel {
    fn validate_state(state: &[f64]) -> PyResult<[f64; 6]> {
        if state.len() != 6 {
            return Err(PyErr::new::<exceptions::TypeError, _>(
                "State must be a list/array of 6 elements [x, y, z, vx, vy, vz]"
            ));
        }
        Ok([state[0], state[1], state[2], state[3], state[4], state[5]])
    }

    fn validate_params(&self, params: &PySpeParams) -> PyResult<()> {
        if params.params.cd < 0.0 && params.params.d_factor < 0.0 {
            return Err(PyErr::new::<exceptions::ValueError, _>(
                "Negative drag and D factor are unphysical"
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PySpeModel {
    #[new]
    pub fn new() -> Self {
        Self { model: SpeModel::default() }
    }

    /// Simulate a single trajectory.
    ///
    /// Args:
    ///     t_eval: Array of time points
    ///     state0: Initial state [x, y, z, vx, vy, vz]
    ///     params: SpeParams
    ///     constants: Constants
    ///
    /// Returns:
    ///     Array of states at each time point
    #[pyo3(signature = (t_eval, state0, params, constants))]
    pub fn simulate(
        &self,
        t_eval: Vec<f64>,
        state0: Vec<f64>,
        params: &PySpeParams,
        constants: &PyConstants,
    ) -> PyResult<Vec<[f64; 6]>> {
        self.validate_params(params)?;

        let state0 = Self::validate_state(&state0)?;
        let trajectory = self.model.simulate(&t_eval, &state0, &params.params, &constants.constants);

        Ok(trajectory)
    }

    /// Simulate multiple trajectories in batch mode.
    ///
    /// Args:
    ///     t_eval: Array of time points
    ///     states0: List of initial states
    ///     params: SpeParams
    ///     constants: Constants
    ///
    /// Returns:
    ///     List of trajectory arrays
    #[pyo3(signature = (t_eval, states0, params, constants))]
    pub fn simulate_batch(
        &self,
        t_eval: Vec<f64>,
        states0: Vec<Vec<f64>>,
        params: &PySpeParams,
        constants: &PyConstants,
    ) -> PyResult<Vec<Vec<[f64; 6]>>> {
        self.validate_params(params)?;

        let states0_clean: Vec<[f64; 6]> = states0
            .iter()
            .map(|s| Self::validate_state(s))
            .collect::<Result<Vec<_>, _>>()?;

        let results = states0_clean
            .iter()
            .map(|state| self.model.simulate(&t_eval, state, &params.params, &constants.constants))
            .collect();

        Ok(results)
    }

    /// Compute RHS (right-hand side) of ODE for a single state.
    pub fn rhs(
        &self,
        t: f64,
        state: Vec<f64>,
        params: &PySpeParams,
        constants: &PyConstants,
    ) -> PyResult<[f64; 6]> {
        let state = Self::validate_state(&state)?;
        let result = self.model.rhs_internal(t, &state, &params.params, &constants.constants);
        Ok(result)
    }

    pub fn __repr__(&self) -> String {
        "SpeModel()".to_string()
    }
}

impl Default for PySpeModel {
    fn default() -> Self {
        Self::new()
    }
}

/// BAP (Bank Angle Proxy) model.
#[pyclass]
pub struct PyBapModel {
    model: BapModel,
}

impl PyBapModel {
    fn validate_state(state: &[f64]) -> PyResult<[f64; 6]> {
        if state.len() != 6 {
            return Err(PyErr::new::<exceptions::TypeError, _>(
                "State must be a list/array of 6 elements [x, y, z, vx, vy, vz]"
            ));
        }
        Ok([state[0], state[1], state[2], state[3], state[4], state[5]])
    }

    fn validate_params(&self, params: &PyBapParams) -> PyResult<()> {
        if params.params.cd < 0.0 {
            return Err(PyErr::new::<exceptions::ValueError, _>(
                "Negative drag coefficient is unphysical"
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyBapModel {
    #[new]
    pub fn new() -> Self {
        Self { model: BapModel::default() }
    }

    /// Simulate a single trajectory.
    #[pyo3(signature = (t_eval, state0, params, constants))]
    pub fn simulate(
        &self,
        t_eval: Vec<f64>,
        state0: Vec<f64>,
        params: &PyBapParams,
        constants: &PyConstants,
    ) -> PyResult<Vec<[f64; 6]>> {
        self.validate_params(params)?;

        let state0 = Self::validate_state(&state0)?;
        let trajectory = self.model.simulate(&t_eval, &state0, &params.params, &constants.constants);

        Ok(trajectory)
    }

    /// Simulate multiple trajectories in batch mode.
    #[pyo3(signature = (t_eval, states0, params, constants))]
    pub fn simulate_batch(
        &self,
        t_eval: Vec<f64>,
        states0: Vec<Vec<f64>>,
        params: &PyBapParams,
        constants: &PyConstants,
    ) -> PyResult<Vec<Vec<[f64; 6]>>> {
        self.validate_params(params)?;

        let states0_clean: Vec<[f64; 6]> = states0
            .iter()
            .map(|s| Self::validate_state(s))
            .collect::<Result<Vec<_>, _>>()?;

        let results = states0_clean
            .iter()
            .map(|state| self.model.simulate(&t_eval, state, &params.params, &constants.constants))
            .collect();

        Ok(results)
    }

    /// Compute RHS (right-hand side) of ODE for a single state.
    pub fn rhs(
        &self,
        t: f64,
        state: Vec<f64>,
        params: &PyBapParams,
        constants: &PyConstants,
    ) -> PyResult<[f64; 6]> {
        let state = Self::validate_state(&state)?;
        let result = self.model.rhs_internal(t, &state, &params.params, &constants.constants);
        Ok(result)
    }

    pub fn __repr__(&self) -> String {
        "BapModel()".to_string()
    }
}

impl Default for PyBapModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Trajectory Metrics
// ============================================================================

/// Trajectory metrics and statistics.
#[pyclass]
pub struct PyTrajectoryMetrics {
    metrics: TrajectoryMetrics,
}

#[pymethods]
impl PyTrajectoryMetrics {
    /// Compute metrics from raw trajectory data.
    #[staticmethod]
    #[pyo3(signature = (t, x, y, z, constants))]
    pub fn from_trajectory(
        t: &[f64],
        x: &[f64],
        y: &[f64],
        z: &[f64],
        constants: &PyConstants,
    ) -> PyResult<Self> {
        match TrajectoryMetrics::from_trajectory(t, x, y, z, &constants.constants) {
            Ok(metrics) => Ok(Self { metrics }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    /// Simulate and compute metrics (SPE model).
    #[staticmethod]
    #[pyo3(signature = (t_eval, state0, params, constants))]
    pub fn simulate_spe(
        t_eval: &[f64],
        state0: &[f64; 6],
        params: &PySpeParams,
        constants: &PyConstants,
    ) -> PyResult<Self> {
        match TrajectoryMetrics::simulate_spe(t_eval, state0, &params.params, &constants.constants) {
            Ok(metrics) => Ok(Self { metrics }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    /// Simulate and compute metrics (BAP model).
    #[staticmethod]
    #[pyo3(signature = (t_eval, state0, params, constants))]
    pub fn simulate_bap(
        t_eval: &[f64],
        state0: &[f64; 6],
        params: &PyBapParams,
        constants: &PyConstants,
    ) -> PyResult<Self> {
        match TrajectoryMetrics::simulate_bap(t_eval, state0, &params.params, &constants.constants) {
            Ok(metrics) => Ok(Self { metrics }),
            Err(e) => Err(PyErr::new::<exceptions::ValueError, _>(e)),
        }
    }

    // Getters for metrics data

    #[getter]
    pub fn t(&self) -> Vec<f64> { self.metrics.t.clone() }

    #[getter]
    pub fn x(&self) -> Vec<f64> { self.metrics.x.clone() }

    #[getter]
    pub fn y(&self) -> Vec<f64> { self.metrics.y.clone() }

    #[getter]
    pub fn z(&self) -> Vec<f64> { self.metrics.z.clone() }

    #[getter]
    pub fn vx(&self) -> Vec<f64> { self.metrics.vx.clone() }

    #[getter]
    pub fn vy(&self) -> Vec<f64> { self.metrics.vy.clone() }

    #[getter]
    pub fn vz(&self) -> Vec<f64> { self.metrics.vz.clone() }

    #[getter]
    pub fn ax(&self) -> Vec<f64> { self.metrics.ax.clone() }

    #[getter]
    pub fn ay(&self) -> Vec<f64> { self.metrics.ay.clone() }

    #[getter]
    pub fn az(&self) -> Vec<f64> { self.metrics.az.clone() }

    #[getter]
    pub fn kinetic_energy(&self) -> Vec<f64> { self.metrics.kinetic_energy.clone() }

    #[getter]
    pub fn potential_energy(&self) -> Vec<f64> { self.metrics.potential_energy.clone() }

    #[getter]
    pub fn total_energy(&self) -> Vec<f64> { self.metrics.total_energy.clone() }

    #[getter]
    pub fn dE_dt(&self) -> Vec<f64> { self.metrics.dE_dt.clone() }

    #[getter]
    pub fn speed(&self) -> Vec<f64> { self.metrics.speed.clone() }

    #[getter]
    pub fn speed_xy(&self) -> Vec<f64> { self.metrics.speed_xy.clone() }

    // Derived metrics

    #[getter]
    pub fn flight_time(&self) -> f64 {
        self.metrics.flight_time()
    }

    #[getter]
    pub fn horizontal_distance(&self) -> f64 {
        self.metrics.horizontal_distance()
    }

    #[getter]
    pub fn vertical_displacement(&self) -> f64 {
        self.metrics.vertical_displacement()
    }

    #[getter]
    pub fn mean_speed(&self) -> f64 {
        self.metrics.mean_speed()
    }

    #[getter]
    pub fn mean_speed_xy(&self) -> f64 {
        self.metrics.mean_speed_xy()
    }

    #[getter]
    pub fn mean_energy_dissipation(&self) -> f64 {
        self.metrics.mean_energy_dissipation()
    }

    pub fn summary(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("n_points", self.metrics.t.len())?;
            dict.set_item("flight_time", self.flight_time())?;
            dict.set_item("horizontal_distance", self.horizontal_distance())?;
            dict.set_item("vertical_displacement", self.vertical_displacement())?;
            dict.set_item("mean_speed", self.mean_speed())?;
            dict.set_item("mean_speed_xy", self.mean_speed_xy())?;
            dict.set_item("mean_energy_dissipation", self.mean_energy_dissipation())?;
            Ok(dict.into())
        })
    }

    /// Check if trajectory is physically valid.
    pub fn is_valid(&self, constants: &PyConstants) -> bool {
        self.metrics.is_valid(&constants.constants)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TrajectoryMetrics(n_points={}, flight_time={:.3f}s)",
            self.metrics.t.len(), self.flight_time()
        )
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

/// Mean Squared Error loss function.
#[pyclass]
pub struct PyMseLoss {
    loss: MseLoss,
}

#[pymethods]
impl PyMseLoss {
    #[new]
    pub fn new() -> Self {
        Self { loss: MseLoss }
    }

    pub fn compute_loss(&self, simulated: &PyTrajectoryMetrics, reference: &PyTrajectoryMetrics) -> f64 {
        self.loss.compute_loss(&simulated.metrics, &reference.metrics)
    }

    pub fn __repr__(&self) -> String {
        "MseLoss()".to_string()
    }
}

impl Default for PyMseLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Weighted MSE loss function.
#[pyclass]
pub struct PyWeightedMseLoss {
    loss: WeightedMseLoss,
}

#[pymethods]
impl PyWeightedMseLoss {
    #[new]
    pub fn default() -> Self {
        Self { loss: WeightedMseLoss::default() }
    }

    #[pyo3(signature = (
        position_weight=1.0,
        velocity_weight=0.1,
        acceleration_weight=0.01,
        energy_weight=0.05,
        start_weight=1.5,
        end_weight=1.2,
        middle_weight=0.8,
    ))]
    pub fn custom(
        position_weight: f64,
        velocity_weight: f64,
        acceleration_weight: f64,
        energy_weight: f64,
        start_weight: f64,
        end_weight: f64,
        middle_weight: f64,
    ) -> Self {
        Self {
            loss: WeightedMseLoss {
                position_weight,
                velocity_weight,
                acceleration_weight,
                energy_weight,
                start_weight,
                end_weight,
                middle_weight,
            },
        }
    }

    pub fn compute_loss(&self, simulated: &PyTrajectoryMetrics, reference: &PyTrajectoryMetrics) -> f64 {
        self.loss.compute_loss(&simulated.metrics, &reference.metrics)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "WeightedMseLoss(pos={}, vel={}, acc={}, energy={}, start={}, end={}, middle={})",
            self.loss.position_weight, self.loss.velocity_weight, self.loss.acceleration_weight,
            self.loss.energy_weight, self.loss.start_weight, self.loss.end_weight, self.loss.middle_weight
        )
    }
}

#[pyclass]
pub struct PyCompositeLoss {
    loss: CompositeLoss,
}

#[pymethods]
impl PyCompositeLoss {
    #[new]
    pub fn new(constants: &PyConstants) -> Self {
        Self { loss: CompositeLoss::new(constants.constants.clone()) }
    }

    pub fn compute_loss(&self, simulated: &PyTrajectoryMetrics, reference: &PyTrajectoryMetrics) -> f64 {
        self.loss.compute_loss(&simulated.metrics, &reference.metrics)
    }

    pub fn __repr__(&self) -> String {
        "CompositeLoss(MSE + Energy + Boundary + Stability)".to_string()
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute MSE between two trajectories.
#[pyfunction]
#[pyo3(name = "trajectory_mse")]
pub fn py_trajectory_mse(simulated: &PyTrajectoryMetrics, reference: &PyTrajectoryMetrics) -> PyResult<f64> {
    trajectory_mse(&simulated.metrics, &reference.metrics)
        .map_err(|e| PyErr::new::<exceptions::ValueError, _>(e))
}

/// Batch simulate SPE trajectories (convenience function).
#[pyfunction]
#[pyo3(name = "batch_simulate_spe")]
pub fn py_batch_simulate_spe(
    t_eval: &[f64],
    states0: Vec<[f64; 6]>,
    params: &PySpeParams,
    constants: &PyConstants,
) -> PyResult<Vec<Vec<[f64; 6]>>> {
    Ok(batch_simulate_spe(t_eval, &states0, &params.params, &constants.constants))
}

/// Batch simulate BAP trajectories (convenience function).
#[pyfunction]
#[pyo3(name = "batch_simulate_bap")]
pub fn py_batch_simulate_bap(
    t_eval: &[f64],
    states0: Vec<[f64; 6]>,
    params: &PyBapParams,
    constants: &PyConstants,
) -> PyResult<Vec<Vec<[f64; 6]>>> {
    Ok(batch_simulate_bap(t_eval, &states0, &params.params, &constants.constants))
}

/// Get track metadata (omega values for each track).
#[pyfunction]
#[pyo3(name = "get_track_metadata")]
pub fn py_get_track_metadata() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        let meta = build_track_meta();
        for (id, info) in &meta {
            let info_dict = PyDict::new(py);
            info_dict.set_item("turns", info.turns)?;
            info_dict.set_item("duration", info.duration)?;
            info_dict.set_item("omega", info.omega)?;
            dict.set_item(format!("{}", id), info_dict)?;
        }
        Ok(dict.into())
    })
}

/// Check Python version compatibility.
#[pyfunction]
#[pyo3(name = "version_info")]
pub fn py_version_info() -> (u32, u32, u32) {
    (
        env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0),
        env!("CARGO_PKG_VERSION_PATCH").parse().unwrap_or(0),
    )
}

// ============================================================================
// Module Definition
// ============================================================================

/// High-performance aerodynamic calculations for boomerang trajectories.
///
/// This module provides optimized Rust implementations of:
/// - SPE (Simplified Physics Equation) and BAP (Bank Angle Proxy) models
/// - RK4 integration for trajectory simulation
/// - Vector operations (Vec3, Quaternion)
/// - Energy/metrics computation
/// - Loss functions for optimization
///
/// Example:
/// -------
/// ```python
/// import boomerang_calc as bc
///
/// constants = bc.Constants.experimental()
/// params = bc.SpeParams.from_array([0.4, 0.5, 0.3, 1.0, 0.5, 1.7])
/// state0 = [0.0, 0.0, 10.0, 5.0, 8.0, -1.0]
/// t_eval = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
///
/// model = bc.SpeModel()
/// trajectory = model.simulate(t_eval, state0, params, constants)
///
/// metrics = bc.TrajectoryMetrics.from_trajectory(
///     t_eval,
///     [s[0] for s in trajectory],
///     [s[1] for s in trajectory],
///     [s[2] for s in trajectory],
///     constants
/// )
/// print(f"Flight time: {metrics.flight_time}s")
/// ```
#[pymodule]
fn boomerang_calc(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Vectors
    m.add_class::<PyVec3>()?;
    m.add_class::<PyQuaternion>()?;

    // Constants
    m.add_class::<PyConstants>()?;

    // Models
    m.add_class::<PySpeParams>()?;
    m.add_class::<PyBapParams>()?;
    m.add_class::<PySpeModel>()?;
    m.add_class::<PyBapModel>()?;

    // Metrics
    m.add_class::<PyTrajectoryMetrics>()?;

    // Loss functions
    m.add_class::<PyMseLoss>()?;
    m.add_class::<PyWeightedMseLoss>()?;
    m.add_class::<PyCompositeLoss>()?;

    // Utility functions
    m.add_function(wrap_pyfunction!(py_trajectory_mse, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_simulate_spe, m)?)?;
    m.add_function(wrap_pyfunction!(py_batch_simulate_bap, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_track_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(py_version_info, m)?)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "CUPT Team")?;
    m.add("__doc__", "High-performance aerodynamic calculations for boomerang trajectories")?;

    Ok(())
}
```
```

Now let me create a Python wrapper script to use this library:
