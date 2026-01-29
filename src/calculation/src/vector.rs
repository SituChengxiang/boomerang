! /// High-performance 3D vector operations for aerodynamic calculations.

use std::f64::consts;

/// 3D vector with utility methods for aerodynamic computations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Creates a new vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Zero vector.
    pub const fn zero() -> Self {
        Self { x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Unit vector in Z direction (up).
    pub const fn up() -> Self {
        Self { x: 0.0, y: 0.0, z: 1.0 }
    }

    /// Unit vector in X direction (right).
    pub const fn right() -> Self {
        Self { x: 1.0, y: 0.0, z: 0.0 }
    }

    /// Unit vector in Y direction (forward).
    pub const fn forward() -> Self {
        Self { x: 0.0, y: 1.0, z: 0.0 }
    }

    /// Magnitude (length) of the vector.
    #[inline(always)]
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared magnitude (avoids sqrt, cheaper for comparisons).
    #[inline(always)]
    pub fn magnitude_sq(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalized vector (unit length).
    pub fn normalized(&self) -> Option<Self> {
        let mag = self.magnitude();
        if mag < 1e-12 {
            None
        } else {
            let inv = 1.0 / mag;
            Some(Self::new(self.x * inv, self.y * inv, self.z * inv))
        }
    }

    /// Dot product.
    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product: self × other
    #[inline(always)]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Element-wise addition.
    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Element-wise subtraction.
    #[inline(always)]
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Scalar multiplication.
    #[inline(always)]
    pub fn mul(&self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    /// Scalar division.
    #[inline(always)]
    pub fn div(&self, scalar: f64) -> Result<Self, &'static str> {
        if scalar.abs() < 1e-12 {
            Err("Division by zero")
        } else {
            let inv = 1.0 / scalar;
            Ok(Self::new(self.x * inv, self.y * inv, self.z * inv))
        }
    }

    /// Project onto another vector.
    pub fn project_onto(&self, onto: &Self) -> Option<Self> {
        let onto_mag_sq = onto.magnitude_sq();
        if onto_mag_sq < 1e-12 {
            None
        } else {
            let n = self.dot(onto) / onto_mag_sq;
            Some(onto.mul(n))
        }
    }

    /// Reflect across a plane with normal.
    pub fn reflect(&self, normal: &Self) -> Option<Self> {
        let normal_norm = normal.normalized()?;
        let dot = self.dot(&normal_norm);
        Some(self.sub(&normal_norm.mul(2.0 * dot)))
    }

    /// Angle between two vectors (in radians).
    /// Returns 0.0 if either vector has zero magnitude.
    pub fn angle_between(&self, other: &Self) -> f64 {
        let mag_self = self.magnitude();
        let mag_other = other.magnitude();
        if mag_self < 1e-12 || mag_other < 1e-12 {
            return 0.0;
        }
        let dot = self.dot(other) / (mag_self * mag_other);
        dot.clamp(-1.0, 1.0).acos()
    }

    /// Linear interpolation between two vectors.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let clamped_t = t.clamp(0.0, 1.0);
        Self::new(
            self.x + (other.x - self.x) * clamped_t,
            self.y + (other.y - self.y) * clamped_t,
            self.z + (other.z - self.z) * clamped_t,
        )
    }

    /// Rodrigues rotation: rotate vector `v` around axis `k` by angle `theta`.
    /// Formula: v_rot = v*cosθ + (k×v)*sinθ + k*(k·v)*(1-cosθ)
    pub fn rotate_around(&self, axis: &Self, theta: f64) -> Option<Self> {
        let axis_norm = axis.normalized()?;
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let dot = self.dot(&axis_norm);
        let k_cross_v = axis_norm.cross(self);

        Some(
            self.mul(cos_theta)
                .add(&k_cross_v.mul(sin_theta))
                .add(&axis_norm.mul(dot * (1.0 - cos_theta)))
        )
    }

    /// Distance to another vector.
    pub fn distance(&self, other: &Self) -> f64 {
        self.sub(other).magnitude()
    }

    /// Squared distance (avoids sqrt).
    pub fn distance_sq(&self, other: &Self) -> f64 {
        self.sub(other).magnitude_sq()
    }

    /// Check if vector is approximately zero.
    pub fn is_zero(&self, eps: f64) -> bool {
        self.magnitude_sq() < eps * eps
    }

    /// Convert to array for NumPy compatibility.
    pub fn to_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// Convert from array.
    pub fn from_array(arr: &[f64; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }
}

/// Quaternion for 3D rotations.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Creates a new quaternion.
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity quaternion (no rotation).
    pub const fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    /// Creates a quaternion from axis-angle representation.
    /// Axis must be normalized.
    pub fn from_axis_angle(axis: &Vec3, theta: f64) -> Result<Self, &'static str> {
        let axis_norm = axis.normalized().ok_or("Axis has zero magnitude")?;
        let half_theta = theta / 2.0;
        let sin_half = half_theta.sin();
        let cos_half = half_theta.cos();

        Ok(Self::new(
            cos_half,
            axis_norm.x * sin_half,
            axis_norm.y * sin_half,
            axis_norm.z * sin_half,
        ))
    }

    /// Creates a quaternion from roll, pitch, yaw (Z-Y-X sequence).
    pub fn from_rpy(roll: f64, pitch: f64, yaw: f64) -> Self {
        let (cy, sy) = (yaw / 2.0).cos(), (yaw / 2.0).sin();
        let (cp, sp) = (pitch / 2.0).cos(), (pitch / 2.0).sin();
        let (cr, sr) = (roll / 2.0).cos(), (roll / 2.0).sin();

        Self::new(
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
        )
    }

    /// Multiply two quaternions.
    pub fn mul(&self, other: &Self) -> Self {
        Self::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }

    /// Rotate a vector by this quaternion.
    pub fn rotate(&self, v: &Vec3) -> Result<Vec3, &'static str> {
        // Convert vector to pure quaternion
        let pure = Self::new(0.0, v.x, v.y, v.z);
        let inv = self.conjugate().ok_or("Unit quaternion expected")?;
        let rotated = self.mul(&pure).mul(&inv);
        Ok(Vec3::new(rotated.x, rotated.y, rotated.z))
    }

    /// Conjugate of quaternion.
    pub fn conjugate(&self) -> Option<Self> {
        // Assume unit quaternion
        if self.magnitude_sq() < 1e-12 {
            None
        } else {
            Some(Self::new(self.w, -self.x, -self.y, -self.z))
        }
    }

    /// Inverse of quaternion.
    pub fn inverse(&self) -> Option<Self> {
        let mag_sq = self.magnitude_sq();
        if mag_sq < 1e-12 {
            None
        } else {
            let mag_sq = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
            let conjugate = Self::new(self.w, -self.x, -self.y, -self.z);
            Some(conjugate.mul(&(1.0 / mag_sq)))
        }
    }

    /// Magnitude (norm) of quaternion.
    pub fn magnitude(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Squared magnitude.
    pub fn magnitude_sq(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalized (unit) quaternion.
    pub fn normalized(&self) -> Option<Self> {
        let mag = self.magnitude();
        if mag < 1e-12 {
            None
        } else {
            let inv = 1.0 / mag;
            Some(Self::new(
                self.w * inv,
                self.x * inv,
                self.y * inv,
                self.z * inv,
            ))
        }
    }

    /// Linear interpolation between quaternions (for smooth rotation).
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let clamped_t = t.clamp(0.0, 1.0);
        Self::new(
            self.w + (other.w - self.w) * clamped_t,
            self.x + (other.x - self.x) * clamped_t,
            self.y + (other.y - self.y) * clamped_t,
            self.z + (other.z - self.z) * clamped_t,
        )
    }

    /// Spherical linear interpolation (slerp) - proper spherical interpolation.
    pub fn slerp(&self, other: &Self, t: f64) -> Result<Self, &'static str> {
        let clamped_t = t.clamp(0.0, 1.0);

        // Ensure quaternions are normalized
        let q1 = self.normalized().ok_or("Input quaternion has zero magnitude")?;
        let q2 = other.normalized().ok_or("Input quaternion has zero magnitude")?;

        // Dot product
        let mut dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        // Use the shortest path
        if dot < 0.0 {
            let neg_q2 = Quaternion::new(-q2.w, -q2.x, -q2.y, -q2.z);
            return q1.slerp(&neg_q2, t);
        }

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            return Ok(q1.lerp(&q2, t));
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();

        // Interlogging coefficients
        let a = (1.0 - clamped_t) / sin_theta;
        let b = clamped_t / sin_theta;

        Ok(Self::new(
            a * q1.w + b * q2.w,
            a * q1.x + b * q2.x,
            a * q1.y + b * q2.y,
            a * q1.z + b * q2.z,
        ))
    }
}

/// Pre-computed direction vectors for common rotations.
pub mod directions {
    use super::Vec3;

    pub const UP: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 1.0 };
    pub const DOWN: Vec3 = Vec3 { x: 0.0, y: 0.0, z: -1.0 };
    pub const RIGHT: Vec3 = Vec3 { x: 1.0, y: 0.0, z: 0.0 };
    pub const LEFT: Vec3 = Vec3 { x: -1.0, y: 0.0, z: 0.0 };
    pub const FORWARD: Vec3 = Vec3 { x: 0.0, y: 1.0, z: 0.0 };
    pub const BACK: Vec3 = Vec3 { x: 0.0, y: -1.0, z: 0.0 };
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_vec3_magnitude() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.magnitude(), 5.0);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let normalized = v.normalized().unwrap();
        assert_relative_eq!(normalized.magnitude(), 1.0);
        assert_relative_eq!(normalized.x, 0.6);
        assert_relative_eq!(normalized.y, 0.8);
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let cross = a.cross(&b);
        assert_relative_eq!(cross, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
    }

    #[test]
    fn test_quaternion_rotate() {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, consts::FRAC_PI_2).unwrap();
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rotated = q.rotate(&v).unwrap();
        assert_relative_eq!(rotated.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(rotated.y, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_quaternion_slerp() {
        let q1 = Quaternion::identity();
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let q2 = Quaternion::from_axis_angle(&axis, consts::PI).unwrap();
        let q_mid = q1.slerp(&q2, 0.5).unwrap();
        // Should be approximately 90 degrees
        assert_relative_eq!(q_mid.w, 0.0, epsilon = 1e-10);
        assert_relative_eq!(q_mid.z.abs(), 1.0, epsilon = 1e-10);
    }
}
