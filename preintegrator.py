"""
Python implementation of IMU preintegration in the style used by LIO-SAM / GTSAM
(Forster et al. 'Preintegration on Manifold').

- Minimal dependencies (numpy only)
- Provides ΔR, Δv, Δp, Jacobians wrt biases, and covariance propagation
- Midpoint integrator with continuous white-noise model

This module focuses on the *preintegrated* deltas between two keyframes.
Gravity is *not* included inside the deltas (as in GTSAM); it is injected
when predicting the next state using the starting pose/vel and gravity.

Usage example is at the bottom.
"""
from __future__ import annotations
import numpy as np

# ----------------------------- SO(3) utilities ----------------------------- #

def skew(v: np.ndarray) -> np.ndarray:
	x, y, z = v
	return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def exp_so3(phi: np.ndarray) -> np.ndarray:
	"""Exponential map from so(3) to SO(3).
	Uses Rodrigues formula; stable for small angles.
	"""
	angle = np.linalg.norm(phi)
	if angle < 1e-8:
		# 1st-order approximation
		return np.eye(3) + skew(phi)
	axis = phi / angle
	K = skew(axis)
	s = np.sin(angle)
	c = np.cos(angle)
	return np.eye(3) + s * K + (1 - c) * (K @ K)


def right_jacobian_SO3(phi: np.ndarray) -> np.ndarray:
	"""Right Jacobian Jr(phi) for SO(3).
	Such that Exp(phi + J_r * eps) ≈ Exp(phi) Exp(eps).
	"""
	angle = np.linalg.norm(phi)
	if angle < 1e-8:
		return np.eye(3) - 0.5 * skew(phi)
	axis = phi / angle
	K = skew(axis)
	s = np.sin(angle)
	c = np.cos(angle)
	return (
		np.eye(3)
		- (1 - c) / angle * K
		+ (angle - s) / (angle) * (K @ K)
	)


# -------------------------- Preintegration class -------------------------- #

class ImuPreintegrator:
	"""IMU preintegration (Forster-style) with midpoint integration.

	State of the preintegrator stores:
	  - dR: SO(3) rotation from i to j (preintegrated)
	  - dv: Δv (body i frame), gravity excluded
	  - dp: Δp (body i frame), gravity excluded
	  - J_*: Jacobians wrt accel/gyro bias at the linearization point
	  - P: covariance of [θ, dv, dp] (15×15 if including biases; here 9×9 for deltas
		   plus separate bias random walk handling; simple and practical)

	Note: Following GTSAM, gravity is not included in deltas; it will be
	accounted for in the state prediction outside this class.
	"""

	def __init__(
		self,
		accel_noise_sigma: float = 0.08,   # m/s^2/√Hz (example)
		gyro_noise_sigma: float = 0.004,   # rad/s/√Hz
		accel_bias_rw_sigma: float = 0.0004,  # m/s^2/√Hz
		gyro_bias_rw_sigma: float = 0.00004,  # rad/s/√Hz
		bias_a_init: np.ndarray | None = None,
		bias_g_init: np.ndarray | None = None,
	):
		self.sigma_a = accel_noise_sigma
		self.sigma_g = gyro_noise_sigma
		self.sigma_ba = accel_bias_rw_sigma
		self.sigma_bg = gyro_bias_rw_sigma

		self.bias_a = np.zeros(3) if bias_a_init is None else bias_a_init.astype(float)
		self.bias_g = np.zeros(3) if bias_g_init is None else bias_g_init.astype(float)

		self.reset()

	# ------------------------------------------------------------------ #
	def reset(self):
		self.dR = np.eye(3)
		self.dv = np.zeros(3)
		self.dp = np.zeros(3)

		# Jacobians wrt biases (per GTSAM notation)
		self.J_r_bg = np.zeros((3, 3))
		self.J_v_ba = np.zeros((3, 3))
		self.J_v_bg = np.zeros((3, 3))
		self.J_p_ba = np.zeros((3, 3))
		self.J_p_bg = np.zeros((3, 3))

		self.P = np.zeros((9, 9))  # Covariance of [θ, dv, dp]
		self.delta_t = 0.0

		# for midpoint integration
		self._last_a = None
		self._last_w = None

	# ------------------------------------------------------------------ #
	def integrate(self, a_meas: np.ndarray, w_meas: np.ndarray, dt: float):
		"""Feed one IMU sample (accel [m/s^2], gyro [rad/s]) over time dt.
		Uses midpoint method when previous sample exists; otherwise Euler.
		"""
		a_meas = a_meas.astype(float)
		w_meas = w_meas.astype(float)
		dt = float(dt)
		assert dt > 0

		if self._last_a is None:
			a0, w0 = a_meas, w_meas
			a1, w1 = a_meas, w_meas
		else:
			a0, w0 = self._last_a, self._last_w
			a1, w1 = a_meas, w_meas

		# Bias corrected
		a0c = a0 - self.bias_a
		a1c = a1 - self.bias_a
		w0c = w0 - self.bias_g
		w1c = w1 - self.bias_g

		# Midpoint values
		am = 0.5 * (a0c + a1c)
		wm = 0.5 * (w0c + w1c)

		# --- Rotation update ---
		dtheta = wm * dt
		Jr = right_jacobian_SO3(dtheta)
		self.dR = self.dR @ exp_so3(dtheta)

		# --- Jacobian update wrt gyro bias ---
		self.J_r_bg = self.J_r_bg - self.dR @ Jr * dt  # approximate

		# --- Specific force in i-frame (gravity excluded here) ---
		a_ib = self.dR @ am

		# --- Velocity / Position update ---
		dv_old = self.dv.copy()
		self.dv = self.dv + a_ib * dt
		self.dp = self.dp + dv_old * dt + 0.5 * a_ib * dt * dt

		# --- Jacobians wrt biases (Forster-style approximations) ---
		Ra = self.dR @ (-0.5 * dt * np.eye(3))  # ∂a_ib/∂ba ≈ -R * 0.5dt
		Rg = self.dR @ (-0.5 * dt * skew(am))   # ∂a_ib/∂bg via rotation change
		self.J_v_ba += Ra * dt
		self.J_v_bg += Rg * dt
		self.J_p_ba += self.J_v_ba * dt + 0.5 * Ra * dt * dt
		self.J_p_bg += self.J_v_bg * dt + 0.5 * Rg * dt * dt

		# --- Covariance propagation ---
		# Error state: x = [θ, dv, dp]
		F = np.eye(9)
		# θ dynamics: θ_{k+1} ≈ θ_k + (-skew(wm)) θ_k dt + noise
		# For small dt, keep identity (higher-order terms omitted).
		# dv depends on a_ib and rotation error -> ∂dv/∂θ ≈ -R * skew(am) * dt
		F[3:6, 0:3] = -self.dR @ skew(am) * dt
		# dp depends on dv and a_ib
		F[6:9, 3:6] = np.eye(3) * dt
		F[6:9, 0:3] = -0.5 * self.dR @ skew(am) * dt * dt

		# Noise mapping (gyro, accel, bias RW)
		G = np.zeros((9, 12))
		# gyro meas noise -> rotation
		G[0:3, 0:3] = np.eye(3) * dt
		# accel meas noise -> dv, dp
		G[3:6, 3:6] = self.dR * dt
		G[6:9, 3:6] = 0.5 * self.dR * dt * dt
		# accel bias RW -> dv, dp
		G[3:6, 6:9] = -self.dR * dt
		G[6:9, 6:9] = -0.5 * self.dR * dt * dt
		# gyro bias RW -> rotation (through θ)
		G[0:3, 9:12] = -np.eye(3) * dt

		Qc = np.diag([
			self.sigma_g**2, self.sigma_g**2, self.sigma_g**2,  # gyro meas
			self.sigma_a**2, self.sigma_a**2, self.sigma_a**2,  # accel meas
			self.sigma_ba**2, self.sigma_ba**2, self.sigma_ba**2,  # accel bias RW
			self.sigma_bg**2, self.sigma_bg**2, self.sigma_bg**2,  # gyro bias RW
		])

		self.P = F @ self.P @ F.T + G @ Qc @ G.T

		# book-keeping
		self.delta_t += dt
		self._last_a = a1
		self._last_w = w1

	# ------------------------------------------------------------------ #
	def delta(self):
		"""Return (ΔR, Δv, Δp, J, P, Δt) where J is a dict of Jacobians.
		ΔR is a 3×3 rotation matrix from i to j.
		Δv, Δp are in i-frame with gravity excluded.
		"""
		J = {
			"J_r_bg": self.J_r_bg.copy(),
			"J_v_ba": self.J_v_ba.copy(),
			"J_v_bg": self.J_v_bg.copy(),
			"J_p_ba": self.J_p_ba.copy(),
			"J_p_bg": self.J_p_bg.copy(),
		}
		return self.dR.copy(), self.dv.copy(), self.dp.copy(), J, self.P.copy(), self.delta_t

	# ------------------------------------------------------------------ #
	@staticmethod
	def predict(
		R_iw: np.ndarray,
		p_iw: np.ndarray,
		v_iw: np.ndarray,
		dR: np.ndarray,
		dv: np.ndarray,
		dp: np.ndarray,
		g_w: np.ndarray,
		dt: float,
	) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""Given start state (R_iw, p_iw, v_iw) at time i and preintegrated
		deltas (dR, dv, dp) from i→j, predict state at j.
		Gravity g_w (in world) is injected here, as in GTSAM/LIO-SAM.
		Returns (R_jw, p_jw, v_jw).
		"""
		R_jw = R_iw @ dR
		v_jw = v_iw + R_iw @ dv + g_w * dt
		p_jw = p_iw + v_iw * dt + R_iw @ dp + 0.5 * g_w * dt * dt
		return R_jw, p_jw, v_jw


# ------------------------------- Example --------------------------------- #
if __name__ == "__main__":
	# Synthetic example: IMU spinning with small yaw rate, level, no gravity inside deltas
	np.set_printoptions(precision=4, suppress=True)

	preint = ImuPreintegrator(
		accel_noise_sigma=0.08,
		gyro_noise_sigma=0.004,
		accel_bias_rw_sigma=0.0004,
		gyro_bias_rw_sigma=0.00004,
		bias_a_init=np.array([0.0, 0.0, 0.0]),
		bias_g_init=np.array([0.0, 0.0, 0.0]),
	)

	dt = 0.005
	for k in range(200):
		a_meas = np.array([0.0, 0.0, 0.0])  # specific force only; gravity handled in predict
		w_meas = np.array([0.0, 0.0, 0.05])  # 0.05 rad/s yaw
		preint.integrate(a_meas, w_meas, dt)

	dR, dv, dp, J, P, T = preint.delta()
	print("Δt:", T)
	print("ΔR:\n", dR)
	print("Δv:", dv)
	print("Δp:", dp)
	print("P diag:", np.diag(P))

	# Predict next state from identity pose/vel with gravity
	R_iw = np.eye(3)
	p_iw = np.zeros(3)
	v_iw = np.zeros(3)
	g_w = np.array([0.0, 0.0, -9.81])

	R_jw, p_jw, v_jw = ImuPreintegrator.predict(R_iw, p_iw, v_iw, dR, dv, dp, g_w, T)
	print("Pred p:", p_jw)
	print("Pred v:", v_jw)
