#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mahalanobis-weighted Gauss–Newton with ANALYTIC Jacobians + sparse linear solver
for joint estimation of {scale s, gravity g, per-keyframe velocities v_i, IMU biases ba,bg}
using IMU preintegration (Forster/LIO-SAM style) and up-to-scale visual positions.

Compared to the previous version, this script:
  - Uses the FULL 9x9 preintegration covariance per edge to whiten residuals and Jacobians
	(Mahalanobis distance), instead of ad-hoc scalar weights.
  - Keeps analytic Jacobians and optional SciPy sparse normal-equation solve.

Run example:
  python mahal_analytic_gn_scale_refine.py \
	--euroc_root /data/MH_01_easy/mav0 --kf_hz 10 --scale_drop 0.3 --iters 8 --sigma_g 0.1

Dependencies: numpy; optional scipy.sparse for speed.
Place this file alongside `preintegrator.py` (must expose ImuPreintegrator, exp_so3, skew).
"""
from __future__ import annotations
import os
import argparse, time
import numpy as np
# Optional SciPy for sparse algebra
try:
	import scipy.sparse as sp
	import scipy.sparse.linalg as spla
	_HAVE_SCIPY = True
except Exception:
	_HAVE_SCIPY = False
from preintegrator import ImuPreintegrator, exp_so3, skew

# ----------------------------- SO(3) helpers ----------------------------- #

def log_so3(R: np.ndarray) -> np.ndarray:
	"""Log map SO(3)->so(3) (rotation vector)."""
	tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
	th = np.arccos(tr)
	if th < 1e-8:
		return np.array([(R[2,1]-R[1,2]), (R[0,2]-R[2,0]), (R[1,0]-R[0,1])]) * 0.5
	return (th / (2.0*np.sin(th))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
# ----------------------------- EuRoC loaders ----------------------------- #

def load_euroc_imu(csv_path: str):
	data = np.genfromtxt(csv_path, delimiter=",", comments="#")
	if data.ndim == 1: data = data[None, :]
	t = data[:, 0].astype(np.float64) * 1e-9
	gyro = data[:, 1:4].astype(np.float64)
	accel = data[:, 4:7].astype(np.float64)
	return t, accel, gyro

def load_euroc_gt(csv_path: str):
	data = np.genfromtxt(csv_path, delimiter=",", comments="#")
	if data.ndim == 1: data = data[None, :]
	t = data[:, 0].astype(np.float64) * 1e-9
	p = data[:, 1:4].astype(np.float64)
	q = data[:, 4:8].astype(np.float64)  # wxyz
	return t, p, q

def q_to_R(qw, qx, qy, qz):
	q = np.array([qw, qx, qy, qz], dtype=float)
	q /= np.linalg.norm(q)
	w,x,y,z = q
	return np.array([
		[1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
		[2*(x*y + z*w),   1-2*(x*x+z*z),   2*(y*z - x*w)],
		[2*(x*z - y*w),   2*(y*z + x*w),   1-2*(x*x+y*y)],
	])

def select_keyframes(t: np.ndarray, hz: float = 10.0) -> np.ndarray:
	idx = [0]; gap = 1.0/hz; last = t[0]
	for i in range(1, len(t)):
		if t[i] - last >= gap:
			idx.append(i); last = t[i]
	if len(idx) < 2:
		idx = [0, len(t)-1]
	return np.asarray(idx, dtype=int)

# ------------------------- Preintegration per edge ------------------------ #

def preint_edge(t0, t1, t_imu, acc, gyro):
	"""Preintegrate IMU over [t0, t1] and return deltas, jacobians, covariance."""
	pre = ImuPreintegrator(bias_a_init=np.zeros(3), bias_g_init=np.zeros(3))
	i = max(0, int(np.searchsorted(t_imu, t0, side='left')) - 1)
	i = max(i, 0); cur = t0
	while cur < t1:
		nxt = min(t_imu[i+1], t1) if (i+1) < len(t_imu) else t1
		dt = float(max(0.0, nxt - cur))
		if dt > 0: pre.integrate(acc[i], gyro[i], dt)
		cur = nxt
		if (i+1) < len(t_imu) and t_imu[i+1] <= t1: i += 1
		else: break
	dR, dv, dp, J, P, T = pre.delta()
	P = 0.5 * (P + P.T)  # symmetrize
	return {
		'dR': dR, 'dv': dv, 'dp': dp,
		'Jr_bg': J["J_r_bg"], 'Jv_ba': J["J_v_ba"], 'Jv_bg': J["J_v_bg"],
		'Jp_ba': J["J_p_ba"], 'Jp_bg': J["J_p_bg"], 'P': P, 'dt': T
	}

# -------------------------- GN problem (analytic + Mahalanobis) ---------- #
class GNProblem:
	def __init__(self, Rw: np.ndarray, tv: np.ndarray, t_kf: np.ndarray, edges: list[dict], g_prior=9.81, sigma_g: float = 0.1):
		self.Rw, self.tv, self.t_kf, self.edges = Rw, tv, t_kf, edges
		self.M = len(t_kf)
		self.g_prior = float(g_prior)
		self.sigma_g = float(sigma_g)
		# variable layout
		self.idx_s = 0
		self.idx_g = 1
		self.idx_ba = 4
		self.idx_bg = 7
		self.idx_v0 = 10  # then 3 per frame

	def pack(self, s: float, g: np.ndarray, ba: np.ndarray, bg: np.ndarray, V: np.ndarray) -> np.ndarray:
		return np.hstack([np.array([s]), g.ravel(), ba.ravel(), bg.ravel(), V.ravel()])

	def unpack(self, x: np.ndarray):
		s = float(x[self.idx_s])
		g = x[self.idx_g:self.idx_g+3]
		ba = x[self.idx_ba:self.idx_ba+3]
		bg = x[self.idx_bg:self.idx_bg+3]
		V = x[self.idx_v0:].reshape(self.M, 3)
		return s, g, ba, bg, V

	def _left_jacobian_inv_SO3(self, phi: np.ndarray) -> np.ndarray:
		th = np.linalg.norm(phi)
		if th < 1e-8: return np.eye(3) + 0.5 * skew(phi)
		a = phi / th
		K = skew(a)
		half = 0.5*th
		cot = (half/np.tan(half))
		return np.eye(3) - 0.5*K + (1.0 - cot/(1.0)) * (K @ K)

	def residual_and_jac(self, x: np.ndarray):
		"""Mahalanobis-whitened residual r and Jacobian J (analytic), using full 9x9 P per edge."""
		s, g, ba, bg, V = self.unpack(x)
		M = self.M
		n_rows = 9*(M-1) + 1
		n_cols = 1 + 3 + 3 + 3 + 3*M
		rows = []; cols = []; data = []
		r = np.zeros(n_rows)
		row = 0
		for i in range(M-1):
			e = self.edges[i]
			dt = e['dt']
			Ri = self.Rw[i]
			Rij_true = self.Rw[i].T @ self.Rw[i+1]
			# bias-corrected preintegrated terms
			dR_corr = e['dR'] @ exp_so3(e['Jr_bg'] @ bg)
			dv_corr = e['dv'] + e['Jv_ba'] @ ba + e['Jv_bg'] @ bg
			dp_corr = e['dp'] + e['Jp_ba'] @ ba + e['Jp_bg'] @ bg
			# residual blocks (unwhitened)
			rR = log_so3(dR_corr.T @ Rij_true)
			rv = Ri.T @ (V[i+1] - V[i] - g*dt) - dv_corr
			pi = s * self.tv[i]
			pj = s * self.tv[i+1]
			rp = Ri.T @ (pj - pi - V[i]*dt - 0.5*g*dt*dt) - dp_corr
			r_edge = np.hstack([rR, rv, rp])  # 9,
			# analytic local Jacobian (9 x 16) with column order: [bg(3), vi(3), vj(3), g(3), ba(3), s(1)]
			Jloc = np.zeros((9, 16))
			Jl_inv = self._left_jacobian_inv_SO3(rR)
			Jloc[0:3, 0:3] = - Jl_inv @ e['Jr_bg']  # d rR / d bg
			RiT = Ri.T
			Jloc[3:6, 3:6]   = -RiT              # d rv / d vi
			Jloc[3:6, 6:9]   =  RiT              # d rv / d vj
			Jloc[3:6, 9:12]  = -RiT * dt         # d rv / d g
			Jloc[3:6, 12:15] = -e['Jv_ba']       # d rv / d ba
			Jloc[3:6, 0:3]  += -e['Jv_bg']       # add d rv / d bg
			Jloc[6:9, 3:6]   = -RiT * dt         # d rp / d vi
			Jloc[6:9, 9:12]  = -RiT * (0.5*dt*dt) # d rp / d g
			dpis = self.tv[i+1] - self.tv[i]
			Jloc[6:9, 15:16] = (RiT @ dpis).reshape(3,1)  # d rp / d s
			Jloc[6:9, 12:15] = -e['Jp_ba']       # d rp / d ba
			Jloc[6:9, 0:3]  += -e['Jp_bg']       # add d rp / d bg
			# Mahalanobis whitening with full 9x9 P
			P = e['P'] + 1e-12*np.eye(9)
			L = np.linalg.cholesky(P)
			Linv = np.linalg.inv(L)
			r_w = Linv @ r_edge
			J_w = Linv @ Jloc
			# write r_w
			r[row:row+9] = r_w
			# map local columns to global indices
			col_bg = self.idx_bg
			col_vi = self.idx_v0 + 3*i
			col_vj = self.idx_v0 + 3*(i+1)
			col_g  = self.idx_g
			col_ba = self.idx_ba
			col_s  = self.idx_s
			global_cols = (
				list(range(col_bg, col_bg+3)) +
				list(range(col_vi, col_vi+3)) +
				list(range(col_vj, col_vj+3)) +
				list(range(col_g,  col_g +3)) +
				list(range(col_ba, col_ba+3)) +
				[col_s]
			)
			# push triplets (sparse)
			for a in range(9):
				# bg(3)
				for b in range(3):
					rows.append(row+a); cols.append(global_cols[0+b]); data.append(J_w[a, 0+b])
				# vi(3)
				for b in range(3):
					rows.append(row+a); cols.append(global_cols[3+b]); data.append(J_w[a, 3+b])
				# vj(3)
				for b in range(3):
					rows.append(row+a); cols.append(global_cols[6+b]); data.append(J_w[a, 6+b])
				# g(3)
				for b in range(3):
					rows.append(row+a); cols.append(global_cols[9+b]); data.append(J_w[a, 9+b])
				# ba(3)
				for b in range(3):
					rows.append(row+a); cols.append(global_cols[12+b]); data.append(J_w[a, 12+b])
				# s(1)
				rows.append(row+a); cols.append(global_cols[15]); data.append(J_w[a, 15])

			row += 9

		# gravity magnitude prior (scalar), whiten by sigma_g
		gnorm = np.linalg.norm(g) + 1e-12
		r[row] = (gnorm - self.g_prior) / self.sigma_g
		dg = (g / gnorm) / self.sigma_g
		for b in range(3):
			rows.append(row); cols.append(self.idx_g+b); data.append(dg[b])

		# build sparse/dense J
		if _HAVE_SCIPY:
			J = sp.coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols))
		else:
			J = np.zeros((n_rows, n_cols))
			for rr, cc, dd in zip(rows, cols, data):
				J[rr, cc] += dd
		return r, J

	def solve_gn(self, x0: np.ndarray, iters=10, lm=1e-6, verbose=True):
		x = x0.copy()
		for it in range(iters):
			r, J = self.residual_and_jac(x)
			if _HAVE_SCIPY:
				H = (J.T @ J).tocoo()
				b = J.T @ r
				H = H + sp.eye(H.shape[0]) * lm
				try:
					dx = -spla.spsolve(H.tocsr(), b)
				except Exception:
					dx = -np.linalg.lstsq(H.toarray(), b, rcond=None)[0]
			else:
				H = J.T @ J + lm * np.eye(J.shape[1])
				b = J.T @ r
				dx = -np.linalg.solve(H, b)
			x_new = x + dx
			r_new, _ = self.residual_and_jac(x_new)
			if verbose:
				print(f"[GN-analytic+Mahalanobis] iter {it}: ||r||={np.linalg.norm(r):.6e} -> {np.linalg.norm(r_new):.6e}, |dx|={np.linalg.norm(dx):.3e}")
			x = x_new
			if np.linalg.norm(dx) < 1e-8:
				break
		return x

# ------------------------------ Linear init ------------------------------ #
def linear_init_fast(Rw, tv, t_kf, edges):
    """
    Fast linear initializer using Schur complement + block tridiagonal solve.
    Reuses precomputed preintegration edges to avoid re-integration.
    Solves for s (scalar), g (3), V (M x 3) from stacked linear equations
    without building dense A. Complexity O(M).
    """
    import numpy as _np
    np = _np  # alias

    M = len(t_kf)
    K = M - 1
    dt = np.array([edges[i]['dt'] for i in range(K)], dtype=float)
    Ri = np.stack([Rw[i] for i in range(K)], axis=0)
    dp = np.stack([edges[i]['dp'] for i in range(K)], axis=0)
    dv = np.stack([edges[i]['dv'] for i in range(K)], axis=0)

    bpi = -(Ri @ dp[..., None]).squeeze(-1)   # (K,3)  = -Ri * Δp_ij
    bvi =  (Ri @ dv[..., None]).squeeze(-1)   # (K,3)  =  Ri * Δv_ij
    dti =  (tv[:K] - tv[1:K+1])               # (K,3)  =  t_i - t_j  (视觉位移差)

    # T = H_vv 的标量三对角结构: diag 和 off=-1 ；H_vv = kron(T, I3)
    diag = np.zeros(M, dtype=float)
    diag[:K] += dt**2 + 1.0   # 来自位置方程(dt*I)^T(dt*I)=dt^2 I 和速度方程(-I)^T(-I)=I
    diag[1:] += 1.0           # 来自下一个速度方程 (I)^T(I)=I
    off = -np.ones(M-1, dtype=float)

    # H_vx（速度对 {g,s} 的耦合）所需量：
    # alpha_i 给 g 的列；Svec_i = dt_i * (t_i - t_j) 给 s 的列
    alpha = np.zeros(M, dtype=float)
    alpha[:K] += 0.5*dt**3 + dt   # 本边贡献
    alpha[1:] += -dt              # 上一边对 v_i 的负贡献
    Svec = np.zeros((M,3), dtype=float)
    Svec[:K] = (dt[:,None] * dti)

    # g_V = A_V^T b  （按 v_i 分块）
    gv = np.zeros((M,3), dtype=float)
    gv[:K] += (dt[:,None] * bpi) - bvi
    gv[1:K] += bvi[:K-1]
	
    # H_xx 与 A_x^T b（只与 {g,s} 有关）
    Hgg_scalar = float(np.sum(dt**2 + 0.25*dt**4))                    # 3x3 = 该标量 * I
    Hgs = np.sum(0.5*(dt**2)[:,None] * dti, axis=0)                  # 3x1
    Hss = float(np.sum(np.einsum('ij,ij->i', dti, dti)))             # 1x1
    bg_vec = np.sum(-dt[:,None]*bvi + 0.5*(dt**2)[:,None]*bpi, axis=0)  # 3x1
    bs_s   = float(np.sum(np.einsum('ij,ij->i', dti, bpi)))             # 1x1

    # —— 三对角 LU 分解（一次） + 多 RHS 求解 ——
    def tridiag_lu(diag, off):
        n = diag.size
        u = np.empty(n, dtype=float)
        l = np.empty(n-1, dtype=float)
        u[0] = diag[0]
        for i in range(1, n):
            l[i-1] = off[i-1] / u[i-1]
            u[i] = diag[i] - l[i-1] * off[i-1]
        return l, u

    def tridiag_solve(l, u, off, d):
        y = np.empty_like(d)
        y[0] = d[0]
        for i in range(1, d.size):
            y[i] = d[i] - l[i-1] * y[i-1]
        x = np.empty_like(d)
        x[-1] = y[-1] / u[-1]
        for i in range(d.size-2, -1, -1):
            x[i] = (y[i] - off[i] * x[i+1]) / u[i]
        return x

    l,u = tridiag_lu(diag, off)

    # z1 = H_vv^{-1} g_V   （3 个 RHS：x/y/z）
    z1 = np.column_stack([tridiag_solve(l,u,off, gv[:,c]) for c in range(3)])  # (M,3)

    # Z2_g: 解 T z_g = alpha ；Z2_s: 解 T z_s = Svec(三列)
    z_g = tridiag_solve(l,u,off, alpha)                             # (M,)
    z_s = np.column_stack([tridiag_solve(l,u,off, Svec[:,c]) for c in range(3)])  # (M,3)

    # y = H_xv H_vv^{-1} g_V = (H_vx)^T z1
    y_g = np.sum(alpha[:,None] * z1, axis=0)                        # (3,)
    y_s = float(np.sum(np.einsum('ij,ij->i', Svec, z1)))            # (1,)

    # 舒尔补 S = H_xx - H_xv H_vv^{-1} H_vx
    Sgg_scalar = Hgg_scalar - float(np.sum(alpha * z_g))            # 3x3 = 该标量 * I
    Sgs = Hgs - np.sum(alpha[:,None] * z_s, axis=0)                 # (3,)
    Sss = Hss - float(np.sum(np.einsum('ij,ij->i', Svec, z_s)))     # (1,)

    # 右端项
    rhs_g = bg_vec - y_g
    rhs_s = bs_s - y_s

    # 解 4x4 的 [a I3, q; q^T, r] · [g; s] = [rhs_g; rhs_s]
    a = Sgg_scalar + 1e-12
    q = Sgs
    r = Sss + 1e-12
    denom = r - (q @ q) / a
    s0 = float((rhs_s - (q @ rhs_g) / a) / (denom + 1e-12))
    g0 = (rhs_g - q * s0) / a

    # 回代速度：V = H_vv^{-1}(g_V - H_vx x) = z1 - (Z2_g * g0 + Z2_s * s0)
    V0 = z1 - (z_g[:,None] * g0 + z_s * s0)
    return s0, g0, V0

# --------------------------------- Main --------------------------------- #

def main():
	t0 = time.time()
	ap = argparse.ArgumentParser()
	ap.add_argument('--euroc_root', required=True, help='e.g., /data/MH_01_easy/mav0')
	ap.add_argument('--imu_csv', default='imu0/data.csv')
	ap.add_argument('--gt_csv',  default=None, help='default tries state_groundtruth_estimate0 or mocap0')
	ap.add_argument('--kf_hz', type=float, default=10.0)
	ap.add_argument('--scale_drop', type=float, default=0.3)
	ap.add_argument('--iters', type=int, default=8)
	ap.add_argument('--sigma_g', type=float, default=0.1, help='std dev for gravity norm prior (m/s^2)')
	args = ap.parse_args()
	root = args.euroc_root
	imu_path = args.imu_csv if os.path.isabs(args.imu_csv) else os.path.join(root, args.imu_csv)
	if args.gt_csv:
		gt_path = args.gt_csv if os.path.isabs(args.gt_csv) else os.path.join(root, args.gt_csv)
		gt_candidates = [gt_path]
	else:
		gt_candidates = [os.path.join(root, 'state_groundtruth_estimate0/data.csv'),os.path.join(root, 'mocap0/data.csv'),]
	print(f"[INFO] Load IMU: {imu_path}")
	t_imu, acc, gyro = load_euroc_imu(imu_path)
	gt_loaded = False
	for p in gt_candidates:
		if os.path.exists(p):
			print(f"[INFO] Load GT : {p}")
			t_gt, p_gt, q_gt = load_euroc_gt(p); gt_loaded = True; break
	if not gt_loaded:
		raise FileNotFoundError('No GT found among: ' + ' | '.join(gt_candidates))
	kf_idx = select_keyframes(t_gt, args.kf_hz)
	t_kf = t_gt[kf_idx]
	p_kf = p_gt[kf_idx]
	q_kf = q_gt[kf_idx]
	Rw = np.stack([q_to_R(q[0], q[1], q[2], q[3]) for q in q_kf], axis=0)
	tv = p_kf / float(args.scale_drop)  # simulate monocular up-to-scale
	t1 = time.time()
	print("prepare time: ", t1-t0, 's')
	# Precompute preintegration per edge (with covariance)
	edges = [preint_edge(t_kf[i], t_kf[i+1], t_imu, acc, gyro) for i in range(len(t_kf)-1)]
	t2 = time.time()
	print("preint_edge time: ", t2-t1, 's')
	# Linear initialization
	s0, g0, V0 = linear_init_fast(Rw, tv, t_kf, edges)
	ba0 = np.zeros(3); bg0 = np.zeros(3)
	print("[INIT] s0=%.6f |g0|=%.4f g0=%s" % (s0, np.linalg.norm(g0), np.array2string(g0, precision=4)))
	t3 = time.time()
	print("linear_init time: ", t3-t2, 's')
	prob = GNProblem(Rw, tv, t_kf, edges, g_prior=9.81, sigma_g=args.sigma_g)
	x0 = prob.pack(s0, g0, ba0, bg0, V0)
	x_opt = prob.solve_gn(x0, iters=args.iters, lm=1e-6, verbose=True)
	s, g, ba, bg, V = prob.unpack(x_opt)
	print("\n==== Mahalanobis-weighted Analytic-GN Result ====")
	print(f"scale s       : {s:.6f}  (true drop ≈ {args.scale_drop})")
	print(f"gravity g     : {g}  |g|={np.linalg.norm(g):.4f}")
	print(f"bias  ba      : {ba}")
	print(f"bias  bg      : {bg}")
	print(f"v[0]          : {V[0]}")
	print(f"#keyframes={len(t_kf)}  edges={len(edges)}  SciPy={_HAVE_SCIPY}")
	t4 = time.time()
	print("solve_gn time: ", t4-t3, 's')

if __name__ == '__main__':
	main()

