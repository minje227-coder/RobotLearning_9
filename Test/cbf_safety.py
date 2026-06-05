"""Dual-arm CBF safety layer (Phase 2).

Plug-and-play safety correction for the joint-space SmolVLA policies of the
dual-arm setup, adapting the VLSA/AEGIS idea (arXiv 2512.11891) to:
  * the OPPOSING arm's end-effector as a DYNAMIC ellipsoid obstacle, and
  * a JOINT-space policy (the paper assumes an EE-velocity / OSC policy).

Pipeline per control step, for a "safety-enabled" arm i with the other arm j:
  1. Model both EEs as ellipsoids (semi-axes `Q_EF`, world-rotated by R_ef).
  2. Along the center-to-center direction n, the ellipsoid extent is
        r(n) = 1 / sqrt(n^T A n),  A = R diag(1/a^2,1/b^2,1/c^2) R^T,
     giving the barrier  h = ||p_i - p_j|| - r_i(n) - r_j(n) - margin.
  3. The policy's joint delta dq implies a nominal EE displacement
        u_nom = J_pos(q) dq          (3x7 positional Jacobian).
  4. Discrete CBF: keep h_next >= (1-alpha) h, i.e. Delta h >= -alpha h.
     With the other arm treated as (locally) static, the linear constraint is
        n . u_i >= -alpha * h.
     The minimum-norm safe displacement is a half-space projection:
        u_safe = u_nom + max(0, (-alpha h) - n.u_nom) * n.
  5. Map only the correction back to joints with a damped pseudo-inverse and
     ADD it to the policy delta (so the policy's full joint intent is kept and
     the layer is inactive when no violation):
        dq_safe = dq + J^T (J J^T + lambda I)^{-1} (u_safe - u_nom).

This is the translational-only regime of the paper (orientation left to the
policy). It needs no QP solver: the single linear constraint has a closed form.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# End-effector ellipsoid semi-axes (meters), matching the paper's Q_ef.
Q_EF = np.array([0.06, 0.12, 0.11])


def _quat_to_mat(quat) -> np.ndarray:
    # robosuite/mujoco obs quats are [x, y, z, w]; scipy expects [x, y, z, w].
    return R.from_quat(np.asarray(quat, dtype=float)).as_matrix()


def _dir_radius(rot_mat: np.ndarray, semi_axes: np.ndarray, n: np.ndarray) -> float:
    """Ellipsoid extent from center along unit direction n (world frame)."""
    A = rot_mat @ np.diag(1.0 / np.square(semi_axes)) @ rot_mat.T
    return float(1.0 / np.sqrt(max(n @ A @ n, 1e-9)))


def barrier_h(p_self, R_self, p_other, R_other, semi_axes=Q_EF, margin=0.03):
    """Return (h, n, dist). h>0 safe, h<0 ellipsoids overlap (incl. margin)."""
    d = np.asarray(p_self, float) - np.asarray(p_other, float)
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return -1.0, np.array([1.0, 0.0, 0.0]), dist
    n = d / dist
    r_self = _dir_radius(R_self, semi_axes, n)
    r_other = _dir_radius(R_other, semi_axes, n)
    return dist - r_self - r_other - margin, n, dist


def positional_jacobian(sim, robot) -> np.ndarray:
    """3x7 positional Jacobian of the grip site w.r.t. this robot's joints."""
    site = robot.gripper.important_sites["grip_site"]
    jidx = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    return np.asarray(sim.data.get_site_jacp(site), dtype=float).reshape(3, -1)[:, jidx]


def cbf_correct_action(
    sim,
    robot_self,
    p_self,
    quat_self,
    p_other,
    quat_other,
    dq_action,
    *,
    semi_axes=Q_EF,
    margin=0.03,
    alpha=0.8,
    damping=0.05,
    action_scale=1.0,
):
    """Return (dq_safe, info). dq_action is the 8-D policy action (7 joints + gripper).

    `action_scale` converts a unit of action[:7] into the joint displacement that
    actually reaches the EE (set after calibrating J@dq vs measured EE motion).
    The gripper channel (index 7) is passed through unchanged.
    """
    dq_action = np.asarray(dq_action, dtype=float)
    dq = dq_action[:7].copy()
    R_self = _quat_to_mat(quat_self)
    R_other = _quat_to_mat(quat_other)
    h, n, dist = barrier_h(p_self, R_self, p_other, R_other, semi_axes, margin)

    Jp = positional_jacobian(sim, robot_self)
    u_nom = Jp @ (dq * action_scale)                # nominal EE displacement (m)
    slack = (-alpha * h) - float(n @ u_nom)         # >0 means constraint violated
    info = {"h": h, "dist": dist, "intervened": False, "slack": slack}
    if slack <= 0.0:
        return dq_action, info

    u_safe = u_nom + slack * n
    du = u_safe - u_nom                             # = slack * n
    # damped least-squares: dq_corr = J^T (J J^T + lambda I)^-1 du
    JJt = Jp @ Jp.T + damping * np.eye(3)
    dq_corr = Jp.T @ np.linalg.solve(JJt, du)
    dq_corr /= max(action_scale, 1e-9)              # back to action units

    out = dq_action.copy()
    out[:7] = np.clip(dq + dq_corr, -0.2, 0.2)
    info["intervened"] = True
    return out, info
