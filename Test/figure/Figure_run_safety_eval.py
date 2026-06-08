#!/usr/bin/env python3
"""Phase 1 baseline safety-evaluation harness for the dual-arm setup.

Runs N episodes (one per seed) of the dual-arm milk->trash task with both
robots driven by their own SmolVLA policies (NO safety / CBF layer = the
"VLSA 0" baseline row of the experiment matrix), and aggregates:

  - TSR  (task success rate)          : env._check_success()
  - CAR  (collision avoidance rate)   : fraction of episodes with NO
                                         robot0<->robot1 mujoco contact
  - safe-success rate                  : success AND collision-free

Collision = any mujoco contact whose two geoms belong to different robots
(robot0_* / gripper0_* vs robot1_* / gripper1_*).

Reuses the (stable) helpers from test_model_grasp.py and re-implements only
the per-episode rollout state machine so we can hook contact detection and
return metrics instead of writing a video.

Run on `kanu` in the `lerobot` conda env with MUJOCO_GL=egl, e.g.:

  ssh kanu "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && \
    conda activate lerobot && cd ~/workspace/RobotLearning_9/Test && \
    MUJOCO_GL=egl python run_safety_eval.py \
      --robot0-policy-path ~/workspace/vlsa_smolvla_robot0_v4/checkpoints/last/pretrained_model \
      --robot1-policy-path ~/workspace/vlsa_smolvla_robot1_v4/checkpoints/last/pretrained_model \
      --seed-start 1 --num-episodes 20 --max-steps 400 \
      --out-csv results/baseline_v4.csv --out-json results/baseline_v4.json'"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import random as _random

import cv2
import imageio
import numpy as np
import torch
from lerobot.utils.constants import ACTION

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "Test"
FIGURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "Train/lerobot/src"))
sys.path.insert(0, str(REPO_ROOT / "Data generation"))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(FIGURE_DIR))

# Stable, side-effect-free helpers reused from the existing harness.
import test_model_grasp as tmg  # noqa: E402
import Figure_cbf_safety as cbf_safety  # noqa: E402

ELLIPSOID_COLORS = {
    "grip_site": (255, 0, 0),
    "link7": (0, 255, 0),
}
ELLIPSOID_INTERVENE_COLORS = {           # robot0 intervene: cyan / magenta
    "grip_site": (0, 255, 255),
    "link7": (255, 0, 255),
}
ELLIPSOID_INTERVENE_COLORS_R1 = {        # robot1 intervene: yellow / orange
    "grip_site": (255, 255, 0),
    "link7": (255, 140, 0),
}
ELLIPSOID_THICKNESS = 2
ELLIPSOID_SURFACE_ALPHA = 0.55
ELLIPSOID_SURFACE_AMBIENT = 0.30
ELLIPSOID_SURFACE_DIFFUSE = 0.70
ELLIPSOID_LAT_STEPS = 12
ELLIPSOID_LON_STEPS = 24
SUCCESS_BOX_COLOR = (0, 255, 0)
ROBOT0_NOMINAL_COLOR = (255, 200, 0)
ROBOT0_CORRECTED_COLOR = (255, 0, 0)
ROBOT1_NOMINAL_COLOR = (180, 180, 255)
ROBOT1_CORRECTED_COLOR = (0, 0, 255)
VIDEO_RENDER_CAMERAS = ("sideview", "birdview")
SIDEVIEW_Z_LIFT = 0.15
TRAJECTORY_DRAW_START = None
TRAJECTORY_DRAW_END = None
TRAJECTORY_DIVERGENCE_THRESHOLD = 0.01


# --------------------------------------------------------------------------- #
# Collision detection
# --------------------------------------------------------------------------- #
def _robot_of_geom(model, gid: int):
    """Return 0/1 if geom belongs to robot0/robot1, else None."""
    name = model.geom_id2name(gid)
    if not name:
        return None
    if name.startswith("robot0") or name.startswith("gripper0"):
        return 0
    if name.startswith("robot1") or name.startswith("gripper1"):
        return 1
    return None


def arm_arm_contact(sim):
    """If any active mujoco contact is between a robot0 and a robot1 geom,
    return the (geom0_name, geom1_name) pair; else return None."""
    model = sim.model
    data = sim.data
    for i in range(data.ncon):
        c = data.contact[i]
        a = _robot_of_geom(model, c.geom1)
        b = _robot_of_geom(model, c.geom2)
        if a is not None and b is not None and a != b:
            return (model.geom_id2name(c.geom1), model.geom_id2name(c.geom2))
    return None


def _camera_name2id(model, camera_name: str) -> int:
    return int(model.camera_name2id(camera_name))


def _lift_sideview_camera(sim, z_lift: float = SIDEVIEW_Z_LIFT) -> None:
    if abs(float(z_lift)) < 1e-9:
        return
    try:
        cam_id = _camera_name2id(sim.model, "sideview")
    except Exception:
        return
    sim.model.cam_pos[cam_id, 2] += float(z_lift)
    sim.forward()


def _camera_pose(sim, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    cam_id = _camera_name2id(sim.model, camera_name)
    cam_pos = np.asarray(sim.data.cam_xpos[cam_id], dtype=float)
    cam_rot = np.asarray(sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
    return cam_pos, cam_rot


def _project_world_points(sim, camera_name: str, points_world: np.ndarray, image_shape) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame 3D points into pixel coordinates for a MuJoCo camera."""
    model = sim.model
    cam_id = _camera_name2id(model, camera_name)
    cam_pos, cam_rot = _camera_pose(sim, camera_name)
    centered = np.asarray(points_world, dtype=float) - cam_pos[None, :]
    pts_cam = centered @ cam_rot

    height, width = image_shape[:2]
    fovy = float(model.cam_fovy[cam_id]) * np.pi / 180.0
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy

    z = -pts_cam[:, 2]
    valid = z > 1e-6
    pix = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pix[valid, 0] = fx * (pts_cam[valid, 0] / z[valid]) + (width * 0.5)
        pix[valid, 1] = -fy * (pts_cam[valid, 1] / z[valid]) + (height * 0.5)
    return pix, valid


def _project_world_points_with_depth(sim, camera_name: str, points_world: np.ndarray, image_shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points and also return positive camera-space depth."""
    model = sim.model
    cam_id = _camera_name2id(model, camera_name)
    cam_pos, cam_rot = _camera_pose(sim, camera_name)
    centered = np.asarray(points_world, dtype=float) - cam_pos[None, :]
    pts_cam = centered @ cam_rot

    height, width = image_shape[:2]
    fovy = float(model.cam_fovy[cam_id]) * np.pi / 180.0
    fy = 0.5 * height / np.tan(0.5 * fovy)
    fx = fy

    depth = -pts_cam[:, 2]
    valid = depth > 1e-6
    pix = np.full((points_world.shape[0], 2), np.nan, dtype=np.float32)
    if np.any(valid):
        pix[valid, 0] = fx * (pts_cam[valid, 0] / depth[valid]) + (width * 0.5)
        pix[valid, 1] = -fy * (pts_cam[valid, 1] / depth[valid]) + (height * 0.5)
    return pix, valid, depth


def _ellipsoid_curves(center, quat, semi_axes, n_points: int = 72) -> list[np.ndarray]:
    """Return three world-frame great-circle curves of the ellipsoid."""
    return _ellipsoid_curves_from_rot(center, cbf_safety._quat_to_mat(quat), semi_axes, n_points=n_points)


def _ellipsoid_curves_from_rot(center, rot, semi_axes, n_points: int = 72) -> list[np.ndarray]:
    """Return three world-frame great-circle curves of the ellipsoid."""
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    local_curves = [
        np.stack([semi_axes[0] * np.cos(t), semi_axes[1] * np.sin(t), np.zeros_like(t)], axis=1),
        np.stack([semi_axes[0] * np.cos(t), np.zeros_like(t), semi_axes[2] * np.sin(t)], axis=1),
        np.stack([np.zeros_like(t), semi_axes[1] * np.cos(t), semi_axes[2] * np.sin(t)], axis=1),
    ]
    center = np.asarray(center, dtype=float)
    return [(rot @ curve.T).T + center[None, :] for curve in local_curves]


def _ellipsoid_surface_mesh(semi_axes: np.ndarray, lat_steps: int = ELLIPSOID_LAT_STEPS,
                            lon_steps: int = ELLIPSOID_LON_STEPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude/longitude triangle mesh plus analytic ellipsoid normals."""
    a, b, c = [float(v) for v in semi_axes]
    lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, lat_steps + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, lon_steps, endpoint=False)

    verts_local = []
    normals_local = []
    for lat in lats:
        clat = np.cos(lat)
        slat = np.sin(lat)
        for lon in lons:
            clon = np.cos(lon)
            slon = np.sin(lon)
            x = a * clat * clon
            y = b * clat * slon
            z = c * slat
            verts_local.append([x, y, z])
            n_local = np.array([x / (a * a), y / (b * b), z / (c * c)], dtype=float)
            n_norm = np.linalg.norm(n_local)
            normals_local.append(n_local / max(n_norm, 1e-9))

    faces = []
    for lat_idx in range(lat_steps):
        row0 = lat_idx * lon_steps
        row1 = (lat_idx + 1) * lon_steps
        for lon_idx in range(lon_steps):
            nxt = (lon_idx + 1) % lon_steps
            i00 = row0 + lon_idx
            i01 = row0 + nxt
            i10 = row1 + lon_idx
            i11 = row1 + nxt
            faces.append((i00, i10, i11))
            faces.append((i00, i11, i01))

    return (
        np.asarray(verts_local, dtype=float),
        np.asarray(normals_local, dtype=float),
        np.asarray(faces, dtype=np.int32),
    )


def _draw_visible_polyline(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray, color, closed: bool) -> np.ndarray:
    if valid.sum() < 3:
        return img
    pts = points_2d[valid]
    if len(pts) < 3:
        return img
    pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    if closed and len(pts) >= 3:
        cv2.polylines(img, [pts], True, color, ELLIPSOID_THICKNESS, cv2.LINE_AA)
    else:
        cv2.polylines(img, [pts], False, color, ELLIPSOID_THICKNESS, cv2.LINE_AA)
    return img


def _draw_projected_segments(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray, edges, color) -> np.ndarray:
    out = img
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        pa = tuple(np.round(points_2d[a]).astype(np.int32))
        pb = tuple(np.round(points_2d[b]).astype(np.int32))
        cv2.line(out, pa, pb, color, ELLIPSOID_THICKNESS, cv2.LINE_AA)
    return out


def _draw_projected_dots(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray, color, radius: int = 2) -> np.ndarray:
    out = img
    for pt, ok in zip(points_2d, valid, strict=False):
        if not ok:
            continue
        center = tuple(np.round(pt).astype(np.int32))
        cv2.circle(out, center, radius, color, -1, cv2.LINE_AA)
    return out


def _append_timestep_bar(img: np.ndarray, timestep: int) -> np.ndarray:
    bar_h = max(28, img.shape[0] // 10)
    bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
    label = f"timestep: {timestep}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, bar_h / 42.0)
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    x = max(12, (img.shape[1] - text_w) // 2)
    y = max(text_h + 6, (bar_h + text_h) // 2 - baseline)
    cv2.putText(bar, label, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.concatenate([img, bar], axis=0)


def _trash_success_box_world_corners(sim, env, *, z_pad: float) -> np.ndarray:
    import create_dataset_grasp as create_dataset

    trash_obj = env.env.objects_dict["trash_can_1"]
    trash_body_id = sim.model.body_name2id(trash_obj.root_body)
    trash_pos = np.asarray(sim.data.body_xpos[trash_body_id], dtype=float)
    trash_rot = np.asarray(sim.data.body_xmat[trash_body_id], dtype=float).reshape(3, 3)

    x_half = float(create_dataset.TRASH_INNER_WIDTH) * 0.5
    y_half = float(create_dataset.TRASH_INNER_WIDTH) * 0.5
    z_min = float(create_dataset.TRASH_BOTTOM_THICK) - z_pad
    z_max = float(create_dataset.TRASH_BOTTOM_THICK + create_dataset.TRASH_INNER_DEPTH) + z_pad

    local_corners = np.array([
        [-x_half, -y_half, z_min],
        [ x_half, -y_half, z_min],
        [ x_half,  y_half, z_min],
        [-x_half,  y_half, z_min],
        [-x_half, -y_half, z_max],
        [ x_half, -y_half, z_max],
        [ x_half,  y_half, z_max],
        [-x_half,  y_half, z_max],
    ], dtype=float)
    return (trash_rot @ local_corners.T).T + trash_pos[None, :]


def _extract_policy_queue_actions(policy) -> list[np.ndarray]:
    if policy is None or not hasattr(policy, "model") or not hasattr(policy.model, "_queues"):
        return []
    queue = policy.model._queues.get(ACTION, [])
    out = []
    for tensor in list(queue):
        arr = tensor.detach().cpu().numpy()
        out.append(np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1])[0].copy())
    return out


def configure_policy_replan(policy, horizon: int) -> int | None:
    if policy is None or not hasattr(policy, "model") or not hasattr(policy.model, "config"):
        return None
    current = int(getattr(policy.model.config, "n_action_steps", horizon))
    new_horizon = max(1, min(int(horizon), current))
    policy.model.config.n_action_steps = new_horizon
    return new_horizon


def _get_robot_eef_pose(sim, robot) -> tuple[np.ndarray, np.ndarray]:
    site = robot.gripper.important_sites["grip_site"]
    pos = np.asarray(sim.data.get_site_xpos(site), dtype=float).copy()
    rot = np.asarray(sim.data.get_site_xmat(site), dtype=float).reshape(3, 3)
    quat = np.asarray(cbf_safety.R.from_matrix(rot).as_quat(), dtype=float)
    return pos, quat


def _shadow_action(raw_action: np.ndarray | None) -> np.ndarray:
    if raw_action is None:
        return tmg.FIXED_ACTION.copy()
    return np.asarray(raw_action, dtype=np.float32).copy()


def _restore_shadow_state(shadow_env, flat_state: np.ndarray) -> dict:
    obs = shadow_env.regenerate_obs_from_state(flat_state)
    shadow_env.env.sim.forward()
    return obs


def _step_shadow_env(shadow_env, robot0_action: np.ndarray, robot1_action: np.ndarray, active_arms: set[int]) -> dict:
    import create_dataset_grasp as create_dataset
    import create_dataset_robot1_grasp as create_dataset_robot1

    obs, _, _, _ = shadow_env.step(tmg.make_dual_action(robot0_action, robot1_action))
    if 1 not in active_arms:
        create_dataset.lock_robot1_pose(shadow_env.env.sim, shadow_env.env.robots[1])
    if 0 not in active_arms:
        create_dataset_robot1.lock_robot0_pose(shadow_env.env.sim, shadow_env.env.robots[0])
    return shadow_env.env._get_observations()


def predict_future_eef_trajectories(
    args,
    env,
    raw_seq0: list[np.ndarray],
    raw_seq1: list[np.ndarray],
    safety_arms: set[int],
    active_arms: set[int],
    shadow_nom_env,
    shadow_cor_env,
) -> dict[str, np.ndarray]:
    horizon = min(args.trajectory_horizon, max(len(raw_seq0), len(raw_seq1), 0))
    if horizon <= 0 or shadow_nom_env is None or shadow_cor_env is None:
        return {}

    traj = {
        "robot0_nominal": [],
        "robot1_nominal": [],
        "robot0_corrected": [],
        "robot1_corrected": [],
    }

    flat_state = env.get_sim_state()
    obs_nom = _restore_shadow_state(shadow_nom_env, flat_state)
    obs_cor = _restore_shadow_state(shadow_cor_env, flat_state)

    for step_idx in range(horizon):
        raw0 = raw_seq0[step_idx] if step_idx < len(raw_seq0) else None
        raw1 = raw_seq1[step_idx] if step_idx < len(raw_seq1) else None

        nom0 = _shadow_action(raw0)
        nom1 = _shadow_action(raw1)
        obs_nom = _step_shadow_env(shadow_nom_env, nom0, nom1, active_arms)
        traj["robot0_nominal"].append(np.asarray(obs_nom["robot0_eef_pos"], dtype=float))
        traj["robot1_nominal"].append(np.asarray(obs_nom["robot1_eef_pos"], dtype=float))

        cor0 = _shadow_action(raw0)
        cor1 = _shadow_action(raw1)
        if raw0 is not None and 0 in safety_arms:
            cor0, _ = cbf_safety.cbf_correct_action(
                shadow_cor_env.env.sim,
                shadow_cor_env.env.robots[0],
                shadow_cor_env.env.robots[1],
                obs_cor["robot0_eef_pos"],
                obs_cor["robot0_eef_quat"],
                obs_cor["robot1_eef_pos"],
                obs_cor["robot1_eef_quat"],
                cor0,
                margin=args.cbf_margin,
                alpha=args.cbf_alpha,
                damping=args.cbf_damping,
                action_scale=args.cbf_scale,
                side_preference=np.array([-1.0, 0.0, 0.0]),
            )
        if raw1 is not None and 1 in safety_arms:
            cor1, _ = cbf_safety.cbf_correct_action(
                shadow_cor_env.env.sim,
                shadow_cor_env.env.robots[1],
                shadow_cor_env.env.robots[0],
                obs_cor["robot1_eef_pos"],
                obs_cor["robot1_eef_quat"],
                obs_cor["robot0_eef_pos"],
                obs_cor["robot0_eef_quat"],
                cor1,
                margin=args.cbf_margin,
                alpha=args.cbf_alpha,
                damping=args.cbf_damping,
                action_scale=args.cbf_scale,
                side_preference=np.array([1.0, 0.0, 0.0]),
            )
        obs_cor = _step_shadow_env(shadow_cor_env, cor0, cor1, active_arms)
        traj["robot0_corrected"].append(np.asarray(obs_cor["robot0_eef_pos"], dtype=float))
        traj["robot1_corrected"].append(np.asarray(obs_cor["robot1_eef_pos"], dtype=float))

    return {k: np.asarray(v, dtype=float) for k, v in traj.items() if len(v) > 0}


def _append_trajectory_point(store: dict[str, list[np.ndarray]], name: str, point: np.ndarray) -> None:
    store[name].append(np.asarray(point, dtype=float).copy())


def _materialize_trajectory_store(store: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) for k, v in store.items() if len(v) > 0}


def _trajectory_window_enabled(args) -> bool:
    return (
        args.trajectory_start_step is not None
        and args.trajectory_end_step is not None
        and int(args.trajectory_start_step) <= int(args.trajectory_end_step)
    )


def _step_in_trajectory_window(step: int, args) -> bool:
    return _trajectory_window_enabled(args) and int(args.trajectory_start_step) <= step <= int(args.trajectory_end_step)


def _draw_dashed_polyline(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray, color) -> np.ndarray:
    out = img
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return out
    for seg_i in range(len(idx) - 1):
        if seg_i % 2 == 1:
            continue
        a = idx[seg_i]
        b = idx[seg_i + 1]
        pa = tuple(np.round(points_2d[a]).astype(np.int32))
        pb = tuple(np.round(points_2d[b]).astype(np.int32))
        cv2.line(out, pa, pb, color, ELLIPSOID_THICKNESS, cv2.LINE_AA)
    return out


def _overlay_trajectories_on_camera_image(
    img: np.ndarray,
    sim,
    camera_name: str,
    trajectories: dict[str, np.ndarray],
    *,
    trajectory_robots: set[int],
    divergence_threshold: float,
) -> np.ndarray:
    out = img.copy()
    if not trajectories:
        return out
    color_map = {
        "robot0_nominal": ROBOT0_NOMINAL_COLOR,
        "robot0_corrected": ROBOT0_CORRECTED_COLOR,
        "robot1_nominal": ROBOT1_NOMINAL_COLOR,
        "robot1_corrected": ROBOT1_CORRECTED_COLOR,
    }
    for ridx in sorted(trajectory_robots):
        nominal_name = f"robot{ridx}_nominal"
        corrected_name = f"robot{ridx}_corrected"
        nominal_world = trajectories.get(nominal_name)
        corrected_world = trajectories.get(corrected_name)

        if nominal_world is not None and len(nominal_world) > 0:
            pts_2d, valid = _project_world_points(sim, camera_name, nominal_world, out.shape)
            out = _draw_dashed_polyline(out, pts_2d, valid, color_map[nominal_name])

        if corrected_world is None or len(corrected_world) == 0:
            continue

        start_idx = 0
        if nominal_world is not None and len(nominal_world) > 0:
            overlap = min(len(nominal_world), len(corrected_world))
            diffs = np.linalg.norm(corrected_world[:overlap] - nominal_world[:overlap], axis=1)
            diverged = np.where(diffs > float(divergence_threshold))[0]
            if len(diverged) == 0:
                continue
            start_idx = int(diverged[0])

        corr_slice = corrected_world[start_idx:]
        if len(corr_slice) < 2:
            continue
        pts_2d, valid = _project_world_points(sim, camera_name, corr_slice, out.shape)
        out = _draw_visible_polyline(out, pts_2d, valid, color_map[corrected_name], closed=False)
    return out


def _overlay_ellipsoids_on_camera_image(img: np.ndarray, sim, env, camera_name: str, obs: dict, intervened_arms) -> np.ndarray:
    out = img.copy()
    semi_axes = np.asarray(cbf_safety.Q_EF, dtype=float)
    try:
        verts_local, normals_local, faces = _ellipsoid_surface_mesh(semi_axes)
        cam_pos, _ = _camera_pose(sim, camera_name)
        for ridx in [0, 1]:
            # Per-arm color: only the arm whose CBF actually intervened this step
            # turns to its intervene palette (robot0=cyan/magenta, robot1=yellow/orange);
            # the other arm stays the base color (red/green).
            if ridx in intervened_arms:
                palette = ELLIPSOID_INTERVENE_COLORS if ridx == 0 else ELLIPSOID_INTERVENE_COLORS_R1
            else:
                palette = ELLIPSOID_COLORS
            targets = cbf_safety.collision_targets(
                sim,
                env.env.robots[ridx],
                ridx,
                obs[f"robot{ridx}_eef_pos"],
                obs[f"robot{ridx}_eef_quat"],
                include_jacobian=False,
            )
            for target in targets:
                color = palette.get(target["name"], ELLIPSOID_COLORS["grip_site"])
                verts_world = (target["rot"] @ verts_local.T).T + target["center"][None, :]
                normals_world = (target["rot"] @ normals_local.T).T
                pts_2d, valid, depth = _project_world_points_with_depth(sim, camera_name, verts_world, out.shape)
                P = pts_2d[valid].astype(np.int32)
                if len(P) >= 3:
                    shaded = out.copy()
                    triangles = []
                    for tri in faces:
                        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
                            continue
                        tri_2d = pts_2d[tri]
                        tri_int = np.round(tri_2d).astype(np.int32)
                        area2 = abs(np.cross(tri_2d[1] - tri_2d[0], tri_2d[2] - tri_2d[0]))
                        if area2 < 1.0:
                            continue
                        center_world = verts_world[tri].mean(axis=0)
                        view_dir = cam_pos - center_world
                        view_norm = float(np.linalg.norm(view_dir))
                        if view_norm <= 1e-9:
                            continue
                        view_dir /= view_norm
                        normal = normals_world[tri].mean(axis=0)
                        normal_norm = float(np.linalg.norm(normal))
                        if normal_norm <= 1e-9:
                            continue
                        normal /= normal_norm
                        facing = float(normal @ view_dir)
                        if facing <= 0.0:
                            continue
                        intensity = np.clip(
                            ELLIPSOID_SURFACE_AMBIENT + ELLIPSOID_SURFACE_DIFFUSE * facing,
                            0.0,
                            1.0,
                        )
                        shade_color = tuple(int(np.clip(ch * intensity, 0, 255)) for ch in color)
                        triangles.append((float(depth[tri].mean()), tri_int.reshape(-1, 1, 2), shade_color))

                    triangles.sort(key=lambda item: item[0], reverse=True)
                    for _, tri_poly, shade_color in triangles:
                        cv2.fillConvexPoly(shaded, tri_poly, shade_color, cv2.LINE_AA)

                    hull = cv2.convexHull(P)
                    cv2.addWeighted(shaded, ELLIPSOID_SURFACE_ALPHA, out, 1.0 - ELLIPSOID_SURFACE_ALPHA, 0, dst=out)
                    cv2.polylines(out, [hull], True, color, 1, cv2.LINE_AA)
    except Exception:
        # Keep the original frame if a camera cannot be projected from.
        return img
    return out


def _concat_render_panels(images: list[np.ndarray]) -> np.ndarray:
    if len(images) == 1:
        return images[0]

    h = min(img.shape[0] for img in images)
    resized = []
    for img in images:
        if img.shape[0] == h:
            resized.append(img)
            continue
        w = int(round(img.shape[1] * (h / img.shape[0])))
        resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8))
    return np.concatenate(resized, axis=1)


def build_render_frame_no_labels(env_obs: dict, cameras: list[str], timestep: int) -> np.ndarray:
    def preprocess_image(img):
        return np.ascontiguousarray(img[::-1]).astype(np.uint8)

    imgs = []
    for camera in cameras:
        key = f"{camera}_image"
        if key not in env_obs:
            raise KeyError(f"Missing {key}. Available keys: {sorted(env_obs.keys())}")
        imgs.append(preprocess_image(env_obs[key]))
    return _append_timestep_bar(_concat_render_panels(imgs), timestep)


def _render_one_overlay_camera(env_obs: dict, sim, env, camera_name: str, intervened_arms, *,
                               success_z_pad: float,
                               future_trajectories: dict[str, np.ndarray] | None = None,
                               history_trajectories: dict[str, np.ndarray] | None = None,
                               trajectory_robots: set[int] | None = None,
                               divergence_threshold: float = TRAJECTORY_DIVERGENCE_THRESHOLD,
                               draw_ellipsoids: bool = True) -> np.ndarray:
    def preprocess_image(img):
        return np.ascontiguousarray(img[::-1]).astype(np.uint8)

    key = f"{camera_name}_image"
    if key not in env_obs:
        raise KeyError(f"Missing {key}. Available keys: {sorted(env_obs.keys())}")
    frame = preprocess_image(env_obs[key])
    if draw_ellipsoids:
        frame = _overlay_ellipsoids_on_camera_image(frame, sim, env, camera_name, env_obs, intervened_arms)
    frame = _overlay_trajectories_on_camera_image(
        frame, sim, camera_name, history_trajectories or {},
        trajectory_robots=set(trajectory_robots or {0, 1}),
        divergence_threshold=divergence_threshold,
    )
    frame = _overlay_trajectories_on_camera_image(
        frame, sim, camera_name, future_trajectories or {},
        trajectory_robots=set(trajectory_robots or {0, 1}),
        divergence_threshold=divergence_threshold,
    )
    return frame


def build_render_frame_with_ellipsoids(env_obs: dict, sim, env, cameras: list[str], intervened_arms, *,
                                       timestep: int,
                                       success_z_pad: float,
                                       future_trajectories: dict[str, np.ndarray] | None = None,
                                       history_trajectories: dict[str, np.ndarray] | None = None,
                                       trajectory_robots: set[int] | None = None,
                                       divergence_threshold: float = TRAJECTORY_DIVERGENCE_THRESHOLD,
                                       draw_ellipsoids: bool = True) -> np.ndarray:
    imgs = [
        _render_one_overlay_camera(
            env_obs, sim, env, camera_name, intervened_arms,
            success_z_pad=success_z_pad,
            future_trajectories=future_trajectories,
            history_trajectories=history_trajectories,
            trajectory_robots=trajectory_robots,
            divergence_threshold=divergence_threshold,
            draw_ellipsoids=draw_ellipsoids,
        )
        for camera_name in cameras
    ]
    return _append_timestep_bar(_concat_render_panels(imgs), timestep)


def _object_inside_trash_box(sim, env, object_name: str, *, z_pad: float) -> bool:
    """Return True if the named object lies inside the full trash-can inner box."""
    import create_dataset_grasp as create_dataset

    trash_obj = env.env.objects_dict["trash_can_1"]
    obj = env.env.objects_dict[object_name]

    trash_body_id = sim.model.body_name2id(trash_obj.root_body)
    obj_body_id = sim.model.body_name2id(obj.root_body)

    trash_pos = np.asarray(sim.data.body_xpos[trash_body_id], dtype=float)
    trash_rot = np.asarray(sim.data.body_xmat[trash_body_id], dtype=float).reshape(3, 3)
    obj_pos = np.asarray(sim.data.body_xpos[obj_body_id], dtype=float)

    local = trash_rot.T @ (obj_pos - trash_pos)
    x_half = float(create_dataset.TRASH_INNER_WIDTH) * 0.5
    y_half = float(create_dataset.TRASH_INNER_WIDTH) * 0.5
    z_min = float(create_dataset.TRASH_BOTTOM_THICK) - z_pad
    z_max = float(create_dataset.TRASH_BOTTOM_THICK + create_dataset.TRASH_INNER_DEPTH) + z_pad

    return (
        -x_half <= local[0] <= x_half and
        -y_half <= local[1] <= y_half and
        z_min <= local[2] <= z_max
    )


def evaluate_custom_success(sim, env, *, z_pad: float) -> dict:
    """Success for minje experiments: both milk and orange juice inside the black box."""
    milk_inside = _object_inside_trash_box(sim, env, "milk_1", z_pad=z_pad)
    orange_inside = _object_inside_trash_box(sim, env, "orange_juice_1", z_pad=z_pad)
    return {
        "success": bool(milk_inside and orange_inside),
        "milk_inside": bool(milk_inside),
        "orange_inside": bool(orange_inside),
    }


def object_placement(sim, env, object_name: str) -> dict:
    """Final object placement vs the trash box: world height z and horizontal
    distance from box center (in the box-local frame). Lets us quantify the
    'avoided but dropped on the floor / off-target' failures."""
    trash_obj = env.env.objects_dict["trash_can_1"]
    obj = env.env.objects_dict[object_name]
    tb = sim.model.body_name2id(trash_obj.root_body)
    ob = sim.model.body_name2id(obj.root_body)
    tp = np.asarray(sim.data.body_xpos[tb], dtype=float)
    tr = np.asarray(sim.data.body_xmat[tb], dtype=float).reshape(3, 3)
    op = np.asarray(sim.data.body_xpos[ob], dtype=float)
    local = tr.T @ (op - tp)
    return {
        "z": round(float(op[2]), 3),                              # world height
        "dist_box": round(float(np.hypot(local[0], local[1])), 3),  # horiz. dist from box center
        "local_z": round(float(local[2]), 3),                      # height above box bottom
    }


# --------------------------------------------------------------------------- #
# Single-episode rollout (mirrors test_model_grasp.replay_episode, dual mode,
# without video; adds contact tracking + returns metrics).
# --------------------------------------------------------------------------- #
def run_episode(args, seed: int, policy0, policy1, video_path: Path | None = None,
                recorder=None, record_task: str | None = None, *,
                nominal_policy0=None, nominal_policy1=None,
                override_safety_arms: set[int] | None = None,
                collect_history_only: bool = False) -> dict:
    import create_dataset_grasp as create_dataset
    import create_dataset_robot1_grasp as create_dataset_robot1

    rollout_steps = args.max_steps
    safety_arms = set(args.safety_arms or []) if override_safety_arms is None else set(override_safety_arms)
    active_arms = set(args.active_arms if args.active_arms is not None else [0, 1])
    spec = create_dataset.build_episode_spec(seed)
    env = None
    shadow_nom_env = None
    shadow_cor_env = None
    t0 = time.time()
    frames = []
    overlay_frames = []
    n_intervene = 0
    min_h = float("inf")
    intervened_arms = set()
    nominal_history_trajectories = {}
    history_trajectory_store = {
        "robot0_nominal": [],
        "robot1_nominal": [],
        "robot0_corrected": [],
        "robot1_corrected": [],
    }

    def apply_cbf(action, self_idx, other_idx):
        """Wrap a policy action with the dual-arm CBF safety correction."""
        nonlocal n_intervene, min_h
        if self_idx not in safety_arms:
            return action
        side_preference = np.array([-1.0, 0.0, 0.0]) if self_idx == 0 else np.array([1.0, 0.0, 0.0])
        out, info = cbf_safety.cbf_correct_action(
            env.env.sim, env.env.robots[self_idx], env.env.robots[other_idx],
            obs[f"robot{self_idx}_eef_pos"], obs[f"robot{self_idx}_eef_quat"],
            obs[f"robot{other_idx}_eef_pos"], obs[f"robot{other_idx}_eef_quat"],
            action,
            margin=args.cbf_margin, alpha=args.cbf_alpha,
            damping=args.cbf_damping, action_scale=args.cbf_scale,
            side_preference=side_preference,
        )
        min_h = min(min_h, info["h"])
        if info["intervened"]:
            n_intervene += 1
            intervened_arms.add(self_idx)
        return out
    # Video export is fixed to sideview + birdview.
    render_cameras = list(VIDEO_RENDER_CAMERAS) if video_path is not None else []
    if render_cameras and _trajectory_window_enabled(args) and not collect_history_only:
        ref = run_episode(
            args,
            seed,
            nominal_policy0,
            nominal_policy1,
            video_path=None,
            recorder=None,
            record_task=None,
            nominal_policy0=None,
            nominal_policy1=None,
            override_safety_arms=set(),
            collect_history_only=True,
        )
        nominal_history_trajectories = ref.get("history_trajectories", {})
    try:
        # Register the 6 policy cameras (robot0/robot1 side_left/wrist/side_right)
        # so build_policy_observation can find them, matching test_model_grasp.
        # For video export, only the requested render cameras are added.
        cam_select = "sideview"
        create_dataset.DATASET_CAMERAS = list(dict.fromkeys([*tmg.camera_names_for_env(cam_select), *render_cameras]))
        create_dataset.DEBUG_CAMERAS = []
        env, obs = create_dataset.make_env(spec, args.resolution, save_debug_video=False)
        create_dataset.set_free_joint_xy(
            env.env.sim, "trash_can_1_joint0",
            create_dataset.TRASH_CAN_CENTER[0], create_dataset.TRASH_CAN_CENTER[1],
        )
        _lift_sideview_camera(env.env.sim)
        env.env.sim.forward()
        obs = env.env._get_observations()

        if render_cameras and args.trajectory_horizon > 0:
            prev_cameras = list(create_dataset.DATASET_CAMERAS)
            prev_debug = list(create_dataset.DEBUG_CAMERAS)
            try:
                create_dataset.DATASET_CAMERAS = ["sideview"]
                create_dataset.DEBUG_CAMERAS = []
                shadow_nom_env, _ = create_dataset.make_env(
                    spec,
                    min(64, args.resolution),
                    save_debug_video=False,
                )
                shadow_cor_env, _ = create_dataset.make_env(
                    spec,
                    min(64, args.resolution),
                    save_debug_video=False,
                )
                _lift_sideview_camera(shadow_nom_env.env.sim)
                _lift_sideview_camera(shadow_cor_env.env.sim)
                shadow_nom_env.env.ignore_done = True
                shadow_cor_env.env.ignore_done = True
            finally:
                create_dataset.DATASET_CAMERAS = prev_cameras
                create_dataset.DEBUG_CAMERAS = prev_debug

        if policy0 is not None:
            policy0.reset()
        if policy1 is not None:
            policy1.reset()

        # Per-episode RNG seeding: SmolVLA action generation (flow-matching) draws
        # from the global torch RNG, which otherwise advances across episodes and
        # makes batch results differ from fresh single-episode runs. Reseeding here
        # makes each (seed) episode reproducible and batch == sum of singles.
        _random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Both arms: scripted grasp -> policy handoff (dual-policy / VLSA-0 baseline).
        waypoints0 = create_dataset.solve_waypoints(env, obs)
        waypoints1 = create_dataset_robot1.solve_waypoints(env, obs)
        move_sequence0 = [
            ("move_pregrasp", waypoints0["pregrasp"], create_dataset.GRIPPER_OPEN),
            ("move_grasp", waypoints0["grasp"], create_dataset.GRIPPER_OPEN),
        ]
        move_sequence1 = [
            ("move_pregrasp", waypoints1["pregrasp"], create_dataset_robot1.GRIPPER_OPEN),
            ("move_grasp", waypoints1["grasp"], create_dataset_robot1.GRIPPER_OPEN),
        ]
        phase0 = phase1 = "open_then_approach"
        phase_steps0 = phase_steps1 = 0
        close_hold_steps0 = close_hold_steps1 = 0
        move_index0 = move_index1 = 0
        grasp_done0 = grasp_done1 = False
        policy_active0 = policy_active1 = False
        robot0_start_delay_steps = max(0, int(round(args.robot0_start_delay_sec * args.fps)))
        robot1_start_delay_steps = max(0, int(round(args.robot1_start_delay_sec * args.fps)))
        robot0_started = robot0_start_delay_steps == 0
        robot1_started = robot1_start_delay_steps == 0

        collided = False
        collision_step = -1
        collision_geoms = None
        grasp0_step = -1
        grasp1_step = -1
        actual_steps = 0

        for step in range(rollout_steps):
            actual_steps = step + 1
            cur0 = obs["robot0_joint_pos"]
            cur1 = create_dataset_robot1.get_robot_joint_pos(env, 1)

            robot0_action = tmg.FIXED_ACTION.copy()
            robot1_action = tmg.FIXED_ACTION.copy()
            raw_robot0_action = None
            raw_robot1_action = None
            intervened_arms = set()

            script_robot0 = not policy_active0
            script_robot1 = not policy_active1

            if script_robot0:
                if not robot0_started:
                    robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_OPEN)
                elif not grasp_done0:
                    if phase0 == "open_then_approach":
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_OPEN)
                        if phase_steps0 > create_dataset.OPEN_GRIPPER_INIT_STEPS:
                            move_index0 = 0
                            phase0 = move_sequence0[move_index0][0]
                            phase_steps0 = 0
                    elif phase0 in [name for name, _, _ in move_sequence0]:
                        _, target_qpos0, gripper_cmd0 = move_sequence0[move_index0]
                        robot0_action = create_dataset.make_joint_position_action(target_qpos0, cur0, gripper_cmd=gripper_cmd0)
                        tol0 = create_dataset.PHASE_TOL_OVERRIDE.get(phase0, create_dataset.JOINT_TOL)
                        timeout0 = create_dataset.PHASE_TIMEOUT_OVERRIDE.get(phase0, create_dataset.PHASE_TIMEOUT_STEPS)
                        timed_out0 = timeout0 is not None and phase_steps0 >= timeout0
                        if create_dataset.joint_distance_to(obs, target_qpos0) < tol0 or timed_out0:
                            move_index0 += 1
                            phase0 = "close_gripper" if move_index0 >= len(move_sequence0) else move_sequence0[move_index0][0]
                            phase_steps0 = 0
                    elif phase0 == "close_gripper":
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
                        close_hold_steps0 += 1
                        if close_hold_steps0 >= create_dataset.GRIP_CLOSE_HOLD_STEPS:
                            phase0 = "grasp_hold"
                            grasp_done0 = True
                            grasp0_step = step
                    else:
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
                else:
                    robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
            elif policy_active0:
                raw_robot0_action = tmg.select_robot_action(policy0, env, obs, 0, args.robot0_task)
                robot0_action = apply_cbf(raw_robot0_action.copy(), 0, 1)

            if script_robot1:
                if not robot1_started:
                    robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_OPEN)
                elif not grasp_done1:
                    if phase1 == "open_then_approach":
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_OPEN)
                        if phase_steps1 > create_dataset_robot1.OPEN_GRIPPER_INIT_STEPS:
                            move_index1 = 0
                            phase1 = move_sequence1[move_index1][0]
                            phase_steps1 = 0
                    elif phase1 in [name for name, _, _ in move_sequence1]:
                        _, target_qpos1, gripper_cmd1 = move_sequence1[move_index1]
                        robot1_action = create_dataset_robot1.make_joint_position_action(target_qpos1, cur1, gripper_cmd=gripper_cmd1)
                        tol1 = create_dataset_robot1.PHASE_TOL_OVERRIDE.get(phase1, create_dataset_robot1.JOINT_TOL)
                        timeout1 = create_dataset_robot1.PHASE_TIMEOUT_OVERRIDE.get(phase1, create_dataset_robot1.PHASE_TIMEOUT_STEPS)
                        timed_out1 = timeout1 is not None and phase_steps1 >= timeout1
                        if create_dataset_robot1.joint_distance_to(env, target_qpos1) < tol1 or timed_out1:
                            move_index1 += 1
                            phase1 = "close_gripper" if move_index1 >= len(move_sequence1) else move_sequence1[move_index1][0]
                            phase_steps1 = 0
                    elif phase1 == "close_gripper":
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
                        close_hold_steps1 += 1
                        if close_hold_steps1 >= create_dataset_robot1.GRIP_CLOSE_HOLD_STEPS:
                            phase1 = "grasp_hold"
                            grasp_done1 = True
                            grasp1_step = step
                    else:
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
                else:
                    robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
            elif policy_active1:
                raw_robot1_action = tmg.select_robot_action(policy1, env, obs, 1, args.robot1_task)
                robot1_action = apply_cbf(raw_robot1_action.copy(), 1, 0)

            # Park inactive arms in their passive pose (single-arm ceiling runs).
            if 0 not in active_arms:
                robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_OPEN)
            if 1 not in active_arms:
                robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_OPEN)

            if render_cameras or collect_history_only:
                if _step_in_trajectory_window(step, args):
                    if collect_history_only:
                        _append_trajectory_point(history_trajectory_store, "robot0_nominal", obs["robot0_eef_pos"])
                        _append_trajectory_point(history_trajectory_store, "robot1_nominal", obs["robot1_eef_pos"])
                    else:
                        _append_trajectory_point(history_trajectory_store, "robot0_corrected", obs["robot0_eef_pos"])
                        _append_trajectory_point(history_trajectory_store, "robot1_corrected", obs["robot1_eef_pos"])

                raw_seq0 = []
                raw_seq1 = []
                future_trajectories = {}
                if args.trajectory_horizon > 0:
                    if policy_active0 and policy0 is not None:
                        raw_seq0 = [raw_robot0_action.copy(), *_extract_policy_queue_actions(policy0)]
                    if policy_active1 and policy1 is not None:
                        raw_seq1 = [raw_robot1_action.copy(), *_extract_policy_queue_actions(policy1)]
                    future_trajectories = predict_future_eef_trajectories(
                        args, env, raw_seq0, raw_seq1, safety_arms, active_arms,
                        shadow_nom_env, shadow_cor_env)
                if render_cameras:
                    frames.append(build_render_frame_no_labels(obs, render_cameras, step))
                    history_overlay = dict(nominal_history_trajectories)
                    history_overlay.update(_materialize_trajectory_store(history_trajectory_store))
                    overlay_frames.append(build_render_frame_with_ellipsoids(
                        obs, env.env.sim, env, render_cameras, intervened_arms,
                        timestep=step,
                        success_z_pad=args.success_z_pad,
                        future_trajectories=future_trajectories,
                        history_trajectories=history_overlay,
                        trajectory_robots=set(args.trajectory_robots),
                        divergence_threshold=args.trajectory_divergence_threshold,
                        draw_ellipsoids=args.draw_ellipsoids))

            # Record robot0's trajectory in LeRobot format (for the LeRobot viewer).
            if recorder is not None:
                def _pp(img):
                    return np.ascontiguousarray(np.asarray(img)[::-1]).astype(np.uint8)
                recorder.add_frame({
                    "observation.images.side_left": _pp(obs["sideview_robot0_left_image"]),
                    "observation.images.wrist": _pp(obs["robot0_eye_in_hand_image"]),
                    "observation.images.side_right": _pp(obs["sideview_robot0_right_image"]),
                    "observation.state": tmg.extract_state(env, obs, 0, 8),
                    "action": tmg.normalize_robot_action(robot0_action, "robot0"),
                    "task": record_task or "put the milk into the target area inside the trash can",
                })

            obs, _, env_done, _ = env.step(tmg.make_dual_action(robot0_action, robot1_action))
            if 1 not in active_arms:
                create_dataset.lock_robot1_pose(env.env.sim, env.env.robots[1])
            if 0 not in active_arms:
                create_dataset_robot1.lock_robot0_pose(env.env.sim, env.env.robots[0])
            obs = env.env._get_observations()

            # ---- collision check (robot0 <-> robot1 mujoco contact) ----
            if not collided:
                pair = arm_arm_contact(env.env.sim)
                if pair is not None:
                    collided = True
                    collision_step = step
                    collision_geoms = pair

            # delay / phase bookkeeping
            if not robot0_started:
                robot0_start_delay_steps -= 1
                robot0_started = robot0_start_delay_steps <= 0
            elif script_robot0:
                phase_steps0 += 1
            if not robot1_started:
                robot1_start_delay_steps -= 1
                robot1_started = robot1_start_delay_steps <= 0
            elif script_robot1:
                phase_steps1 += 1

            if policy0 is not None and grasp_done0 and not policy_active0:
                policy_active0 = True
            if policy1 is not None and grasp_done1 and not policy_active1:
                policy_active1 = True

        if recorder is not None:
            recorder.save_episode()
        if collect_history_only:
            return {"history_trajectories": _materialize_trajectory_store(history_trajectory_store)}
        success_info = evaluate_custom_success(env.env.sim, env, z_pad=args.success_z_pad)
        success = success_info["success"]
        _milk_place = object_placement(env.env.sim, env, "milk_1")
        _orange_place = object_placement(env.env.sim, env, "orange_juice_1")
        if video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(video_path), frames, fps=args.fps, codec="libx264")
            overlay_tag = "ellipsoid" if args.draw_ellipsoids else "trajectory"
            overlay_path = video_path.with_name(f"{video_path.stem}_{overlay_tag}{video_path.suffix}")
            if overlay_frames:
                imageio.mimwrite(str(overlay_path), overlay_frames, fps=args.fps, codec="libx264")
        return {
            "seed": seed,
            "success": success,
            "collided": collided,
            "safe_success": success and not collided,
            "collision_step": collision_step,
            "collision_geoms": "" if collision_geoms is None else f"{collision_geoms[0]} | {collision_geoms[1]}",
            "grasp0_step": grasp0_step,
            "grasp1_step": grasp1_step,
            "steps": actual_steps,
            "n_intervene": n_intervene,
            "min_h": None if min_h == float("inf") else round(min_h, 4),
            "wall_sec": round(time.time() - t0, 1),
            "milk_inside": success_info["milk_inside"],
            "orange_inside": success_info["orange_inside"],
            "milk_z": _milk_place["z"], "milk_dist_box": _milk_place["dist_box"], "milk_local_z": _milk_place["local_z"],
            "orange_z": _orange_place["z"], "orange_dist_box": _orange_place["dist_box"], "orange_local_z": _orange_place["local_z"],
            **({"video_overlay": str(overlay_path)} if video_path is not None and overlay_frames else {}),
        }
    finally:
        if shadow_nom_env is not None:
            shadow_nom_env.close()
        if shadow_cor_env is not None:
            shadow_cor_env.close()
        if env is not None:
            env.close()
        spec.bddl_file.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--robot0-policy-path", required=True)
    p.add_argument("--robot1-policy-path", required=True)
    p.add_argument("--robot0-task", default="put the milk into the target area inside the trash can")
    p.add_argument("--robot1-task", default="put the distractor orange juice into the target area inside the trash can")
    p.add_argument("--seed-start", type=int, default=1)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit seed list (overrides seed-start/num-episodes)")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--robot0-start-delay-sec", type=float, default=0.0)
    p.add_argument("--robot1-start-delay-sec", type=float, default=0.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-csv", type=Path, default=Path(__file__).with_name("results") / "baseline.csv")
    p.add_argument("--out-json", type=Path, default=Path(__file__).with_name("results") / "baseline.json")
    p.add_argument("--label", default="VLSA0_baseline_v4", help="label for this run (stored in json)")
    # --- dual-arm CBF safety layer ---
    p.add_argument("--safety-arms", type=int, nargs="*", default=None,
                   help="which arms get the CBF safety layer: none / 0 / 1 / 0 1 (VLSA 0/1/2)")
    p.add_argument("--active-arms", type=int, nargs="*", default=None,
                   help="which arms are active; others are locked in passive pose (default: 0 1)")
    p.add_argument("--cbf-margin", type=float, default=0.04, help="EE ellipsoid inflation margin (m)")
    p.add_argument("--cbf-alpha", type=float, default=0.5, help="discrete CBF decay (0,1]")
    p.add_argument("--cbf-scale", type=float, default=0.166, help="action->EE displacement scale (calibrated)")
    p.add_argument("--cbf-damping", type=float, default=0.05, help="damped pseudo-inverse lambda")
    p.add_argument("--success-z-pad", type=float, default=0.05,
                   help="extra z tolerance (m) when checking whether objects are inside the black box")
    p.add_argument("--policy-replan-horizon", type=int, default=50,
                   help="number of policy actions consumed before forcing a fresh SmolVLA replan")
    p.add_argument("--trajectory-horizon", type=int, default=0,
                   help="how many future policy actions to visualize on the ellipsoid overlay video (default: disabled)")
    p.add_argument("--trajectory-start-step", type=int, default=TRAJECTORY_DRAW_START,
                   help="inclusive start step for executed trajectory overlay (default: disabled)")
    p.add_argument("--trajectory-end-step", type=int, default=TRAJECTORY_DRAW_END,
                   help="inclusive end step for executed trajectory overlay (default: disabled)")
    p.add_argument("--trajectory-robots", type=int, nargs="*", default=[0, 1],
                   help="which robots to draw in the trajectory overlay")
    p.add_argument("--trajectory-divergence-threshold", type=float, default=TRAJECTORY_DIVERGENCE_THRESHOLD,
                   help="show corrected trajectory only after it separates from nominal by this many meters")
    p.add_argument("--draw-ellipsoids", action=argparse.BooleanOptionalAction, default=True,
                   help="whether to draw ellipsoids in the overlay video")
    p.add_argument("--video-seeds", type=int, nargs="*", default=None, help="seeds to also render to mp4")
    p.add_argument("--video-camera", default="sideview",
                   help="deprecated; videos are rendered with sideview and birdview")
    p.add_argument("--video-dir", type=Path, default=Path(__file__).with_name("results") / "videos")
    # --- record rollouts as a LeRobot dataset (for the LeRobot dataset viewer) ---
    p.add_argument("--record-lerobot", type=Path, default=None,
                   help="if set, record robot0's trajectory of every episode into a LeRobot dataset at this dir")
    p.add_argument("--record-repo-id", default="morealcholplz/vlsa-dual-rollouts")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.num_episodes))
    device = torch.device(args.device)

    print(f"[setup] loading policies on {device} ...", flush=True)
    policy0, _ = tmg.load_policy(args.robot0_policy_path, device, "robot0")
    policy1, _ = tmg.load_policy(args.robot1_policy_path, device, "robot1")
    nominal_policy0 = None
    nominal_policy1 = None
    if _trajectory_window_enabled(args):
        print("[setup] loading nominal reference policies ...", flush=True)
        nominal_policy0, _ = tmg.load_policy(args.robot0_policy_path, device, "robot0_nominal")
        nominal_policy1, _ = tmg.load_policy(args.robot1_policy_path, device, "robot1_nominal")
    policy0_horizon = configure_policy_replan(policy0, args.policy_replan_horizon)
    policy1_horizon = configure_policy_replan(policy1, args.policy_replan_horizon)
    if nominal_policy0 is not None:
        configure_policy_replan(nominal_policy0, args.policy_replan_horizon)
    if nominal_policy1 is not None:
        configure_policy_replan(nominal_policy1, args.policy_replan_horizon)
    print(f"[setup] policy_replan_horizon="
          f"{policy0_horizon if policy0_horizon is not None else '-'} / "
          f"{policy1_horizon if policy1_horizon is not None else '-'}", flush=True)

    recorder = None
    if args.record_lerobot is not None:
        import shutil
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if args.record_lerobot.exists():
            shutil.rmtree(args.record_lerobot)
        img_feat = lambda: {"dtype": "video", "shape": (args.resolution, args.resolution, 3),
                            "names": ["height", "width", "channels"]}
        names = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]
        features = {
            "observation.images.side_left": img_feat(),
            "observation.images.wrist": img_feat(),
            "observation.images.side_right": img_feat(),
            "observation.state": {"dtype": "float32", "shape": (8,), "names": names},
            "action": {"dtype": "float32", "shape": (8,), "names": names},
        }
        recorder = LeRobotDataset.create(
            repo_id=args.record_repo_id, root=str(args.record_lerobot),
            robot_type="dual_panda_robot0", fps=args.fps, features=features,
            use_videos=True, vcodec="libx264")

    video_seeds = set(args.video_seeds or [])
    rows = []
    for i, seed in enumerate(seeds):
        vpath = (args.video_dir / f"{args.label}_seed{seed}.mp4") if seed in video_seeds else None
        r = run_episode(args, seed, policy0, policy1, video_path=vpath,
                        recorder=recorder, record_task=f"{args.label}: put the milk into the trash can",
                        nominal_policy0=nominal_policy0, nominal_policy1=nominal_policy1)
        if vpath is not None:
            r["video"] = str(vpath)
        rows.append(r)
        print(f"[{i+1}/{len(seeds)}] seed={seed} success={r['success']} "
              f"collided={r['collided']} safe={r['safe_success']} "
              f"coll_step={r['collision_step']} intervene={r['n_intervene']} "
              f"min_h={r['min_h']} steps={r['steps']} ({r['wall_sec']}s)", flush=True)

    if recorder is not None:
        recorder.finalize()
        print(f"[lerobot] dataset saved to {args.record_lerobot}")

    n = len(rows)
    tsr = sum(r["success"] for r in rows) / n
    car = sum(not r["collided"] for r in rows) / n
    safe = sum(r["safe_success"] for r in rows) / n
    summary = {
        "label": args.label,
        "safety_arms": sorted(set(args.safety_arms or [])),
        "cbf": {"margin": args.cbf_margin, "alpha": args.cbf_alpha,
                "scale": args.cbf_scale, "damping": args.cbf_damping},
        "success_criterion": {
            "type": "both_objects_inside_black_box",
            "objects": ["milk_1", "orange_juice_1"],
            "z_pad": args.success_z_pad,
        },
        "policy_replan_horizon": {
            "requested": args.policy_replan_horizon,
            "robot0": policy0_horizon,
            "robot1": policy1_horizon,
        },
        "robot0_policy": str(args.robot0_policy_path),
        "robot1_policy": str(args.robot1_policy_path),
        "num_episodes": n,
        "seeds": seeds,
        "max_steps": args.max_steps,
        "TSR": round(tsr, 4),
        "CAR": round(car, 4),
        "safe_success_rate": round(safe, 4),
        "n_success": sum(r["success"] for r in rows),
        "n_collided": sum(r["collided"] for r in rows),
        "episodes": rows,
    }

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [k for k in rows[0].keys() if k != "video"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with args.out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== SUMMARY ==========")
    print(f"label   : {args.label}")
    print(f"episodes: {n}  (seeds {seeds[0]}..{seeds[-1]})")
    print(f"TSR (task success rate)      : {tsr*100:.2f}%  ({summary['n_success']}/{n})")
    print(f"CAR (collision avoidance)    : {car*100:.2f}%  ({n-summary['n_collided']}/{n})")
    print(f"safe-success (succ & no coll): {safe*100:.2f}%")
    print(f"csv : {args.out_csv}")
    print(f"json: {args.out_json}")


if __name__ == "__main__":
    main()
