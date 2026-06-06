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

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "Train/lerobot/src"))
sys.path.insert(0, str(BASE_DIR / "Data generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for importing test_model_grasp

# Stable, side-effect-free helpers reused from the existing harness.
import test_model_grasp as tmg  # noqa: E402
import minje_cbf_safety as cbf_safety  # noqa: E402

ELLIPSOID_COLORS = {
    "grip_site": (255, 0, 0),
    "link7": (0, 255, 0),
}
ELLIPSOID_INTERVENE_COLORS = {
    "grip_site": (0, 255, 255),
    "link7": (255, 0, 255),
}
ELLIPSOID_THICKNESS = 2
SUCCESS_BOX_COLOR = (0, 255, 0)
ROBOT0_NOMINAL_COLOR = (255, 200, 0)
ROBOT0_CORRECTED_COLOR = (255, 0, 0)
ROBOT1_NOMINAL_COLOR = (180, 180, 255)
ROBOT1_CORRECTED_COLOR = (0, 0, 255)


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


def _project_world_points(sim, camera_name: str, points_world: np.ndarray, image_shape) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame 3D points into pixel coordinates for a MuJoCo camera."""
    model = sim.model
    data = sim.data
    cam_id = _camera_name2id(model, camera_name)
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=float)
    cam_rot = np.asarray(data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
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


def _overlay_future_trajectories_on_camera_image(
    img: np.ndarray,
    sim,
    camera_name: str,
    trajectories: dict[str, np.ndarray],
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
    for name, pts_world in trajectories.items():
        if len(pts_world) == 0:
            continue
        pts_2d, valid = _project_world_points(sim, camera_name, pts_world, out.shape)
        if name.endswith("nominal"):
            out = _draw_projected_dots(out, pts_2d, valid, color_map[name], radius=2)
        else:
            out = _draw_visible_polyline(out, pts_2d, valid, color_map[name], closed=False)
    return out


def _overlay_ellipsoids_on_camera_image(img: np.ndarray, sim, env, camera_name: str, obs: dict, intervened: bool) -> np.ndarray:
    out = img.copy()
    semi_axes = np.asarray(cbf_safety.Q_EF, dtype=float)
    try:
        palette = ELLIPSOID_INTERVENE_COLORS if intervened else ELLIPSOID_COLORS
        for ridx in [0, 1]:
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
                # Filled, semi-transparent ellipsoid: project the surface points,
                # fill their 2D convex-hull silhouette with alpha, keep a thin edge.
                curves = _ellipsoid_curves_from_rot(target["center"], target["rot"], semi_axes)
                allpts = np.concatenate(curves, axis=0)
                pts_2d, valid = _project_world_points(sim, camera_name, allpts, out.shape)
                P = pts_2d[valid].astype(np.int32)
                if len(P) >= 3:
                    hull = cv2.convexHull(P)
                    fill = out.copy()
                    cv2.fillConvexPoly(fill, hull, color)
                    cv2.addWeighted(fill, 0.38, out, 0.62, 0, dst=out)
                    cv2.polylines(out, [hull], True, color, 1, cv2.LINE_AA)
    except Exception:
        # Keep the original frame if a camera cannot be projected from.
        return img
    return out


def build_render_frame_with_ellipsoids(env_obs: dict, sim, env, camera: str, active_policy_robots: set[int],
                                       intervened: bool, *, success_z_pad: float,
                                       future_trajectories: dict[str, np.ndarray] | None = None) -> np.ndarray:
    def preprocess_image(img):
        return np.ascontiguousarray(img[::-1]).astype(np.uint8)

    def render_one(camera_name: str) -> np.ndarray:
        key = f"{camera_name}_image"
        if key not in env_obs:
            raise KeyError(f"Missing {key}. Available keys: {sorted(env_obs.keys())}")
        frame = preprocess_image(env_obs[key])
        frame = _overlay_ellipsoids_on_camera_image(frame, sim, env, camera_name, env_obs, intervened)
        frame = _overlay_future_trajectories_on_camera_image(
            frame, sim, camera_name, future_trajectories or {})
        try:
            box_corners = _trash_success_box_world_corners(sim, env, z_pad=success_z_pad)
            box_pts_2d, box_valid = _project_world_points(sim, camera_name, box_corners, frame.shape)
            box_edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            frame = _draw_projected_segments(frame, box_pts_2d, box_valid, box_edges, SUCCESS_BOX_COLOR)
        except Exception:
            pass
        return tmg.draw_camera_labels(frame, camera_name, active_policy_robots)

    if camera != "all":
        return render_one(camera)

    preferred = [f"{name}_image" for name in [*tmg.POLICY_CAMERAS, *tmg.RENDER_CAMERAS]]
    imgs = []
    seen = set()
    for key in preferred:
        if key in seen:
            continue
        seen.add(key)
        if key not in env_obs:
            continue
        imgs.append(render_one(key.removesuffix("_image")))
    if not imgs:
        available = sorted([k for k in env_obs.keys() if k.endswith("_image")])
        raise KeyError(f"No '*_image' keys for rendering. Available image keys: {available}")

    h = min(img.shape[0] for img in imgs)
    resized = []
    for img in imgs:
        if img.shape[0] == h:
            resized.append(img)
            continue
        w = int(round(img.shape[1] * (h / img.shape[0])))
        resized.append(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.uint8))

    cols = 3
    rows = (len(resized) + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        chunk = resized[r * cols : (r + 1) * cols]
        if len(chunk) < cols:
            pad = np.zeros_like(chunk[0])
            chunk = chunk + [pad] * (cols - len(chunk))
        row_imgs.append(np.concatenate(chunk, axis=1))
    return np.concatenate(row_imgs, axis=0)


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


# --------------------------------------------------------------------------- #
# Single-episode rollout (mirrors test_model_grasp.replay_episode, dual mode,
# without video; adds contact tracking + returns metrics).
# --------------------------------------------------------------------------- #
def run_episode(args, seed: int, policy0, policy1, video_path: Path | None = None,
                recorder=None, record_task: str | None = None) -> dict:
    import create_dataset_grasp as create_dataset
    import create_dataset_robot1_grasp as create_dataset_robot1

    rollout_steps = args.max_steps
    safety_arms = set(args.safety_arms or [])
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
    step_intervened = False

    def apply_cbf(action, self_idx, other_idx):
        """Wrap a policy action with the dual-arm CBF safety correction."""
        nonlocal n_intervene, min_h, step_intervened
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
            step_intervened = True
        return out
    # For video we also render birdview/backview (top-down clearly shows arm-arm
    # contact); metrics-only runs use the minimal policy-camera set for speed.
    render_camera = args.video_camera if video_path is not None else None
    try:
        # Register the 6 policy cameras (robot0/robot1 side_left/wrist/side_right)
        # so build_policy_observation can find them, matching test_model_grasp.
        cam_select = "all" if video_path is not None else "sideview"
        create_dataset.DATASET_CAMERAS = tmg.camera_names_for_env(cam_select)
        create_dataset.DEBUG_CAMERAS = []
        env, obs = create_dataset.make_env(spec, args.resolution, save_debug_video=False)
        create_dataset.set_free_joint_xy(
            env.env.sim, "trash_can_1_joint0",
            create_dataset.TRASH_CAN_CENTER[0], create_dataset.TRASH_CAN_CENTER[1],
        )
        env.env.sim.forward()
        obs = env.env._get_observations()

        if render_camera is not None and args.trajectory_horizon > 0:
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
            step_intervened = False

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

            if render_camera is not None:
                raw_seq0 = []
                raw_seq1 = []
                if policy_active0 and policy0 is not None:
                    raw_seq0 = [raw_robot0_action.copy(), *_extract_policy_queue_actions(policy0)]
                if policy_active1 and policy1 is not None:
                    raw_seq1 = [raw_robot1_action.copy(), *_extract_policy_queue_actions(policy1)]
                future_trajectories = predict_future_eef_trajectories(
                    args, env, raw_seq0, raw_seq1, safety_arms, active_arms,
                    shadow_nom_env, shadow_cor_env)
                label_robots = {idx for idx, (pol, act) in enumerate(
                    [(policy0, policy_active0), (policy1, policy_active1)]) if pol is not None and act}
                frames.append(tmg.build_render_frame(obs, render_camera, label_robots))
                overlay_frames.append(build_render_frame_with_ellipsoids(
                    obs, env.env.sim, env, render_camera, label_robots, step_intervened,
                    success_z_pad=args.success_z_pad,
                    future_trajectories=future_trajectories))

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
        success_info = evaluate_custom_success(env.env.sim, env, z_pad=args.success_z_pad)
        success = success_info["success"]
        if video_path is not None and frames:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(str(video_path), frames, fps=args.fps, codec="libx264")
            overlay_path = video_path.with_name(f"{video_path.stem}_ellipsoid{video_path.suffix}")
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
    p.add_argument("--trajectory-horizon", type=int, default=50,
                   help="how many future policy actions to visualize on the ellipsoid overlay video")
    p.add_argument("--video-seeds", type=int, nargs="*", default=None, help="seeds to also render to mp4")
    p.add_argument("--video-camera", default="all", help="render camera for video (all/sideview/birdview/...)")
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
    policy0_horizon = configure_policy_replan(policy0, args.policy_replan_horizon)
    policy1_horizon = configure_policy_replan(policy1, args.policy_replan_horizon)
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
                        recorder=recorder, record_task=f"{args.label}: put the milk into the trash can")
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
