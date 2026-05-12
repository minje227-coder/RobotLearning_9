#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import pathlib
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "Train/lerobot/src"))
sys.path.insert(0, str(BASE_DIR / "vlsa-aegis/safelibero"))

import imageio
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.problems.dual_tabletop_manipulation as dual_mod  # noqa: E402
from robosuite.controllers import load_controller_config

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:  # LeRobot 0.5.1 requires Python >= 3.12; libero env is Python 3.8.
    LeRobotDataset = None
    LEROBOT_IMPORT_ERROR = exc
else:
    LEROBOT_IMPORT_ERROR = None


TASK_DESCRIPTION = "put the distractor orange juice into the target area inside the trash can"

ROBOT_X_OFFSET = 0.48
HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
# Inactive robot0 should stay out of the workspace, same idea as
# ROBOT1_PASSIVE_QPOS in create_dataset.py.
ROBOT0_PASSIVE_QPOS = [0.0, -1.3, 0.0, -2.35, 0.0, 1.0, 0.785]
ROBOT0_PASSIVE_GRIPPER = [0.020833, -0.020833]
ROBOT0_PASSIVE_GRIPPER_ACTION = -1.0

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

MILK_CENTER = (-0.13, -0.35)
MILK_RADIUS = 0.1
DISTRACTOR_MILK_CENTER = (-MILK_CENTER[0], MILK_CENTER[1])
DISTRACTOR_MILK_RADIUS = MILK_RADIUS

TRASH_CAN_CENTER = (0.00, 0.35)
TARGET_CENTER_LOCAL = (0.00, 0.00)
TARGET_RADIUS_LOCAL = 0.07
TARGET_BOX_HALF_SIZE = 0.03

TRASH_INNER_WIDTH = 0.3
TRASH_INNER_DEPTH = 0.13
TRASH_WALL_THICK = 0.01
TRASH_BOTTOM_THICK = 0.010
TRASH_MASS = 1000.0

RESOLUTION_DEFAULT = 256
FPS_DEFAULT = 10
MAX_STEPS_DEFAULT = 700

OPEN_GRIPPER_INIT_STEPS = 8
GRIP_CLOSE_HOLD_STEPS = 12
GRIP_OPEN_HOLD_STEPS = 10
RETREAT_HOLD_STEPS = 20
JOINT_TOL = 0.15
JOINT_TOL_MID = 0.2
PHASE_TIMEOUT_STEPS = 40
PHASE_TOL_OVERRIDE = {"move_preplace": 0.25}
PHASE_TIMEOUT_OVERRIDE = {"return_home": 15}

IK_MAX_ITERS = 250
IK_POS_TOL = 0.003
IK_AXIS_TOL = 0.02
IK_DAMPING = 1e-4
IK_AXIS_WEIGHT = 1.5
IK_SECONDARY_AXIS_WEIGHT = 0.6
IK_NULLSPACE_GAIN = 0.05
IK_MAX_DELTA_Q = 0.15
TARGET_DOWN_AXIS_WORLD = [0.0, 0.0, -1.0]
TARGET_FORWARD_AXIS_WORLD = [1.0, 0.0, 0.0]

GRASP_Z_OFFSET = -0.00
PREGRASP_Z_OFFSET = 0.25
POSTGRASP_LIFT_Z_OFFSET = 0.4
PREPLACE_Z_OFFSET = PREGRASP_Z_OFFSET
RETREAT_Z_OFFSET = 0.3

JOINT_POS_OUTPUT_MAX = 0.2
JOINT_POS_OUTPUT_MIN = -0.2
JOINT_POS_KP = 120
JOINT_POS_DAMPING_RATIO = 1.5
JOINT_POS_RAMP_RATIO = 0.05
BIRDVIEW_CAM_POS = [0.0, 0.0, 2.5]

DATASET_CAMERAS = ["sideview_robot1_left", "robot1_eye_in_hand", "sideview_robot1_right"]
DEBUG_CAMERAS = []

TRASH_XML_PATH = pathlib.Path(
    BASE_DIR / "vlsa-aegis/safelibero/libero/libero/assets/stable_scanned_objects/trash_can/trash_can.xml"
)


@dataclass
class EpisodeSpec:
    seed: int
    bddl_file: pathlib.Path
    milk_xy: list[float]
    orange_juice_xy: list[float]
    target_local_xy: list[float]
    target_global_xy: list[float]


@dataclass
class RolloutResult:
    success: bool
    num_steps: int
    frames: list[dict]
    debug_frames: list[np.ndarray]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--push-to-hub", action="store_true")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-video", action="store_true")
    parser.add_argument("--video-debug-dir", type=pathlib.Path, default=None)
    parser.add_argument(
        "--image-rotation",
        choices=["vertical_flip", "rotate_180", "none"],
        default="vertical_flip",
    )
    parser.add_argument("--vcodec", default="auto")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    return parser.parse_args()


def sample_point_in_disk(center_xy, radius, rng):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = radius * math.sqrt(rng.uniform(0.0, 1.0))
    return [center_xy[0] + r * math.cos(theta), center_xy[1] + r * math.sin(theta)]


def tiny_range(center, half_width=0.001):
    return [center - half_width, center + half_width]


def write_trash_can_xml(target_local_xy):
    iw_half = TRASH_INNER_WIDTH / 2
    wt_half = TRASH_WALL_THICK / 2
    bt_half = TRASH_BOTTOM_THICK / 2
    wall_off = iw_half + wt_half
    outer_half = iw_half + TRASH_WALL_THICK
    inner_d_half = TRASH_INNER_DEPTH / 2
    wall_zc = TRASH_BOTTOM_THICK + inner_d_half
    top_z = TRASH_BOTTOM_THICK + TRASH_INNER_DEPTH

    target_half = min(TARGET_BOX_HALF_SIZE, iw_half - 0.01)
    target_x = float(np.clip(target_local_xy[0], -iw_half + target_half, iw_half - target_half))
    target_y = float(np.clip(target_local_xy[1], -iw_half + target_half, iw_half - target_half))

    geom_attr = (
        'solimp="0.998 0.998 0.001" solref="0.001 1" '
        'density="200" friction="0.95 0.3 0.1" group="1" rgba="0.20 0.20 0.20 1"'
    )
    TRASH_XML_PATH.write_text(
        f"""<mujoco model="trash_can">
  <worldbody>
    <body>
      <body name="object">
        <inertial pos="0 0 {wall_zc}" mass="{TRASH_MASS}" diaginertia="0.005 0.005 0.005"/>
        <site type="box" pos="{target_x} {target_y} {wall_zc}" quat="1 0 0 0" size="{target_half} {target_half} {inner_d_half}" group="0" rgba="0 0.8 0 0" name="contain_region"/>
        <geom {geom_attr} type="box" pos="0 0 {bt_half}" size="{outer_half} {outer_half} {bt_half}"/>
        <geom {geom_attr} type="box" pos="0 {wall_off} {wall_zc}" size="{iw_half} {wt_half} {inner_d_half}"/>
        <geom {geom_attr} type="box" pos="0 -{wall_off} {wall_zc}" size="{iw_half} {wt_half} {inner_d_half}"/>
        <geom {geom_attr} type="box" pos="{wall_off} 0 {wall_zc}" size="{wt_half} {outer_half} {inner_d_half}"/>
        <geom {geom_attr} type="box" pos="-{wall_off} 0 {wall_zc}" size="{wt_half} {outer_half} {inner_d_half}"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top_z}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{iw_half} {iw_half} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return target_x, target_y


def build_episode_spec(seed: int) -> EpisodeSpec:
    rng = random.Random(seed)
    milk_xy = sample_point_in_disk(MILK_CENTER, MILK_RADIUS, rng)
    target_local_xy = sample_point_in_disk(TARGET_CENTER_LOCAL, TARGET_RADIUS_LOCAL, rng)
    orange_juice_xy = sample_point_in_disk(DISTRACTOR_MILK_CENTER, DISTRACTOR_MILK_RADIUS, rng)
    target_x, target_y = write_trash_can_xml(target_local_xy)

    milk_x = tiny_range(milk_xy[0])
    milk_y = tiny_range(milk_xy[1])
    orange_juice_x = tiny_range(orange_juice_xy[0])
    orange_juice_y = tiny_range(orange_juice_xy[1])
    trash_x = tiny_range(TRASH_CAN_CENTER[0])
    trash_y = tiny_range(TRASH_CAN_CENTER[1])

    dual_mod.ROBOT_X_OFFSET = ROBOT_X_OFFSET
    dual_mod.HOME_QPOS = HOME_QPOS

    problem_name = "LIBERO_Dual_Tabletop_Manipulation"
    bddl_content = f"""(define (problem {problem_name})
  (:domain robosuite)
  (:language {TASK_DESCRIPTION})
  (:regions
      (milk_region
          (:target main_table)
          (:ranges (({milk_x[0]} {milk_y[0]} {milk_x[1]} {milk_y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (orange_juice_region
          (:target main_table)
          (:ranges (({orange_juice_x[0]} {orange_juice_y[0]} {orange_juice_x[1]} {orange_juice_y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (trash_can_region
          (:target main_table)
          (:ranges (({trash_x[0]} {trash_y[0]} {trash_x[1]} {trash_y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (contain_region
          (:target trash_can_1)
      )
  )
  (:fixtures main_table - table)
  (:objects milk_1 - milk  orange_juice_1 - orange_juice  trash_can_1 - trash_can)
  (:obj_of_interest orange_juice_1 trash_can_1)
  (:init
    (On milk_1 main_table_milk_region)
    (On orange_juice_1 main_table_orange_juice_region)
    (On trash_can_1 main_table_trash_can_region)
  )
  (:goal
    (And (In orange_juice_1 trash_can_1_contain_region))
  )
)"""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False)
    tmp.write(bddl_content)
    tmp.close()

    target_global_xy = [TRASH_CAN_CENTER[0] + target_x, TRASH_CAN_CENTER[1] + target_y]
    return EpisodeSpec(
        seed=seed,
        bddl_file=pathlib.Path(tmp.name),
        milk_xy=milk_xy,
        orange_juice_xy=orange_juice_xy,
        target_local_xy=[target_x, target_y],
        target_global_xy=target_global_xy,
    )


def make_env(spec: EpisodeSpec, resolution: int, save_debug_video: bool):
    camera_names = DATASET_CAMERAS + (DEBUG_CAMERAS if save_debug_video else [])
    env = OffScreenRenderEnv(
        bddl_file_name=str(spec.bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
        camera_names=camera_names,
        controller_configs={
            **load_controller_config(default_controller="JOINT_POSITION"),
            "input_max": JOINT_POS_OUTPUT_MAX,
            "input_min": JOINT_POS_OUTPUT_MIN,
            "output_max": JOINT_POS_OUTPUT_MAX,
            "output_min": JOINT_POS_OUTPUT_MIN,
            "kp": JOINT_POS_KP,
            "damping_ratio": JOINT_POS_DAMPING_RATIO,
            "ramp_ratio": JOINT_POS_RAMP_RATIO,
        },
        robots=["Panda", "Panda"],
        env_configuration="single-arm-opposed",
        table_full_size=(1.1, 1.15, 0.05),
    )
    if "birdview" in camera_names:
        birdview_cam_id = env.env.sim.model.camera_name2id("birdview")
        env.env.sim.model.cam_pos[birdview_cam_id][:] = BIRDVIEW_CAM_POS
    env.seed(spec.seed)
    obs = env.reset()
    if "birdview" in camera_names:
        birdview_cam_id = env.env.sim.model.camera_name2id("birdview")
        env.env.sim.model.cam_pos[birdview_cam_id][:] = BIRDVIEW_CAM_POS
    env.env.sim.forward()
    return env, obs


def set_free_joint_xy(sim, joint_name, x, y):
    joint_id = sim.model.joint_name2id(joint_name)
    qpos_adr = sim.model.jnt_qposadr[joint_id]
    qpos = sim.data.qpos[qpos_adr : qpos_adr + 7].copy()
    qpos[0] = x
    qpos[1] = y
    sim.data.set_joint_qpos(joint_name, qpos)


def lock_robot0_pose(sim, robot):
    sim.data.qpos[robot._ref_joint_pos_indexes] = np.asarray(ROBOT0_PASSIVE_QPOS, dtype=float)
    sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    if robot.has_gripper:
        sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = np.asarray(
            ROBOT0_PASSIVE_GRIPPER, dtype=float
        )
        sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
    sim.forward()


def axis_alignment_error(target_ori_mat, current_ori_mat, axis_index=2):
    target_axis = np.asarray(target_ori_mat, dtype=float)[:3, axis_index]
    current_axis = np.asarray(current_ori_mat, dtype=float)[:3, axis_index]
    return np.cross(current_axis, target_axis)


def make_target_ori_from_axes(down_axis_world, forward_axis_world):
    z_axis = np.asarray(down_axis_world, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_hint = np.asarray(forward_axis_world, dtype=float)
    x_hint = x_hint - np.dot(x_hint, z_axis) * z_axis
    if np.linalg.norm(x_hint) < 1e-8:
        x_hint = np.array([1.0, 0.0, 0.0], dtype=float)
        x_hint = x_hint - np.dot(x_hint, z_axis) * z_axis
    x_axis = x_hint / np.linalg.norm(x_hint)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def solve_ik_for_pose(sim, robot, target_xyz, target_ori_mat, initial_qpos):
    qpos_backup = sim.data.qpos.copy()
    qvel_backup = sim.data.qvel.copy()
    joint_qpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    joint_qvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    eef_site_name = robot.gripper.important_sites["grip_site"]
    seed_qpos = np.asarray(initial_qpos, dtype=float).copy()

    sim.data.qpos[joint_qpos_idx] = seed_qpos
    sim.data.qvel[joint_qvel_idx] = 0.0
    sim.forward()

    q = sim.data.qpos[joint_qpos_idx].copy()
    target_xyz = np.asarray(target_xyz, dtype=float)
    target_ori_mat = np.asarray(target_ori_mat, dtype=float)
    for _ in range(IK_MAX_ITERS):
        eef_xyz = np.asarray(sim.data.get_site_xpos(eef_site_name), dtype=float)
        eef_ori_mat = np.asarray(sim.data.get_site_xmat(eef_site_name), dtype=float)
        pos_err = target_xyz - eef_xyz
        axis_err = axis_alignment_error(target_ori_mat, eef_ori_mat, axis_index=2)
        secondary_axis_err = axis_alignment_error(target_ori_mat, eef_ori_mat, axis_index=0)
        if np.linalg.norm(pos_err) < IK_POS_TOL and np.linalg.norm(axis_err) < IK_AXIS_TOL:
            break
        jac_pos = sim.data.get_site_jacp(eef_site_name).reshape(3, -1)[:, joint_qvel_idx]
        jac_ori = sim.data.get_site_jacr(eef_site_name).reshape(3, -1)[:, joint_qvel_idx]
        jac = np.vstack([jac_pos, IK_AXIS_WEIGHT * jac_ori, IK_SECONDARY_AXIS_WEIGHT * jac_ori])
        err = np.concatenate(
            [pos_err, IK_AXIS_WEIGHT * axis_err, IK_SECONDARY_AXIS_WEIGHT * secondary_axis_err]
        )
        jtj = jac.T @ jac + IK_DAMPING * np.eye(jac.shape[1])
        dq = np.linalg.solve(jtj, jac.T @ err)
        dq += IK_NULLSPACE_GAIN * (seed_qpos - q)
        dq = np.clip(dq, -IK_MAX_DELTA_Q, IK_MAX_DELTA_Q)
        q = q + dq
        sim.data.qpos[joint_qpos_idx] = q
        sim.data.qvel[joint_qvel_idx] = 0.0
        sim.forward()

    solved_q = sim.data.qpos[joint_qpos_idx].copy()
    sim.data.qpos[:] = qpos_backup
    sim.data.qvel[:] = qvel_backup
    sim.forward()
    return solved_q


def make_joint_position_action(target_qpos, current_qpos, gripper_cmd):
    delta = np.asarray(target_qpos, dtype=float) - np.asarray(current_qpos, dtype=float)
    delta = np.clip(delta, JOINT_POS_OUTPUT_MIN, JOINT_POS_OUTPUT_MAX)
    return np.concatenate([delta, [gripper_cmd]]).astype(np.float32)


def get_robot_joint_pos(env, robot_index):
    robot = env.env.robots[robot_index]
    return np.asarray(env.env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=float)


def get_robot_gripper_qpos(env, robot_index):
    robot = env.env.robots[robot_index]
    return np.asarray(env.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=float)


def joint_distance_to(env, target_qpos):
    qpos = get_robot_joint_pos(env, 1)
    return float(np.max(np.abs(qpos - np.asarray(target_qpos, dtype=float))))


def interpolate_joint_waypoint(start_qpos, end_qpos, alpha=0.5):
    start_qpos = np.asarray(start_qpos, dtype=float)
    end_qpos = np.asarray(end_qpos, dtype=float)
    return (1.0 - alpha) * start_qpos + alpha * end_qpos


def make_dual_action(robot1_action):
    return list(ROBOT0_PASSIVE_QPOS) + [ROBOT0_PASSIVE_GRIPPER_ACTION] + list(robot1_action)


def extract_state(env):
    return np.concatenate(
        [
            get_robot_joint_pos(env, 1).astype(np.float32),
            [float(np.mean(get_robot_gripper_qpos(env, 1)))],
        ]
    ).astype(np.float32)


def preprocess_image(img, rotation):
    if rotation == "vertical_flip":
        img = img[::-1]
    elif rotation == "rotate_180":
        img = img[::-1, ::-1]
    elif rotation == "none":
        img = img
    else:
        raise ValueError(f"Unknown image rotation: {rotation}")
    return np.ascontiguousarray(img).astype(np.uint8)


def make_frame(env, obs, robot1_action, image_rotation):
    side_left_img = preprocess_image(obs["sideview_robot1_left_image"], image_rotation)
    wrist_img = preprocess_image(obs["robot1_eye_in_hand_image"], image_rotation)
    side_right_img = preprocess_image(obs["sideview_robot1_right_image"], image_rotation)
    state = extract_state(env)
    action = np.asarray(robot1_action, dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"Expected state shape (8,), got {state.shape}")
    if action.shape != (8,):
        raise ValueError(f"Expected action shape (8,), got {action.shape}")
    return {
        "observation.images.side_left": side_left_img,
        "observation.images.wrist": wrist_img,
        "observation.images.side_right": side_right_img,
        "observation.state": state,
        "action": action,
        "task": TASK_DESCRIPTION,
    }


def make_debug_frame(obs, image_rotation):
    images = [
        preprocess_image(obs["sideview_robot1_left_image"], image_rotation),
        preprocess_image(obs["robot1_eye_in_hand_image"], image_rotation),
        preprocess_image(obs["sideview_robot1_right_image"], image_rotation),
    ]
    return np.concatenate(images, axis=1)


def solve_waypoints(env, obs):
    sim = env.env.sim
    robot1 = env.env.robots[1]
    orange_juice_pos = np.asarray(obs["orange_juice_1_pos"], dtype=float)
    milk_pick_xyz = orange_juice_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])
    milk_above_xyz = orange_juice_pos + np.array([0.0, 0.0, PREGRASP_Z_OFFSET])
    postgrasp_lift_xyz = orange_juice_pos + np.array([0.0, 0.0, POSTGRASP_LIFT_Z_OFFSET])
    trash_target_xyz = np.asarray(sim.data.get_site_xpos("trash_can_1_contain_region"), dtype=float)
    trash_above_xyz = trash_target_xyz + np.array([0.0, 0.0, PREPLACE_Z_OFFSET])
    retreat_xyz = trash_target_xyz + np.array([0.0, 0.0, RETREAT_Z_OFFSET])

    home_qpos = get_robot_joint_pos(env, 1)
    target_ori_mat = make_target_ori_from_axes(TARGET_DOWN_AXIS_WORLD, TARGET_FORWARD_AXIS_WORLD)
    pregrasp_qpos = solve_ik_for_pose(sim, robot1, milk_above_xyz, target_ori_mat, home_qpos)
    grasp_qpos = solve_ik_for_pose(sim, robot1, milk_pick_xyz, target_ori_mat, pregrasp_qpos)
    postgrasp_lift_qpos = solve_ik_for_pose(sim, robot1, postgrasp_lift_xyz, target_ori_mat, grasp_qpos)
    lift_qpos = solve_ik_for_pose(sim, robot1, milk_above_xyz, target_ori_mat, postgrasp_lift_qpos)
    preplace_qpos = solve_ik_for_pose(sim, robot1, trash_above_xyz, target_ori_mat, lift_qpos)
    preplace_mid_qpos = interpolate_joint_waypoint(lift_qpos, preplace_qpos)
    place_qpos = solve_ik_for_pose(sim, robot1, trash_target_xyz, target_ori_mat, preplace_qpos)
    retreat_qpos = solve_ik_for_pose(sim, robot1, retreat_xyz, target_ori_mat, place_qpos)
    return {
        "pregrasp": pregrasp_qpos,
        "grasp": grasp_qpos,
        "postgrasp_lift": postgrasp_lift_qpos,
        "lift": lift_qpos,
        "preplace_mid": preplace_mid_qpos,
        "preplace": preplace_qpos,
        "place": place_qpos,
        "retreat": retreat_qpos,
        "home": np.asarray(HOME_QPOS, dtype=float),
    }


def run_scripted_episode(env, obs, max_steps, image_rotation, save_debug_video):
    set_free_joint_xy(env.env.sim, "trash_can_1_joint0", TRASH_CAN_CENTER[0], TRASH_CAN_CENTER[1])
    lock_robot0_pose(env.env.sim, env.env.robots[0])
    env.env.sim.forward()
    obs = env.env._get_observations()

    for key in ["sideview_robot1_left_image", "robot1_eye_in_hand_image", "sideview_robot1_right_image"]:
        if key not in obs:
            raise KeyError(f"Missing expected observation key: {key}. Available: {list(obs)}")

    waypoints = solve_waypoints(env, obs)
    move_sequence = [
        ("move_pregrasp", waypoints["pregrasp"], GRIPPER_OPEN),
        ("move_grasp", waypoints["grasp"], GRIPPER_OPEN),
    ]
    post_grasp_sequence = [
        ("back_to_pregrasp", waypoints["postgrasp_lift"], GRIPPER_CLOSE),
        ("lift_milk", waypoints["lift"], GRIPPER_CLOSE),
        ("move_preplace_mid", waypoints["preplace_mid"], GRIPPER_CLOSE),
        ("move_preplace", waypoints["preplace"], GRIPPER_CLOSE),
        ("move_place", waypoints["place"], GRIPPER_CLOSE),
    ]
    retreat_sequence = [
        ("retreat", waypoints["retreat"], GRIPPER_OPEN),
        ("return_home", waypoints["home"], GRIPPER_OPEN),
    ]

    frames = []
    debug_frames = []
    phase = "open_then_approach"
    phase_steps = 0
    close_hold_steps = 0
    open_hold_steps = 0
    retreat_hold_steps = 0
    done = False
    move_index = 0
    post_grasp_index = 0
    retreat_index = 0

    for _ in range(max_steps):
        cur = get_robot_joint_pos(env, 1)
        if phase == "open_then_approach":
            robot1_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_OPEN)
            if phase_steps > OPEN_GRIPPER_INIT_STEPS:
                move_index = 0
                phase = move_sequence[move_index][0]
                phase_steps = 0
        elif phase in [name for name, _, _ in move_sequence]:
            _, target_qpos, gripper_cmd = move_sequence[move_index]
            robot1_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
            tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
            timeout = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
            timed_out = timeout is not None and phase_steps >= timeout
            if joint_distance_to(env, target_qpos) < tol or timed_out:
                move_index += 1
                if move_index >= len(move_sequence):
                    phase = "close_gripper"
                else:
                    phase = move_sequence[move_index][0]
                phase_steps = 0
        elif phase == "close_gripper":
            robot1_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_CLOSE)
            close_hold_steps += 1
            if close_hold_steps >= GRIP_CLOSE_HOLD_STEPS:
                post_grasp_index = 0
                phase = post_grasp_sequence[post_grasp_index][0]
                phase_steps = 0
        elif phase in [name for name, _, _ in post_grasp_sequence]:
            _, target_qpos, gripper_cmd = post_grasp_sequence[post_grasp_index]
            robot1_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
            tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
            timeout = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
            timed_out = timeout is not None and phase_steps >= timeout
            if joint_distance_to(env, target_qpos) < tol or timed_out:
                post_grasp_index += 1
                if post_grasp_index >= len(post_grasp_sequence):
                    phase = "open_gripper"
                else:
                    phase = post_grasp_sequence[post_grasp_index][0]
                phase_steps = 0
        elif phase == "open_gripper":
            robot1_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_OPEN)
            open_hold_steps += 1
            if open_hold_steps >= GRIP_OPEN_HOLD_STEPS:
                retreat_index = 0
                phase = retreat_sequence[retreat_index][0]
                phase_steps = 0
        elif phase == "retreat_hold":
            robot1_action = make_joint_position_action(waypoints["home"], cur, gripper_cmd=GRIPPER_OPEN)
            retreat_hold_steps += 1
            if retreat_hold_steps >= RETREAT_HOLD_STEPS:
                done = True
        else:
            _, target_qpos, gripper_cmd = retreat_sequence[retreat_index]
            robot1_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
            tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
            timeout = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
            timed_out = timeout is not None and phase_steps >= timeout
            if joint_distance_to(env, target_qpos) < tol or timed_out:
                retreat_index += 1
                if retreat_index >= len(retreat_sequence):
                    phase = "retreat_hold"
                else:
                    phase = retreat_sequence[retreat_index][0]
                phase_steps = 0

        frames.append(make_frame(env, obs, robot1_action, image_rotation))
        if save_debug_video:
            debug_frames.append(make_debug_frame(obs, image_rotation))

        obs, _, _, _ = env.step(make_dual_action(robot1_action))
        lock_robot0_pose(env.env.sim, env.env.robots[0])
        obs = env.env._get_observations()
        phase_steps += 1
        if done:
            break

    return RolloutResult(success=bool(env.env._check_success()), num_steps=len(frames), frames=frames, debug_frames=debug_frames)


def save_raw_episode(path: pathlib.Path, frames: list[dict]) -> None:
    side_left = np.stack([frame["observation.images.side_left"] for frame in frames]).astype(np.uint8)
    wrist = np.stack([frame["observation.images.wrist"] for frame in frames]).astype(np.uint8)
    side_right = np.stack([frame["observation.images.side_right"] for frame in frames]).astype(np.uint8)
    state = np.stack([frame["observation.state"] for frame in frames]).astype(np.float32)
    action = np.stack([frame["action"] for frame in frames]).astype(np.float32)
    np.savez_compressed(
        path,
        side_left=side_left,
        wrist=wrist,
        side_right=side_right,
        state=state,
        action=action,
        task=np.array(TASK_DESCRIPTION),
    )


def generate_raw_attempt(
    attempt_index: int,
    episode_seed: int,
    raw_dir: pathlib.Path,
    lock_path: pathlib.Path,
    resolution: int,
    max_steps: int,
    image_rotation: str,
    save_debug_video: bool,
    debug_dir: pathlib.Path | None,
    fps: int,
) -> dict:
    env = None
    spec = None
    try:
        # trash_can.xml is a shared asset. Lock through environment creation so
        # another worker cannot overwrite the target site while MuJoCo is loading.
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            spec = build_episode_spec(episode_seed)
            env, obs = make_env(spec, resolution, save_debug_video)
            fcntl.flock(lock_file, fcntl.LOCK_UN)

        result = run_scripted_episode(
            env=env,
            obs=obs,
            max_steps=max_steps,
            image_rotation=image_rotation,
            save_debug_video=save_debug_video,
        )
        raw_path = None
        debug_path = None
        if result.success:
            raw_path = raw_dir / f"episode_attempt_{attempt_index:06d}_seed_{episode_seed}.npz"
            save_raw_episode(raw_path, result.frames)
            if debug_dir is not None and result.debug_frames:
                debug_path = debug_dir / f"attempt_{attempt_index:06d}_seed_{episode_seed}.mp4"
                imageio.mimwrite(debug_path, result.debug_frames, fps=fps)
        return {
            "attempt_index": attempt_index,
            "saved_episode_index": None,
            "seed": episode_seed,
            "success": result.success,
            "saved": result.success,
            "num_steps": result.num_steps,
            "raw_path": str(raw_path) if raw_path else None,
            "debug_path": str(debug_path) if debug_path else None,
            "milk_xy": spec.milk_xy,
            "orange_juice_xy": spec.orange_juice_xy,
            "trash_can_center": list(TRASH_CAN_CENTER),
            "target_local_xy": spec.target_local_xy,
            "target_global_xy": spec.target_global_xy,
        }
    except Exception as exc:
        return {
            "attempt_index": attempt_index,
            "saved_episode_index": None,
            "seed": episode_seed,
            "success": False,
            "saved": False,
            "error": repr(exc),
        }
    finally:
        if env is not None:
            env.close()
        if spec is not None:
            spec.bddl_file.unlink(missing_ok=True)


def create_raw_dataset_parallel(args, raw_dir: pathlib.Path, debug_dir: pathlib.Path | None, log_path: pathlib.Path):
    max_attempts = args.max_attempts or max(args.num_episodes * 3, args.num_episodes + args.num_workers)
    lock_path = args.output_dir / ".trash_xml.lock"
    saved_episodes = 0
    attempted_episodes = 0

    with log_path.open("w", encoding="utf-8") as log_file:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {}

            def submit_next():
                nonlocal attempted_episodes
                if attempted_episodes >= max_attempts:
                    return
                attempt_index = attempted_episodes
                episode_seed = args.seed + attempt_index
                attempted_episodes += 1
                fut = executor.submit(
                    generate_raw_attempt,
                    attempt_index,
                    episode_seed,
                    raw_dir,
                    lock_path,
                    args.resolution,
                    args.max_steps,
                    args.image_rotation,
                    args.save_debug_video,
                    debug_dir,
                    args.fps,
                )
                futures[fut] = attempt_index

            for _ in range(min(args.num_workers, max_attempts)):
                submit_next()

            while futures and saved_episodes < args.num_episodes:
                done, _ = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    futures.pop(fut)
                    record = fut.result()
                    if record.get("success") and saved_episodes < args.num_episodes:
                        record["saved_episode_index"] = saved_episodes
                        saved_episodes += 1
                    elif record.get("success"):
                        record["saved"] = False
                        raw_path = record.get("raw_path")
                        if raw_path:
                            pathlib.Path(raw_path).unlink(missing_ok=True)
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()
                    print(
                        f"[episode] attempt={record['attempt_index']} seed={record['seed']} "
                        f"success={record.get('success')} saved={saved_episodes}/{args.num_episodes} "
                        f"steps={record.get('num_steps')}"
                    )
                    if saved_episodes < args.num_episodes:
                        submit_next()

            for fut in futures:
                fut.cancel()

    if saved_episodes < args.num_episodes:
        raise RuntimeError(
            f"Only saved {saved_episodes}/{args.num_episodes} successful episodes after {attempted_episodes} attempts."
        )
    return saved_episodes, attempted_episodes


def create_dataset(args):
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists. Use --overwrite to replace it: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = {
        "observation.images.side_left": {
            "dtype": "image",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.side_right": {
            "dtype": "image",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["action"],
        },
    }

    dataset = None
    raw_dir = args.output_dir / "raw_episodes"
    if LeRobotDataset is None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] LeRobotDataset is unavailable in this env: {LEROBOT_IMPORT_ERROR!r}")
        print(f"[info] Writing successful episodes as raw NPZ files under: {raw_dir}")
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            root=args.output_dir,
            robot_type="dual_panda_robot1",
            fps=args.fps,
            features=features,
            use_videos=True,
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
            vcodec=args.vcodec,
        )

    if args.save_debug_video:
        debug_dir = args.video_debug_dir or (args.output_dir.parent / "side_wrist_debug_videos")
        debug_dir.mkdir(parents=True, exist_ok=True)
    else:
        debug_dir = None

    log_path = args.output_dir.parent / f"{args.output_dir.name}_generation_log.jsonl"
    saved_episodes = 0
    attempted_episodes = 0

    if dataset is None and args.num_workers > 1:
        saved_episodes, attempted_episodes = create_raw_dataset_parallel(args, raw_dir, debug_dir, log_path)
        print(f"Saved output: {args.output_dir}")
        print(f"Generation log: {log_path}")
        print(f"saved_episodes={saved_episodes} attempted_episodes={attempted_episodes}")
        print("[next] Convert raw episodes to LeRobotDataset with Python >= 3.12 / lerobot env.")
        return

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            while saved_episodes < args.num_episodes:
                if args.max_attempts is not None and attempted_episodes >= args.max_attempts:
                    raise RuntimeError(
                        f"Only saved {saved_episodes}/{args.num_episodes} successful episodes "
                        f"after {attempted_episodes} attempts."
                    )
                episode_seed = args.seed + attempted_episodes
                attempted_episodes += 1
                spec = build_episode_spec(episode_seed)
                env = None
                result = None
                try:
                    env, obs = make_env(spec, args.resolution, args.save_debug_video)
                    result = run_scripted_episode(
                        env=env,
                        obs=obs,
                        max_steps=args.max_steps,
                        image_rotation=args.image_rotation,
                        save_debug_video=args.save_debug_video,
                    )
                    if result.success:
                        if dataset is None:
                            raw_path = raw_dir / f"episode_{saved_episodes:06d}_seed_{episode_seed}.npz"
                            save_raw_episode(raw_path, result.frames)
                        else:
                            for frame in result.frames:
                                dataset.add_frame(frame)
                            dataset.save_episode()
                        if debug_dir is not None and result.debug_frames:
                            debug_path = debug_dir / f"episode_{saved_episodes:04d}_seed_{episode_seed}.mp4"
                            imageio.mimwrite(debug_path, result.debug_frames, fps=args.fps)
                        saved_episodes += 1
                    log_record = {
                        "attempt_index": attempted_episodes - 1,
                        "saved_episode_index": saved_episodes - 1 if result.success else None,
                        "seed": episode_seed,
                        "success": result.success,
                        "saved": result.success,
                        "num_steps": result.num_steps,
                        "milk_xy": spec.milk_xy,
                        "orange_juice_xy": spec.orange_juice_xy,
                        "trash_can_center": list(TRASH_CAN_CENTER),
                        "target_local_xy": spec.target_local_xy,
                        "target_global_xy": spec.target_global_xy,
                    }
                    print(
                        f"[episode] attempt={attempted_episodes - 1} seed={episode_seed} "
                        f"success={result.success} saved={saved_episodes}/{args.num_episodes} steps={result.num_steps}"
                    )
                except Exception as exc:
                    log_record = {
                        "attempt_index": attempted_episodes - 1,
                        "saved_episode_index": None,
                        "seed": episode_seed,
                        "success": False,
                        "saved": False,
                        "error": repr(exc),
                    }
                    print(f"[episode] attempt={attempted_episodes - 1} seed={episode_seed} error={exc!r}")
                finally:
                    if env is not None:
                        env.close()
                    spec.bddl_file.unlink(missing_ok=True)
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()
    finally:
        if dataset is not None:
            dataset.finalize()

    if args.push_to_hub and dataset is None:
        raise RuntimeError("Cannot push raw episodes to Hub. Convert to LeRobotDataset first.")

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["lerobot", "smolvla", "sideview", "wrist", "libero", "dual-tabletop"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Saved output: {args.output_dir}")
    print(f"Generation log: {log_path}")
    print(f"saved_episodes={saved_episodes} attempted_episodes={attempted_episodes}")
    if dataset is None:
        print("[next] Convert raw episodes to LeRobotDataset with Python >= 3.12 / lerobot env.")


def main():
    args = parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
