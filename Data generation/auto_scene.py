#!/usr/bin/env python3
import argparse
import math
import os
import pathlib
import random
import sys
import tempfile

sys.path.insert(0, os.path.expanduser("~/RobotLearning_9/vlsa-aegis/safelibero"))

import cv2
import imageio
import numpy as np

from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.problems.dual_tabletop_manipulation as dual_mod  # noqa
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

# ==============================================================================
# 환경 설정 (직접 수정)
# ==============================================================================
ROBOT_X_OFFSET = 0.28
HOME_QPOS = [0.0, -1.3, 0.0, -2.35, 0.0, 1.0, 0.785]

# milk 초기 위치: main table 기준 중심 + 반경(원형 샘플링)
MILK_CENTER = (-0.35, -0.35)
MILK_RADIUS = 0.14
DISTRACTOR_MILK_CENTER = (0.40, 0.50)
DISTRACTOR_MILK_RADIUS = 0.14

# trash_can 배치 위치: main table 기준 중심
TRASH_CAN_CENTER = (0.00, 0.35)

# trash_can 내부 목표점: trash_can 중심 기준 local xy + 반경(원형 샘플링)
TARGET_CENTER_LOCAL = (0.00, 0.00)
TARGET_RADIUS_LOCAL = 0.1

# 목표 허용 박스 반폭 (원형 샘플링 점 주변의 작은 목표 영역)
TARGET_BOX_HALF_SIZE = 0.03

# trash_can 크기 (m)
TRASH_INNER_WIDTH = 0.3
TRASH_INNER_DEPTH = 0.13
TRASH_WALL_THICK = 0.01
TRASH_BOTTOM_THICK = 0.010
TRASH_MASS = 1000.0

MAX_STEPS = 30
RESOLUTION = 256

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--video-out",
    type=str,
    default=str(pathlib.Path(__file__).with_name("AUTO_SCENE.mp4")),
)
args = parser.parse_args()


def sample_point_in_disk(center_xy, radius, rng):
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r = radius * math.sqrt(rng.uniform(0.0, 1.0))
    return [center_xy[0] + r * math.cos(theta), center_xy[1] + r * math.sin(theta)]


def tiny_range(center, half_width=0.001):
    return [center - half_width, center + half_width]


rng = random.Random(args.seed)
milk_xy = sample_point_in_disk(MILK_CENTER, MILK_RADIUS, rng)
target_local_xy = sample_point_in_disk(TARGET_CENTER_LOCAL, TARGET_RADIUS_LOCAL, rng)
milk_xy_2 = sample_point_in_disk(DISTRACTOR_MILK_CENTER, DISTRACTOR_MILK_RADIUS, rng)
trash_can_center_2 = (TRASH_CAN_CENTER[0], -TRASH_CAN_CENTER[1])

milk_x = tiny_range(milk_xy[0])
milk_y = tiny_range(milk_xy[1])
milk_x_2 = tiny_range(milk_xy_2[0])
milk_y_2 = tiny_range(milk_xy_2[1])
trash_x = tiny_range(TRASH_CAN_CENTER[0])
trash_y = tiny_range(TRASH_CAN_CENTER[1])
trash_x_2 = tiny_range(trash_can_center_2[0])
trash_y_2 = tiny_range(trash_can_center_2[1])

# ==============================================================================
# trash_can.xml 동적 생성: 내부 목표 영역을 sampled local target으로 이동
# ==============================================================================
TRASH_XML_PATH = pathlib.Path(
    os.path.expanduser(
        "~/RobotLearning_9/vlsa-aegis/safelibero/libero/libero/assets/stable_scanned_objects/trash_can/trash_can.xml"
    )
)
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

# ==============================================================================
# dual problem 설정: robot0가 target, robot1 / milk_2 / trash_can_2는 잉여
# ==============================================================================
dual_mod.ROBOT_X_OFFSET = ROBOT_X_OFFSET
dual_mod.HOME_QPOS = HOME_QPOS

problem_name = "LIBERO_Dual_Tabletop_Manipulation"
bddl_content = f"""(define (problem {problem_name})
  (:domain robosuite)
  (:language put the milk into the target area inside the trash can)
  (:regions
      (milk_region
          (:target main_table)
          (:ranges (({milk_x[0]} {milk_y[0]} {milk_x[1]} {milk_y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (milk_region_2
          (:target main_table)
          (:ranges (({milk_x_2[0]} {milk_y_2[0]} {milk_x_2[1]} {milk_y_2[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (trash_can_region
          (:target main_table)
          (:ranges (({trash_x[0]} {trash_y[0]} {trash_x[1]} {trash_y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (trash_can_region_2
          (:target main_table)
          (:ranges (({trash_x_2[0]} {trash_y_2[0]} {trash_x_2[1]} {trash_y_2[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (contain_region
          (:target trash_can_1)
      )
  )
  (:fixtures main_table - table)
  (:objects milk_1 milk_2 - milk  trash_can_1 trash_can_2 - trash_can)
  (:obj_of_interest milk_1 trash_can_1)
  (:init
    (On milk_1 main_table_milk_region)
    (On milk_2 main_table_milk_region_2)
    (On trash_can_1 main_table_trash_can_region)
    (On trash_can_2 main_table_trash_can_region_2)
  )
  (:goal
    (And (In milk_1 trash_can_1_contain_region))
  )
)"""
tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False)
tmp.write(bddl_content)
tmp.close()
bddl_file = pathlib.Path(tmp.name)

CAMERAS = [
    "agentview",
    "birdview",
    "frontview",
    "sideview",
    "robot0_eye_in_hand",
    "backview",
]
LABELS = [
    "Agent View",
    "Bird View",
    "Front View",
    "Side View",
    "Hand Cam",
    "Back View",
]

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_file),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    robots=["Panda", "Panda"],
    env_configuration="single-arm-opposed",
    table_full_size=(1.1, 1.15, 0.05),
)

cam_id = env.env.sim.model.camera_name2id("birdview")
env.env.sim.model.cam_pos[cam_id][:] = [0.0, 0.0, 2.5]

env.seed(args.seed)
obs = env.reset()
env.env.sim.model.cam_pos[cam_id][:] = [0.0, 0.0, 2.5]
env.env.sim.forward()


def set_free_joint_xy(sim, joint_name, x, y):
    joint_id = sim.model.joint_name2id(joint_name)
    qpos_adr = sim.model.jnt_qposadr[joint_id]
    qpos = sim.data.qpos[qpos_adr : qpos_adr + 7].copy()
    qpos[0] = x
    qpos[1] = y
    sim.data.set_joint_qpos(joint_name, qpos)


set_free_joint_xy(env.env.sim, "trash_can_1_joint0", TRASH_CAN_CENTER[0], TRASH_CAN_CENTER[1])
set_free_joint_xy(env.env.sim, "trash_can_2_joint0", trash_can_center_2[0], trash_can_center_2[1])
env.env.sim.forward()
BIRDVIEW_WORLD_TO_CAMERA = get_camera_transform_matrix(
    sim=env.env.sim,
    camera_name="birdview",
    camera_height=RESOLUTION,
    camera_width=RESOLUTION,
)

sim_model = env.env.sim.model
gid = sim_model.geom_name2id("table_visual")
mat_id = sim_model.geom_matid[gid]
table_gray = (140, 140, 140)

if mat_id >= 0:
    texids = sim_model.mat_texid[mat_id]
    if hasattr(texids, "__iter__"):
        tex_id_list = [int(t) for t in texids if int(t) >= 0]
    else:
        tex_id_list = [int(texids)] if int(texids) >= 0 else []

    for tid in tex_id_list:
        adr = int(sim_model.tex_adr[tid])
        nch = int(sim_model.tex_nchannel[tid]) if hasattr(sim_model, "tex_nchannel") else 3
        npx = int(sim_model.tex_height[tid]) * int(sim_model.tex_width[tid]) * nch
        flat = list(table_gray[:nch]) * (npx // nch)
        sim_model.tex_data[adr : adr + npx] = flat

    sim_model.mat_rgba[mat_id] = [
        table_gray[0] / 255,
        table_gray[1] / 255,
        table_gray[2] / 255,
        1.0,
    ]

env.env.sim.forward()

trash_goal_global = [
    TRASH_CAN_CENTER[0] + target_x,
    TRASH_CAN_CENTER[1] + target_y,
]


def add_label(img, label):
    out = img.copy()
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return out


def project_world_xy_to_birdview_px(xy, z=0.91):
    pts = np.array([[xy[0], xy[1], z]], dtype=float)
    pixels = project_points_from_world_to_camera(
        points=pts,
        world_to_camera_transform=BIRDVIEW_WORLD_TO_CAMERA,
        camera_height=RESOLUTION,
        camera_width=RESOLUTION,
    )
    row, col = pixels[0]
    return int(col), int(row)


def project_circle_to_birdview(center_xy, radius, z=0.91, num_points=96):
    pts = []
    for i in range(num_points):
        theta = 2.0 * math.pi * i / num_points
        pts.append([
            center_xy[0] + radius * math.cos(theta),
            center_xy[1] + radius * math.sin(theta),
            z,
        ])
    pixels = project_points_from_world_to_camera(
        points=np.asarray(pts, dtype=float),
        world_to_camera_transform=BIRDVIEW_WORLD_TO_CAMERA,
        camera_height=RESOLUTION,
        camera_width=RESOLUTION,
    )
    return np.asarray([[int(col), int(row)] for row, col in pixels], dtype=np.int32)


def draw_birdview_regions(img):
    out = img.copy()
    cv2.polylines(out, [project_circle_to_birdview(MILK_CENTER, MILK_RADIUS)], True, (80, 220, 80), 2)

    target_center_global = (
        TRASH_CAN_CENTER[0] + TARGET_CENTER_LOCAL[0],
        TRASH_CAN_CENTER[1] + TARGET_CENTER_LOCAL[1],
    )
    cv2.polylines(
        out,
        [project_circle_to_birdview(target_center_global, TARGET_RADIUS_LOCAL)],
        True,
        (80, 80, 255),
        2,
    )

    sampled_target_px = project_world_xy_to_birdview_px(trash_goal_global)
    cv2.circle(out, sampled_target_px, 4, (80, 80, 255), -1)
    return out


def make_grid(obs_dict):
    views = []
    for cam, label in zip(CAMERAS, LABELS):
        img = obs_dict[f"{cam}_image"][::-1]
        if cam == "birdview":
            img = draw_birdview_regions(img)
        views.append(add_label(img, label))
    row1 = np.concatenate(views[:3], axis=1)
    row2 = np.concatenate(views[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


frames = []
for _ in range(MAX_STEPS):
    obs, _, done, _ = env.step([0.0] * 6 + [-1.0] + [0.0] * 6 + [-1.0])
    frames.append(make_grid(obs))
    if done:
        break

imageio.mimwrite(args.video_out, frames, fps=10)
print(f"Saved: {args.video_out}")
print(f"milk_xy={milk_xy}")
print(f"milk_xy_2={milk_xy_2}")
print(f"trash_can_center={list(TRASH_CAN_CENTER)}")
print(f"trash_can_center_2={list(trash_can_center_2)}")
print(f"target_local_xy={[target_x, target_y]}")
print(f"target_global_xy={trash_goal_global}")

env.close()
bddl_file.unlink(missing_ok=True)
