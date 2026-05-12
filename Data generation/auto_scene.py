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
from robosuite.controllers import load_controller_config
from robosuite.utils.camera_utils import (
    get_camera_transform_matrix,
    project_points_from_world_to_camera,
)

# ==============================================================================
# 환경 설정 (직접 수정)
# ==============================================================================
ROBOT_X_OFFSET = 0.48
HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
ROBOT1_PASSIVE_QPOS = [0.0, -1.3, 0.0, -2.35, 0.0, 1.0, 0.785]
ROBOT1_PASSIVE_GRIPPER = [0.020833, -0.020833]
ROBOT1_PASSIVE_GRIPPER_ACTION = -1.0   # robot1 gripper 열어둠 (반전 적용)

# gripper 액션 부호 (robosuite Panda 관례 반대로 들어가서 여기서 한 번에 정의)
GRIPPER_OPEN  = -1.0   # gripper 열기 (approach / release 때)
GRIPPER_CLOSE =  1.0   # gripper 닫기 (집은 채 이동)

# milk 초기 위치: main table 기준 중심 + 반경(원형 샘플링)
MILK_CENTER = (-0.13, -0.35)
MILK_RADIUS = 0.06
# 잉여 salad_dressing은 MILK_CENTER의 y축 대칭 (x 부호 반전, y 유지)
SALAD_DRESSING_CENTER = (-MILK_CENTER[0], MILK_CENTER[1])
SALAD_DRESSING_RADIUS = MILK_RADIUS

# trash_can 배치 위치: main table 기준 중심
TRASH_CAN_CENTER = (0.00, 0.35)

# trash_can 내부 목표점: trash_can 중심 기준 local xy + 반경(원형 샘플링)
TARGET_CENTER_LOCAL = (0.00, 0.00)
TARGET_RADIUS_LOCAL = 0.07

# 목표 허용 박스 반폭 (원형 샘플링 점 주변의 작은 목표 영역)
TARGET_BOX_HALF_SIZE = 0.03

# trash_can 크기 (m)
TRASH_INNER_WIDTH = 0.3
TRASH_INNER_DEPTH = 0.13
TRASH_WALL_THICK = 0.01
TRASH_BOTTOM_THICK = 0.010
TRASH_MASS = 1000.0

# trajectory 테스트 제어 상수
MAX_STEPS = 700                 # 전체 scripted trajectory 최대 step 수 (fps=10)
RESOLUTION = 256                # 렌더링 해상도
OPEN_GRIPPER_INIT_STEPS = 8     # 시작 직후 gripper를 열린 상태로 유지하는 step 수
GRIP_CLOSE_HOLD_STEPS = 12      # 집기 위해 gripper를 닫은 채 유지하는 step 수
GRIP_OPEN_HOLD_STEPS = 10       # 놓기 위해 gripper를 연 채 유지하는 step 수
RETREAT_HOLD_STEPS = 20         # retreat 자세에 머물 step 수 (영상 끝 정지 화면용)
JOINT_TOL = 0.15                # 정식 waypoint 도착 판정 최대 오차
#JOINT_TOL_MID = 0.4            # 중간 보간점(_mid)은 빡세게 안 가도 되게 — 큰 값일수록 빨리 통과
PHASE_TIMEOUT_STEPS = 40        # 한 waypoint에 이 step 안에 못 도달하면 강제로 다음 phase 진행
# 특정 waypoint만 별도 tol을 주고싶을 때 (없으면 위 기본값 사용)
PHASE_TOL_OVERRIDE = {
    "move_preplace": 0.25,
}
# 특정 waypoint만 별도 timeout을 주고싶을 때 (없으면 PHASE_TIMEOUT_STEPS 사용)
PHASE_TIMEOUT_OVERRIDE = {
    "return_home": 15,
}
IK_MAX_ITERS = 250              # waypoint당 IK 반복 최대 횟수
IK_POS_TOL = 0.003              # IK 종료용 end-effector 위치 오차
IK_AXIS_TOL = 0.02              # IK 종료용 end-effector 접근축 오차
IK_DAMPING = 1e-4               # damped least-squares IK damping
IK_AXIS_WEIGHT = 1.5            # 접근축(아래 향함) hard-ish 강제
IK_SECONDARY_AXIS_WEIGHT = 0.6  # 보조 축(yaw 고정)도 충분히 강제
IK_NULLSPACE_GAIN = 0.05        # IK 해가 튀지 않도록 seed 자세로 약하게 끌어당기는 비율
IK_MAX_DELTA_Q = 0.15           # IK 1회 반복당 허용할 최대 joint 업데이트 크기
TARGET_DOWN_AXIS_WORLD = [0.0, 0.0, -1.0]  # grip_site 로컬 z축이 향해야 하는 world 방향
TARGET_FORWARD_AXIS_WORLD = [1.0, 0.0, 0.0]  # grip_site 로컬 x축이 대체로 향해야 하는 world 방향
GRASP_Z_OFFSET = -0.00           # milk 중심에서 grasp waypoint까지의 z 오프셋 (음수면 깊이 들어감)
PREGRASP_Z_OFFSET = 0.25          # milk 중심에서 pregrasp waypoint까지의 z 오프셋
POSTGRASP_LIFT_Z_OFFSET = 0.4   # 잡은 직후 수직 상승 waypoint까지의 z 오프셋 (milk 중심 기준)
PREPLACE_Z_OFFSET = PREGRASP_Z_OFFSET        # trash target에서 preplace waypoint까지의 z 오프셋
RETREAT_Z_OFFSET = 0.3         # trash target에서 retreat waypoint까지의 z 오프셋
WAYPOINT_INTERP_ALPHA = 0.5     # 인접한 두 joint waypoint 사이에 넣을 중간점 비율

# JOINT_POSITION controller custom config
JOINT_POS_OUTPUT_MAX = 0.2     # 한 step에 허용할 최대 joint position 명령 크기
JOINT_POS_OUTPUT_MIN = -0.2    # 한 step에 허용할 최소 joint position 명령 크기
JOINT_POS_KP = 120              # joint position 추종 stiffness
JOINT_POS_DAMPING_RATIO = 1.5   # joint damping 비율
JOINT_POS_RAMP_RATIO = 0.05     # 목표 qpos로 올라가는 내부 ramp 비율
BIRDVIEW_CAM_POS = [0.0, 0.0, 2.5]

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
salad_dressing_xy = sample_point_in_disk(SALAD_DRESSING_CENTER, SALAD_DRESSING_RADIUS, rng)

milk_x = tiny_range(milk_xy[0])
milk_y = tiny_range(milk_xy[1])
salad_dressing_x = tiny_range(salad_dressing_xy[0])
salad_dressing_y = tiny_range(salad_dressing_xy[1])
trash_x = tiny_range(TRASH_CAN_CENTER[0])
trash_y = tiny_range(TRASH_CAN_CENTER[1])

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
# dual problem 설정: robot0가 target, robot1 / salad_dressing_1는 잉여 (distractor)
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
      (salad_dressing_region
          (:target main_table)
          (:ranges (({salad_dressing_x[0]} {salad_dressing_y[0]} {salad_dressing_x[1]} {salad_dressing_y[1]})))
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
  (:objects milk_1 - milk  salad_dressing_1 - salad_dressing  trash_can_1 - trash_can)
  (:obj_of_interest milk_1 trash_can_1)
  (:init
    (On milk_1 main_table_milk_region)
    (On salad_dressing_1 main_table_salad_dressing_region)
    (On trash_can_1 main_table_trash_can_region)
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
    "sideview_robot0_left",
    "sideview_robot0_right",
    "sideview_robot1_left",
    "sideview_robot1_right",
    "robot0_eye_in_hand",
    "backview",
]
LABELS = [
    "Agent View",
    "Bird View",
    "Front View",
    "Side View",
    "Side R0 Left",
    "Side R0 Right",
    "Side R1 Left",
    "Side R1 Right",
    "Hand Cam",
    "Back View",
]

env = OffScreenRenderEnv(
    bddl_file_name=str(bddl_file),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    controller_configs={
        **load_controller_config(default_controller="JOINT_POSITION"),
        "input_max":  JOINT_POS_OUTPUT_MAX,   # input/output 범위 동일 → scale identity
        "input_min":  JOINT_POS_OUTPUT_MIN,
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

birdview_cam_id = env.env.sim.model.camera_name2id("birdview")
env.env.sim.model.cam_pos[birdview_cam_id][:] = BIRDVIEW_CAM_POS

env.seed(args.seed)
obs = env.reset()
env.env.sim.model.cam_pos[birdview_cam_id][:] = BIRDVIEW_CAM_POS
env.env.sim.forward()


def set_free_joint_xy(sim, joint_name, x, y):
    joint_id = sim.model.joint_name2id(joint_name)
    qpos_adr = sim.model.jnt_qposadr[joint_id]
    qpos = sim.data.qpos[qpos_adr : qpos_adr + 7].copy()
    qpos[0] = x
    qpos[1] = y
    sim.data.set_joint_qpos(joint_name, qpos)


def lock_robot1_pose(sim, robot):
    sim.data.qpos[robot._ref_joint_pos_indexes] = np.asarray(ROBOT1_PASSIVE_QPOS, dtype=float)
    sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    if robot.has_gripper:
        sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = np.asarray(
            ROBOT1_PASSIVE_GRIPPER, dtype=float
        )
        sim.data.qvel[robot._ref_gripper_joint_vel_indexes] = 0.0
    sim.forward()


set_free_joint_xy(env.env.sim, "trash_can_1_joint0", TRASH_CAN_CENTER[0], TRASH_CAN_CENTER[1])
lock_robot1_pose(env.env.sim, env.env.robots[1])
env.env.sim.forward()
BIRDVIEW_WORLD_TO_CAMERA = get_camera_transform_matrix(
    sim=env.env.sim,
    camera_name="birdview",
    camera_height=RESOLUTION,
    camera_width=RESOLUTION,
)
WAYPOINT_CAMERA_NAMES = ["frontview", "sideview", "backview"]
WAYPOINT_WORLD_TO_CAMERA = {
    cam: get_camera_transform_matrix(
        sim=env.env.sim,
        camera_name=cam,
        camera_height=RESOLUTION,
        camera_width=RESOLUTION,
    )
    for cam in WAYPOINT_CAMERA_NAMES
}

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


def project_world_points_to_camera_px(camera_name, points_xyz):
    pixels = project_points_from_world_to_camera(
        points=np.asarray(points_xyz, dtype=float),
        world_to_camera_transform=WAYPOINT_WORLD_TO_CAMERA[camera_name],
        camera_height=RESOLUTION,
        camera_width=RESOLUTION,
    )
    return [(int(col), int(row)) for row, col in pixels]


def draw_birdview_regions(img):
    out = img.copy()
    # milk_1 sampling disk (green)
    cv2.polylines(out, [project_circle_to_birdview(MILK_CENTER, MILK_RADIUS)], True, (80, 220, 80), 2)
    # salad_dressing_1 (distractor) sampling disk (yellow-green, dashed-style 다른 색)
    cv2.polylines(
        out,
        [project_circle_to_birdview(SALAD_DRESSING_CENTER, SALAD_DRESSING_RADIUS)],
        True,
        (80, 220, 220),
        2,
    )

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


def draw_reference_points_on_camera(img, camera_name, reference_points):
    out = img.copy()
    if camera_name == "birdview":
        pixels = [
            project_world_xy_to_birdview_px((xyz[0], xyz[1]), z=xyz[2])
            for _, xyz, _ in reference_points
        ]
    else:
        pixels = project_world_points_to_camera_px(
            camera_name,
            [xyz for _, xyz, _ in reference_points],
        )
    for (name, _, color), (px, py) in zip(reference_points, pixels):
        if 0 <= px < RESOLUTION and 0 <= py < RESOLUTION:
            cv2.circle(out, (px, py), 5, color, -1)
            cv2.putText(
                out,
                name,
                (px + 6, py + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                out,
                name,
                (px + 6, py + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
    return out


def draw_orientation_axes_on_camera(img, camera_name, origin_xyz, ori_mat, axis_length=0.12):
    out = img.copy()
    origin_xyz = np.asarray(origin_xyz, dtype=float)
    ori_mat = np.asarray(ori_mat, dtype=float)
    axis_specs = [
        ("x", (255, 80, 80), origin_xyz + ori_mat[:, 0] * axis_length),
        ("y", (80, 255, 80), origin_xyz + ori_mat[:, 1] * axis_length),
        ("z", (80, 80, 255), origin_xyz + ori_mat[:, 2] * axis_length),
    ]
    points_xyz = [origin_xyz] + [end_xyz for _, _, end_xyz in axis_specs]
    if camera_name == "birdview":
        pixels = [
            project_world_xy_to_birdview_px((xyz[0], xyz[1]), z=xyz[2])
            for xyz in points_xyz
        ]
    else:
        pixels = project_world_points_to_camera_px(camera_name, points_xyz)
    origin_px = pixels[0]
    for (axis_name, color, _), end_px in zip(axis_specs, pixels[1:]):
        ox, oy = origin_px
        ex, ey = end_px
        if 0 <= ox < RESOLUTION and 0 <= oy < RESOLUTION and 0 <= ex < RESOLUTION and 0 <= ey < RESOLUTION:
            cv2.line(out, (ox, oy), (ex, ey), color, 2)
            cv2.putText(
                out,
                axis_name,
                (ex + 4, ey + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                out,
                axis_name,
                (ex + 4, ey + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
    return out


def draw_waypoints_on_camera(img, camera_name, waypoint_xyzs):
    out = img.copy()
    overlay_specs = [
        ("pregrasp", waypoint_xyzs["pregrasp"], (80, 220, 80)),
        ("grasp", waypoint_xyzs["grasp"], (0, 255, 255)),
        ("preplace", waypoint_xyzs["preplace"], (255, 180, 80)),
        ("place", waypoint_xyzs["place"], (80, 80, 255)),
        ("retreat", waypoint_xyzs["retreat"], (220, 80, 220)),
    ]
    pixels = project_world_points_to_camera_px(
        camera_name,
        [xyz for _, xyz, _ in overlay_specs],
    )
    for (name, _, color), (px, py) in zip(overlay_specs, pixels):
        if 0 <= px < RESOLUTION and 0 <= py < RESOLUTION:
            cv2.circle(out, (px, py), 5, color, -1)
            cv2.putText(
                out,
                name,
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                out,
                name,
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
    return out


def clamp_norm(vec, max_norm):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    if norm <= max_norm:
        return vec
    return vec * (max_norm / norm)


def make_dual_action(robot0_action):
    return list(robot0_action) + list(ROBOT1_PASSIVE_QPOS) + [ROBOT1_PASSIVE_GRIPPER_ACTION]


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
        jac = np.vstack(
            [
                jac_pos,
                IK_AXIS_WEIGHT * jac_ori,
                IK_SECONDARY_AXIS_WEIGHT * jac_ori,
            ]
        )
        err = np.concatenate(
            [
                pos_err,
                IK_AXIS_WEIGHT * axis_err,
                IK_SECONDARY_AXIS_WEIGHT * secondary_axis_err,
            ]
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


def get_grip_site_orientation_for_qpos(sim, robot, qpos):
    qpos_backup = sim.data.qpos.copy()
    qvel_backup = sim.data.qvel.copy()
    joint_qpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    joint_qvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    eef_site_name = robot.gripper.important_sites["grip_site"]
    sim.data.qpos[joint_qpos_idx] = np.asarray(qpos, dtype=float)
    sim.data.qvel[joint_qvel_idx] = 0.0
    sim.forward()
    ori_mat = np.asarray(sim.data.get_site_xmat(eef_site_name), dtype=float).copy()
    sim.data.qpos[:] = qpos_backup
    sim.data.qvel[:] = qvel_backup
    sim.forward()
    return ori_mat


def get_grip_site_pose_for_qpos(sim, robot, qpos):
    qpos_backup = sim.data.qpos.copy()
    qvel_backup = sim.data.qvel.copy()
    joint_qpos_idx = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
    joint_qvel_idx = np.asarray(robot._ref_joint_vel_indexes, dtype=int)
    eef_site_name = robot.gripper.important_sites["grip_site"]
    sim.data.qpos[joint_qpos_idx] = np.asarray(qpos, dtype=float)
    sim.data.qvel[joint_qvel_idx] = 0.0
    sim.forward()
    pos = np.asarray(sim.data.get_site_xpos(eef_site_name), dtype=float).copy()
    ori_mat = np.asarray(sim.data.get_site_xmat(eef_site_name), dtype=float).copy()
    sim.data.qpos[:] = qpos_backup
    sim.data.qvel[:] = qvel_backup
    sim.forward()
    return pos, ori_mat


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


def make_joint_position_action(target_qpos, current_qpos, gripper_cmd):
    """JOINT_POSITION 컨트롤러는 delta 모드이므로 (target - current) 보냄."""
    delta = np.asarray(target_qpos, dtype=float) - np.asarray(current_qpos, dtype=float)
    delta = np.clip(delta, JOINT_POS_OUTPUT_MIN, JOINT_POS_OUTPUT_MAX)
    return np.concatenate([delta, [gripper_cmd]])


def joint_distance_to(obs_dict, target_qpos):
    qpos = np.asarray(obs_dict["robot0_joint_pos"], dtype=float)
    return float(np.max(np.abs(qpos - np.asarray(target_qpos, dtype=float))))


def interpolate_joint_waypoint(start_qpos, end_qpos, alpha=WAYPOINT_INTERP_ALPHA):
    start_qpos = np.asarray(start_qpos, dtype=float)
    end_qpos = np.asarray(end_qpos, dtype=float)
    return (1.0 - alpha) * start_qpos + alpha * end_qpos


def print_waypoint_ik_debug(sim, robot, name, target_xyz, target_ori_mat, solved_qpos):
    actual_xyz, actual_ori_mat = get_grip_site_pose_for_qpos(sim, robot, solved_qpos)
    pos_err = float(np.linalg.norm(np.asarray(target_xyz, dtype=float) - actual_xyz))
    axis_err = float(np.linalg.norm(axis_alignment_error(np.asarray(target_ori_mat, dtype=float), actual_ori_mat)))
    print(f"{name}_target_xyz={np.asarray(target_xyz, dtype=float).tolist()}")
    print(f"{name}_actual_xyz={actual_xyz.tolist()}")
    print(f"{name}_pos_err={pos_err:.6f}")
    print(f"{name}_axis_err={axis_err:.6f}")


def make_grid(obs_dict, waypoint_xyzs, ik_reference_points, ik_reference_pose):
    views = []
    for cam, label in zip(CAMERAS, LABELS):
        img = obs_dict[f"{cam}_image"][::-1]
        if cam == "birdview":
            img = draw_birdview_regions(img)
            img = draw_reference_points_on_camera(img, cam, ik_reference_points)
            img = draw_orientation_axes_on_camera(img, cam, ik_reference_pose[0], ik_reference_pose[1])
        elif cam in WAYPOINT_CAMERA_NAMES:
            img = draw_reference_points_on_camera(img, cam, ik_reference_points)
            img = draw_orientation_axes_on_camera(img, cam, ik_reference_pose[0], ik_reference_pose[1])
            img = draw_waypoints_on_camera(img, cam, waypoint_xyzs)
        views.append(add_label(img, label))
    # Build a 2-row grid with dynamic column count.
    num_views = len(views)
    if num_views == 0:
        raise ValueError("No camera views available for grid rendering.")
    cols = (num_views + 1) // 2
    row1_views = views[:cols]
    row2_views = views[cols:]
    if len(row2_views) < cols:
        pad_img = np.zeros_like(views[0])
        row2_views.extend([pad_img] * (cols - len(row2_views)))
    row1 = np.concatenate(row1_views, axis=1)
    row2 = np.concatenate(row2_views, axis=1)
    return np.concatenate([row1, row2], axis=0)


frames = []
actions = []

milk_pos = np.asarray(obs["milk_1_pos"], dtype=float)
milk_pick_xyz = milk_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])
milk_above_xyz = milk_pos + np.array([0.0, 0.0, PREGRASP_Z_OFFSET])
postgrasp_lift_xyz = milk_pos + np.array([0.0, 0.0, POSTGRASP_LIFT_Z_OFFSET])
trash_target_xyz = np.asarray(
    env.env.sim.data.get_site_xpos("trash_can_1_contain_region"),
    dtype=float,
)
trash_above_xyz = trash_target_xyz + np.array([0.0, 0.0, PREPLACE_Z_OFFSET])
retreat_xyz = trash_target_xyz + np.array([0.0, 0.0, RETREAT_Z_OFFSET])
WAYPOINT_XYZS = {
    "pregrasp": milk_above_xyz,
    "grasp": milk_pick_xyz,
    "preplace": trash_above_xyz,
    "place": trash_target_xyz,
    "retreat": retreat_xyz,
}
robot0 = env.env.robots[0]
home_qpos = np.asarray(obs["robot0_joint_pos"], dtype=float)
target_ori_mat = make_target_ori_from_axes(TARGET_DOWN_AXIS_WORLD, TARGET_FORWARD_AXIS_WORLD)
pregrasp_qpos = solve_ik_for_pose(env.env.sim, robot0, milk_above_xyz, target_ori_mat, home_qpos)
grasp_qpos = solve_ik_for_pose(env.env.sim, robot0, milk_pick_xyz, target_ori_mat, pregrasp_qpos)
postgrasp_lift_qpos = solve_ik_for_pose(env.env.sim, robot0, postgrasp_lift_xyz, target_ori_mat, grasp_qpos)
lift_qpos = solve_ik_for_pose(env.env.sim, robot0, milk_above_xyz, target_ori_mat, postgrasp_lift_qpos)
preplace_qpos = solve_ik_for_pose(env.env.sim, robot0, trash_above_xyz, target_ori_mat, lift_qpos)
place_qpos = solve_ik_for_pose(env.env.sim, robot0, trash_target_xyz, target_ori_mat, preplace_qpos)
retreat_qpos = solve_ik_for_pose(env.env.sim, robot0, retreat_xyz, target_ori_mat, place_qpos)

# IK 수렴 검증: 각 waypoint별 pos/axis 오차 + 실제 grip_site z축 (down 고정 확인)
for _name, _xyz, _q, _ori_target in [
    ("pregrasp",       milk_above_xyz,      pregrasp_qpos,       target_ori_mat),
    ("grasp",          milk_pick_xyz,       grasp_qpos,          target_ori_mat),
    ("postgrasp_lift", postgrasp_lift_xyz,  postgrasp_lift_qpos, target_ori_mat),
    ("lift",           milk_above_xyz,      lift_qpos,           target_ori_mat),
    ("preplace",       trash_above_xyz,     preplace_qpos,       target_ori_mat),
    ("place",          trash_target_xyz,    place_qpos,          target_ori_mat),
    ("retreat",        retreat_xyz,         retreat_qpos,        target_ori_mat),
]:
    print_waypoint_ik_debug(env.env.sim, robot0, _name, _xyz, _ori_target, _q)
    _ori = get_grip_site_orientation_for_qpos(env.env.sim, robot0, _q)
    print(f"{_name}_eef_z_axis_world={_ori[:, 2].tolist()}  eef_x={_ori[:, 0].round(2).tolist()}")

IK_REFERENCE_NAME = robot0.gripper.important_sites["grip_site"]


def get_ik_reference_points(sim):
    return [
        ("grip_site", np.asarray(sim.data.get_site_xpos(IK_REFERENCE_NAME), dtype=float), (80, 255, 80)),
    ]


def get_ik_reference_pose(sim):
    return (
        np.asarray(sim.data.get_site_xpos(IK_REFERENCE_NAME), dtype=float),
        np.asarray(sim.data.get_site_xmat(IK_REFERENCE_NAME), dtype=float),
    )


move_sequence = [
    ("move_pregrasp", pregrasp_qpos, GRIPPER_OPEN),
    ("move_grasp",    grasp_qpos,    GRIPPER_OPEN),
]
post_grasp_sequence = [
    # 잡은 직후 POSTGRASP_LIFT_Z_OFFSET 만큼 수직 상승 (z 독립 조절)
    ("back_to_pregrasp", postgrasp_lift_qpos, GRIPPER_CLOSE),
    ("lift_milk",        lift_qpos,           GRIPPER_CLOSE),
    ("move_preplace",    preplace_qpos,       GRIPPER_CLOSE),
    ("move_place",       place_qpos,          GRIPPER_CLOSE),
]
_home_target = np.asarray(HOME_QPOS, dtype=float)
retreat_sequence = [
    ("retreat",     retreat_qpos,  GRIPPER_OPEN),
    ("return_home", _home_target,  GRIPPER_OPEN),
]

phase = "open_then_approach"
_prev_phase = None
phase_steps = 0
close_hold_steps = 0
open_hold_steps = 0
retreat_hold_steps = 0
done = False
move_index = 0
post_grasp_index = 0
retreat_index = 0

for _ in range(MAX_STEPS):
    if phase != _prev_phase:
        print(f"[phase] step={len(actions):3d} -> {phase}")
        _prev_phase = phase
    cur = obs["robot0_joint_pos"]
    if phase == "open_then_approach":
        robot0_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_OPEN)
        if phase_steps > OPEN_GRIPPER_INIT_STEPS:
            move_index = 0
            phase = move_sequence[move_index][0]
            phase_steps = 0

    elif phase in [name for name, _, _ in move_sequence]:
        _, target_qpos, gripper_cmd = move_sequence[move_index]
        robot0_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
        _tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
        _to = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
        _timed_out = (_to is not None) and (phase_steps >= _to)
        if joint_distance_to(obs, target_qpos) < _tol or _timed_out:
            if _timed_out:
                print(f"[timeout] {phase}: jd={joint_distance_to(obs, target_qpos):.3f}, advancing")
            move_index += 1
            if move_index >= len(move_sequence):
                phase = "close_gripper"
            else:
                phase = move_sequence[move_index][0]
            phase_steps = 0

    elif phase == "close_gripper":
        robot0_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_CLOSE)
        close_hold_steps += 1
        if close_hold_steps >= GRIP_CLOSE_HOLD_STEPS:
            post_grasp_index = 0
            phase = post_grasp_sequence[post_grasp_index][0]
            phase_steps = 0

    elif phase in [name for name, _, _ in post_grasp_sequence]:
        _, target_qpos, gripper_cmd = post_grasp_sequence[post_grasp_index]
        robot0_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
        _tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
        _to = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
        _timed_out = (_to is not None) and (phase_steps >= _to)
        if joint_distance_to(obs, target_qpos) < _tol or _timed_out:
            if _timed_out:
                print(f"[timeout] {phase}: jd={joint_distance_to(obs, target_qpos):.3f}, advancing")
            post_grasp_index += 1
            if post_grasp_index >= len(post_grasp_sequence):
                phase = "open_gripper"
            else:
                phase = post_grasp_sequence[post_grasp_index][0]
            phase_steps = 0

    elif phase == "open_gripper":
        robot0_action = make_joint_position_action(cur, cur, gripper_cmd=GRIPPER_OPEN)
        open_hold_steps += 1
        if open_hold_steps >= GRIP_OPEN_HOLD_STEPS:
            retreat_index = 0
            phase = retreat_sequence[retreat_index][0]
            phase_steps = 0

    elif phase == "retreat_hold":
        # 마지막 home 자세 유지 (gripper 열어둠)
        robot0_action = make_joint_position_action(_home_target, cur, gripper_cmd=GRIPPER_OPEN)
        retreat_hold_steps += 1
        if retreat_hold_steps >= RETREAT_HOLD_STEPS:
            done = True

    else:
        _, target_qpos, gripper_cmd = retreat_sequence[retreat_index]
        robot0_action = make_joint_position_action(target_qpos, cur, gripper_cmd=gripper_cmd)
        _tol = PHASE_TOL_OVERRIDE.get(phase, JOINT_TOL_MID if phase.endswith("_mid") else JOINT_TOL)
        _to = PHASE_TIMEOUT_OVERRIDE.get(phase, PHASE_TIMEOUT_STEPS)
        _timed_out = (_to is not None) and (phase_steps >= _to)
        if joint_distance_to(obs, target_qpos) < _tol or _timed_out:
            if _timed_out:
                print(f"[timeout] {phase}: jd={joint_distance_to(obs, target_qpos):.3f}, advancing")
            retreat_index += 1
            if retreat_index >= len(retreat_sequence):
                phase = "retreat_hold"
            else:
                phase = retreat_sequence[retreat_index][0]
            phase_steps = 0

    action = make_dual_action(robot0_action)
    actions.append(action)
    obs, _, env_done, _ = env.step(action)
    lock_robot1_pose(env.env.sim, env.env.robots[1])
    obs = env.env._get_observations()
    frames.append(
        make_grid(
            obs,
            WAYPOINT_XYZS,
            get_ik_reference_points(env.env.sim),
            get_ik_reference_pose(env.env.sim),
        )
    )
    phase_steps += 1
    if done:   # env_done은 무시 — goal 달성해도 retreat까지 끝까지 진행
        break

imageio.mimwrite(args.video_out, frames, fps=10)
print(f"Saved: {args.video_out}")
print(f"num_actions={len(actions)}")
print(f"final_success={env.env._check_success()}")
print(f"pregrasp_qpos={pregrasp_qpos.tolist()}")
print(f"grasp_qpos={grasp_qpos.tolist()}")
print(f"preplace_qpos={preplace_qpos.tolist()}")
print(f"place_qpos={place_qpos.tolist()}")
print(f"milk_xy={milk_xy}")
print(f"salad_dressing_xy={salad_dressing_xy}")
print(f"trash_can_center={list(TRASH_CAN_CENTER)}")
print(f"target_local_xy={[target_x, target_y]}")
print(f"target_global_xy={trash_goal_global}")
print(f"ik_grip_site={IK_REFERENCE_NAME}")

env.close()
bddl_file.unlink(missing_ok=True)
