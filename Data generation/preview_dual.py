#!/usr/bin/env python3
import sys, os, pathlib, tempfile, argparse
sys.path.insert(0, os.path.expanduser('~/RobotLearning_9/vlsa-aegis/safelibero'))

import numpy as np
import imageio
import cv2
from libero.libero.envs import OffScreenRenderEnv
import libero.libero.envs.problems.dual_tabletop_manipulation as dual_mod  # noqa

# ==============================================================================
# 환경 설정 (직접 수정)
# ==============================================================================
# 로봇 배치 (None이면 기존 책상 바깥쪽; 값 주면 책상 위에 robot0=-X, robot1=+X)
ROBOT_X_OFFSET = 0.28

# None이면 dual 문제 클래스 / 로봇 기본 home pose 사용
#HOME_QPOS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
HOME_QPOS = [0.0, -1.3, 0.0, -2.356, 0.0, 1.0, 0.785]

# 객체 위치 ([min, max] 범위; min==max면 고정 위치)
# milk, trash_can: main table 중심 기준
MILK_X   = [-0.4, -0.4]
MILK_Y   = [-0.5, -0.5]
MILK_REF = "table"          # "table" or "robot"


TRASH_CAN_X = [-0.0, -0.0]
TRASH_CAN_Y = [0.5, 0.5]

# trash_can 크기 (m)
TRASH_INNER_WIDTH  = 0.4    # 내부 가로/세로 (xy)
TRASH_INNER_DEPTH  = 0.13   # 내부 깊이 (z, 즉 안쪽 높이)
TRASH_WALL_THICK   = 0.01   # 벽 두께
TRASH_BOTTOM_THICK = 0.010   # 바닥 두께
TRASH_MASS         = 1000.0     # 질량 (kg)

# 렌더링
MAX_STEPS  = 30
RESOLUTION = 256

# ==============================================================================
# CLI: robots 개수만
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("robots", type=int, choices=[1, 2], help="1 or 2")
args = parser.parse_args()

# ==============================================================================
# trash_can.xml 동적 생성 (크기 반영)
# ==============================================================================
TRASH_XML_PATH = pathlib.Path(os.path.expanduser(
    '~/RobotLearning_9/vlsa-aegis/safelibero/libero/libero/assets/stable_scanned_objects/trash_can/trash_can.xml'
))
iw_half      = TRASH_INNER_WIDTH / 2
wt_half      = TRASH_WALL_THICK / 2
bt_half      = TRASH_BOTTOM_THICK / 2
wall_off     = iw_half + wt_half
outer_half   = iw_half + TRASH_WALL_THICK
inner_d_half = TRASH_INNER_DEPTH / 2
wall_zc      = TRASH_BOTTOM_THICK + inner_d_half
top_z        = TRASH_BOTTOM_THICK + TRASH_INNER_DEPTH

_geom_attr = 'solimp="0.998 0.998 0.001" solref="0.001 1" density="200" friction="0.95 0.3 0.1" group="1" rgba="0.20 0.20 0.20 1"'
TRASH_XML_PATH.write_text(f"""<mujoco model="trash_can">
  <worldbody>
    <body>
      <body name="object">
        <inertial pos="0 0 {wall_zc}" mass="{TRASH_MASS}" diaginertia="0.005 0.005 0.005"/>
        <site type="box" pos="0 0 {wall_zc}" quat="1 0 0 0" size="{iw_half} {iw_half} {inner_d_half}" group="0" rgba="0 0.8 0 0" name="contain_region"/>
        <geom {_geom_attr} type="box" pos="0 0 {bt_half}"             size="{outer_half} {outer_half} {bt_half}"/>
        <geom {_geom_attr} type="box" pos="0 {wall_off} {wall_zc}"    size="{iw_half} {wt_half} {inner_d_half}"/>
        <geom {_geom_attr} type="box" pos="0 -{wall_off} {wall_zc}"   size="{iw_half} {wt_half} {inner_d_half}"/>
        <geom {_geom_attr} type="box" pos="{wall_off} 0 {wall_zc}"    size="{wt_half} {outer_half} {inner_d_half}"/>
        <geom {_geom_attr} type="box" pos="-{wall_off} 0 {wall_zc}"   size="{wt_half} {outer_half} {inner_d_half}"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 0"      name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {top_z}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{iw_half} {iw_half} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
""")

# ==============================================================================
# dual_mod 모듈 변수 주입
# ==============================================================================
dual_mod.ROBOT_X_OFFSET = ROBOT_X_OFFSET
dual_mod.HOME_QPOS = HOME_QPOS

# ==============================================================================
# 객체 위치 범위 처리
# ==============================================================================
TABLE_LENGTH = 2.0   # main table x 크기 (env 생성의 table_full_size[0]과 일치해야 함)
rx = -(0.16 + TABLE_LENGTH/2) if MILK_REF == "robot" else 0.0

def _fix(v):  # min==max면 MuJoCo geom size 0 오류 방지
    return [v[0], v[1]] if v[0] != v[1] else [v[0] - 0.001, v[1] + 0.001]

mx = _fix([MILK_X[0] + rx, MILK_X[1] + rx])
my = _fix(MILK_Y)
mx2 = _fix([-mx[1], -mx[0]])
my2 = _fix([-my[1], -my[0]])
tx = _fix(TRASH_CAN_X)
ty = _fix(TRASH_CAN_Y)
# trash_can_2: main table 기준 y축 반대편 (x는 동일)
t2x = tx
t2y = _fix([-TRASH_CAN_Y[1], -TRASH_CAN_Y[0]])

# ==============================================================================
# 로봇 / 액션 / 출력 파일
# ==============================================================================
if args.robots == 2:
    VIDEO_OUT = pathlib.Path(__file__).parent / "preview_dual_compare.mp4"
    robots    = ["Panda", "Panda"]
    env_cfg   = "single-arm-opposed"
    action    = [0.0] * 6 + [-1.0] + [0.0] * 6 + [-1.0]
else:
    VIDEO_OUT = pathlib.Path(__file__).parent / "preview_single_compare.mp4"
    robots    = ["Panda"]
    env_cfg   = "default"
    action    = [0.0] * 6 + [-1.0]

problem_name = "LIBERO_Dual_Tabletop_Manipulation"

bddl_content = f"""(define (problem {problem_name})
  (:domain robosuite)
  (:language put milk in the trash can)
  (:regions
      (milk_region
          (:target main_table)
          (:ranges (({mx[0]} {my[0]} {mx[1]} {my[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (milk_region_2
          (:target main_table)
          (:ranges (({mx2[0]} {my2[0]} {mx2[1]} {my2[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (trash_can_region
          (:target main_table)
          (:ranges (({tx[0]} {ty[0]} {tx[1]} {ty[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (trash_can_region_2
          (:target main_table)
          (:ranges (({t2x[0]} {t2y[0]} {t2x[1]} {t2y[1]})))
          (:yaw_rotation ((0.0 0.0)))
      )
      (contain_region
          (:target trash_can_1)
      )
  )
  (:fixtures main_table - table)
  (:objects milk_1 milk_2 - milk  trash_can_1 trash_can_2 - trash_can)
  (:obj_of_interest milk_1 trash_can_1 trash_can_2)
  (:init
    (On milk_1 main_table_milk_region)
    (On milk_2 main_table_milk_region_2)
    (On trash_can_1 main_table_trash_can_region)
    (On trash_can_2 main_table_trash_can_region_2)
  )
  (:goal (And (On milk_1 main_table_milk_region)))
)"""
_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.bddl', delete=False)
_tmp.write(bddl_content)
_tmp.close()
BDDL_FILE = pathlib.Path(_tmp.name)

# ==============================================================================
# 환경 생성 + 렌더링
# ==============================================================================
CAMERAS = ['agentview', 'birdview', 'frontview', 'sideview', 'robot0_eye_in_hand', 'backview']
LABELS  = ['Agent View', 'Bird View', 'Front View', 'Side View', 'Hand Cam', 'Back View']

env = OffScreenRenderEnv(
    bddl_file_name=str(BDDL_FILE),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    robots=robots,
    env_configuration=env_cfg,
    table_full_size=(2.0, 2.0, 0.05),
)

if args.robots == 2:
    cam_id = env.env.sim.model.camera_name2id('birdview')
    env.env.sim.model.cam_pos[cam_id][:] = [-0.2, 0.0, 5.0]

env.seed(0)
obs = env.reset()

def set_free_joint_xy(sim, joint_name, x, y):
    joint_id = sim.model.joint_name2id(joint_name)
    qpos_adr = sim.model.jnt_qposadr[joint_id]
    qpos = sim.data.qpos[qpos_adr:qpos_adr + 7].copy()
    qpos[0] = x
    qpos[1] = y
    sim.data.set_joint_qpos(joint_name, qpos)

set_free_joint_xy(env.env.sim, "trash_can_1_joint0", float(TRASH_CAN_X[0]), float(TRASH_CAN_Y[0]))
set_free_joint_xy(env.env.sim, "trash_can_2_joint0", float(TRASH_CAN_X[0]), -float(TRASH_CAN_Y[0]))
env.env.sim.forward()

# 책상 무늬 제거 — table_visual 머티리얼이 사용하는 텍스처 픽셀을 회색으로 덮어쓰기
sim_model = env.env.sim.model
gid = sim_model.geom_name2id('table_visual')
mat_id = sim_model.geom_matid[gid]
TABLE_GRAY = (140, 140, 140)   # RGB 0~255

if mat_id >= 0:
    # mat_texid는 mujoco 3.x에서 [nmat, ntexrole] 형태일 수 있음
    texids = sim_model.mat_texid[mat_id]
    if hasattr(texids, '__iter__'):
        tex_id_list = [int(t) for t in texids if int(t) >= 0]
    else:
        tex_id_list = [int(texids)] if int(texids) >= 0 else []

    for tid in tex_id_list:
        # tex_data: 전체 텍스처 픽셀 평탄 배열, tex_adr[tid]부터 시작
        adr = int(sim_model.tex_adr[tid])
        nch = int(sim_model.tex_nchannel[tid]) if hasattr(sim_model, 'tex_nchannel') else 3
        npx = int(sim_model.tex_height[tid]) * int(sim_model.tex_width[tid]) * nch
        flat = list(TABLE_GRAY[:nch]) * (npx // nch)
        sim_model.tex_data[adr:adr + npx] = flat

    sim_model.mat_rgba[mat_id] = [TABLE_GRAY[0]/255, TABLE_GRAY[1]/255, TABLE_GRAY[2]/255, 1.0]

env.env.sim.forward()

def add_label(img, label):
    img = img.copy()
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return img

def make_grid(obs):
    views = []
    for cam, label in zip(CAMERAS, LABELS):
        img = obs[f"{cam}_image"][::-1]
        views.append(add_label(img, label))
    row1 = np.concatenate(views[:3], axis=1)
    row2 = np.concatenate(views[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)

frames = []
for _ in range(MAX_STEPS):
    obs, _, done, _ = env.step(action)
    frames.append(make_grid(obs))
    if done:
        break

imageio.mimwrite(str(VIDEO_OUT), frames, fps=10)
print(f"Saved: {VIDEO_OUT}")
env.close()
