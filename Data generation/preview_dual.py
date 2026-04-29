#!/usr/bin/env python3
import sys, os, pathlib
sys.path.insert(0, os.path.expanduser('~/RobotLearning_9/vlsa-aegis/safelibero'))

import numpy as np
import imageio
import cv2
from libero.libero.envs import OffScreenRenderEnv

import libero.libero.envs.problems.dual_tabletop_manipulation as dual_mod  # noqa

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("robots", type=int, choices=[1, 2], help="1 or 2")
parser.add_argument("--side-depth",    type=float, default=0.8, help="side table x 깊이 (m), 기본=table_length/2")
parser.add_argument("--side-width",    type=float, default=0.7,  help="side table y 너비 (m)")
parser.add_argument("--side-y-offset", type=float, default=0.5, help="main_table 중심에서 y 거리 (m), 기본=table_width/2+side_width/2")
parser.add_argument("--side-x-offset", type=float, default=0.0,  help="로봇 기준 x 추가 오프셋 (m)")
# 객체 위치 (main_table 중심 기준, x_min y_min x_max y_max)
parser.add_argument("--basket-x", type=float, nargs=2, default=[-0.0, -0.0], metavar=('MIN','MAX'), help="바구니 x 범위")
parser.add_argument("--basket-y", type=float, nargs=2, default=[-0.3, -0.3], metavar=('MIN','MAX'), help="바구니 y 범위")
parser.add_argument("--milk-x",   type=float, nargs=2, default=[-0.0,-0.0], metavar=('MIN','MAX'), help="우유 x 범위")
parser.add_argument("--milk-y",   type=float, nargs=2, default=[0.4,  0.4], metavar=('MIN','MAX'), help="우유 y 범위")
parser.add_argument("--milk-ref", type=str,   default="robot", choices=["table","robot"],
                    help="우유 위치 기준 (table/robot)")
args = parser.parse_args()

dual_mod.SIDE_DEPTH    = args.side_depth
dual_mod.SIDE_WIDTH    = args.side_width
dual_mod.SIDE_Y_OFFSET = args.side_y_offset
dual_mod.SIDE_X_OFFSET = args.side_x_offset

MAX_STEPS  = 30
RESOLUTION = 256

import tempfile
TABLE_LENGTH = 0.6
rx = -(0.16 + TABLE_LENGTH/2) if args.milk_ref == "robot" else 0.0

def _fix(v):  # min==max면 MuJoCo geom size 0 오류 방지
    return [v[0], v[1]] if v[0] != v[1] else [v[0] - 0.001, v[1] + 0.001]

bx, by = _fix(args.basket_x), _fix(args.basket_y)
mx = _fix([args.milk_x[0] + rx, args.milk_x[1] + rx])
my = _fix(args.milk_y)

if args.robots == 2:
    VIDEO_OUT = pathlib.Path(__file__).parent / "preview_dual.mp4"
    robots    = ["Panda", "Panda"]
    env_cfg   = "single-arm-opposed"
    action    = [0.0] * 6 + [-1.0] + [0.0] * 6 + [-1.0]
else:
    VIDEO_OUT = pathlib.Path(__file__).parent / "preview_single.mp4"
    robots    = ["Panda"]
    env_cfg   = "default"
    action    = [0.0] * 6 + [-1.0]

problem_name = "LIBERO_Dual_Tabletop_Manipulation"

bddl_content = f"""(define (problem {problem_name})
  (:domain robosuite)
  (:language scene)
  (:regions
      (object_region
          (:target main_table)
          (:ranges (({mx[0]} {my[0]} {mx[1]} {my[1]})))
      )
      (box_region
          (:target main_table)
          (:ranges (({bx[0]} {by[0]} {bx[1]} {by[1]})))
      )
      (contain_region
          (:target basket_1)
      )
  )
  (:fixtures main_table - table)
  (:objects milk_1 - milk  basket_1 - basket)
  (:obj_of_interest milk_1 basket_1)
  (:init
    (On milk_1 main_table_object_region)
    (On basket_1 main_table_box_region)
  )
  (:goal (And (In milk_1 basket_1_contain_region)))
)"""
_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.bddl', delete=False)
_tmp.write(bddl_content)
_tmp.close()
BDDL_FILE = pathlib.Path(_tmp.name)

CAMERAS = ['agentview', 'birdview', 'frontview', 'sideview', 'robot0_eye_in_hand', 'backview']
LABELS  = ['Agent View', 'Bird View', 'Front View', 'Side View', 'Hand Cam', 'Back View']

env = OffScreenRenderEnv(
    bddl_file_name=str(BDDL_FILE),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    robots=robots,
    env_configuration=env_cfg,
    table_full_size=(0.6, 1.7, 0.05),
)

# dual 모드일 때 side table이 보이도록 birdview 카메라 높이 조정
if args.robots == 2:
    cam_id = env.env.sim.model.camera_name2id('birdview')
    env.env.sim.model.cam_pos[cam_id][:] = [-0.2, 0.0, 5.0]
env.seed(0)
obs = env.reset()

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
