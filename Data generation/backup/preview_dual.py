#!/usr/bin/env python3
import sys, os, pathlib
sys.path.insert(0, os.path.expanduser('~/RobotLearning_9/vlsa-aegis/safelibero'))

import numpy as np
import imageio
import cv2
from libero.libero.envs import OffScreenRenderEnv

import libero.libero.envs.problems.dual_tabletop_manipulation  # noqa

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("robots", type=int, choices=[1, 2], help="1 or 2")
args = parser.parse_args()

MAX_STEPS  = 30
RESOLUTION = 256

if args.robots == 2:
    BDDL_FILE = pathlib.Path(__file__).parent / "dual_scene.bddl"
    VIDEO_OUT  = pathlib.Path(__file__).parent / "preview_dual.mp4"
    robots     = ["Panda", "Panda"]
    env_cfg    = "single-arm-opposed"
    action     = [0.0] * 6 + [-1.0] + [0.0] * 6 + [-1.0]
else:
    BDDL_FILE = pathlib.Path(__file__).parent / "test_scene.bddl"
    VIDEO_OUT  = pathlib.Path(__file__).parent / "preview_single.mp4"
    robots     = ["Panda"]
    env_cfg    = "default"
    action     = [0.0] * 6 + [-1.0]

CAMERAS = ['agentview', 'birdview', 'frontview', 'sideview', 'robot0_eye_in_hand', 'backview']
LABELS  = ['Agent View', 'Bird View', 'Front View', 'Side View', 'Hand Cam', 'Back View']

env = OffScreenRenderEnv(
    bddl_file_name=str(BDDL_FILE),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    robots=robots,
    env_configuration=env_cfg,
    table_full_size=(0.6, 2.0, 0.05),
)
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
