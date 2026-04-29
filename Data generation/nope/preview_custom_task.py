#!/usr/bin/env python3
import sys, os, pathlib
sys.path.insert(0, os.path.expanduser('~/RobotLearning_9/vlsa-aegis/safelibero'))

import numpy as np
import imageio
import cv2
from libero.libero.envs import OffScreenRenderEnv

BDDL_FILE = pathlib.Path(__file__).parent / "test_scene.bddl"
VIDEO_OUT  = pathlib.Path(__file__).parent / "test_preview.mp4"
MAX_STEPS  = 30
RESOLUTION = 256

CAMERAS = ['agentview', 'birdview', 'frontview', 'sideview', 'robot0_eye_in_hand', 'backview']
LABELS  = ['Agent View', 'Bird View', 'Front View', 'Side View', 'Hand Cam', 'Back View']

env = OffScreenRenderEnv(
    bddl_file_name=str(BDDL_FILE),
    camera_heights=RESOLUTION,
    camera_widths=RESOLUTION,
    camera_names=CAMERAS,
    table_full_size=(0.6, 2.0, 0.05),  # (길이, 가로, 높이) 기본값: (1.0, 1.2, 0.05)
)
env.seed(0)
env.reset()
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
    # 2행 3열 grid
    row1 = np.concatenate(views[:3], axis=1)
    row2 = np.concatenate(views[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)

frames = []
for _ in range(MAX_STEPS):
    obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
    frames.append(make_grid(obs))
    if done:
        break

imageio.mimwrite(str(VIDEO_OUT), frames, fps=10)
print(f"Saved: {VIDEO_OUT}")
env.close()
