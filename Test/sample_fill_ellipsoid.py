#!/usr/bin/env python3
"""Sample: render one frame with SEMI-TRANSPARENT FILLED ellipsoids (instead of
wireframe). Monkeypatches minje_run_safety_eval's ellipsoid overlay, runs a short
rollout, and saves a single birdview frame as PNG."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
import imageio
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "Train/lerobot/src"))
sys.path.insert(0, str(BASE / "Data generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import minje_run_safety_eval as mse        # noqa: E402
import test_model_grasp as tmg             # noqa: E402

FILL_ALPHA = 0.38
COLORS_NORMAL = {"grip_site": (255, 0, 0), "link7": (0, 255, 0)}      # red / green
COLORS_INTERVENE = {"grip_site": (0, 255, 255), "link7": (255, 0, 255)}  # cyan / magenta when CBF acts


def _ellipsoid_surface(center, rot, semi, nu=30, nv=16):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    uu, vv = np.meshgrid(u, v)
    local = np.stack([(np.cos(uu) * np.sin(vv)).ravel(),
                      (np.sin(uu) * np.sin(vv)).ravel(),
                      np.cos(vv).ravel()], axis=1) * semi
    return (rot @ local.T).T + np.asarray(center, float)


def filled_overlay(img, sim, env, camera_name, obs, intervened):
    out = img.copy()
    semi = np.asarray(mse.cbf_safety.Q_EF, dtype=float)
    palette = COLORS_INTERVENE if intervened else COLORS_NORMAL
    try:
        for ridx in [0, 1]:
            targets = mse.cbf_safety.collision_targets(
                sim, env.env.robots[ridx], ridx,
                obs[f"robot{ridx}_eef_pos"], obs[f"robot{ridx}_eef_quat"],
                include_jacobian=False)
            for t in targets:
                color = palette.get(t["name"], (255, 0, 0))
                pts3d = _ellipsoid_surface(t["center"], t["rot"], semi)
                pts2d, valid = mse._project_world_points(sim, camera_name, pts3d, out.shape)
                P = pts2d[valid].astype(np.int32)
                if len(P) >= 3:
                    hull = cv2.convexHull(P)
                    ov = out.copy()
                    cv2.fillConvexPoly(ov, hull, color)
                    cv2.addWeighted(ov, FILL_ALPHA, out, 1 - FILL_ALPHA, 0, dst=out)
                    cv2.polylines(out, [hull], True, color, 1, cv2.LINE_AA)
    except Exception:
        return img
    return out


# --- swap in the filled renderer ---
mse._overlay_ellipsoids_on_camera_image = filled_overlay

P0 = str(Path.home() / "workspace/vlsa_smolvla_robot0_v4/checkpoints/015000/pretrained_model")
P1 = str(Path.home() / "workspace/vlsa_smolvla_robot1_v4/checkpoints/015000/pretrained_model")
sys.argv = ["x",
            "--device", "cuda", "--seeds", "1",
            "--robot0-policy-path", P0, "--robot1-policy-path", P1,
            "--robot0-task", "put the milk into the black box",
            "--robot1-task", "put the orange juice into the black box",
            "--max-steps", "500", "--robot1-start-delay-sec", "8",
            "--video-camera", "all", "--video-seeds", "1",
            "--video-dir", "/tmp/fill_sample",
            "--safety-arms", "0", "1",
            "--policy-replan-horizon", "50", "--trajectory-horizon", "0",
            "--out-csv", "/tmp/fs.csv", "--out-json", "/tmp/fs.json", "--label", "FILL"]
args = mse.parse_args()
device = torch.device("cuda")
p0, _ = tmg.load_policy(args.robot0_policy_path, device, "r0")
p1, _ = tmg.load_policy(args.robot1_policy_path, device, "r1")

vpath = Path("/tmp/fill_sample/FILL_seed1.mp4")
mse.run_episode(args, 1, p0, p1, video_path=vpath)

# Save the filled (_ellipsoid) video to a persistent location.
import shutil
ell = vpath.with_name("FILL_seed1_ellipsoid.mp4")
dest = BASE / "Test/minje_safety_test/filled_ellipsoid_sample_intervene.mp4"
shutil.copy(str(ell), str(dest))
print("SAVED_VIDEO", dest)
