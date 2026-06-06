#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "Train/lerobot/src"))
sys.path.insert(0, str(BASE_DIR / "Data generation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import create_dataset_grasp as create_dataset  # noqa: E402
import minje_cbf_safety as cbf_safety  # noqa: E402
import minje_run_safety_eval as mse  # noqa: E402
import test_model_grasp as tmg  # noqa: E402


CANDIDATES = [
    ("grip_site", (255, 0, 0)),
    ("gripper_eef", (255, 255, 0)),
    ("right_hand", (0, 255, 0)),
    ("link7", (0, 165, 255)),
    ("link6", (255, 0, 255)),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results") / "ellipsoid_candidates_seed1_all.png",
    )
    return p.parse_args()


def _preprocess_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1]).astype(np.uint8)


def _ellipsoid_curves_from_rot(center: np.ndarray, rot: np.ndarray, semi_axes: np.ndarray, n_points: int = 72):
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    local_curves = [
        np.stack([semi_axes[0] * np.cos(t), semi_axes[1] * np.sin(t), np.zeros_like(t)], axis=1),
        np.stack([semi_axes[0] * np.cos(t), np.zeros_like(t), semi_axes[2] * np.sin(t)], axis=1),
        np.stack([np.zeros_like(t), semi_axes[1] * np.cos(t), semi_axes[2] * np.sin(t)], axis=1),
    ]
    center = np.asarray(center, dtype=float)
    return [(rot @ curve.T).T + center[None, :] for curve in local_curves]


def _body_pose(sim, body_name: str):
    body_id = sim.model.body_name2id(body_name)
    center = np.asarray(sim.data.body_xpos[body_id], dtype=float)
    rot = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    return center, rot


def _candidate_poses(sim, obs: dict, ridx: int):
    poses = {}
    poses["grip_site"] = (
        np.asarray(obs[f"robot{ridx}_eef_pos"], dtype=float),
        cbf_safety._quat_to_mat(obs[f"robot{ridx}_eef_quat"]),
    )
    poses["gripper_eef"] = _body_pose(sim, f"gripper{ridx}_eef")
    poses["right_hand"] = _body_pose(sim, f"robot{ridx}_right_hand")
    poses["link7"] = _body_pose(sim, f"robot{ridx}_link7")
    poses["link6"] = _body_pose(sim, f"robot{ridx}_link6")
    return poses


def _draw_center_marker(img: np.ndarray, sim, camera_name: str, center: np.ndarray, color):
    pts_2d, valid = mse._project_world_points(sim, camera_name, center[None, :], img.shape)
    if valid[0]:
        pt = tuple(np.round(pts_2d[0]).astype(np.int32))
        cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)
    return img


def _overlay_candidates(img: np.ndarray, sim, camera_name: str, obs: dict) -> np.ndarray:
    out = img.copy()
    semi_axes = np.asarray(cbf_safety.Q_EF, dtype=float)
    try:
        for ridx in [0, 1]:
            poses = _candidate_poses(sim, obs, ridx)
            for name, color in CANDIDATES:
                center, rot = poses[name]
                for curve in _ellipsoid_curves_from_rot(center, rot, semi_axes):
                    pts_2d, valid = mse._project_world_points(sim, camera_name, curve, out.shape)
                    out = mse._draw_visible_polyline(out, pts_2d, valid, color, closed=True)
                out = _draw_center_marker(out, sim, camera_name, center, color)
    except Exception:
        return img
    return out


def _draw_legend(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    x0, y0 = 10, 12
    box_w, box_h = 18, 12
    for idx, (name, color) in enumerate(CANDIDATES):
        y = y0 + idx * 20
        cv2.rectangle(out, (x0, y), (x0 + box_w, y + box_h), color, -1)
        cv2.putText(
            out,
            name,
            (x0 + box_w + 8, y + box_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _render_one(camera_name: str, obs: dict, sim) -> np.ndarray:
    key = f"{camera_name}_image"
    if key not in obs:
        raise KeyError(f"Missing {key}. Available keys: {sorted(obs.keys())}")
    frame = _preprocess_image(obs[key])
    frame = _overlay_candidates(frame, sim, camera_name, obs)
    frame = tmg.draw_camera_labels(frame, camera_name, {0, 1})
    return frame


def build_frame(obs: dict, sim) -> np.ndarray:
    preferred = [f"{name}_image" for name in [*tmg.POLICY_CAMERAS, *tmg.RENDER_CAMERAS]]
    imgs = []
    seen = set()
    for key in preferred:
        if key in seen or key not in obs:
            continue
        seen.add(key)
        imgs.append(_render_one(key.removesuffix("_image"), obs, sim))
    if not imgs:
        raise RuntimeError("No camera images available for rendering.")

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
            chunk = chunk + [np.zeros_like(chunk[0])] * (cols - len(chunk))
        row_imgs.append(np.concatenate(chunk, axis=1))
    frame = np.concatenate(row_imgs, axis=0)
    return _draw_legend(frame)


def main():
    args = parse_args()
    create_dataset.DATASET_CAMERAS = tmg.camera_names_for_env("all")
    create_dataset.DEBUG_CAMERAS = []

    spec = create_dataset.build_episode_spec(args.seed)
    env, _ = create_dataset.make_env(spec, args.resolution, save_debug_video=False)
    try:
        create_dataset.set_free_joint_xy(
            env.env.sim,
            "trash_can_1_joint0",
            create_dataset.TRASH_CAN_CENTER[0],
            create_dataset.TRASH_CAN_CENTER[1],
        )
        env.env.sim.forward()
        obs = env.env._get_observations()
        frame = build_frame(obs, env.env.sim)
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frame)
    print(args.out)


if __name__ == "__main__":
    main()
