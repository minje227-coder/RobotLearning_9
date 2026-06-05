#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from huggingface_hub import snapshot_download
import imageio
import numpy as np
import torch

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "Train/lerobot/src"))
sys.path.insert(0, str(BASE_DIR / "Data generation"))

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


DEFAULT_POLICY_PATH = BASE_DIR / "outputs/my_smolvla/checkpoints/100000/pretrained_model"
DATASET_CAMERAS = ["all", "sideview", "robot0_eye_in_hand", "robot1_eye_in_hand", "frontview", "agentview", "birdview", "backview"]
POLICY_CAMERAS = [
    "sideview_robot0_left",
    "robot0_eye_in_hand",
    "sideview_robot0_right",
    "sideview_robot1_left",
    "robot1_eye_in_hand",
    "sideview_robot1_right",
]
RENDER_CAMERAS = ["sideview", "birdview", "backview"]
FIXED_ACTION = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
FPS_DEFAULT = 10
RESOLUTION_DEFAULT = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot0-policy-path",
        default=None,
        help="Local pretrained_model path or Hugging Face model repo id for robot0. If omitted, robot0 uses fixed action.",
    )
    parser.add_argument(
        "--robot1-policy-path",
        default=None,
        help="Local pretrained_model path or Hugging Face model repo id for robot1. If omitted, robot1 uses fixed action.",
    )
    parser.add_argument(
        "--policy-path",
        default=None,
        help="Backward-compatible alias to set both --robot0-policy-path and --robot1-policy-path.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default="pick up and place the milk in the trash can")
    parser.add_argument(
        "--robot0-task",
        default="put the milk into the target area inside the trash can",
        help="Language instruction for robot0 policy inference.",
    )
    parser.add_argument(
        "--robot1-task",
        default="put the distractor orange juice into the target area inside the trash can",
        help="Language instruction for robot1 policy inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Environment seed.",
    )
    parser.add_argument("--video-out", type=Path, default=Path(__file__).with_name("test_data_episode1.mp4"))
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--robot0-start-delay-sec", type=float, default=0.0, help="Delay before robot0 scripted grasp starts.")
    parser.add_argument("--robot1-start-delay-sec", type=float, default=0.0, help="Delay before robot1 scripted grasp starts.")
    parser.add_argument("--vcodec", default="libx264")
    parser.add_argument("--camera", choices=DATASET_CAMERAS, default="sideview")
    return parser.parse_args()


def resolve_policy_root(policy_ref: str | Path) -> Path:
    policy_ref = str(policy_ref)
    local_path = Path(policy_ref).expanduser()
    if local_path.exists():
        return local_path
    print(f"[info] downloading HF model repo: {policy_ref}")
    return Path(
        snapshot_download(
            repo_id=policy_ref,
            repo_type="model",
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "policy_preprocessor*.json",
                "policy_preprocessor*.safetensors",
                "policy_postprocessor*.json",
                "policy_postprocessor*.safetensors",
                "train_config.json",
            ],
        )
    )


def normalize_robot_action(action: np.ndarray, name: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (8,):
        raise ValueError(f"Expected {name} action shape (8,), got {action.shape}")

    action = action.copy()
    action[:7] = np.clip(action[:7], -0.2, 0.2)
    action[7] = 1.0 if action[7] > 0.0 else -1.0
    return action


def make_dual_action(robot0_action: np.ndarray, robot1_action: np.ndarray) -> np.ndarray:
    robot0_action = np.asarray(robot0_action, dtype=np.float32)
    if robot0_action.shape != (8,):
        raise ValueError(f"Expected robot0 action shape (8,), got {robot0_action.shape}")
    robot1_action = np.asarray(robot1_action, dtype=np.float32)
    if robot1_action.shape != (8,):
        raise ValueError(f"Expected robot1 action shape (8,), got {robot1_action.shape}")

    return np.concatenate(
        [
            normalize_robot_action(robot0_action, "robot0"),
            normalize_robot_action(robot1_action, "robot1"),
        ]
    ).astype(np.float32)


class LoadedPolicy:
    def __init__(self, policy_root: Path, device: torch.device):
        with (policy_root / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.policy_type = cfg["type"]
        self.input_features = cfg.get("input_features", {})
        self.state_dim = max(8, int(self.input_features.get("observation.state", {}).get("shape", [8])[0]))
        self.policy_class = get_policy_class(self.policy_type)
        self.model = self.policy_class.from_pretrained(str(policy_root))
        self.model.to(device)
        self.model.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.model.config,
            str(policy_root),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        self.device = device

    def reset(self) -> None:
        self.model.reset()

    def infer(self, obs: dict[str, np.ndarray], task: str | None = None) -> np.ndarray:
        policy_obs = prepare_observation_for_inference(obs, self.device, task=task, robot_type=None)
        policy_obs = self.preprocessor(policy_obs)
        with torch.inference_mode():
            action = self.model.select_action(policy_obs)
        action = self.postprocessor(action)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_state(env, obs: dict, robot_idx: int, state_dim: int) -> np.ndarray:
    joint_key = f"robot{robot_idx}_joint_pos"
    gripper_key = f"robot{robot_idx}_gripper_qpos"

    if joint_key in obs:
        joint = np.asarray(obs[joint_key], dtype=np.float32)
    else:
        robot = env.env.robots[robot_idx]
        joint = np.asarray(env.env.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float32)

    if gripper_key in obs:
        gripper = np.asarray(obs[gripper_key], dtype=np.float32)
    else:
        robot = env.env.robots[robot_idx]
        gripper = np.asarray(env.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float32)

    base_state = np.concatenate([joint, [float(np.mean(gripper))]]).astype(np.float32)
    if base_state.shape[0] >= state_dim:
        return base_state[:state_dim]
    pad = np.zeros(state_dim - base_state.shape[0], dtype=np.float32)
    return np.concatenate([base_state, pad]).astype(np.float32)


def build_policy_observation(
    env,
    env_obs: dict,
    robot_idx: int,
    state_dim: int,
    input_features: dict,
    image_rotation: str = "vertical_flip",
) -> dict[str, np.ndarray]:
    def preprocess_image(img):
        if image_rotation == "vertical_flip":
            img = img[::-1]
        return np.ascontiguousarray(img).astype(np.uint8)

    side_left_key = f"sideview_robot{robot_idx}_left_image"
    wrist_key = f"robot{robot_idx}_eye_in_hand_image"
    side_right_key = f"sideview_robot{robot_idx}_right_image"

    if side_left_key not in env_obs:
        side_left_key = "sideview_robot0_left_image"
    if wrist_key not in env_obs:
        wrist_key = "robot0_eye_in_hand_image"
    if side_right_key not in env_obs:
        side_right_key = "sideview_robot0_right_image"

    side_left = preprocess_image(env_obs[side_left_key])
    wrist = preprocess_image(env_obs[wrist_key])
    side_right = preprocess_image(env_obs[side_right_key])

    obs = {
        "observation.state": extract_state(env, env_obs, robot_idx, state_dim),
        # Match create_dataset.py / create_dataset_robot1.py camera mapping.
        "observation.images.side_left": side_left,
        "observation.images.wrist": wrist,
        "observation.images.side_right": side_right,
    }

    return obs


def policy_label(camera_name: str, active_policy_robots: set[int]) -> str | None:
    labels = []
    if camera_name == "sideview_robot0_left" and 0 in active_policy_robots:
        labels.append("policy0 side_left")
    elif camera_name == "sideview_robot0_right" and 0 in active_policy_robots:
        labels.append("policy0 side_right")
    elif camera_name == "sideview_robot1_left" and 1 in active_policy_robots:
        labels.append("policy1 side_left")
    elif camera_name == "sideview_robot1_right" and 1 in active_policy_robots:
        labels.append("policy1 side_right")
    elif camera_name == "robot0_eye_in_hand" and 0 in active_policy_robots:
        labels.append("policy0 wrist")
    elif camera_name == "robot1_eye_in_hand" and 1 in active_policy_robots:
        labels.append("policy1 wrist")
    if not labels:
        return None
    return ", ".join(labels)


def draw_label(img: np.ndarray, label: str, position: str) -> np.ndarray:
    img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(img.shape[0], img.shape[1]) / 430.0)
    thickness = max(1, int(round(scale * 2)))
    margin = max(6, int(round(scale * 10)))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
    box_w = min(text_w + margin * 2, img.shape[1])
    box_h = text_h + baseline + margin * 2
    y0 = 0 if position == "top" else img.shape[0] - box_h
    y1 = y0 + box_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (box_w, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    cv2.putText(img, label, (margin, y0 + margin + text_h), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def draw_camera_labels(img: np.ndarray, camera_name: str, active_policy_robots: set[int]) -> np.ndarray:
    img = draw_label(img, camera_name, "top")
    policy = policy_label(camera_name, active_policy_robots)
    if policy is not None:
        img = draw_label(img, policy, "bottom")
    return img


def build_render_frame(env_obs: dict, camera: str, active_policy_robots: set[int]) -> np.ndarray:
    def preprocess_image(img):
        return np.ascontiguousarray(img[::-1]).astype(np.uint8)

    if camera != "all":
        key = f"{camera}_image"
        if key not in env_obs:
            raise KeyError(f"Missing {key}. Available keys: {sorted(env_obs.keys())}")
        return draw_camera_labels(preprocess_image(env_obs[key]), camera, active_policy_robots)

    preferred = [f"{name}_image" for name in [*POLICY_CAMERAS, *RENDER_CAMERAS]]
    imgs = []
    seen = set()
    for key in preferred:
        if key in seen:
            continue
        seen.add(key)
        if key not in env_obs:
            continue
        camera_name = key.removesuffix("_image")
        imgs.append(draw_camera_labels(preprocess_image(env_obs[key]), camera_name, active_policy_robots))
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


def camera_names_for_env(camera: str) -> list[str]:
    if camera == "all":
        return list(dict.fromkeys([*POLICY_CAMERAS, *RENDER_CAMERAS]))
    return list(dict.fromkeys([*POLICY_CAMERAS, camera]))


def load_policy(policy_ref: str | Path | None, device: torch.device, name: str) -> tuple[LoadedPolicy | None, Path | None]:
    if policy_ref is None:
        print(f"[info] {name}_policy=fixed_action")
        return None, None
    policy_root = resolve_policy_root(policy_ref)
    return LoadedPolicy(policy_root, device), policy_root


def select_robot_action(policy: LoadedPolicy | None, env, obs: dict, robot_idx: int, task: str | None) -> np.ndarray:
    if policy is None:
        return FIXED_ACTION.copy()
    policy_obs = build_policy_observation(env, obs, robot_idx, policy.state_dim, policy.input_features)
    return policy.infer(policy_obs, task=task)


def replay_episode(args) -> None:
    import create_dataset_grasp as create_dataset
    import create_dataset_robot1_grasp as create_dataset_robot1

    episode_seed = args.seed
    rollout_steps = args.max_steps

    policy0_ref = args.policy_path if args.policy_path is not None else args.robot0_policy_path
    policy1_ref = args.policy_path if args.policy_path is not None else args.robot1_policy_path
    device = torch.device(args.device)
    policy0, policy0_root = load_policy(policy0_ref, device, "robot0")
    policy1, policy1_root = load_policy(policy1_ref, device, "robot1")
    active_policy_robots = {idx for idx, policy in enumerate([policy0, policy1]) if policy is not None}
    single_policy_mode = len(active_policy_robots) == 1
    if single_policy_mode:
        single_policy_robot = 0 if policy0 is not None else 1
        print("[info] single policy detected; only the policy-loaded robot will use scripted grasp before policy control")

    spec = create_dataset.build_episode_spec(episode_seed)
    env = None

    def lock_passive_robot_for_single_policy() -> None:
        if not single_policy_mode:
            return
        if single_policy_robot == 0:
            create_dataset.lock_robot1_pose(env.env.sim, env.env.robots[1])
        else:
            create_dataset_robot1.lock_robot0_pose(env.env.sim, env.env.robots[0])

    try:
        env_cameras = camera_names_for_env(args.camera)
        create_dataset.DATASET_CAMERAS = env_cameras
        create_dataset.DEBUG_CAMERAS = []
        env, obs = create_dataset.make_env(spec, args.resolution, save_debug_video=False)
        print(f"[info] env_cameras={env_cameras}")
        set_free_joint_xy = create_dataset.set_free_joint_xy
        set_free_joint_xy(
            env.env.sim,
            "trash_can_1_joint0",
            create_dataset.TRASH_CAN_CENTER[0],
            create_dataset.TRASH_CAN_CENTER[1],
        )
        env.env.sim.forward()
        lock_passive_robot_for_single_policy()
        obs = env.env._get_observations()

        if policy0 is not None:
            policy0.reset()
        if policy1 is not None:
            policy1.reset()

        if single_policy_mode:
            if single_policy_robot == 0:
                waypoints0 = create_dataset.solve_waypoints(env, obs)
                move_sequence0 = [
                    ("move_pregrasp", waypoints0["pregrasp"], create_dataset.GRIPPER_OPEN),
                    ("move_grasp", waypoints0["grasp"], create_dataset.GRIPPER_OPEN),
                ]
                phase0 = "open_then_approach"
                phase1 = ""
                phase_steps0 = 0
                phase_steps1 = 0
                close_hold_steps0 = 0
                close_hold_steps1 = 0
                move_index0 = 0
                move_index1 = 0
                grasp_done0 = False
                grasp_done1 = True
                move_sequence1 = []
            else:
                waypoints1 = create_dataset_robot1.solve_waypoints(env, obs)
                move_sequence1 = [
                    ("move_pregrasp", waypoints1["pregrasp"], create_dataset_robot1.GRIPPER_OPEN),
                    ("move_grasp", waypoints1["grasp"], create_dataset_robot1.GRIPPER_OPEN),
                ]
                phase0 = ""
                phase1 = "open_then_approach"
                phase_steps0 = 0
                phase_steps1 = 0
                close_hold_steps0 = 0
                close_hold_steps1 = 0
                move_index0 = 0
                move_index1 = 0
                grasp_done0 = True
                grasp_done1 = False
                move_sequence0 = []
        else:
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
            phase0 = "open_then_approach"
            phase1 = "open_then_approach"
            phase_steps0 = 0
            phase_steps1 = 0
            close_hold_steps0 = 0
            close_hold_steps1 = 0
            move_index0 = 0
            move_index1 = 0
            grasp_done0 = False
            grasp_done1 = False
        policy_active0 = False
        policy_active1 = False
        robot0_start_delay_steps = max(0, int(round(args.robot0_start_delay_sec * args.fps)))
        robot1_start_delay_steps = max(0, int(round(args.robot1_start_delay_sec * args.fps)))
        robot0_started = robot0_start_delay_steps == 0
        robot1_started = robot1_start_delay_steps == 0
        frames = []

        for step in range(rollout_steps):
            label_robots = {
                idx
                for idx, (policy, is_active) in enumerate(
                    [(policy0, policy_active0), (policy1, policy_active1)]
                )
                if policy is not None and is_active
            }
            frames.append(build_render_frame(obs, args.camera, label_robots))
            cur0 = obs["robot0_joint_pos"]
            cur1 = create_dataset_robot1.get_robot_joint_pos(env, 1)

            robot0_action = FIXED_ACTION.copy()
            robot1_action = FIXED_ACTION.copy()

            script_robot0 = ((not single_policy_mode) or single_policy_robot == 0) and not policy_active0
            script_robot1 = ((not single_policy_mode) or single_policy_robot == 1) and not policy_active1

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
                            if move_index0 >= len(move_sequence0):
                                phase0 = "close_gripper"
                            else:
                                phase0 = move_sequence0[move_index0][0]
                            phase_steps0 = 0
                    elif phase0 == "close_gripper":
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
                        close_hold_steps0 += 1
                        if close_hold_steps0 >= create_dataset.GRIP_CLOSE_HOLD_STEPS:
                            phase0 = "grasp_hold"
                            grasp_done0 = True
                    elif phase0 == "grasp_hold":
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
                    else:
                        robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
                else:
                    robot0_action = create_dataset.make_joint_position_action(cur0, cur0, gripper_cmd=create_dataset.GRIPPER_CLOSE)
            elif policy_active0:
                robot0_action = select_robot_action(policy0, env, obs, 0, args.robot0_task)

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
                            if move_index1 >= len(move_sequence1):
                                phase1 = "close_gripper"
                            else:
                                phase1 = move_sequence1[move_index1][0]
                            phase_steps1 = 0
                    elif phase1 == "close_gripper":
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
                        close_hold_steps1 += 1
                        if close_hold_steps1 >= create_dataset_robot1.GRIP_CLOSE_HOLD_STEPS:
                            phase1 = "grasp_hold"
                            grasp_done1 = True
                    elif phase1 == "grasp_hold":
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
                    else:
                        robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
                else:
                    robot1_action = create_dataset_robot1.make_joint_position_action(cur1, cur1, gripper_cmd=create_dataset_robot1.GRIPPER_CLOSE)
            elif policy_active1:
                robot1_action = select_robot_action(policy1, env, obs, 1, args.robot1_task)

            obs, _, env_done, _ = env.step(make_dual_action(robot0_action, robot1_action))
            lock_passive_robot_for_single_policy()
            obs = env.env._get_observations()

            if script_robot0 and not robot0_started:
                robot0_start_delay_steps -= 1
                if robot0_start_delay_steps <= 0:
                    robot0_started = True
                    print(f"[info] robot0 start delay elapsed at step {step}; starting scripted grasp")
            elif script_robot0:
                phase_steps0 += 1
            if script_robot1 and not robot1_started:
                robot1_start_delay_steps -= 1
                if robot1_start_delay_steps <= 0:
                    robot1_started = True
                    print(f"[info] robot1 start delay elapsed at step {step}; starting scripted grasp")
            elif script_robot1:
                phase_steps1 += 1

            if policy0 is not None and grasp_done0 and not policy_active0:
                policy_active0 = True
                print(f"[info] robot0 reached grasp_hold at step {step}; enabling policy0")
            if policy1 is not None and grasp_done1 and not policy_active1:
                policy_active1 = True
                print(f"[info] robot1 reached grasp_hold at step {step}; enabling policy1")

            if env_done:
                print(f"[info] env_done=True at step {step}; continuing replay actions.")

        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video_out, frames, fps=args.fps, codec=args.vcodec)
        print(f"Saved: {args.video_out}")
        print(f"robot0_policy_root={policy0_root}")
        print(f"robot1_policy_root={policy1_root}")
        print(f"seed={episode_seed} steps={rollout_steps}")
        print(f"final_success={env.env._check_success()}")
    finally:
        if env is not None:
            env.close()
        spec.bddl_file.unlink(missing_ok=True)


def main():
    replay_episode(parse_args())


if __name__ == "__main__":
    main()
