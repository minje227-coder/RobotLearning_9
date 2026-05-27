#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
import imageio
import numpy as np
import pandas as pd


DEFAULT_DATASET_ROOT = Path(__file__).with_name("lerobot_robot0_dataset_v3")
DEFAULT_ROBOT1_DATASET_ROOT = Path(__file__).with_name("lerobot_robot1_dataset_v3")
DATASET_CAMERAS = ["sideview", "sideview_robot0_left", "robot0_eye_in_hand", "sideview_robot0_right"]
FPS_DEFAULT = 10
RESOLUTION_DEFAULT = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot0-dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Local LeRobot dataset path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--robot1-dataset-root",
        default=str(DEFAULT_ROBOT1_DATASET_ROOT),
        help="Local LeRobot dataset path or Hugging Face repo id.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Backward-compatible alias for --robot0-dataset-root. Accepts local path or HF repo id.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=1,
        help="Backward-compatible episode index used for both robots unless per-robot indices are set.",
    )
    parser.add_argument("--robot0-episode-index", type=int, default=None)
    parser.add_argument("--robot1-episode-index", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Environment seed. Defaults to episode-index; episode 1 in dataset_1 was seed 1.",
    )
    parser.add_argument("--video-out", type=Path, default=Path(__file__).with_name("test_data_episode1.mp4"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=RESOLUTION_DEFAULT)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--vcodec", default="libx264")
    parser.add_argument("--camera", choices=DATASET_CAMERAS, default="sideview_robot0_left")
    return parser.parse_args()


def resolve_dataset_root(dataset_ref: str | Path) -> Path:
    dataset_ref = str(dataset_ref)
    local_path = Path(dataset_ref).expanduser()
    if local_path.exists():
        return local_path

    print(f"[info] downloading HF dataset repo: {dataset_ref}")
    return Path(
        snapshot_download(
            repo_id=dataset_ref,
            repo_type="dataset",
            allow_patterns=[
                "data/**/*.parquet",
                "meta/**/*.json",
                "meta/**/*.parquet",
            ],
        )
    )


def load_episode_actions(dataset_root: Path, episode_index: int) -> np.ndarray:
    data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_parquet(data_path)
    episode = df[df["episode_index"] == episode_index].sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"episode_index={episode_index} not found in {data_path}")

    actions = np.stack(episode["action"].to_numpy()).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Expected episode actions shape [T, 8], got {actions.shape}")
    return actions


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


def replay_episode(args) -> None:
    from create_dataset import TRASH_CAN_CENTER, build_episode_spec, make_env, set_free_joint_xy

    robot0_episode_index = args.episode_index if args.robot0_episode_index is None else args.robot0_episode_index
    robot1_episode_index = args.episode_index if args.robot1_episode_index is None else args.robot1_episode_index
    episode_seed = robot0_episode_index if args.seed is None else args.seed
    robot0_dataset_ref = args.dataset_root if args.dataset_root is not None else args.robot0_dataset_root
    robot0_dataset_root = resolve_dataset_root(robot0_dataset_ref)
    robot1_dataset_root = resolve_dataset_root(args.robot1_dataset_root)
    robot0_actions = load_episode_actions(robot0_dataset_root, robot0_episode_index)
    robot1_actions = load_episode_actions(robot1_dataset_root, robot1_episode_index)
    if len(robot0_actions) != len(robot1_actions):
        min_len = min(len(robot0_actions), len(robot1_actions))
        print(
            f"[warn] action length mismatch: robot0={len(robot0_actions)} "
            f"robot1={len(robot1_actions)}. Replaying first {min_len} steps."
        )
        robot0_actions = robot0_actions[:min_len]
        robot1_actions = robot1_actions[:min_len]
    if args.max_steps is not None:
        robot0_actions = robot0_actions[: args.max_steps]
        robot1_actions = robot1_actions[: args.max_steps]

    spec = build_episode_spec(episode_seed)
    env = None
    try:
        env, obs = make_env(spec, args.resolution, save_debug_video=False)
        set_free_joint_xy(env.env.sim, "trash_can_1_joint0", TRASH_CAN_CENTER[0], TRASH_CAN_CENTER[1])
        env.env.sim.forward()
        obs = env.env._get_observations()

        camera_aliases = {
            "sideview": "sideview_robot0_left",
            "sideview_robot0_left": "sideview_robot0_left",
            "sideview_robot0_right": "sideview_robot0_right",
            "robot0_eye_in_hand": "robot0_eye_in_hand",
        }
        image_key = f"{camera_aliases[args.camera]}_image"
        if image_key not in obs:
            raise KeyError(f"Missing {image_key}. Available keys: {sorted(obs.keys())}")

        frames = []
        for step, (robot0_action, robot1_action) in enumerate(zip(robot0_actions, robot1_actions)):
            frames.append(np.ascontiguousarray(obs[image_key][::-1]).astype(np.uint8))
            obs, _, env_done, _ = env.step(make_dual_action(robot0_action, robot1_action))
            obs = env.env._get_observations()
            if env_done:
                print(f"[info] env_done=True at step {step}; continuing replay actions.")

        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video_out, frames, fps=args.fps, codec=args.vcodec)
        print(f"Saved: {args.video_out}")
        print(f"robot0_dataset_root={robot0_dataset_root}")
        print(f"robot1_dataset_root={robot1_dataset_root}")
        print(
            f"robot0_episode_index={robot0_episode_index} "
            f"robot1_episode_index={robot1_episode_index} seed={episode_seed} steps={len(robot0_actions)}"
        )
        print(f"final_success={env.env._check_success()}")
    finally:
        if env is not None:
            env.close()
        spec.bddl_file.unlink(missing_ok=True)


def main():
    replay_episode(parse_args())


if __name__ == "__main__":
    main()
