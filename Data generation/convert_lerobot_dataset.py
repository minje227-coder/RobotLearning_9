#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_FPS = 10
DEFAULT_RESOLUTION = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=True)
    parser.add_argument("--vcodec", default="auto")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    return parser.parse_args()


def load_raw_episode(path: pathlib.Path):
    data = np.load(path)
    required = ["side", "wrist", "state", "action", "task"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    side = data["side"]
    wrist = data["wrist"]
    state = data["state"]
    action = data["action"]
    task = str(data["task"].item())

    if side.ndim != 4 or side.shape[-1] != 3:
        raise ValueError(f"{path} side shape must be [T,H,W,3], got {side.shape}")
    if wrist.shape != side.shape:
        raise ValueError(f"{path} wrist shape {wrist.shape} does not match side {side.shape}")
    if state.shape != (side.shape[0], 8):
        raise ValueError(f"{path} state shape must be [T,8], got {state.shape}")
    if action.shape != (side.shape[0], 8):
        raise ValueError(f"{path} action shape must be [T,8], got {action.shape}")

    return side, wrist, state.astype(np.float32), action.astype(np.float32), task


def create_dataset(args):
    if not args.raw_dir.exists():
        raise FileNotFoundError(args.raw_dir)
    episode_paths = sorted(args.raw_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No raw episode_*.npz files found in {args.raw_dir}")

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists. Use --overwrite: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    features = {
        "observation.images.side": {
            "dtype": "image",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (args.resolution, args.resolution, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["action"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        robot_type="dual_panda_robot0",
        fps=args.fps,
        features=features,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
        vcodec=args.vcodec,
    )

    saved = 0
    try:
        for episode_path in episode_paths:
            side, wrist, state, action, task = load_raw_episode(episode_path)
            for idx in range(side.shape[0]):
                dataset.add_frame(
                    {
                        "observation.images.side": side[idx],
                        "observation.images.wrist": wrist[idx],
                        "observation.state": state[idx],
                        "action": action[idx],
                        "task": task,
                    }
                )
            dataset.save_episode()
            saved += 1
            print(f"[convert] saved episode {saved}/{len(episode_paths)} from {episode_path.name}")
    finally:
        dataset.finalize()

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["lerobot", "smolvla", "sideview", "wrist", "libero", "dual-tabletop"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Saved LeRobotDataset: {args.output_dir}")
    print(f"episodes={saved}")


def main():
    args = parse_args()
    create_dataset(args)


if __name__ == "__main__":
    main()
