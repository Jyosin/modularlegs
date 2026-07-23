from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import gymnasium as gym
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sbx
from stable_baselines3.common.monitor import Monitor

from cube_contact_task import CubePushSim, CubeRobotVisionWrapper, CubeTaskConfig, CubeVisionConfig
from modularlegs.utils.files import load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train quadruped cube pushing from robot-centric camera input.")
    parser.add_argument(
        "--config",
        default="shape_experiments/sim_train_shape_quadruped_10k_local",
        help="Config path under config/, or a full config path.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--asset-file", default="quadrupedX4air1s.xml")
    parser.add_argument(
        "--camera-mode",
        choices=("level", "mounted"),
        required=True,
        help="level keeps the robot camera horizontal; mounted follows robot body pitch/roll/yaw.",
    )
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--views", nargs="+", default=["front", "back", "left", "right"])
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--no-proprioception",
        action="store_true",
        help="Use only images. By default the policy also receives robot proprioception.",
    )
    return parser.parse_args()


def make_cfg(args):
    cfg = load_cfg(args.config, alg="sbx")
    output_dir = os.path.abspath(args.output_dir)

    cfg.logging.data_dir = output_dir
    cfg.sim.asset_file = args.asset_file
    cfg.sim.render = False
    cfg.sim.render_size = [args.image_width, args.image_height]
    cfg.sim.randomize_orientation = False
    cfg.sim.random_latency_scheme = False
    cfg.sim.randomize_mass = False
    cfg.sim.randomize_friction = False
    cfg.sim.noisy_actions = False
    cfg.sim.noisy_observations = False
    cfg.robot.noisy = False
    cfg.robot.randomize_start_pos = False
    cfg.trainer.total_steps = args.total_steps
    cfg.trainer.seed = args.seed
    cfg.run_name = f"quadruped_cube_vision_goal_{args.camera_mode}"
    return cfg


def make_env(cfg, args):
    task_cfg = CubeTaskConfig(observe_cube=False)
    vision_cfg = CubeVisionConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        views=tuple(args.views),
        camera_mode=args.camera_mode,
        include_proprioception=not args.no_proprioception,
    )
    env = CubePushSim(cfg, task_cfg)
    env = CubeRobotVisionWrapper(env, vision_cfg)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=args.max_episode_steps)
    return Monitor(env, filename=os.path.join(os.path.abspath(args.output_dir), "monitor.csv"))


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg = make_cfg(args)
    OmegaConf.save(cfg, os.path.join(output_dir, "running_config.yaml"))
    OmegaConf.save(
        OmegaConf.create(
            {
                "camera_mode": args.camera_mode,
                "image_height": args.image_height,
                "image_width": args.image_width,
                "views": args.views,
                "include_proprioception": not args.no_proprioception,
                "observation": (
                    "robot_centric_rgb_images_plus_proprioception_channel"
                    if not args.no_proprioception
                    else "robot_centric_rgb_images_only"
                ),
            }
        ),
        os.path.join(output_dir, "vision_config.yaml"),
    )

    if os.path.isabs(args.asset_file):
        copied_asset = os.path.join(output_dir, os.path.basename(args.asset_file))
        if not os.path.exists(copied_asset):
            shutil.copy2(args.asset_file, copied_asset)

    env = make_env(cfg, args)
    model = sbx.CrossQ(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=output_dir,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        train_freq=10,
        gradient_steps=1,
        seed=args.seed,
    )

    start = time.time()
    model.learn(total_timesteps=args.total_steps, progress_bar=False)
    model.save(os.path.join(output_dir, "rl_model_last"))
    env.close()

    print("saved_model", os.path.join(output_dir, "rl_model_last.zip"))
    print("elapsed_sec", round(time.time() - start, 1))


if __name__ == "__main__":
    main()
