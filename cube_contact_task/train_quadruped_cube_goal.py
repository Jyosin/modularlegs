from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import gymnasium as gym
from omegaconf import OmegaConf
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sbx
from cube_contact_task import CubePushSim, CubeTaskConfig
from modularlegs.utils.files import load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train quadruped to push a cube to a target.")
    parser.add_argument(
        "--config",
        default="shape_experiments/sim_train_shape_quadruped_10k_local",
        help="Config path under config/, or a full config path.",
    )
    parser.add_argument(
        "--output-dir",
        default="exp/quadruped_cube_goal",
        help="Directory for logs and the final model.",
    )
    parser.add_argument("--total-steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--asset-file", default="quadrupedX4air1s.xml")
    return parser.parse_args()


def make_cfg(args):
    cfg = load_cfg(args.config, alg="sbx")
    output_dir = os.path.abspath(args.output_dir)
    asset_path = args.asset_file

    cfg.logging.data_dir = output_dir
    cfg.sim.asset_file = asset_path
    cfg.sim.render = False
    cfg.sim.render_size = [960, 544]
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
    cfg.run_name = "quadruped_cube_goal"
    return cfg


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cfg = make_cfg(args)
    OmegaConf.save(cfg, os.path.join(output_dir, "running_config.yaml"))

    if os.path.isabs(args.asset_file):
        copied_asset = os.path.join(output_dir, os.path.basename(args.asset_file))
        if not os.path.exists(copied_asset):
            shutil.copy2(args.asset_file, copied_asset)

    env = Monitor(
        gym.wrappers.TimeLimit(
            CubePushSim(cfg, CubeTaskConfig()),
            max_episode_steps=args.max_episode_steps,
        ),
        filename=os.path.join(output_dir, "monitor.csv"),
    )
    model = sbx.CrossQ(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=output_dir,
        learning_starts=5000,
        batch_size=64,
        buffer_size=100000,
        train_freq=10,
        gradient_steps=1,
        policy_kwargs={"net_arch": [64, 64]},
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
