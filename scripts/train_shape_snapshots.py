import argparse
import contextlib
import glob
import json
import os
import shutil
from collections import defaultdict

import gymnasium as gym
import imageio.v2 as imageio
import imageio.v3 as iio
import numpy as np
import sbx
from stable_baselines3.common.logger import configure

from modularlegs.envs.env_sim import ZeroSim
from modularlegs.utils.files import load_cfg
from modularlegs.utils.train import load_model, save_rollout

try:
    from generate_shape_variants import SHAPE_VARIANTS
except ImportError:
    from scripts.generate_shape_variants import SHAPE_VARIANTS


BASE_EXPERIMENTS = [
    ("single", "config/shape_experiments/sim_train_shape_original.yaml"),
    ("quadruped", "config/shape_experiments/sim_train_shape_quadruped.yaml"),
    ("quadruped_angle", "config/shape_experiments/sim_train_shape_quadruped_angle.yaml"),
    ("quadruped_angle_large", "config/shape_experiments/sim_train_shape_quadruped_angle_large.yaml"),
    ("quadruped_angle_weird", "config/shape_experiments/sim_train_shape_quadruped_angle_weird.yaml"),
    ("quadruped_one_leg_up", "config/shape_experiments/sim_train_shape_quadruped_one_leg_up.yaml"),
    ("quadruped_rear_leg_up", "config/shape_experiments/sim_train_shape_quadruped_rear_leg_up.yaml"),
    ("extra_balls", "config/shape_experiments/sim_train_shape_extra_balls.yaml"),
]
SHAPE_VARIANT_TEMPLATE = "config/shape_experiments/sim_train_shape_chain5.yaml"
SHAPE_EXPERIMENTS = [
    (asset_name.removesuffix("_air1s"), SHAPE_VARIANT_TEMPLATE, f"{asset_name}.xml")
    for asset_name in SHAPE_VARIANTS
]
EXPERIMENTS = BASE_EXPERIMENTS + SHAPE_EXPERIMENTS

DEFAULT_TARGET_STEPS = 1_000_000
DEFAULT_SNAPSHOT_INTERVAL = 100_000
DEFAULT_VIDEO_STEPS = 120
GITHUB_ORIGINAL_OUTPUT_ROOT = "exp/shape_experiments_github_original"
GITHUB_ORIGINAL_COMMANDS = [0.6, 0, 2]
GITHUB_ORIGINAL_REWARD_PARAMS = [0.7, 0.3, 0, 0.0, -0.02, -0.01, -0.000002]


def episode_max_steps(conf):
    return None if conf.agent.done_version is None else 1000


def make_train_env(conf):
    base_env = ZeroSim(conf)
    max_episode_steps = episode_max_steps(conf)
    env = (
        base_env
        if max_episode_steps is None
        else gym.wrappers.TimeLimit(base_env, max_episode_steps=max_episode_steps)
    )
    return base_env, env


def apply_github_original_shape_method(conf):
    if conf.agent.reward_version != "cheat_isaac_general":
        return conf
    conf.agent.done_version = "ballance_up"
    conf.agent.predefined_commands = list(GITHUB_ORIGINAL_COMMANDS)
    conf.agent.reward_params = list(GITHUB_ORIGINAL_REWARD_PARAMS)
    return conf


def load_experiment_cfg(
    cfg_name,
    name,
    asset_file=None,
    output_root=None,
    github_original_method=False,
):
    conf = load_cfg(cfg_name, alg="sbx")
    if asset_file is not None:
        conf.sim.asset_file = asset_file
        conf.logging.data_dir = os.path.join("exp", "shape_experiments", name)
    if output_root is not None:
        conf.logging.data_dir = os.path.join(output_root, name)
    if github_original_method:
        conf = apply_github_original_shape_method(conf)
    return conf


def prepare_visual_conf(conf, vis_dir, name):
    from modularlegs import LEG_ROOT_DIR
    from modularlegs.utils.model import XMLCompiler

    conf.trainer.mode = "play"
    conf.trainer.device = "cpu"
    conf.sim.render = False
    conf.sim.render_size = [640, 360]
    conf.sim.randomize_orientation = False
    conf.sim.random_latency_scheme = False
    conf.sim.randomize_mass = False
    conf.sim.randomize_friction = False
    conf.sim.randomize_rolling_friction = False
    conf.sim.randomize_damping = False
    conf.sim.noisy_actions = False
    conf.sim.noisy_observations = False
    conf.sim.noisy_init = False
    conf.sim.randomize_ini_vel = False

    source_asset = conf.sim.asset_file
    source_xml = (
        source_asset
        if os.path.isabs(source_asset)
        else os.path.join(
            LEG_ROOT_DIR, "modularlegs", "sim", "assets", "robots", source_asset
        )
    )
    render_xml = os.path.abspath(os.path.join(vis_dir, f"{name}_no_shadow.xml"))
    compiler = XMLCompiler(source_xml)
    compiler.remove_shadow()
    compiler.save(render_xml)
    conf.sim.asset_file = render_xml
    return conf


def record_snapshot(
    name,
    cfg_name,
    asset_file,
    model_path,
    total_steps,
    video_steps,
    output_root=None,
    github_original_method=False,
):
    conf = load_experiment_cfg(
        cfg_name,
        name,
        asset_file,
        output_root=output_root,
        github_original_method=github_original_method,
    )
    vis_dir = os.path.join(conf.logging.data_dir, "visualization")
    save_dir = os.path.join(vis_dir, f"rl_model_{total_steps}")
    os.makedirs(save_dir, exist_ok=True)
    conf = prepare_visual_conf(conf, vis_dir, name)

    base_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=video_steps)
    model = load_model(model_path, env, sbx.CrossQ, device="cpu")

    obs, _ = env.reset()
    preview_path = os.path.join(save_dir, "preview.png")
    video_path = os.path.join(save_dir, "episode_000.mp4")
    video_ref_path = os.path.join(save_dir, "rollout_video_refs.json")
    iio.imwrite(preview_path, base_env.render())
    writer = imageio.get_writer(
        video_path,
        fps=int(1 / conf.robot.dt),
        macro_block_size=1,
    )
    rollout = defaultdict(list)

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        try:
            for frame_idx in range(video_steps):
                action, _ = model.predict(obs, deterministic=True)
                rollout["observations"].append(np.expand_dims(np.asarray(obs), axis=0))
                rollout["actions"].append(np.expand_dims(np.asarray(action), axis=0))

                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                rollout["rewards"].append(np.asarray([reward]))
                rollout["dones"].append(np.asarray([done]))
                writer.append_data(base_env.render())
                if done:
                    break
        finally:
            writer.close()
            env.close()

    save_rollout(rollout, save_dir)
    with open(video_ref_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_model": model_path,
                "video_path": os.path.relpath(video_path, save_dir),
                "video_env_index": 0,
                "fps": int(1 / conf.robot.dt),
                "video_every_n_steps": 1,
                "frame_index_semantics": "one frame is written for each rollout step in this single-env snapshot",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"recorded {name} snapshot at {total_steps}: {save_dir}")


def get_existing_max_step(out_dir):
    steps = []
    for path in glob.glob(os.path.join(out_dir, "rl_model_*.zip")):
        stem = os.path.basename(path).removesuffix(".zip")
        suffix = stem.removeprefix("rl_model_")
        if suffix.isdigit():
            steps.append(int(suffix))
    return max(steps) if steps else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL)
    parser.add_argument("--video-steps", type=int, default=DEFAULT_VIDEO_STEPS)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Remove existing checkpoints for selected experiments before training")
    parser.add_argument("--output-root", default=None, help="Override experiment output root, e.g. exp/shape_experiments_github_original")
    parser.add_argument("--github-original-method", action="store_true", help="Use the original shape-training reward settings for cheat_isaac_general configs")
    args = parser.parse_args()
    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot-interval must be positive")
    if args.video_steps <= 0:
        raise ValueError("--video-steps must be positive")

    experiments = EXPERIMENTS
    if args.only:
        requested = set(args.only)
        experiments = [item for item in EXPERIMENTS if item[0] in requested]

    for item in experiments:
        name, cfg_name, *asset_file = item
        asset_file = asset_file[0] if asset_file else None
        output_root = args.output_root
        if args.github_original_method and output_root is None:
            output_root = GITHUB_ORIGINAL_OUTPUT_ROOT
        conf = load_experiment_cfg(
            cfg_name,
            name,
            asset_file,
            output_root=output_root,
            github_original_method=args.github_original_method,
        )
        out_dir = conf.logging.data_dir
        if args.fresh and os.path.isdir(out_dir):
            for pattern in ["rl_model_*.zip", "rl_model_last.zip", "progress.csv"]:
                for path in glob.glob(os.path.join(out_dir, pattern)):
                    os.remove(path)
            shutil.rmtree(os.path.join(out_dir, "visualization"), ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        start_steps = get_existing_max_step(out_dir)
        if start_steps >= args.target_steps:
            print(f"skip {name}: already has {start_steps} steps")
            continue

        _, env = make_train_env(conf)
        model_path = os.path.join(out_dir, "rl_model_last.zip")
        model = load_model(model_path if os.path.exists(model_path) else None, env, sbx.CrossQ)
        model.set_logger(configure(out_dir, ["stdout", "csv", "tensorboard"]))

        total_steps = start_steps
        while total_steps < args.target_steps:
            chunk_steps = min(args.snapshot_interval, args.target_steps - total_steps)
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
            total_steps += chunk_steps
            snapshot_model = os.path.join(out_dir, f"rl_model_{total_steps}.zip")
            model.save(snapshot_model)
            model.save(model_path)
            if not args.no_video:
                record_snapshot(
                    name,
                    cfg_name,
                    asset_file,
                    snapshot_model,
                    total_steps,
                    args.video_steps,
                    output_root=output_root,
                    github_original_method=args.github_original_method,
                )
            print(f"saved {name} snapshot at {total_steps} steps")

        env.close()


if __name__ == "__main__":
    main()
