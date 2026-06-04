import argparse
import contextlib
import glob
import os

import gymnasium as gym
import imageio.v3 as iio
import sbx
from stable_baselines3.common.logger import configure

from modularlegs import LEG_ROOT_DIR
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.envs.gym.rendering import RecordVideo
from modularlegs.utils.files import load_cfg
from modularlegs.utils.model import XMLCompiler
from modularlegs.utils.train import load_model


EXPERIMENTS = [
    ("single", "config/shape_experiments/sim_train_shape_original.yaml"),
    ("quadruped", "config/shape_experiments/sim_train_shape_quadruped.yaml"),
    ("quadruped_angle", "config/shape_experiments/sim_train_shape_quadruped_angle.yaml"),
    ("extra_balls", "config/shape_experiments/sim_train_shape_extra_balls.yaml"),
]

DEFAULT_TARGET_STEPS = 1_000_000
DEFAULT_SNAPSHOT_INTERVAL = 100_000
DEFAULT_VIDEO_STEPS = 120


def make_train_env(conf):
    base_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=1000)
    return base_env, env


def prepare_visual_conf(conf, vis_dir, name):
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


def record_snapshot(name, cfg_name, model_path, total_steps, video_steps):
    conf = load_cfg(cfg_name, alg="sbx")
    vis_dir = os.path.join(conf.logging.data_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    conf = prepare_visual_conf(conf, vis_dir, name)

    base_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=video_steps)
    env = RecordVideo(
        env,
        video_folder=vis_dir,
        step_trigger=lambda step: step == 0,
        video_length=video_steps,
        name_prefix=f"step_{total_steps}",
        fps=int(1 / conf.robot.dt),
        disable_logger=True,
    )
    model = load_model(model_path, env, sbx.CrossQ, device="cpu")

    obs, _ = env.reset()
    iio.imwrite(os.path.join(vis_dir, f"step_{total_steps}_preview.png"), base_env.render())
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        for _ in range(video_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    env.close()


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
    args = parser.parse_args()
    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot-interval must be positive")
    if args.video_steps <= 0:
        raise ValueError("--video-steps must be positive")

    experiments = EXPERIMENTS
    if args.only:
        requested = set(args.only)
        experiments = [item for item in EXPERIMENTS if item[0] in requested]

    for name, cfg_name in experiments:
        conf = load_cfg(cfg_name, alg="sbx")
        out_dir = conf.logging.data_dir
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
                record_snapshot(name, cfg_name, snapshot_model, total_steps, args.video_steps)
            print(f"saved {name} snapshot at {total_steps} steps")

        env.close()


if __name__ == "__main__":
    main()
