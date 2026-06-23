import argparse
import contextlib
import glob
import json
import os
import shutil
from collections import defaultdict

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sbx
import yaml
from omegaconf import OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from modularlegs import LEG_ROOT_DIR
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.utils.files import load_cfg
from modularlegs.utils.logger import plot_learning_curve
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
DEFAULT_RECORD_STEPS = 120
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


def to_serializable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]
    return x


def save_run_conf(conf, cfg_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(cfg_name):
        shutil.copy(cfg_name, out_dir)
    with open(os.path.join(out_dir, "running_config.yaml"), "w") as file:
        yaml.dump(OmegaConf.to_container(conf, resolve=True), file, default_flow_style=False)
    with open(os.path.join(out_dir, "note.txt"), "w") as f:
        f.write(str(getattr(conf.trainer, "notes", "")))

    asset_files = (
        conf.sim.asset_file
        if isinstance(conf.sim.asset_file, (list, tuple))
        else [conf.sim.asset_file]
    )
    asset_log_dir = os.path.join(out_dir, "assets")
    os.makedirs(asset_log_dir, exist_ok=True)
    for asset_file in asset_files:
        xml_file = (
            asset_file
            if os.path.isabs(asset_file)
            else os.path.join(LEG_ROOT_DIR, "modularlegs", "sim", "assets", "robots", asset_file)
        )
        if os.path.exists(xml_file):
            shutil.copy(xml_file, asset_log_dir)


def prepare_record_conf(conf):
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
    render_xml = os.path.abspath(
        os.path.join(
            conf.logging.data_dir,
            f"{os.path.basename(os.path.normpath(conf.logging.data_dir))}_no_shadow.xml",
        )
    )
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
    record_steps,
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
    model_path = os.path.abspath(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    save_dir = os.path.join(os.path.dirname(model_path), model_stem)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "_source_model.txt"), "w") as f:
        f.write(model_path + "\n")
    conf = prepare_record_conf(conf)

    base_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=record_steps)
    model = load_model(model_path, env, sbx.CrossQ, device="cpu")

    obs, _ = env.reset()
    constructed_obs = obs
    video_path = os.path.join(save_dir, "episode_000.mp4")
    obslog_path = os.path.join(save_dir, "episode_000.obs.jsonl")
    video_ref_path = os.path.join(save_dir, "rollout_video_refs.json")
    writer = imageio.get_writer(
        video_path,
        fps=int(1 / conf.robot.dt),
        macro_block_size=1,
    )
    rollout = defaultdict(list)
    obslog_fp = open(obslog_path, "w", encoding="utf-8")

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        try:
            for step_idx in range(record_steps):
                action, _ = model.predict(obs, deterministic=True)
                rollout["observations"].append(np.expand_dims(np.asarray(constructed_obs), axis=0))
                rollout["actions"].append(np.expand_dims(np.asarray(action), axis=0))

                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                constructed_obs = obs
                rollout["rewards"].append(np.asarray([reward]))
                rollout["dones"].append(np.asarray([done]))
                rollout["frame_idx"].append(np.asarray([step_idx], dtype=np.int32))
                writer.append_data(base_env.render())
                obslog_fp.write(
                    json.dumps(
                        {
                            "step_idx": step_idx,
                            "frame_idx": step_idx,
                            "video_env_index": 0,
                            "obs": to_serializable(constructed_obs),
                            "action": to_serializable(action),
                            "reward": to_serializable(reward),
                            "done": to_serializable(done),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if done:
                    break
        finally:
            obslog_fp.close()
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
                "frame_index_semantics": "rollout['frame_idx'][t][i] == frame index in video, or -1 if no frame was written for env i",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"recorded {name} model {os.path.basename(model_path)}: {save_dir}")


def get_existing_max_step(out_dir):
    steps = []
    for path in glob.glob(os.path.join(out_dir, "rl_model_*.zip")):
        stem = os.path.basename(path).removesuffix(".zip")
        suffix = stem.removeprefix("rl_model_")
        suffix = suffix.removesuffix("_steps")
        if suffix.isdigit():
            steps.append(int(suffix))
    return max(steps) if steps else 0


def checkpoint_paths(out_dir):
    paths = []
    for path in glob.glob(os.path.join(out_dir, "rl_model_*_steps.zip")):
        stem = os.path.basename(path).removesuffix(".zip")
        suffix = stem.removeprefix("rl_model_").removesuffix("_steps")
        if suffix.isdigit():
            paths.append((int(suffix), path))
    return [path for _, path in sorted(paths)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--target-steps", type=int, default=DEFAULT_TARGET_STEPS)
    parser.add_argument("--snapshot-interval", type=int, default=DEFAULT_SNAPSHOT_INTERVAL, help="Checkpoint save frequency; matches train.py default at 100000")
    parser.add_argument("--record-steps", type=int, default=DEFAULT_RECORD_STEPS)
    parser.add_argument("--video-steps", type=int, default=None, help="Deprecated alias for --record-steps")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Remove existing checkpoints for selected experiments before training")
    parser.add_argument("--output-root", default=None, help="Override experiment output root, e.g. exp/shape_experiments_github_original")
    parser.add_argument("--github-original-method", action="store_true", help="Use the original shape-training reward settings for cheat_isaac_general configs")
    args = parser.parse_args()
    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot-interval must be positive")
    if args.video_steps is not None:
        args.record_steps = args.video_steps
    if args.record_steps <= 0:
        raise ValueError("--record-steps must be positive")

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
            for pattern in ["rl_model_*.zip", "rl_model_last.zip", "progress.csv", "events.out.tfevents.*", "curve.png"]:
                for path in glob.glob(os.path.join(out_dir, pattern)):
                    os.remove(path)
            for path in glob.glob(os.path.join(out_dir, "rl_model_*_steps")):
                shutil.rmtree(path, ignore_errors=True)
            shutil.rmtree(os.path.join(out_dir, "rl_model_last"), ignore_errors=True)
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
        save_run_conf(conf, cfg_name, out_dir)

        checkpoint_callback = CheckpointCallback(
            save_freq=args.snapshot_interval,
            save_path=out_dir,
            name_prefix="rl_model",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            model.learn(
                total_timesteps=args.target_steps - start_steps,
                callback=[checkpoint_callback],
                reset_num_timesteps=start_steps == 0,
            )
        model.save(model_path)
        try:
            plot_learning_curve(
                os.path.join(out_dir, "progress.csv"),
                os.path.join(out_dir, "curve.png"),
            )
        except Exception as exc:
            print(f"warning: failed to plot learning curve for {name}: {exc}")

        if not args.no_video:
            models_to_record = checkpoint_paths(out_dir)
            if os.path.exists(model_path):
                models_to_record.append(model_path)
            for snapshot_model in models_to_record:
                record_snapshot(
                    name,
                    cfg_name,
                    asset_file,
                    snapshot_model,
                    args.record_steps,
                    output_root=output_root,
                    github_original_method=args.github_original_method,
                )
        print(f"saved {name} through {args.target_steps} steps")

        env.close()


if __name__ == "__main__":
    main()
