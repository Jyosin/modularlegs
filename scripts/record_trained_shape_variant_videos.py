import argparse
import contextlib
import os

import gymnasium as gym
import imageio.v3 as iio
import sbx

from modularlegs import LEG_ROOT_DIR
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.envs.gym.rendering import RecordVideo
from modularlegs.utils.files import load_cfg
from modularlegs.utils.model import XMLCompiler
from modularlegs.utils.train import load_model

try:
    from generate_shape_variants import SHAPE_VARIANTS
except ImportError:
    from scripts.generate_shape_variants import SHAPE_VARIANTS


SHAPE_VARIANT_TEMPLATE = "config/shape_experiments/sim_train_shape_chain5.yaml"


def prepare_visual_conf(name, asset_file):
    conf = load_cfg(SHAPE_VARIANT_TEMPLATE, alg="sbx")
    conf.sim.asset_file = asset_file
    conf.logging.data_dir = os.path.join("exp", "shape_experiments", name)
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

    vis_dir = os.path.join(conf.logging.data_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    source_xml = os.path.join(LEG_ROOT_DIR, "modularlegs", "sim", "assets", "robots", asset_file)
    render_xml = os.path.abspath(os.path.join(vis_dir, f"{name}_no_shadow.xml"))
    compiler = XMLCompiler(source_xml)
    compiler.remove_shadow()
    compiler.save(render_xml)
    conf.sim.asset_file = render_xml
    return conf, vis_dir


def record_video(name, asset_file, model_step, video_steps):
    model_path = os.path.join(
        "exp", "shape_experiments", name, f"rl_model_{model_step}.zip"
    )
    if not os.path.exists(model_path):
        print(f"skip {name}: missing {model_path}")
        return

    conf, vis_dir = prepare_visual_conf(name, asset_file)
    base_env = ZeroSim(conf)
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=video_steps)
    env = RecordVideo(
        env,
        video_folder=vis_dir,
        step_trigger=lambda step: step == 0,
        video_length=video_steps,
        name_prefix=f"trained_{model_step}",
        fps=int(1 / conf.robot.dt),
        disable_logger=True,
    )
    model = load_model(model_path, env, sbx.CrossQ, device="cpu")

    obs, _ = env.reset()
    iio.imwrite(os.path.join(vis_dir, f"trained_{model_step}_preview.png"), base_env.render())
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        for _ in range(video_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    env.close()
    print(f"saved {name}: {vis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--model-step", type=int, default=1_000_000)
    parser.add_argument("--video-steps", type=int, default=120)
    args = parser.parse_args()

    requested = set(args.only) if args.only else None
    for asset_name in SHAPE_VARIANTS:
        name = asset_name.removesuffix("_air1s")
        if requested is not None and name not in requested:
            continue
        record_video(name, f"{asset_name}.xml", args.model_step, args.video_steps)


if __name__ == "__main__":
    main()
