import os

import gymnasium as gym
import imageio.v3 as iio
import sbx

from modularlegs.envs.env_sim import ZeroSim
from modularlegs.envs.gym.rendering import RecordVideo
from modularlegs import LEG_ROOT_DIR
from modularlegs.utils.files import load_cfg
from modularlegs.utils.model import XMLCompiler
from modularlegs.utils.train import load_model


EXPERIMENTS = [
    ("original", "config/shape_experiments/sim_train_shape_original.yaml"),
    ("comr0", "config/shape_experiments/sim_train_shape_comr0.yaml"),
    ("chopped0", "config/shape_experiments/sim_train_shape_chopped0.yaml"),
]


def prepare_visual_conf(conf):
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
    return conf


def main():
    for name, cfg_name in EXPERIMENTS:
        conf = prepare_visual_conf(load_cfg(cfg_name, alg="sbx"))
        out_dir = conf.logging.data_dir
        model_path = os.path.join(out_dir, "rl_model_last.zip")
        vis_dir = os.path.join(out_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

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

        base_env = ZeroSim(conf)
        env = gym.wrappers.TimeLimit(base_env, max_episode_steps=120)
        env = RecordVideo(
            env,
            video_folder=vis_dir,
            step_trigger=lambda step: step == 0,
            video_length=120,
            fps=int(1 / conf.robot.dt),
            disable_logger=True,
            full_name=f"{name}_rollout.mp4",
        )
        model = load_model(model_path, env, sbx.CrossQ, device="cpu")

        obs, _ = env.reset()
        iio.imwrite(os.path.join(vis_dir, f"{name}_preview.png"), base_env.render())

        for _ in range(120):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()
        print(f"saved {name}: {vis_dir}")


if __name__ == "__main__":
    main()
