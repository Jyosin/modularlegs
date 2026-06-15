import argparse
import os
import shutil
import subprocess
import sys

from omegaconf import OmegaConf

from modularlegs.utils.files import load_cfg

try:
    from generate_shape_variants import SHAPE_VARIANTS
except ImportError:
    from scripts.generate_shape_variants import SHAPE_VARIANTS


TEMPLATE_CONFIG = "shape_experiments/sim_train_shape_chain5"
RECORD_CONFIG_DIR = os.path.join("exp", "shape_experiments", "_record_configs")
DEFAULT_OUTPUT_DIR = "recordings"


def build_record_config(name, asset_file, model_step, record_steps, output_dir):
    model_path = os.path.abspath(
        os.path.join("exp", "shape_experiments", name, f"rl_model_{model_step}.zip")
    )
    if not os.path.exists(model_path):
        return None

    conf = load_cfg(TEMPLATE_CONFIG, alg="sbx")
    conf.sim.asset_file = asset_file
    conf.sim.render = False
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
    conf.sim.render_size = [640, 360]

    conf.logging.data_dir = os.path.join("exp", "shape_experiments", name)
    conf.trainer.mode = "record"
    conf.trainer.device = "cpu"
    conf.trainer.load_run = model_path
    conf.trainer.wandb_on = False
    conf.trainer.joystick = False
    conf.trainer.record.camera_mode = "follow"
    conf.trainer.record.record_steps = int(record_steps)
    conf.trainer.record.num_envs = 1
    conf.trainer.record.video = True
    conf.trainer.record.video_name = f"{name}_follow.mp4"
    conf.trainer.record.save_obs_jsonl = False
    conf.record = {"camera_mode": "follow"}

    os.makedirs(RECORD_CONFIG_DIR, exist_ok=True)
    config_path = os.path.abspath(os.path.join(RECORD_CONFIG_DIR, f"{name}_record_follow.yaml"))
    OmegaConf.save(conf, config_path)
    return config_path, model_path


def record_one(name, asset_file, model_step, record_steps, output_dir):
    built = build_record_config(name, asset_file, model_step, record_steps, output_dir)
    if built is None:
        print(f"skip {name}: missing exp/shape_experiments/{name}/rl_model_{model_step}.zip")
        return

    config_path, model_path = built
    cmd = [sys.executable, "modularlegs/scripts/train_sbx_record_with_video.py", config_path]
    print(f"record {name}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    source_video = os.path.join(
        os.path.dirname(model_path),
        model_stem,
        f"{name}_follow.mp4",
    )
    if not os.path.exists(source_video):
        raise FileNotFoundError(f"expected video was not created: {source_video}")

    os.makedirs(output_dir, exist_ok=True)
    final_video = os.path.join(output_dir, f"{name}_follow.mp4")
    shutil.copy2(source_video, final_video)
    print(f"saved {final_video}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--model-step", type=int, default=1_000_000)
    parser.add_argument("--record-steps", type=int, default=120)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    requested = set(args.only) if args.only else None
    for asset_name in SHAPE_VARIANTS:
        name = asset_name.removesuffix("_air1s")
        if requested is not None and name not in requested:
            continue
        record_one(
            name=name,
            asset_file=f"{asset_name}.xml",
            model_step=args.model_step,
            record_steps=args.record_steps,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
