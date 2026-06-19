import argparse
import os
import sys

if sys.platform != "darwin":
    os.environ.setdefault("MUJOCO_GL", "egl")
elif os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("MUJOCO_GL")

import imageio.v2 as imageio
import sbx

from modularlegs import LEG_ROOT_DIR
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.utils.files import load_cfg
from modularlegs.utils.model import XMLCompiler
from modularlegs.utils.train import load_model


DEFAULT_CONFIG = "shape_experiments/sim_train_shape_quadruped_one_leg_up"


def _experiment_name(conf):
    return os.path.basename(os.path.normpath(conf.logging.data_dir))


def _robot_xml_path(asset_file):
    if os.path.isabs(asset_file):
        return asset_file
    return os.path.join(
        LEG_ROOT_DIR,
        "modularlegs",
        "sim",
        "assets",
        "robots",
        asset_file,
    )


def prepare_record_conf(conf, output_dir, width, height):
    conf.trainer.mode = "play"
    conf.trainer.device = "cpu"
    conf.sim.render = False
    conf.sim.render_size = [width, height]
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
    conf.logging.print_data = False

    source_xml = _robot_xml_path(conf.sim.asset_file)
    render_xml = os.path.abspath(
        os.path.join(output_dir, f"{_experiment_name(conf)}_no_shadow.xml")
    )
    compiler = XMLCompiler(source_xml)
    compiler.remove_shadow()
    compiler.save(render_xml)
    conf.sim.asset_file = render_xml
    return conf


def _get_body_pos(env, body_name=None):
    env = env.unwrapped
    if body_name is not None:
        try:
            return env.data.body(body_name).xpos.copy()
        except Exception:
            pass

    for candidate in ["torso0", "l0", "r0"]:
        try:
            return env.data.body(candidate).xpos.copy()
        except Exception:
            pass

    try:
        return env.data.qpos[:3].copy()
    except Exception:
        return None


def update_follow_camera(
    env,
    body_name=None,
    distance=3.0,
    elevation=-20.0,
    azimuth=90.0,
    z_offset=0.3,
):
    env = env.unwrapped
    pos = _get_body_pos(env, body_name=body_name)
    if pos is None:
        return False

    lookat = pos.copy()
    lookat[2] += z_offset

    cam = None
    if hasattr(env, "viewer") and hasattr(env.viewer, "cam"):
        cam = env.viewer.cam
    elif hasattr(env, "mujoco_renderer"):
        renderer = env.mujoco_renderer
        if hasattr(renderer, "viewer") and hasattr(renderer.viewer, "cam"):
            cam = renderer.viewer.cam
    elif hasattr(env, "renderer") and hasattr(env.renderer, "cam"):
        cam = env.renderer.cam

    if cam is None:
        return False

    cam.lookat[:] = lookat
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return True


def record_policy_video(args):
    conf = load_cfg(args.config, alg="sbx")
    name = args.name or _experiment_name(conf)
    output_dir = args.output_dir or os.path.join(conf.logging.data_dir, "visualization")
    os.makedirs(output_dir, exist_ok=True)

    model_path = args.model_path or os.path.join(
        conf.logging.data_dir,
        f"rl_model_{args.model_step}.zip",
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    conf = prepare_record_conf(conf, output_dir, args.width, args.height)
    fps = int(round(1 / conf.robot.dt))
    video_steps = int(round(args.seconds * fps))
    video_path = args.video_path or os.path.join(
        output_dir,
        f"{name}_{args.model_step}_{int(args.seconds)}s"
        f"{'_follow' if args.camera_mode == 'follow' else ''}.mp4",
    )

    env = ZeroSim(conf)
    model = load_model(model_path, env, sbx.CrossQ, device=args.device)

    obs, _ = env.reset()
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for _ in range(video_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            if args.camera_mode == "follow":
                update_follow_camera(
                    env,
                    body_name=args.camera_follow_body,
                    distance=args.camera_distance,
                    elevation=args.camera_elevation,
                    azimuth=args.camera_azimuth,
                    z_offset=args.camera_z_offset,
                )
            writer.append_data(env.render())
            if args.stop_on_done and (done or truncated):
                break
    finally:
        writer.close()
        env.close()

    print(video_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--name", default=None)
    parser.add_argument("--model-step", type=int, default=1_000_000)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--stop-on-done", action="store_true")
    parser.add_argument("--camera-mode", choices=["fixed", "follow"], default="fixed")
    parser.add_argument("--camera-follow-body", default=None)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-z-offset", type=float, default=0.3)
    args = parser.parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    record_policy_video(args)


if __name__ == "__main__":
    main()
