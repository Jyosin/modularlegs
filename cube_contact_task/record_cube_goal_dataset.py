from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sbx
from cube_contact_task import CubePushSim, CubeTaskConfig
from modularlegs.utils.model import XMLCompiler


CAMERA_PRESETS = {
    "fixed": None,
    "follow": {"distance": 3.0, "elevation": -20.0, "azimuth": 90.0, "z_offset": 0.3},
    "front": {"distance": 2.6, "elevation": -18.0, "azimuth": 180.0, "z_offset": 0.25},
    "side": {"distance": 2.6, "elevation": -18.0, "azimuth": 90.0, "z_offset": 0.25},
    "top": {"distance": 3.2, "elevation": -89.0, "azimuth": 90.0, "z_offset": 0.0},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record synchronized multi-view videos and state data for cube-goal rollouts."
    )
    parser.add_argument("--run-dir", required=True, help="Training run directory with running_config.yaml.")
    parser.add_argument("--model", default=None, help="Model zip path. Defaults to RUN_DIR/rl_model_last.zip.")
    parser.add_argument("--output-dir", default=None, help="Dataset output directory.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--views", nargs="+", default=["fixed", "follow", "front", "side", "top"])
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--shadow-size",
        type=int,
        default=0,
        help="MuJoCo shadow map size. Default 0 records no-shadow videos; use 4096 for shadow videos.",
    )
    return parser.parse_args()


def make_record_xml(source_asset, output_dir, shadow_size):
    os.makedirs(output_dir, exist_ok=True)
    suffix = "no_shadow" if shadow_size <= 0 else f"shadow_{shadow_size}"
    xml_path = os.path.abspath(os.path.join(output_dir, f"record_{suffix}.xml"))
    compiler = XMLCompiler(source_asset)
    if shadow_size <= 0:
        compiler.remove_shadow()
    else:
        quality = compiler.root.find(".//visual/quality")
        if quality is None:
            from lxml import etree

            visual = compiler.root.find(".//visual")
            if visual is None:
                visual = etree.SubElement(compiler.root, "visual")
            quality = etree.SubElement(visual, "quality")
        quality.set("shadowsize", str(shadow_size))
    compiler.save(xml_path)
    return xml_path


def get_camera(env):
    env = env.unwrapped
    if hasattr(env, "viewer") and hasattr(env.viewer, "cam"):
        return env.viewer.cam
    if hasattr(env, "mujoco_renderer"):
        renderer = env.mujoco_renderer
        if hasattr(renderer, "viewer") and hasattr(renderer.viewer, "cam"):
            return renderer.viewer.cam
    if hasattr(env, "renderer") and hasattr(env.renderer, "cam"):
        return env.renderer.cam
    return None


def body_position(env, body_name=None):
    env = env.unwrapped
    if body_name is not None:
        try:
            return env.data.body(body_name).xpos.copy()
        except Exception:
            pass
    for candidate in ("base", "torso", "trunk", "body", "chassis", "root", "l0", "r0"):
        try:
            return env.data.body(candidate).xpos.copy()
        except Exception:
            pass
    return env.data.qpos[:3].copy()


def apply_camera(env, view):
    preset = CAMERA_PRESETS.get(view)
    if preset is None:
        return
    cam = get_camera(env)
    if cam is None:
        return
    lookat = body_position(env)
    lookat[2] += preset["z_offset"]
    cam.lookat[:] = lookat
    cam.distance = preset["distance"]
    cam.elevation = preset["elevation"]
    cam.azimuth = preset["azimuth"]


def robot_joint_state(env):
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    qpos_addr = env.model.jnt_qposadr[env.joint_idx]
    qvel_addr = env.model.jnt_dofadr[env.joint_idx]
    return {
        "qpos": qpos,
        "qvel": qvel,
        "dof_pos": qpos[qpos_addr].copy(),
        "dof_vel": qvel[qvel_addr].copy(),
    }


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    return value


def record_episode(model, cfg, output_dir, episode_idx, args):
    episode_dir = os.path.join(output_dir, f"episode_{episode_idx:03d}")
    video_dir = os.path.join(episode_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    base_env = CubePushSim(cfg, CubeTaskConfig())
    env = gym.wrappers.TimeLimit(base_env, max_episode_steps=args.steps)
    obs, info = env.reset(seed=args.seed + episode_idx)

    writers = {}
    video_paths = {}
    for view in args.views:
        path = os.path.join(video_dir, f"{view}.mp4")
        writers[view] = imageio.get_writer(path, fps=args.fps, macro_block_size=1)
        video_paths[view] = os.path.relpath(path, episode_dir)

    rollout = defaultdict(list)
    states = defaultdict(list)
    obslog_path = os.path.join(episode_dir, "episode.obs.jsonl")
    frame_indices = []
    cube_start_xy = np.asarray(info["cube_xy"], dtype=np.float32).copy()
    target_xy = np.asarray(info["target_xy"], dtype=np.float32).copy()

    cube_end_xy = cube_start_xy.copy()
    try:
        with open(obslog_path, "w", encoding="utf-8") as obslog:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                for step_idx in range(args.steps):
                    for view, writer in writers.items():
                        apply_camera(base_env, view)
                        writer.append_data(base_env.render())

                    action, _ = model.predict(obs, deterministic=args.deterministic)
                    state = robot_joint_state(base_env)
                    cube_xy = base_env._cube_xy().copy()
                    cube_vel_xy = base_env._cube_vel_xy().copy()
                    cube_target_distance = base_env._cube_target_distance()

                    rollout["observations"].append(np.asarray(obs, dtype=np.float32))
                    rollout["actions"].append(np.asarray(action, dtype=np.float32))
                    states["qpos"].append(state["qpos"])
                    states["qvel"].append(state["qvel"])
                    states["dof_pos"].append(state["dof_pos"])
                    states["dof_vel"].append(state["dof_vel"])
                    states["cube_xy"].append(cube_xy)
                    states["cube_vel_xy"].append(cube_vel_xy)
                    states["target_xy"].append(target_xy.copy())
                    states["cube_target_distance"].append(cube_target_distance)
                    states["robot_xy"].append(np.asarray(base_env.data.qpos[:2], dtype=np.float32).copy())
                    frame_indices.append(step_idx)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    rollout["rewards"].append(float(reward))
                    rollout["dones"].append(done)
                    states["contact"].append(bool(info.get("cube_contact_reward", 0.0) > 0.0))
                    states["success"].append(bool(info.get("cube_success_reward", 0.0) > 0.0))

                    obslog.write(
                        json.dumps(
                            {
                                "step_idx": step_idx,
                                "frame_idx": step_idx,
                                "obs": serializable(obs),
                                "action": serializable(action),
                                "reward": serializable(reward),
                                "done": done,
                                "cube_xy": serializable(info.get("cube_xy")),
                                "target_xy": serializable(info.get("target_xy")),
                                "cube_target_distance": serializable(info.get("cube_target_distance")),
                                "contact": bool(info.get("cube_contact_reward", 0.0) > 0.0),
                                "success": bool(info.get("cube_success_reward", 0.0) > 0.0),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if done:
                        break
                cube_end_xy = np.asarray(base_env._cube_xy(), dtype=np.float32).copy()
    finally:
        for writer in writers.values():
            writer.close()
        env.close()

    target_distance_start = float(np.linalg.norm(target_xy - cube_start_xy))
    target_distance_end = float(np.linalg.norm(target_xy - cube_end_xy))

    rollout_np = {
        "observations": np.asarray(rollout["observations"], dtype=np.float32),
        "actions": np.asarray(rollout["actions"], dtype=np.float32),
        "rewards": np.asarray(rollout["rewards"], dtype=np.float32),
        "dones": np.asarray(rollout["dones"], dtype=bool),
    }
    np.savez_compressed(os.path.join(episode_dir, "rolloutN.npz"), **rollout_np)
    np.savez_compressed(
        os.path.join(episode_dir, "states.npz"),
        qpos=np.asarray(states["qpos"], dtype=np.float32),
        qvel=np.asarray(states["qvel"], dtype=np.float32),
        dof_pos=np.asarray(states["dof_pos"], dtype=np.float32),
        dof_vel=np.asarray(states["dof_vel"], dtype=np.float32),
        cube_xy=np.asarray(states["cube_xy"], dtype=np.float32),
        cube_vel_xy=np.asarray(states["cube_vel_xy"], dtype=np.float32),
        target_xy=np.asarray(states["target_xy"], dtype=np.float32),
        cube_target_distance=np.asarray(states["cube_target_distance"], dtype=np.float32),
        robot_xy=np.asarray(states["robot_xy"], dtype=np.float32),
        contact=np.asarray(states["contact"], dtype=bool),
        success=np.asarray(states["success"], dtype=bool),
        frame_idx=np.asarray(frame_indices, dtype=np.int32),
    )

    metadata = {
        "episode_index": episode_idx,
        "seed": args.seed + episode_idx,
        "deterministic": args.deterministic,
        "steps_recorded": int(len(rollout_np["actions"])),
        "fps": args.fps,
        "shadow_size": args.shadow_size,
        "views": args.views,
        "video_paths": video_paths,
        "model_path": os.path.abspath(args.model),
        "running_config": os.path.abspath(os.path.join(args.run_dir, "running_config.yaml")),
        "cube_start_xy": cube_start_xy.tolist(),
        "cube_end_xy": cube_end_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "target_distance_start": target_distance_start,
        "target_distance_end": target_distance_end,
        "target_distance_delta": target_distance_start - target_distance_end,
        "cube_delta_xy": (cube_end_xy - cube_start_xy).tolist(),
        "cube_delta_norm": float(np.linalg.norm(cube_end_xy - cube_start_xy)),
        "contact_steps": int(np.asarray(states["contact"], dtype=bool).sum()),
        "success_steps": int(np.asarray(states["success"], dtype=bool).sum()),
        "files": {
            "rollout": "rolloutN.npz",
            "states": "states.npz",
            "obslog": "episode.obs.jsonl",
        },
    }
    with open(os.path.join(episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def main():
    args = parse_args()
    args.run_dir = os.path.abspath(args.run_dir)
    args.model = os.path.abspath(args.model or os.path.join(args.run_dir, "rl_model_last.zip"))
    output_dir = os.path.abspath(args.output_dir or os.path.join(args.run_dir, "dataset"))
    os.makedirs(output_dir, exist_ok=True)

    cfg = OmegaConf.load(os.path.join(args.run_dir, "running_config.yaml"))
    cfg.trainer.mode = "play"
    cfg.trainer.device = args.device
    cfg.sim.render = False
    cfg.sim.render_size = [args.width, args.height]
    cfg.sim.randomize_orientation = False
    cfg.sim.random_latency_scheme = False
    cfg.sim.randomize_mass = False
    cfg.sim.randomize_friction = False
    cfg.sim.randomize_rolling_friction = False
    cfg.sim.randomize_damping = False
    cfg.sim.noisy_actions = False
    cfg.sim.noisy_observations = False
    cfg.sim.noisy_init = False
    cfg.sim.randomize_ini_vel = False
    cfg.sim.asset_file = make_record_xml(cfg.sim.asset_file, output_dir, args.shadow_size)

    env_for_model = gym.wrappers.TimeLimit(CubePushSim(cfg, CubeTaskConfig()), max_episode_steps=args.steps)
    model = sbx.CrossQ.load(args.model, env=env_for_model, device=args.device)
    env_for_model.close()

    summaries = []
    for episode_idx in range(args.episodes):
        summary = record_episode(model, cfg, output_dir, episode_idx, args)
        summaries.append(summary)
        print(
            f"episode_{episode_idx:03d}: "
            f"dist {summary['target_distance_start']:.3f}->{summary['target_distance_end']:.3f}, "
            f"cube_delta={summary['cube_delta_norm']:.3f}, "
            f"contacts={summary['contact_steps']}, success={summary['success_steps']}"
        )

    with open(os.path.join(output_dir, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": summaries}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
