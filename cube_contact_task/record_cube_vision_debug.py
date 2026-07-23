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
from cube_contact_task import CubePushSim, CubeRobotVisionWrapper, CubeTaskConfig, CubeVisionConfig
from cube_contact_task.record_cube_goal_dataset import apply_camera, make_record_xml, robot_joint_state


def parse_args():
    parser = argparse.ArgumentParser(description="Record cube vision-policy input and shadow state videos.")
    parser.add_argument("--run-dir", required=True, help="Training run directory with running_config.yaml.")
    parser.add_argument("--model", default=None, help="Optional model zip. If missing, random actions are used.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--camera-mode", choices=("level", "mounted"), default=None)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=64)
    parser.add_argument("--views", nargs="+", default=["front", "back", "left", "right"])
    parser.add_argument("--state-view", default="follow", choices=["fixed", "follow", "front", "side", "top"])
    parser.add_argument("--state-width", type=int, default=960)
    parser.add_argument("--state-height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--shadow-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_vision_config(run_dir, args):
    vision_path = os.path.join(run_dir, "vision_config.yaml")
    data = OmegaConf.load(vision_path) if os.path.exists(vision_path) else OmegaConf.create({})
    return CubeVisionConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        views=tuple(args.views),
        camera_mode=args.camera_mode or data.get("camera_mode", "level"),
        include_proprioception=True,
    )


def make_cfg(args):
    cfg = OmegaConf.load(os.path.join(args.run_dir, "running_config.yaml"))
    cfg.trainer.mode = "play"
    cfg.trainer.device = args.device
    cfg.sim.render = False
    cfg.sim.render_size = [args.state_width, args.state_height]
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
    cfg.sim.asset_file = make_record_xml(cfg.sim.asset_file, args.output_dir, args.shadow_size)
    return cfg


def make_env(cfg, vision_cfg, steps):
    base_env = CubePushSim(cfg, CubeTaskConfig(observe_cube=False))
    env = CubeRobotVisionWrapper(base_env, vision_cfg)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=steps)
    return env, base_env


def state_panel(dof_pos, dof_vel, width, height):
    frame = np.full((height, width, 3), 245, dtype=np.uint8)
    margin = 28
    usable_width = width - 2 * margin
    values = np.concatenate([np.asarray(dof_pos), np.asarray(dof_vel)])
    if values.size == 0:
        return frame
    values = np.clip(values, -3.0, 3.0) / 3.0
    bar_h = max(4, (height - 2 * margin) // values.size)
    center = margin + usable_width // 2
    frame[:, center - 1 : center + 1] = 80
    for idx, value in enumerate(values):
        y0 = margin + idx * bar_h
        y1 = min(height - margin, y0 + max(2, bar_h - 2))
        x1 = int(center + value * usable_width / 2)
        lo, hi = sorted((center, x1))
        color = np.array([45, 120, 220], dtype=np.uint8) if idx < len(dof_pos) else np.array([220, 95, 45], dtype=np.uint8)
        frame[y0:y1, lo:hi] = color
    return frame


def save_episode(args, cfg, vision_cfg, model, episode_idx):
    episode_dir = os.path.join(args.output_dir, f"episode_{episode_idx:03d}")
    video_dir = os.path.join(episode_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env, base_env = make_env(cfg, vision_cfg, args.steps)
    obs, info = env.reset(seed=args.seed + episode_idx)

    writers = {
        "vision_rgb": imageio.get_writer(os.path.join(video_dir, "vision_rgb.mp4"), fps=args.fps, macro_block_size=1),
        "proprioception_channel": imageio.get_writer(
            os.path.join(video_dir, "proprioception_channel.mp4"), fps=args.fps, macro_block_size=1
        ),
        f"state_shadow_{args.state_view}": imageio.get_writer(
            os.path.join(video_dir, f"state_shadow_{args.state_view}.mp4"), fps=args.fps, macro_block_size=1
        ),
        "joint_state_panel": imageio.get_writer(
            os.path.join(video_dir, "joint_state_panel.mp4"), fps=args.fps, macro_block_size=1
        ),
    }
    rollout = defaultdict(list)
    states = defaultdict(list)
    cube_start_xy = np.asarray(info["cube_xy"], dtype=np.float32)
    cube_end_xy = cube_start_xy.copy()
    target_xy = np.asarray(info["target_xy"], dtype=np.float32)

    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            for step_idx in range(args.steps):
                vision_obs = np.asarray(obs)
                writers["vision_rgb"].append_data(vision_obs[..., :3])
                proprio = vision_obs[..., 3] if vision_obs.shape[-1] > 3 else np.zeros(vision_obs.shape[:2], dtype=np.uint8)
                writers["proprioception_channel"].append_data(np.repeat(proprio[..., None], 3, axis=2))

                apply_camera(base_env, args.state_view)
                writers[f"state_shadow_{args.state_view}"].append_data(base_env.render())

                joint_state = robot_joint_state(base_env)
                writers["joint_state_panel"].append_data(
                    state_panel(joint_state["dof_pos"], joint_state["dof_vel"], args.state_width, args.state_height)
                )

                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=args.deterministic)

                rollout["actions"].append(np.asarray(action, dtype=np.float32))
                states["qpos"].append(joint_state["qpos"])
                states["qvel"].append(joint_state["qvel"])
                states["dof_pos"].append(joint_state["dof_pos"])
                states["dof_vel"].append(joint_state["dof_vel"])
                states["cube_xy"].append(base_env._cube_xy().copy())
                states["target_xy"].append(target_xy.copy())
                states["robot_xy"].append(np.asarray(base_env.data.qpos[:2], dtype=np.float32).copy())

                obs, reward, terminated, truncated, info = env.step(action)
                rollout["rewards"].append(float(reward))
                rollout["dones"].append(bool(terminated or truncated))
                states["contact"].append(bool(info.get("cube_contact_reward", 0.0) > 0.0))
                states["success"].append(bool(info.get("cube_success_reward", 0.0) > 0.0))
                if terminated or truncated:
                    break
            cube_end_xy = np.asarray(base_env._cube_xy(), dtype=np.float32)
    finally:
        for writer in writers.values():
            writer.close()
        env.close()

    np.savez_compressed(
        os.path.join(episode_dir, "states.npz"),
        qpos=np.asarray(states["qpos"], dtype=np.float32),
        qvel=np.asarray(states["qvel"], dtype=np.float32),
        dof_pos=np.asarray(states["dof_pos"], dtype=np.float32),
        dof_vel=np.asarray(states["dof_vel"], dtype=np.float32),
        cube_xy=np.asarray(states["cube_xy"], dtype=np.float32),
        target_xy=np.asarray(states["target_xy"], dtype=np.float32),
        robot_xy=np.asarray(states["robot_xy"], dtype=np.float32),
        contact=np.asarray(states["contact"], dtype=bool),
        success=np.asarray(states["success"], dtype=bool),
        actions=np.asarray(rollout["actions"], dtype=np.float32),
        rewards=np.asarray(rollout["rewards"], dtype=np.float32),
        dones=np.asarray(rollout["dones"], dtype=bool),
    )
    summary = {
        "episode_index": episode_idx,
        "seed": args.seed + episode_idx,
        "steps_recorded": len(rollout["actions"]),
        "model_path": None if args.model is None else os.path.abspath(args.model),
        "action_source": "random" if model is None else "model",
        "camera_mode": vision_cfg.camera_mode,
        "vision_views": list(vision_cfg.views),
        "shadow_size": args.shadow_size,
        "cube_start_xy": cube_start_xy.tolist(),
        "cube_end_xy": cube_end_xy.tolist(),
        "target_xy": target_xy.tolist(),
        "contact_steps": int(np.asarray(states["contact"], dtype=bool).sum()),
        "success_steps": int(np.asarray(states["success"], dtype=bool).sum()),
        "videos": {
            "vision_rgb": "videos/vision_rgb.mp4",
            "proprioception_channel": "videos/proprioception_channel.mp4",
            "state_shadow": f"videos/state_shadow_{args.state_view}.mp4",
            "joint_state_panel": "videos/joint_state_panel.mp4",
        },
    }
    with open(os.path.join(episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    args = parse_args()
    args.run_dir = os.path.abspath(args.run_dir)
    args.output_dir = os.path.abspath(args.output_dir or os.path.join(args.run_dir, "vision_debug_shadow"))
    args.model = os.path.abspath(args.model) if args.model else os.path.join(args.run_dir, "rl_model_last.zip")
    if not os.path.exists(args.model):
        print(f"model not found, recording random actions: {args.model}")
        args.model = None

    os.makedirs(args.output_dir, exist_ok=True)
    vision_cfg = load_vision_config(args.run_dir, args)
    cfg = make_cfg(args)
    model = None
    if args.model is not None:
        model_env, _ = make_env(cfg, vision_cfg, args.steps)
        model = sbx.CrossQ.load(args.model, env=model_env, device=args.device)
        model_env.close()

    summaries = []
    for episode_idx in range(args.episodes):
        summary = save_episode(args, cfg, vision_cfg, model, episode_idx)
        summaries.append(summary)
        print(
            f"episode_{episode_idx:03d}: source={summary['action_source']} "
            f"steps={summary['steps_recorded']} contacts={summary['contact_steps']} "
            f"success={summary['success_steps']}"
        )

    with open(os.path.join(args.output_dir, "debug_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": summaries}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
