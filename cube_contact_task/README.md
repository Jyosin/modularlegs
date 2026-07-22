# Cube Goal Task

This directory isolates the cube manipulation task from the older locomotion code.

What it adds:

- A movable MuJoCo cube named `push_cube` in the existing robot scene.
- A lightweight cube with lower friction so the quadruped can move it by contact.
- A visible target marker named `cube_target`.
- Random cube and target respawns around the robot at reset.
- A reward based on reducing cube-to-target distance, with contact and approach shaping.
- Extra observations: cube position, target position, cube-to-target vector, and cube XY velocity.

Run a smoke test:

```bash
python -m cube_contact_task.smoke_cube_push
```

Train the quadruped locally with SBX CrossQ:

```bash
python cube_contact_task/train_quadruped_cube_goal.py \
  --output-dir exp/quadruped_cube_goal_100k \
  --total-steps 100000
```

Train 2M steps on the server:

```bash
python cube_contact_task/train_quadruped_cube_goal.py \
  --output-dir exp/quadruped_cube_goal_2m \
  --total-steps 2000000
```

Train vision-only policies from robot-centric cameras:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python cube_contact_task/train_quadruped_cube_vision_goal.py \
  --output-dir exp/quadruped_cube_vision_goal_level_2m \
  --total-steps 2000000 \
  --seed 0 \
  --camera-mode level

JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python cube_contact_task/train_quadruped_cube_vision_goal.py \
  --output-dir exp/quadruped_cube_vision_goal_mounted_2m \
  --total-steps 2000000 \
  --seed 0 \
  --camera-mode mounted
```

Or start both server jobs in the background:

```bash
bash cube_contact_task/run_vision_training_server.sh
```

The default vision policy receives four robot-centric RGB views (`front back left right`) stitched horizontally plus robot proprioception. Proprioception is the existing robot body/joint state observation: projected gravity, body angular velocity, joint positions, joint velocities, and previous action history according to the configured observation version. It still does not receive cube position or target position as numeric observations. `level` keeps the camera horizon level using robot yaw; `mounted` follows the robot body's full orientation, so the view tilts as the robot moves.

For a pure image ablation, add:

```bash
--no-proprioception
```

Record synchronized dataset episodes after training:

```bash
python cube_contact_task/record_cube_goal_dataset.py \
  --run-dir exp/quadruped_cube_goal_2m \
  --output-dir exp/quadruped_cube_goal_2m/dataset_sample \
  --episodes 1 \
  --steps 500 \
  --seed 0 \
  --deterministic \
  --views fixed follow front side top
```

Record matching shadow videos by using the same seed and a separate output directory:

```bash
python cube_contact_task/record_cube_goal_dataset.py \
  --run-dir exp/quadruped_cube_goal_2m \
  --output-dir exp/quadruped_cube_goal_2m/dataset_sample_shadow \
  --episodes 1 \
  --steps 500 \
  --seed 0 \
  --deterministic \
  --views fixed follow front side top \
  --shadow-size 4096
```

Each recorded episode contains:

- `videos/*.mp4`: multiple synchronized camera views for the same action sequence.
- `rolloutN.npz`: standard arrays `observations`, `actions`, `rewards`, `dones`.
- `states.npz`: `qpos`, `qvel`, `dof_pos`, `dof_vel`, `cube_xy`, `cube_vel_xy`, `target_xy`, `cube_target_distance`, `robot_xy`, `contact`, `success`, `frame_idx`.
- `episode.obs.jsonl`: step-by-step JSON sidecar.
- `metadata.json`: model/config paths, cube start/end coordinates, target coordinates, distance change, video paths, and contact/success counts.

The branch for this work is:

```bash
cube-contact-reward
```

Push it with:

```bash
git push -u origin cube-contact-reward
```
