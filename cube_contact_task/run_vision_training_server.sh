#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

mkdir -p exp/quadruped_cube_vision_goal_level_2m
mkdir -p exp/quadruped_cube_vision_goal_mounted_2m

nohup env JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python cube_contact_task/train_quadruped_cube_vision_goal.py \
    --output-dir exp/quadruped_cube_vision_goal_level_2m \
    --total-steps 2000000 \
    --seed 0 \
    --camera-mode level \
  > exp/quadruped_cube_vision_goal_level_2m/train.log 2>&1 &
echo "level_pid=$!"

nohup env JAX_PLATFORMS=cpu MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python cube_contact_task/train_quadruped_cube_vision_goal.py \
    --output-dir exp/quadruped_cube_vision_goal_mounted_2m \
    --total-steps 2000000 \
    --seed 0 \
    --camera-mode mounted \
  > exp/quadruped_cube_vision_goal_mounted_2m/train.log 2>&1 &
echo "mounted_pid=$!"
