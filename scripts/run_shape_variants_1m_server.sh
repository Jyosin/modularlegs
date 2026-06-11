#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

python scripts/generate_shape_variants.py

python -u scripts/train_shape_snapshots.py \
  --only \
  chain5 arc5 zigzag5 branch_y comb5 cross4 offset_cross t_shape \
  t_wide t_offset t_long_stem l_shape l_hook l_stair front_fork double_tail \
  --target-steps 1000000 \
  --snapshot-interval 100000 \
  --video-steps 120 \
  --no-video
