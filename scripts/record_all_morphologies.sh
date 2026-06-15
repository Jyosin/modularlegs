#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

python scripts/generate_shape_variants.py
python scripts/record_all_morphologies.py "$@"
