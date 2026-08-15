# Quadruped four-view input sample

This sample was generated from
`config/cube_contact_task/quadruped_four_view_input.yaml` with seed `20260815`.
It contains 100 simulation steps at 20 fps. Actions are random because no trained
four-view policy was loaded.

The policy-facing input is `episode_000/videos/vision_rgb.mp4`: four 64 x 64
robot-centric views (`front`, `back`, `left`, `right`) stitched horizontally into
a 256 x 64 RGB frame. The four `analysis_*.mp4` files preserve the corresponding
full-resolution views for inspection. `proprioception_channel.mp4` visualizes the
fourth input channel separately.

Regenerate the sample with an arm64 environment containing the project
dependencies:

```bash
python -c "from modularlegs.utils.files import load_cfg; from omegaconf import OmegaConf; c=load_cfg('cube_contact_task/quadruped_four_view_input', alg='sbx'); OmegaConf.save(c, 'exp/quadruped_four_view_input_sample/running_config.yaml')"

MUJOCO_GL=glfw python cube_contact_task/record_cube_vision_debug.py \
  --run-dir exp/quadruped_four_view_input_sample \
  --model exp/does_not_exist.zip \
  --output-dir quadruped_four_view_inputs \
  --episodes 1 --steps 100 --seed 20260815 \
  --camera-mode level --views front back left right --state-view follow
```
