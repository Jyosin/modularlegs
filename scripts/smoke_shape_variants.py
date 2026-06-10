import os

import imageio.v3 as iio
import numpy as np

from modularlegs import LEG_ROOT_DIR
from modularlegs.envs.env_sim import ZeroSim
from modularlegs.utils.files import load_cfg
from modularlegs.utils.model import XMLCompiler
from scripts.generate_shape_variants import SHAPE_VARIANTS


def smoke_variant(asset_name, steps):
    name = asset_name.removesuffix("_air1s")
    cfg = load_cfg("shape_experiments/sim_train_shape_chain5", alg="sbx")
    vis_dir = os.path.join("exp", "shape_experiments", name, "visualization")
    os.makedirs(vis_dir, exist_ok=True)

    source_xml = os.path.join(
        LEG_ROOT_DIR, "modularlegs", "sim", "assets", "robots", f"{asset_name}.xml"
    )
    render_xml = os.path.abspath(os.path.join(vis_dir, f"{name}_no_shadow.xml"))
    compiler = XMLCompiler(source_xml)
    compiler.remove_shadow()
    compiler.save(render_xml)

    cfg.sim.asset_file = render_xml
    cfg.logging.data_dir = os.path.join("exp", "shape_experiments", name)
    cfg.sim.render = False
    cfg.sim.randomize_orientation = False
    cfg.sim.randomize_mass = False
    cfg.sim.randomize_friction = False
    cfg.sim.noisy_actions = False
    cfg.sim.noisy_observations = False
    cfg.sim.random_latency_scheme = False

    env = ZeroSim(cfg)
    obs, _ = env.reset()
    done = truncated = False
    for step in range(steps):
        obs, _, done, truncated, _ = env.step(env.action_space.sample())
        if done or truncated:
            break

    preview = os.path.join(vis_dir, f"{name}_preview.png")
    iio.imwrite(preview, env.render())
    result = {
        "name": name,
        "steps": step + 1,
        "obs_shape": np.shape(obs),
        "action_shape": env.action_space.shape,
        "done": done,
        "truncated": truncated,
        "preview": preview,
    }
    env.close()
    return result


def main():
    for asset_name in SHAPE_VARIANTS:
        result = smoke_variant(asset_name, steps=5)
        print(
            "{name}: ok steps={steps} obs={obs_shape} action={action_shape} "
            "done={done} truncated={truncated} preview={preview}".format(**result)
        )


if __name__ == "__main__":
    main()
