import numpy as np

from cube_contact_task.env import CubePushSim
from modularlegs.utils.files import load_cfg


def main():
    cfg = load_cfg("shape_experiments/sim_train_shape_quadruped_10k_local", alg="sbx")
    cfg.sim.render = False
    cfg.sim.randomize_orientation = False
    cfg.sim.random_latency_scheme = False
    cfg.sim.randomize_mass = False
    cfg.sim.randomize_friction = False
    cfg.sim.noisy_actions = False
    cfg.sim.noisy_observations = False

    env = CubePushSim(cfg)
    obs, _ = env.reset()
    print("obs shape:", obs.shape)
    print("action space:", env.action_space)
    print("initial cube xy:", env._cube_xy())
    print("target xy:", env._target_xy)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"step={step:02d} reward={float(reward): .4f} "
            f"cube_xy={np.round(info['cube_xy'], 3)} "
            f"target_xy={np.round(info['target_xy'], 3)} "
            f"target_dist={info['cube_target_distance']: .3f} "
            f"cube_delta={np.round(info['cube_delta_xy'], 4)} "
            f"contact={info['cube_contact_reward'] > 0} "
            f"done={done} truncated={truncated}"
        )

    env.close()


if __name__ == "__main__":
    main()
