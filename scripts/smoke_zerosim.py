import numpy as np

from modularlegs.envs.env_sim import ZeroSim
from modularlegs.utils.files import load_cfg


def main():
    cfg = load_cfg("sim_smoke_m3air1s", alg="sbx")
    cfg.sim.render = False

    env = ZeroSim(cfg)
    obs, _ = env.reset()

    print("obs shape:", obs.shape)
    print("action space:", env.action_space)

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, _ = env.step(action)
        print(
            f"step={step} reward={float(reward):.4f} "
            f"done={done} truncated={truncated} obs_shape={np.shape(obs)}"
        )

    env.close()


if __name__ == "__main__":
    main()
