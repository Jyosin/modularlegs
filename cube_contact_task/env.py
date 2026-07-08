from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from gymnasium.spaces import Box
from lxml import etree

from modularlegs.envs.env_sim import ZeroSim


@dataclass
class CubeTaskConfig:
    cube_name: str = "push_cube"
    cube_joint_name: str = "push_cube_freejoint"
    target_name: str = "cube_target"
    cube_pos: tuple[float, float, float] = (0.35, 0.0, 0.075)
    target_pos: tuple[float, float, float] = (0.55, 0.0, 0.01)
    cube_size: tuple[float, float, float] = (0.075, 0.075, 0.075)
    cube_mass: float = 0.08
    cube_friction: tuple[float, float, float] = (0.55, 0.01, 0.001)
    cube_rgba: tuple[float, float, float, float] = (0.1, 0.45, 0.95, 1.0)
    target_rgba: tuple[float, float, float, float] = (0.1, 0.9, 0.25, 0.45)
    randomize_task: bool = True
    cube_spawn_radius: tuple[float, float] = (0.4, 0.6)
    target_spawn_radius: tuple[float, float] = (0.35, 0.75)
    min_cube_target_distance: float = 0.22
    max_cube_target_distance: float = 0.65
    goal_radius: float = 0.1
    target_progress_reward_scale: float = 40.0
    target_distance_penalty_scale: float = 0.2
    success_reward: float = 10.0
    contact_reward: float = 1.0
    approach_reward_scale: float = 2.0
    proximity_reward_scale: float = 0.05
    proximity_radius: float = 0.25
    action_penalty_scale: float = 0.0005
    use_base_reward: bool = False
    observe_cube: bool = True


class CubePushSim(ZeroSim):
    """ZeroSim variant with a movable cube and reward for pushing it."""

    def __init__(self, cfg, task_config: CubeTaskConfig | None = None):
        self.cube_task = task_config or CubeTaskConfig()
        self._target_xy = np.asarray(self.cube_task.target_pos[:2], dtype=np.float32)
        self._last_cube_xy = None
        self._last_cube_reward = 0.0
        self._last_cube_info = {}
        super().__init__(cfg)

        self._base_observation_shape = self.observation_space.shape
        if self.cube_task.observe_cube:
            extra_dim = 8
            self.observation_space = Box(
                -np.inf,
                np.inf,
                (self._base_observation_shape[0] + extra_dim,),
                dtype=np.float32,
            )

    def _load_asset(self, asset_file_name):
        super()._load_asset(asset_file_name)
        self._add_task_objects_to_xml()
        self.xml_string = self.xml_compiler.get_string()

    def _add_task_objects_to_xml(self):
        root = self.xml_compiler.root
        asset = root.find("./asset")
        worldbody = root.find("./worldbody")
        if asset is None or worldbody is None:
            raise ValueError("MuJoCo XML must contain asset and worldbody nodes.")

        material_name = "push_cube_mat"
        if asset.find(f"./material[@name='{material_name}']") is None:
            rgba = " ".join(str(v) for v in self.cube_task.cube_rgba)
            etree.SubElement(
                asset,
                "material",
                name=material_name,
                rgba=rgba,
                specular="0.2",
                shininess="0.25",
            )

        target_material_name = "cube_target_mat"
        if asset.find(f"./material[@name='{target_material_name}']") is None:
            rgba = " ".join(str(v) for v in self.cube_task.target_rgba)
            etree.SubElement(
                asset,
                "material",
                name=target_material_name,
                rgba=rgba,
                specular="0.1",
                shininess="0.1",
            )

        old_cube = worldbody.find(f"./body[@name='{self.cube_task.cube_name}']")
        if old_cube is not None:
            worldbody.remove(old_cube)
        old_target = worldbody.find(f"./body[@name='{self.cube_task.target_name}']")
        if old_target is not None:
            worldbody.remove(old_target)

        cube_body = etree.SubElement(
            worldbody,
            "body",
            name=self.cube_task.cube_name,
            pos=self._vec_to_string(self.cube_task.cube_pos),
        )
        etree.SubElement(cube_body, "freejoint", name=self.cube_task.cube_joint_name)
        etree.SubElement(
            cube_body,
            "geom",
            name=f"{self.cube_task.cube_name}_geom",
            type="box",
            size=self._vec_to_string(self.cube_task.cube_size),
            mass=str(self.cube_task.cube_mass),
            material=material_name,
            condim="6",
            friction=self._vec_to_string(self.cube_task.cube_friction),
            priority="2",
        )
        target_body = etree.SubElement(
            worldbody,
            "body",
            name=self.cube_task.target_name,
            pos=self._vec_to_string(self.cube_task.target_pos),
        )
        etree.SubElement(
            target_body,
            "geom",
            name=f"{self.cube_task.target_name}_geom",
            type="cylinder",
            size=f"{self.cube_task.goal_radius} 0.004",
            material=target_material_name,
            contype="0",
            conaffinity="0",
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._last_cube_xy = self._cube_xy().copy()
        self._last_cube_reward = 0.0
        self._last_cube_info = {}
        info.update(
            {
                "cube_xy": self._last_cube_xy.copy(),
                "target_xy": self._target_xy.copy(),
                "cube_target_distance": self._cube_target_distance(),
            }
        )
        return self._append_cube_obs(obs), info

    def step(self, action, clip=True):
        if self.num_envs > 1:
            raise NotImplementedError("CubePushSim currently supports one env per instance.")

        self._wait_until_motor_on()
        actions_scaled = action * self.action_scale
        if clip:
            if self.clip_actions_list is None:
                action_clipped = np.clip(actions_scaled, self.clip_actions_min, self.clip_actions)
            else:
                action_clipped = np.clip(actions_scaled, -self.clip_actions_list, self.clip_actions_list)
        else:
            action_clipped = actions_scaled

        self.last_action = action_clipped
        self.last_action_flat = action_clipped.flatten()
        joint_action = self._action_remap(action_clipped)
        if self.frozen_joints is not None:
            joint_action[self.frozen_joints] = 0

        if self.control_mode == "position":
            target_action = joint_action + self.default_dof_pos
            target_action = self.action_filter.filter(target_action) if self.filter_action else target_action
            target_action = self.action_melter.filter(target_action) if self.action_melter is not None else target_action
            info = self._perform_action(target_action)
        elif self.control_mode == "incremental":
            target_action = joint_action + self.brain.dof_pos
            target_action = self.action_filter.filter(target_action) if self.filter_action else target_action
            info = self._perform_action(target_action)
        elif self.control_mode == "velocity":
            info = self._perform_action(pos=np.zeros_like(joint_action), vel=joint_action)
        elif self.control_mode == "advanced":
            assert self.step_info is not None, "set_step_info should be called before step"
            info = self._perform_action(
                pos=self.step_info["pos"],
                vel=self.step_info["vel"],
                kps=self.step_info["kps"] if "kps" in self.step_info else self.kps,
                kds=self.step_info["kds"] if "kds" in self.step_info else self.kds,
            )
        else:
            raise ValueError(f"Unsupported control_mode: {self.control_mode}")

        if self.step_count % self.resampling_time == 0 and self.step_count > 0:
            self._resample_commands()

        self._update_observable_data()
        obs = self.brain.get_observations()
        reward = self._last_cube_reward

        self.step_count += 1
        truncated = self._is_truncated()
        self._log_data()
        self._check_input()

        done, done_info = self.brain.get_done_info()
        info.update(
            {
                "policy_switch": self.policy_switch,
                "upsidedown": self.brain.is_upsidedown(),
            }
        )
        info.update(done_info)
        info.update(self._last_cube_info)
        return self._append_cube_obs(obs), reward, done, truncated, info

    def _perform_action(self, pos, vel=None, kps=None, kds=None):
        cube_before = self._cube_xy().copy()
        cube_target_distance_before = self._cube_target_distance()
        robot_cube_distance_before = self._nearest_robot_geom_distance_to_cube()
        info = super()._perform_action(pos, vel=vel, kps=kps, kds=kds)
        cube_after = self._cube_xy().copy()
        cube_target_distance_after = self._cube_target_distance()
        robot_cube_distance_after = self._nearest_robot_geom_distance_to_cube()

        cube_delta = cube_after - cube_before
        target_progress_reward = self.cube_task.target_progress_reward_scale * (
            cube_target_distance_before - cube_target_distance_after
        )
        target_distance_penalty = (
            self.cube_task.target_distance_penalty_scale * cube_target_distance_after
        )
        success_bonus = (
            self.cube_task.success_reward
            if cube_target_distance_after <= self.cube_task.goal_radius
            else 0.0
        )
        contact_bonus = self.cube_task.contact_reward if self._robot_touching_cube() else 0.0
        approach_reward = self.cube_task.approach_reward_scale * (
            robot_cube_distance_before - robot_cube_distance_after
        )
        proximity_reward = self.cube_task.proximity_reward_scale * max(
            0.0,
            self.cube_task.proximity_radius - robot_cube_distance_after,
        )
        action_penalty = self.cube_task.action_penalty_scale * float(np.sum(np.square(self.last_action_flat)))
        self._last_cube_reward = (
            target_progress_reward
            + success_bonus
            + contact_bonus
            + approach_reward
            + proximity_reward
            - target_distance_penalty
            - action_penalty
        )
        self._last_cube_info = {
            "cube_xy": cube_after.copy(),
            "target_xy": self._target_xy.copy(),
            "cube_delta_xy": cube_delta.copy(),
            "robot_cube_distance": robot_cube_distance_after,
            "cube_target_distance": cube_target_distance_after,
            "cube_target_progress_reward": target_progress_reward,
            "cube_target_distance_penalty": target_distance_penalty,
            "cube_success_reward": success_bonus,
            "cube_contact_reward": contact_bonus,
            "cube_approach_reward": approach_reward,
            "cube_proximity_reward": proximity_reward,
            "cube_action_penalty": action_penalty,
            "cube_reward": self._last_cube_reward,
        }
        info.update(self._last_cube_info)
        return info

    def reset_model(self):
        super().reset_model()
        self._sample_task_positions()
        self._reset_target_marker()
        self._reset_cube_state()

    def _reset_cube_state(self):
        joint_id = self.model.joint(self.cube_task.cube_joint_name).id
        qpos_addr = self.model.jnt_qposadr[joint_id]
        qvel_addr = self.model.jnt_dofadr[joint_id]
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[qpos_addr : qpos_addr + 3] = np.asarray(
            [self._cube_start_xy[0], self._cube_start_xy[1], self.cube_task.cube_pos[2]]
        )
        qpos[qpos_addr + 3 : qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
        qvel[qvel_addr : qvel_addr + 6] = 0.0
        self.set_state(qpos, qvel)

    def _reset_target_marker(self):
        body_id = self.model.body(self.cube_task.target_name).id
        self.model.body_pos[body_id] = np.asarray(
            [self._target_xy[0], self._target_xy[1], self.cube_task.target_pos[2]],
            dtype=np.float64,
        )
        mujoco.mj_forward(self.model, self.data)

    def _sample_task_positions(self):
        if not self.cube_task.randomize_task:
            self._cube_start_xy = np.asarray(self.cube_task.cube_pos[:2], dtype=np.float32)
            self._target_xy = np.asarray(self.cube_task.target_pos[:2], dtype=np.float32)
            return

        for _ in range(100):
            cube_xy = self._sample_xy_in_radius(*self.cube_task.cube_spawn_radius)
            target_xy = self._sample_xy_in_radius(*self.cube_task.target_spawn_radius)
            cube_target_distance = np.linalg.norm(target_xy - cube_xy)
            if (
                self.cube_task.min_cube_target_distance
                <= cube_target_distance
                <= self.cube_task.max_cube_target_distance
            ):
                self._cube_start_xy = cube_xy
                self._target_xy = target_xy
                return

        self._cube_start_xy = np.asarray(self.cube_task.cube_pos[:2], dtype=np.float32)
        self._target_xy = np.asarray(self.cube_task.target_pos[:2], dtype=np.float32)

    def _sample_xy_in_radius(self, min_radius, max_radius):
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        radius = self.np_random.uniform(min_radius, max_radius)
        return np.asarray([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)

    def _append_cube_obs(self, obs):
        if not self.cube_task.observe_cube:
            return obs
        cube_xy = self._cube_xy()
        robot_xy = np.asarray(self.data.qpos[:2], dtype=np.float32)
        cube_vel_xy = self._cube_vel_xy()
        extra = np.concatenate(
            [
                cube_xy - robot_xy,
                self._target_xy - robot_xy,
                self._target_xy - cube_xy,
                cube_vel_xy,
            ]
        ).astype(np.float32)
        return np.concatenate([np.asarray(obs, dtype=np.float32), extra])

    def _cube_xy(self):
        body_id = self.model.body(self.cube_task.cube_name).id
        return np.asarray(self.data.xpos[body_id][:2], dtype=np.float32)

    def _cube_vel_xy(self):
        joint_id = self.model.joint(self.cube_task.cube_joint_name).id
        qvel_addr = self.model.jnt_dofadr[joint_id]
        return np.asarray(self.data.qvel[qvel_addr : qvel_addr + 2], dtype=np.float32)

    def _cube_target_distance(self):
        return float(np.linalg.norm(self._target_xy - self._cube_xy()))

    def _robot_touching_cube(self):
        cube_geom_id = self.model.geom(f"{self.cube_task.cube_name}_geom").id
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom[0]), int(contact.geom[1])
            if cube_geom_id not in (geom1, geom2):
                continue
            other_geom_id = geom2 if geom1 == cube_geom_id else geom1
            if self.model.geom(other_geom_id).name != "floor":
                return True
        return False

    def _nearest_robot_geom_distance_to_cube(self):
        cube_geom_id = self.model.geom(f"{self.cube_task.cube_name}_geom").id
        target_geom_id = self.model.geom(f"{self.cube_task.target_name}_geom").id
        floor_geom_id = self.model.geom("floor").id
        cube_pos = np.asarray(self.data.geom_xpos[cube_geom_id], dtype=np.float32)
        min_distance = np.inf
        for geom_id in range(self.model.ngeom):
            if geom_id in (cube_geom_id, target_geom_id, floor_geom_id):
                continue
            distance = float(np.linalg.norm(self.data.geom_xpos[geom_id] - cube_pos))
            min_distance = min(min_distance, distance)
        return min_distance

    @staticmethod
    def _vec_to_string(vec):
        return " ".join(str(v) for v in vec)
