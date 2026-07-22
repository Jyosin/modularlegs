from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict


@dataclass
class CubeVisionConfig:
    image_height: int = 64
    image_width: int = 64
    views: tuple[str, ...] = ("front", "back", "left", "right")
    camera_mode: str = "level"
    camera_distance: float = 0.45
    lookahead: float = 0.65
    camera_height: float = 0.22
    include_proprioception: bool = True


class CubeRobotVisionWrapper(ObservationWrapper):
    """Replace low-dimensional task observations with robot-centric camera images."""

    def __init__(self, env, vision_config: CubeVisionConfig | None = None):
        super().__init__(env)
        self.vision_config = vision_config or CubeVisionConfig()
        if self.vision_config.camera_mode not in {"level", "mounted"}:
            raise ValueError("camera_mode must be 'level' or 'mounted'")
        if not self.vision_config.views:
            raise ValueError("At least one camera view is required.")

        height = self.vision_config.image_height
        width = self.vision_config.image_width * len(self.vision_config.views)
        self.image_space = Box(0, 255, shape=(height, width, 3), dtype=np.uint8)
        if self.vision_config.include_proprioception:
            self.observation_space = Dict(
                {
                    "image": self.image_space,
                    "proprioception": Box(
                        -np.inf,
                        np.inf,
                        shape=env.observation_space.shape,
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = self.image_space

    def observation(self, observation):
        image = self._render_robot_views()
        if not self.vision_config.include_proprioception:
            return image
        return {
            "image": image,
            "proprioception": np.asarray(observation, dtype=np.float32),
        }

    def _render_robot_views(self):
        frames = [self._render_view(view_name) for view_name in self.vision_config.views]
        return np.concatenate(frames, axis=1).astype(np.uint8)

    def _render_view(self, view_name):
        base_env = self.unwrapped
        viewer = base_env.mujoco_renderer._get_viewer("rgb_array")
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        direction = self._view_direction(view_name)
        azimuth, elevation = self._direction_to_camera_angles(direction)
        robot_pos = np.asarray(base_env.data.qpos[:3], dtype=np.float64)
        lookat = robot_pos + direction * self.vision_config.lookahead
        lookat[2] = robot_pos[2] + self.vision_config.camera_height

        cam.lookat[:] = lookat
        cam.distance = self.vision_config.camera_distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        frame = base_env.render()
        return self._center_crop_or_downsample(frame)

    def _view_direction(self, view_name):
        local = {
            "front": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "back": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        }.get(view_name)
        if local is None:
            raise ValueError(f"Unsupported view '{view_name}'. Use front/back/left/right.")

        quat = np.asarray(self.unwrapped.data.qpos[3:7], dtype=np.float64)
        if self.vision_config.camera_mode == "level":
            yaw = _yaw_from_wxyz(quat)
            rot = np.array(
                [
                    [math.cos(yaw), -math.sin(yaw), 0.0],
                    [math.sin(yaw), math.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            direction = rot @ local
            direction[2] = 0.0
        else:
            mat = np.zeros(9, dtype=np.float64)
            mujoco.mju_quat2Mat(mat, quat)
            direction = mat.reshape(3, 3) @ local

        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return local
        return direction / norm

    @staticmethod
    def _direction_to_camera_angles(direction):
        dx, dy, dz = direction
        azimuth = math.degrees(math.atan2(dy, dx)) - 180.0
        horizontal = max(1e-9, math.hypot(dx, dy))
        elevation = -math.degrees(math.atan2(dz, horizontal))
        return azimuth, elevation

    def _center_crop_or_downsample(self, frame):
        target_h = self.vision_config.image_height
        target_w = self.vision_config.image_width
        height, width = frame.shape[:2]
        if height == target_h and width == target_w:
            return frame

        y_idx = np.linspace(0, height - 1, target_h).astype(np.int64)
        x_idx = np.linspace(0, width - 1, target_w).astype(np.int64)
        return frame[np.ix_(y_idx, x_idx)]


def _yaw_from_wxyz(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
