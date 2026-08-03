from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


@dataclass
class CubeVisionConfig:
    image_height: int = 64
    image_width: int = 64
    views: tuple[str, ...] = ("front", "back", "left", "right")
    camera_mode: str = "level"
    camera_distance: float = 0.45
    lookahead: float = 0.65
    camera_height: float = 0.22
    front_overhead: bool = True
    front_overhead_distance: float = 1.25
    front_overhead_elevation: float = -89.0
    front_overhead_height: float = 0.05
    include_proprioception: bool = True
    proprioception_clip: float = 5.0


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
        channels = 4 if self.vision_config.include_proprioception else 3
        self.observation_space = Box(0, 255, shape=(height, width, channels), dtype=np.uint8)

    def observation(self, observation):
        image = self._render_robot_views()
        if not self.vision_config.include_proprioception:
            return image
        proprio_map = self._encode_proprioception(observation, image.shape[:2])
        return np.concatenate([image, proprio_map[..., None]], axis=2).astype(np.uint8)

    def _render_robot_views(self):
        frames = [self.render_view(view_name) for view_name in self.vision_config.views]
        return np.concatenate(frames, axis=1).astype(np.uint8)

    def render_view(self, view_name, resize=True):
        base_env = self.unwrapped
        viewer = base_env.mujoco_renderer._get_viewer("rgb_array")
        cam = viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        direction = self._view_direction(view_name)
        azimuth, elevation = self._direction_to_camera_angles(direction)
        robot_pos = np.asarray(base_env.data.qpos[:3], dtype=np.float64)
        if view_name == "front" and self.vision_config.front_overhead:
            lookat = robot_pos.copy()
            lookat[2] = robot_pos[2] + self.vision_config.front_overhead_height
            cam.lookat[:] = lookat
            cam.distance = self.vision_config.front_overhead_distance
            cam.azimuth = azimuth
            cam.elevation = self.vision_config.front_overhead_elevation
            frame = base_env.render()
            return self._center_crop_or_downsample(frame) if resize else frame

        lookat = robot_pos + direction * self.vision_config.lookahead
        lookat[2] = robot_pos[2] + self.vision_config.camera_height

        cam.lookat[:] = lookat
        cam.distance = self.vision_config.camera_distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        frame = base_env.render()
        return self._center_crop_or_downsample(frame) if resize else frame

    def _view_direction(self, view_name):
        # The quadruped asset's visual forward axis is +Y, not +X.
        local = {
            "front": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "back": np.array([0.0, -1.0, 0.0], dtype=np.float64),
            "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
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

    def _encode_proprioception(self, observation, image_shape):
        height, width = image_shape
        proprio = np.asarray(observation, dtype=np.float32).reshape(-1)
        if proprio.size == 0:
            return np.zeros((height, width), dtype=np.uint8)

        clip = self.vision_config.proprioception_clip
        encoded = np.clip(proprio, -clip, clip)
        encoded = ((encoded / clip) * 0.5 + 0.5) * 255.0
        encoded = encoded.astype(np.uint8)

        x_idx = np.floor(np.linspace(0, encoded.size, width, endpoint=False)).astype(np.int64)
        x_idx = np.clip(x_idx, 0, encoded.size - 1)
        row = encoded[x_idx]
        return np.repeat(row[None, :], height, axis=0)


def _yaw_from_wxyz(quat):
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)
