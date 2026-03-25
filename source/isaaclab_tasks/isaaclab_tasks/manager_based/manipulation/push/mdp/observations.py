# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the push task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation of the object as a quaternion (w, x, y, z) in the world frame."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_quat_w


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The orientation of the end-effector as a quaternion (w, x, y, z) in the world frame."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[..., 0, :]


def ee_to_object_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The vector from the end-effector to the object in the world frame."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return object_pos_w - ee_pos_w


def object_to_target_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """The vector from the object to the target in the world frame."""
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    target_pos_w = target.data.root_pos_w[:, :3]
    return target_pos_w - object_pos_w


def target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """The position of the target in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_w = target.data.root_pos_w[:, :3]
    target_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w)
    return target_pos_b


def target_orientation_in_world_frame(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """The orientation of the target as a quaternion (w, x, y, z) in the world frame."""
    target: RigidObject = env.scene[target_cfg.name]
    return target.data.root_quat_w


def deformable_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the deformable object center (mean of nodes) in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: DeformableObject = env.scene[object_cfg.name]
    object_pos_w = obj.data.nodal_pos_w.mean(dim=1)  # (num_envs, 3)
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def deformable_ee_to_object_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The vector from the end-effector to the deformable object center in the world frame."""
    obj: DeformableObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.nodal_pos_w.mean(dim=1)  # (num_envs, 3)
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return object_pos_w - ee_pos_w


def deformable_object_to_target_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """The vector from the deformable object center to the target in the world frame."""
    obj: DeformableObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    object_pos_w = obj.data.nodal_pos_w.mean(dim=1)  # (num_envs, 3)
    target_pos_w = target.data.root_pos_w[:, :3]
    return target_pos_w - object_pos_w


def visualize_object_orientation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    visualizer_cfg: Any | None = None,
) -> torch.Tensor:
    """Visualize the orientation of the object and target using arrows."""
    if not hasattr(env, "_object_orientation_visualizer"):
        if visualizer_cfg is not None:
            from isaaclab.markers import VisualizationMarkers
            import copy
            # Resolve the prim path by replacing the regex namespace with env_0
            resolved_cfg = copy.copy(visualizer_cfg)
            resolved_cfg.prim_path = resolved_cfg.prim_path.format(ENV_REGEX_NS="/World/envs/env_0")
            env._object_orientation_visualizer = VisualizationMarkers(resolved_cfg)  # type: ignore
        else:
            return torch.empty((env.num_envs, 0), device=env.device)

    visualizer: Any = env._object_orientation_visualizer  # type: ignore
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # Get positions and orientations
    obj_pos = object.data.root_pos_w[:, :3]
    obj_quat = object.data.root_quat_w
    tar_pos = target.data.root_pos_w[:, :3]
    tar_quat = target.data.root_quat_w

    # Update visualizer
    # We use two markers: 0 for object, 1 for target
    # The visualizer expects (M, 3) where M is total number of markers across all envs
    marker_indices = torch.tensor([0, 1], device=obj_pos.device, dtype=torch.int32).repeat(env.num_envs)
    visualizer.visualize(
        translations=torch.stack([obj_pos, tar_pos], dim=1).view(-1, 3),
        orientations=torch.stack([obj_quat, tar_quat], dim=1).view(-1, 4),
        marker_indices=marker_indices,
    )

    return torch.empty((env.num_envs, 0), device=env.device)
