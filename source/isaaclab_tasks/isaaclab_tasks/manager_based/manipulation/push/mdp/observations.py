# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the push task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
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
