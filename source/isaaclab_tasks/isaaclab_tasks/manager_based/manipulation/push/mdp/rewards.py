# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the push task.

These rewards are inspired by the MuJoCo Playground push_cube.py implementation
(DeepMind Technologies Limited, Apache License 2.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for moving the end-effector close to the object.

    Uses a tanh-kernel to provide a smooth reward signal. The reward is 1 when the
    end-effector is at the object and decreases as the distance increases.

    Args:
        env: The environment.
        std: The standard deviation of the tanh kernel.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        ee_frame_cfg: The end-effector frame configuration. Defaults to SceneEntityCfg("ee_frame").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_dist = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_dist / std)


def undesired_robot_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_link_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ee_cylinder"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize contacts between the robot (excluding the end-effector) and any other object.

    Args:
        env: The environment.
        threshold: The contact force threshold to consider a contact.
        object_cfg: The object configuration (not used in this version). Defaults to SceneEntityCfg("object").
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        ee_link_cfg: The end-effector link configuration. Defaults to SceneEntityCfg("robot", body_names="ee").
        sensor_cfg: The contact sensor configuration. Defaults to SceneEntityCfg("contact_forces").

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get net contact forces: (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w

    # Get the indices of the bodies to exclude (end-effector)
    # Note: sensor_cfg.body_ids are indices into the sensor's tracked bodies.
    # If the sensor tracks all robot bodies, we need to find the index of "ee".
    ee_link_indices = ee_link_cfg.body_ids

    # Create a mask for all bodies except the end-effector
    if contact_forces is None or contact_forces.numel() == 0:
        return torch.zeros(env.num_envs, device=contact_sensor.device)

    num_bodies = contact_forces.shape[1]
    mask = torch.ones(num_bodies, dtype=torch.bool, device=contact_sensor.device)
    # Handle multiple indices if ee_link_cfg.body_ids is a list/tensor
    mask[ee_link_indices] = False

    # Sum the contact force magnitudes for all other bodies
    non_ee_contact_forces = contact_forces[:, mask, :]
    non_ee_contact_mag = torch.norm(non_ee_contact_forces, dim=-1).max(dim=-1)[0]
    undesired_contact_mask = non_ee_contact_mag > threshold

    # Optional debug logging: summarize undesired contacts across vectorized environments.
    # if torch.any(undesired_contact_mask):
    #     max_force = non_ee_contact_mag.max().item()
    #     num_triggered = undesired_contact_mask.sum().item()
    #     print(
    #         f"Warning: Undesired robot contact detected in {num_triggered} env(s); "
    #         f"max non-EE contact force={max_force:.2f} (threshold={threshold:.2f})."
    #     )

    # Return 1.0 if any non-EE body has contact force above threshold, 0.0 otherwise
    return undesired_contact_mask.float()


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward the agent for pushing the object close to the target position.

    Uses a tanh-kernel to provide a smooth reward signal. The reward is 1 when the
    object is at the target and decreases as the distance increases.

    Args:
        env: The environment.
        std: The standard deviation of the tanh kernel.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    # Object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w[:, :3]
    # Target position: (num_envs, 3)
    target_pos_w = target.data.root_pos_w[:, :3]
    # Distance of the object to the target (only XY plane for pushing): (num_envs,)
    distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)

    return 1 - torch.tanh(distance / std)


def object_goal_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward the agent for aligning the object orientation with the target orientation.

    Uses a tanh-kernel based on the orientation error (angle between quaternions).

    Args:
        env: The environment.
        std: The standard deviation of the tanh kernel (in radians).
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    # Object quaternion: (num_envs, 4) in (w, x, y, z) format
    object_quat_w = object.data.root_quat_w
    # Target quaternion: (num_envs, 4) in (w, x, y, z) format
    target_quat_w = target.data.root_quat_w

    # Compute orientation error as the angle between the two quaternions
    # q_diff = q_object * q_target_inv
    # For unit quaternions: q_inv = (w, -x, -y, -z)
    target_quat_inv = torch.cat([target_quat_w[:, :1], -target_quat_w[:, 1:]], dim=1)

    # Quaternion multiplication: q1 * q2
    # q_diff = object_quat * target_quat_inv
    w1, x1, y1, z1 = object_quat_w.unbind(dim=1)
    w2, x2, y2, z2 = target_quat_inv.unbind(dim=1)
    q_diff_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_diff_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_diff_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_diff_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Normalize the quaternion difference
    q_diff = torch.stack([q_diff_w, q_diff_x, q_diff_y, q_diff_z], dim=1)
    q_diff_norm = torch.norm(q_diff, dim=1, keepdim=True).clamp(min=1e-8)
    q_diff = q_diff / q_diff_norm

    # Orientation error: 2 * asin(|q_diff[1:]|)
    vec_norm = torch.norm(q_diff[:, 1:], dim=1).clamp(max=1.0)
    ori_error = 2.0 * torch.asin(vec_norm)

    return 1 - torch.tanh(ori_error / std)


def object_at_goal(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.03,
    ori_threshold: float = 0.1745,  # ~10 degrees in radians
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Bonus reward when the object reaches the target position and orientation.

    Args:
        env: The environment.
        pos_threshold: Position threshold in meters. Defaults to 0.03 (3cm).
        ori_threshold: Orientation threshold in radians. Defaults to ~10 degrees.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Reward tensor of shape (num_envs,) with 1.0 for success, 0.0 otherwise.
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # Position check
    object_pos_w = object.data.root_pos_w[:, :3]
    target_pos_w = target.data.root_pos_w[:, :3]
    pos_distance = torch.norm(object_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
    pos_success = pos_distance < pos_threshold

    # Orientation check
    object_quat_w = object.data.root_quat_w
    target_quat_w = target.data.root_quat_w

    target_quat_inv = torch.cat([target_quat_w[:, :1], -target_quat_w[:, 1:]], dim=1)
    w1, x1, y1, z1 = object_quat_w.unbind(dim=1)
    w2, x2, y2, z2 = target_quat_inv.unbind(dim=1)
    q_diff_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_diff_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_diff_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_diff_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    q_diff = torch.stack([q_diff_w, q_diff_x, q_diff_y, q_diff_z], dim=1)
    q_diff_norm = torch.norm(q_diff, dim=1, keepdim=True).clamp(min=1e-8)
    q_diff = q_diff / q_diff_norm

    vec_norm = torch.norm(q_diff[:, 1:], dim=1).clamp(max=1.0)
    ori_error = 2.0 * torch.asin(vec_norm)
    ori_success = ori_error < ori_threshold

    return (pos_success & ori_success).float()


def object_goal_distance_velocity(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward the agent for the velocity of the object towards the target.

    Args:
        env: The environment.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # Object position and velocity: (num_envs, 3)
    object_pos_w = object.data.root_pos_w[:, :3]
    object_vel_w = object.data.root_lin_vel_w[:, :3]
    # Target position: (num_envs, 3)
    target_pos_w = target.data.root_pos_w[:, :3]

    # Vector from object to target
    to_target = target_pos_w[:, :2] - object_pos_w[:, :2]
    to_target_dist = torch.norm(to_target, dim=1, keepdim=True).clamp(min=1e-6)
    to_target_unit = to_target / to_target_dist

    # Project velocity onto the unit vector towards the target
    vel_towards_target = torch.sum(object_vel_w[:, :2] * to_target_unit, dim=1)

    return vel_towards_target


def object_goal_distance_raw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward the agent for the distance between the object and the target.

    Args:
        env: The environment.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # Object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w[:, :3]
    # Target position: (num_envs, 3)
    target_pos_w = target.data.root_pos_w[:, :3]

    # Distance of the object to the target: (num_envs,)
    return torch.norm(object_pos_w - target_pos_w, dim=1)


def object_ee_distance_raw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for the distance between the end-effector and the object.

    Args:
        env: The environment.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        ee_frame_cfg: The end-effector frame configuration. Defaults to SceneEntityCfg("ee_frame").

    Returns:
        Reward tensor of shape (num_envs,).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Object position: (num_envs, 3)
    object_pos_w = object.data.root_pos_w[:, :3]
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]

    # Distance of the end-effector to the object: (num_envs,)
    return torch.norm(object_pos_w - ee_w, dim=1)
