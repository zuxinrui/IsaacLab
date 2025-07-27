# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.envs.mdp.actions as actions
from isaaclab.envs.mdp.events import reset_joints_by_offset, reset_root_state_uniform
import isaaclab.envs.mdp.observations as observations
import isaaclab.envs.mdp.rewards as rewards
import isaaclab.envs.mdp.terminations as terminations
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

from isaaclab.envs import ManagerBasedEnv

##
# Actions
##

JointPositionActionCfg = actions.JointPositionActionCfg
"""Joint position action for the Lynx arm."""

##
# Observations
##

joint_pos_rel = observations.joint_pos_rel
"""Joint position observation for the Lynx arm."""

def end_effector_pos_w(env) -> torch.Tensor:
    """End-effector position in world frame."""
    # Find the index of the end-effector body
    ee_idx, _ = env.scene["robot"].find_bodies("tool_link")
    return env.scene["robot"].data.body_pos_w[:, ee_idx[0], :]

def target_pos_w(env) -> torch.Tensor:
    """Target position in world frame."""
    return env.scene["target"].data.root_pose_w[:, :3]

##
# Rewards
##

is_alive = rewards.is_alive
"""Reward for staying alive."""

is_terminated = rewards.is_terminated
"""Penalty for terminating."""

def l2_distance_from_target(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 distance from target."""
    # Get asset position
    if asset_cfg.body_names:
        asset_idx, _ = env.scene[asset_cfg.name].find_bodies(asset_cfg.body_names[0])
        asset_pos = env.scene[asset_cfg.name].data.body_pos_w[:, asset_idx[0], :]
    else:
        asset_pos = env.scene[asset_cfg.name].data.pos_w[:, 0, :]

    # Get target position
    if target_cfg.body_names:
        target_idx, _ = env.scene[target_cfg.name].find_bodies(target_cfg.body_names[0])
        target_pos = env.scene[target_cfg.name].data.body_pos_w[:, target_idx[0], :]
    else:
        target_pos = env.scene[target_cfg.name].data.root_pose_w[:, :3]

    return torch.linalg.norm(asset_pos - target_pos, dim=-1)

def joint_pos_limit_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty for joint positions exceeding limits."""
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    joint_pos_limits = env.scene[asset_cfg.name].data.joint_pos_limits
    lower_limits = joint_pos_limits[:, :, 0]
    upper_limits = joint_pos_limits[:, :, 1]

    # Calculate how much the joint positions exceed the limits
    lower_exceedance = torch.relu(lower_limits - joint_pos)
    upper_exceedance = torch.relu(joint_pos - upper_limits)

    # Sum the exceedances and return as penalty
    return torch.sum(lower_exceedance + upper_exceedance, dim=-1)

def joint_vel_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L2 norm of joint velocities."""
    joint_vel = env.scene[asset_cfg.name].data.joint_vel
    return torch.linalg.norm(joint_vel, dim=-1)

##
# Terminations
##

time_out = terminations.time_out
"""Termination for time out."""

def l2_distance_from_target_less_than(env, asset_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Termination when L2 distance from target is less than a threshold."""
    distance = l2_distance_from_target(env, asset_cfg, target_cfg)
    return distance < threshold

def joint_pos_out_of_bounds(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Termination when joint positions are out of bounds."""
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    joint_pos_limits = env.scene[asset_cfg.name].data.joint_pos_limits
    lower_limits = joint_pos_limits[:, :, 0]
    upper_limits = joint_pos_limits[:, :, 1]

    # Check if any joint position is outside the limits
    out_of_bounds = (joint_pos < lower_limits) | (joint_pos > upper_limits)
    return torch.any(out_of_bounds, dim=-1)

##
# Events
##

reset_joints_by_offset = reset_joints_by_offset
reset_scene_entity_by_offset = reset_root_state_uniform
def reset_root_state_uniform_sphere(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    center: tuple[float, float, float],
    radius: float,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position within a sphere and random velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position uniformly within a sphere around the given center and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a center and radius for the spherical sampling of position. The velocity ranges are
    dictionaries for each axis and rotation. The keys of the dictionary are ``x``, ``y``, ``z``, ``roll``,
    ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``. If the dictionary does not
    contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # positions
    # sample uniform points in a sphere
    # sample uniform points in a sphere
    # Generate random points in a 3D Gaussian distribution
    gaussian_samples = math_utils.sample_gaussian(0.0, 1.0, (len(env_ids), 3), device=asset.device)
    # Normalize these points to get points on the surface of a unit sphere
    unit_sphere_samples = math_utils.normalize(gaussian_samples)
    # Multiply by a random radius sampled from [0, radius] raised to the power of 1/3 (to account for volume)
    random_radii = radius * torch.pow(torch.rand(len(env_ids), 1, device=asset.device), 1/3)
    rand_samples = unit_sphere_samples * random_radii
    # Add the center offset
    rand_samples += torch.tensor(center, device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples

    # orientations (randomly sample from SO(3))
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)