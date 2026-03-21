# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the ball-in-a-cup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def undesired_robot_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Penalize contacts between the robot and any other object, excluding ball and string segments.

    Args:
        env: The environment.
        threshold: The contact force threshold to consider a contact.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        sensor_cfg: The contact sensor configuration. Defaults to SceneEntityCfg("contact_forces").

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get net contact forces: (num_envs, num_bodies, 3)
    contact_forces = contact_sensor.data.net_forces_w

    if contact_forces is None or contact_forces.numel() == 0:
        return torch.zeros(env.num_envs, device=contact_sensor.device)

    # Get the indices of the bodies to exclude (ball and string segments)
    # We assume the robot asset is used and we look for body names
    robot: Articulation = env.scene[robot_cfg.name]
    body_names = robot.data.body_names

    # Find indices of bodies to exclude from penalty
    exclude_names = ["ball"] + [f"string_seg_{i}" for i in range(20)] # covers up to 20 segments
    exclude_indices = [i for i, name in enumerate(body_names) if any(ex in name for ex in exclude_names)]

    # Map robot body indices to sensor body indices
    # sensor_cfg.body_ids contains the indices of robot bodies tracked by the sensor
    # We need to find which of these tracked bodies are NOT in the exclude list
    tracked_body_ids = sensor_cfg.body_ids
    if isinstance(tracked_body_ids, slice):
        # If it's a slice, we need to resolve it to a list of indices
        tracked_body_ids = list(range(*tracked_body_ids.indices(len(body_names))))
    
    # Create a mask for tracked bodies that should trigger the penalty
    # contact_forces shape is (num_envs, num_bodies_in_sensor, 3)
    # We need to ensure the mask matches the number of bodies in the sensor
    num_bodies_in_sensor = contact_forces.shape[1]
    mask = torch.ones(num_bodies_in_sensor, dtype=torch.bool, device=contact_sensor.device)
    
    for i, body_id in enumerate(tracked_body_ids):
        if i >= num_bodies_in_sensor:
            break
        if body_id in exclude_indices:
            mask[i] = False

    # Sum the contact force magnitudes for all other bodies
    relevant_contact_forces = contact_forces[:, mask, :]
    if relevant_contact_forces.shape[1] == 0:
        return torch.zeros(env.num_envs, device=contact_sensor.device)
        
    relevant_contact_mag = torch.norm(relevant_contact_forces, dim=-1).max(dim=-1)[0]
    undesired_contact_mask = relevant_contact_mag > threshold

    return undesired_contact_mask.float()


def _resolve_body_name(body_cfg: SceneEntityCfg, fallback: str) -> str:
    if isinstance(body_cfg.body_names, list) and len(body_cfg.body_names) > 0:
        return body_cfg.body_names[0]
    if isinstance(body_cfg.body_names, str) and len(body_cfg.body_names) > 0:
        return body_cfg.body_names
    return fallback


def _cup_ball_kinematics(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    cup_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute key geometric terms used by multiple reward functions.

    Returns:
        h, d_perp, v_parallel, ball_pos, cup_pos
    """
    robot: Articulation = env.scene[robot_cfg.name]

    cup_name = _resolve_body_name(cup_cfg, "cup")
    ball_name = _resolve_body_name(ball_cfg, "ball")

    cup_i = robot.data.body_names.index(cup_name)
    ball_i = robot.data.body_names.index(ball_name)

    cup_pos = robot.data.body_pos_w[:, cup_i, :]
    ball_pos = robot.data.body_pos_w[:, ball_i, :]
    cup_vel = robot.data.body_lin_vel_w[:, cup_i, :]
    ball_vel = robot.data.body_lin_vel_w[:, ball_i, :]
    cup_quat = robot.data.body_quat_w[:, cup_i, :]

    cup_axis = quat_apply(cup_quat, cup_pos.new_tensor([1.0, 0.0, 0.0]).repeat(cup_quat.shape[0], 1))
    cup_axis = cup_axis / cup_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    rel = ball_pos - cup_pos
    h = torch.sum(rel * cup_axis, dim=-1)
    rel_perp = rel - h.unsqueeze(-1) * cup_axis
    d_perp = torch.norm(rel_perp, dim=-1)

    v_rel = ball_vel - cup_vel
    v_parallel = torch.sum(v_rel * cup_axis, dim=-1)
    return h, d_perp, v_parallel, ball_pos, cup_pos


def _cup_ball_kinematics_extended(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    cup_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cup-ball geometric relations.

    Returns:
        h: ball height along cup axis
        d_perp: radial distance to cup axis
        v_parallel: relative velocity along cup axis
        rel_speed: norm of relative velocity ball wrt cup
        upright_cos: dot(cup_up_axis, world_up)
        ball_pos_w: ball world position
        cup_pos_w: cup world position
    """
    robot: Articulation = env.scene[robot_cfg.name]

    cup_name = _resolve_body_name(cup_cfg, "cup")
    ball_name = _resolve_body_name(ball_cfg, "ball")

    cup_i = robot.data.body_names.index(cup_name)
    ball_i = robot.data.body_names.index(ball_name)

    cup_pos = robot.data.body_pos_w[:, cup_i, :]
    ball_pos = robot.data.body_pos_w[:, ball_i, :]
    cup_vel = robot.data.body_lin_vel_w[:, cup_i, :]
    ball_vel = robot.data.body_lin_vel_w[:, ball_i, :]
    cup_quat = robot.data.body_quat_w[:, cup_i, :]

    # local x-axis of cup in world frame (based on existing _cup_ball_kinematics)
    cup_axis = quat_apply(cup_quat, cup_pos.new_tensor([1.0, 0.0, 0.0]).repeat(cup_quat.shape[0], 1))
    cup_axis = cup_axis / cup_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    world_up = torch.zeros_like(cup_axis)
    world_up[:, 2] = 1.0
    upright_cos = torch.sum(cup_axis * world_up, dim=-1)

    rel = ball_pos - cup_pos
    h = torch.sum(rel * cup_axis, dim=-1)
    rel_perp = rel - h.unsqueeze(-1) * cup_axis
    d_perp = torch.norm(rel_perp, dim=-1)

    v_rel = ball_vel - cup_vel
    v_parallel = torch.sum(v_rel * cup_axis, dim=-1)
    rel_speed = torch.norm(v_rel, dim=-1)

    return h, d_perp, v_parallel, rel_speed, upright_cos, ball_pos, cup_pos


def _smooth_upright_gate(
    upright_cos: torch.Tensor,
    cos_min: float = 0.5,
) -> torch.Tensor:
    """Soft gate for cup uprightness based on dot(cup_up_axis, world_up).

    Args:
        upright_cos: cosine between cup up axis and world up, shape [num_envs]
        cos_min: gate starts activating above this value

    Returns:
        gate in [0, 1]
    """
    return torch.clamp((upright_cos - cos_min) / max(1.0 - cos_min, 1e-6), 0.0, 1.0)


def regularization_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small shared smoothness penalty used in all intention rewards."""
    robot: Articulation = env.scene[robot_cfg.name]
    action_penalty = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    joint_vel_penalty = torch.sum(torch.square(robot.data.joint_vel), dim=1)
    return -(action_penalty + joint_vel_penalty)


def reward_lift(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_lift: float = 0.02,
    dh_lift: float = 0.10,
) -> torch.Tensor:
    """Task-1: reward lifting the ball above a baseline along cup axis."""
    h, _, _, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    base = torch.clamp((h - h_lift) / max(dh_lift, 1e-6), 0.0, 1.0)
    return base


def reward_above_rim(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_rim: float = 0.08,
    k_h: float = 20.0,
) -> torch.Tensor:
    """Task-2: reward being at/above cup rim height."""
    h, _, _, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    base = torch.sigmoid(k_h * (h - h_rim))
    return base


def reward_near_opening(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_gate: float = 0.04,
    dh_gate: float = 0.06,
    sigma_d: float = 0.04,
) -> torch.Tensor:
    """Task-3: reward lateral alignment near opening with height gating."""
    h, d_perp, _, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    g_h = torch.clamp((h - h_gate) / max(dh_gate, 1e-6), 0.0, 1.0)
    base = g_h * torch.exp(-(d_perp * d_perp) / max(sigma_d * sigma_d, 1e-9))
    return base


def reward_downward_entry(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_rim: float = 0.08,
    dh_entry: float = 0.04,
    sigma_d: float = 0.03,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """Task-4: reward near-rim, near-axis, downward approach."""
    h, d_perp, v_parallel, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    g_hrim = torch.clamp((h - h_rim) / max(dh_entry, 1e-6), 0.0, 1.0)
    g_d = torch.exp(-(d_perp * d_perp) / max(sigma_d * sigma_d, 1e-9))
    g_v = torch.clamp((-v_parallel) / max(v_scale, 1e-6), 0.0, 1.0)
    base = g_hrim * g_d * g_v
    return base


def reward_catch_sparse(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_low: float = 0.01,
    h_high: float = 0.09,
    d_enter: float = 0.025,
    v_down_min: float = 0.10,
) -> torch.Tensor:
    """Task-5: sparse catch proxy (inside opening corridor + descending)."""
    h, d_perp, v_parallel, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    in_height = (h > h_low) & (h < h_high)
    near_axis = d_perp < d_enter
    downward = v_parallel < -v_down_min
    base = (in_height & near_axis & downward).float()
    return base


def reward_above_cup_base(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
) -> torch.Tensor:
    """Task-6: reward cup being above base height.

    r_6(z) = (1 + tanh(7.5 * z)) / 2
    """
    _, _, _, _, cup_pos = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    z = cup_pos[:, 2]
    base = (1.0 + torch.tanh(7.5 * z)) / 2.0
    return base


def reward_ball_in_cup(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    z_low: float = 0.01,
    z_high: float = 0.09,
    r_cup: float = 0.025,
) -> torch.Tensor:
    """Task-7: reward ball being inside the cup.

    in_cup = (z_low < z < z_high) AND (x^2 + y^2 < r_cup^2)
    where (x, y, z) is the ball position relative to the cup in cup frame.
    """
    h, d_perp, _, _, _ = _cup_ball_kinematics(env, robot_cfg, cup_cfg, ball_cfg)
    # h is the projection of rel_pos onto cup_axis (z in cup frame)
    # d_perp is the norm of the perpendicular component (sqrt(x^2 + y^2) in cup frame)
    in_height = (h > z_low) & (h < z_high)
    in_radius = d_perp < r_cup
    base = (in_height & in_radius).float()
    return base


def reward_swing_up(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    target_height: float = 0.25,
) -> torch.Tensor:
    """Encourage lifting the ball high relative to the cup base in world-z.

    This is an early-stage curriculum reward. It is intentionally broad.
    """
    (
        _,
        _,
        _,
        _,
        _,
        ball_pos_w,
        cup_base_pos_w,
    ) = _cup_ball_kinematics_extended(env, robot_cfg, cup_cfg, ball_cfg)

    z_rel_world = ball_pos_w[:, 2] - cup_base_pos_w[:, 2]
    base = torch.clamp(z_rel_world / max(target_height, 1e-6), 0.0, 1.0)
    return base


def reward_above_base_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_base: float = 0.0,
    dh: float = 0.05,
    upright_cos_min: float = 0.5,
) -> torch.Tensor:
    """Reward ball being above cup base along the cup axis, only when cup is upright."""
    h, _, _, _, upright_cos, _, _ = _cup_ball_kinematics_extended(env, robot_cfg, cup_cfg, ball_cfg)
    g_upright = _smooth_upright_gate(upright_cos, upright_cos_min)
    g_h = torch.clamp((h - h_base) / max(dh, 1e-6), 0.0, 1.0)
    base = g_upright * g_h
    return base


def reward_above_rim_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_rim: float = 0.08,
    dh: float = 0.04,
    upright_cos_min: float = 0.6,
) -> torch.Tensor:
    """Reward ball reaching above the rim height along the cup axis, with upright gating."""
    h, _, _, _, upright_cos, _, _ = _cup_ball_kinematics_extended(env, robot_cfg, cup_cfg, ball_cfg)
    g_upright = _smooth_upright_gate(upright_cos, upright_cos_min)
    g_h = torch.clamp((h - h_rim) / max(dh, 1e-6), 0.0, 1.0)
    base = g_upright * g_h
    return base


def reward_near_opening_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_rim: float = 0.08,
    dh: float = 0.04,
    sigma_d: float = 0.03,
    upright_cos_min: float = 0.6,
) -> torch.Tensor:
    """Reward ball being near the cup opening when the cup is upright."""
    h, d_perp, _, _, upright_cos, _, _ = _cup_ball_kinematics_extended(env, robot_cfg, cup_cfg, ball_cfg)
    g_upright = _smooth_upright_gate(upright_cos, upright_cos_min)
    g_h = torch.clamp((h - h_rim) / max(dh, 1e-6), 0.0, 1.0)
    g_d = torch.exp(-(d_perp * d_perp) / max(sigma_d * sigma_d, 1e-9))
    base = g_upright * g_h * g_d
    return base


def reward_downward_entry_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_rim: float = 0.08,
    dh_entry: float = 0.04,
    sigma_d: float = 0.03,
    v_scale: float = 1.0,
    upright_cos_min: float = 0.65,
) -> torch.Tensor:
    """High-value shaping reward: ball is above rim, near axis, and descending into the cup."""
    h, d_perp, v_parallel, _, upright_cos, _, _ = _cup_ball_kinematics_extended(
        env, robot_cfg, cup_cfg, ball_cfg
    )
    g_upright = _smooth_upright_gate(upright_cos, upright_cos_min)
    g_h = torch.clamp((h - h_rim) / max(dh_entry, 1e-6), 0.0, 1.0)
    g_d = torch.exp(-(d_perp * d_perp) / max(sigma_d * sigma_d, 1e-9))
    g_v = torch.clamp((-v_parallel) / max(v_scale, 1e-6), 0.0, 1.0)
    base = g_upright * g_h * g_d * g_v
    return base


def reward_catch_success_upright(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    h_low: float = 0.01,
    h_high: float = 0.09,
    d_inner: float = 0.025,
    rel_speed_max: float = 0.5,
    upright_cos_min: float = 0.75,
) -> torch.Tensor:
    """Sparse success: ball is inside cup volume, cup is upright, and relative speed is small."""
    h, d_perp, _, rel_speed, upright_cos, _, _ = _cup_ball_kinematics_extended(
        env, robot_cfg, cup_cfg, ball_cfg
    )

    in_height = (h > h_low) & (h < h_high)
    near_axis = d_perp < d_inner
    upright = upright_cos > upright_cos_min
    low_speed = rel_speed < rel_speed_max

    base = (in_height & near_axis & upright & low_speed).float()
    return base


def joint_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.07,
    soft_margin: float = 0.03,
) -> torch.Tensor:
    """Penalize robot joints if they are below a certain height.

    The penalty is soft between (height_threshold) and (height_threshold + soft_margin).
    Below height_threshold, the penalty is 1.0.
    Above height_threshold + soft_margin, the penalty is 0.0.

    Args:
        env: The environment.
        asset_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        height_threshold: The height below which the penalty is maximum. Defaults to 0.07.
        soft_margin: The margin over which the penalty transitions from 1.0 to 0.0. Defaults to 0.03.

    Returns:
        Penalty tensor of shape (num_envs,).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    # Get joint positions in world frame: (num_envs, num_bodies, 3)
    # We use body_pos_w because joints are typically associated with bodies
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    # Get z-heights: (num_envs, num_bodies)
    z_heights = body_pos_w[..., 2]

    # Calculate penalty: 1.0 if z < height_threshold, 0.0 if z > height_threshold + soft_margin
    # Linear interpolation in between
    penalty = torch.clamp((height_threshold + soft_margin - z_heights) / max(soft_margin, 1e-6), 0.0, 1.0)

    # Sum or max the penalty across joints? Max is usually better for "stay away" constraints
    return torch.max(penalty, dim=-1)[0]
