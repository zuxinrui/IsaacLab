# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the push task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object falls off the table.

    Args:
        env: The environment.
        minimum_height: The minimum height below which the episode terminates. Defaults to -0.05.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] < minimum_height


def object_reached_goal(
    env: ManagerBasedRLEnv,
    pos_threshold: float = 0.03,
    ori_threshold: float = 0.1745,  # ~10 degrees in radians
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Terminate when the object reaches the target position and orientation.

    Args:
        env: The environment.
        pos_threshold: Position threshold in meters. Defaults to 0.03 (3cm).
        ori_threshold: Orientation threshold in radians. Defaults to ~10 degrees.
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        target_cfg: The target configuration. Defaults to SceneEntityCfg("target").

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments should terminate.
    """
    object: RigidObject = env.scene[object_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # Position check (XY plane only for pushing)
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

    return pos_success & ori_success
