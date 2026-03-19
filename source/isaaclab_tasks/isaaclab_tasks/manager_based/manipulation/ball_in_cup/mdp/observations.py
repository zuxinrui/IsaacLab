# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the ball-in-a-cup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cup_ball_features(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["cup"]),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
) -> torch.Tensor:
    """Return concise geometric features between cup and ball.

    Output columns: [h, d_perp, v_parallel, ball_height_world, cup_pos_rel, ball_pos_rel].
    """
    robot: Articulation = env.scene[robot_cfg.name]

    cup_name = "cup"
    ball_name = "ball"
    if isinstance(cup_cfg.body_names, list) and len(cup_cfg.body_names) > 0:
        cup_name = cup_cfg.body_names[0]
    if isinstance(ball_cfg.body_names, list) and len(ball_cfg.body_names) > 0:
        ball_name = ball_cfg.body_names[0]

    cup_i = robot.data.body_names.index(cup_name)
    ball_i = robot.data.body_names.index(ball_name)

    cup_pos = robot.data.body_pos_w[:, cup_i, :]
    ball_pos = robot.data.body_pos_w[:, ball_i, :]
    cup_vel = robot.data.body_lin_vel_w[:, cup_i, :]
    ball_vel = robot.data.body_lin_vel_w[:, ball_i, :]
    cup_quat = robot.data.body_quat_w[:, cup_i, :]

    # relative to robot base
    robot_pos = robot.data.root_pos_w
    cup_pos_rel = cup_pos - robot_pos
    ball_pos_rel = ball_pos - robot_pos

    cup_axis = quat_apply(cup_quat, cup_pos.new_tensor([1.0, 0.0, 0.0]).repeat(cup_quat.shape[0], 1))
    cup_axis = cup_axis / cup_axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    rel = ball_pos - cup_pos
    h = torch.sum(rel * cup_axis, dim=-1, keepdim=True)
    rel_perp = rel - h * cup_axis
    d_perp = torch.norm(rel_perp, dim=-1, keepdim=True)

    v_rel = ball_vel - cup_vel
    v_parallel = torch.sum(v_rel * cup_axis, dim=-1, keepdim=True)
    ball_height_world = ball_pos[:, 2:3]

    return torch.cat([h, d_perp, v_parallel, ball_height_world, cup_pos_rel, ball_pos_rel], dim=-1)
