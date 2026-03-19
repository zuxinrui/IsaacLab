# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for the ball-in-a-cup task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_out_of_workspace(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ball"]),
    z_min: float = -0.25,
    radial_max: float = 1.0,
) -> torch.Tensor:
    """Terminate episode when the ball drifts too far or drops below floor margin."""
    robot: Articulation = env.scene[ball_cfg.name]
    ball_name = "ball"
    if isinstance(ball_cfg.body_names, list) and len(ball_cfg.body_names) > 0:
        ball_name = ball_cfg.body_names[0]
    elif isinstance(ball_cfg.body_names, str) and len(ball_cfg.body_names) > 0:
        ball_name = ball_cfg.body_names

    ball_i = robot.data.body_names.index(ball_name)
    # use environment-local position (not world position), otherwise vectorized envs
    # away from origin terminate immediately.
    ball_pos_w = robot.data.body_pos_w[:, ball_i, :]
    ball_pos = ball_pos_w - env.scene.env_origins
    radial = torch.linalg.norm(ball_pos[:, :2], dim=-1)

    return (ball_pos[:, 2] < z_min) | (radial > radial_max)
