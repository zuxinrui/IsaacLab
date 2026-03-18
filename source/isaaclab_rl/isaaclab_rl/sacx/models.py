# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal task-conditioned SAC-X models.

This module intentionally contains a compact, skeleton-quality implementation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, out_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _one_hot(task_ids: torch.Tensor, num_tasks: int) -> torch.Tensor:
    return F.one_hot(task_ids.long(), num_classes=num_tasks).float()


class TaskConditionedActor(nn.Module):
    """Shared actor conditioned on task one-hot vector."""

    def __init__(self, obs_dim: int, action_dim: int, num_tasks: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks

        self.backbone = _make_mlp(obs_dim + num_tasks, 2 * action_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_oh = _one_hot(task_ids, self.num_tasks).to(obs.device)
        x = torch.cat([obs, task_oh], dim=-1)
        out = self.backbone(x)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20.0, max=2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs, task_ids)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)
        # standard tanh-squash correction
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(self, obs: torch.Tensor, task_ids: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            mean, _ = self(obs, task_ids)
            return torch.tanh(mean)
        return self.sample(obs, task_ids)[0]


class TwinCritic(nn.Module):
    """Twin Q-network for one intention/task."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        in_dim = obs_dim + action_dim
        self.q1 = _make_mlp(in_dim, 1, hidden_dims)
        self.q2 = _make_mlp(in_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACXModels(nn.Module):
    """Container for shared actor + per-task critics and target critics."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_tasks: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks

        self.actor = TaskConditionedActor(obs_dim, action_dim, num_tasks, hidden_dims)
        self.critics = nn.ModuleList([TwinCritic(obs_dim, action_dim, hidden_dims) for _ in range(num_tasks)])
        self.target_critics = nn.ModuleList([copy.deepcopy(c) for c in self.critics])

        for target in self.target_critics:
            target.requires_grad_(False)

