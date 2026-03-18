# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal SAC-X agent update logic.

The implementation is intentionally skeleton-level and kept compact.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .models import SACXModels


class SACXAgent:
    """Minimal SAC-X update routines.

    Provided routines:
    - selected-task critic update (for one sampled intention)
    - shared actor update (task-conditioned actor)
    """

    def __init__(
        self,
        models: SACXModels,
        device: str | torch.device,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha: float = 0.2,
    ):
        self.models = models
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.models.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.models.actor.parameters(), lr=actor_lr)
        self.critic_optimizers = [torch.optim.Adam(c.parameters(), lr=critic_lr) for c in self.models.critics]

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, task_ids: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.models.actor.act(obs, task_ids, deterministic=deterministic)

    def _polyak_update(self, task_id: int):
        critic = self.models.critics[task_id]
        target = self.models.target_critics[task_id]
        for target_p, p in zip(target.parameters(), critic.parameters(), strict=True):
            target_p.data.mul_(1.0 - self.tau)
            target_p.data.add_(self.tau * p.data)

    def _min_q_by_task(self, obs: torch.Tensor, action: torch.Tensor, task_ids: torch.Tensor, *, target: bool) -> torch.Tensor:
        values = torch.zeros((obs.shape[0], 1), device=obs.device)
        critics = self.models.target_critics if target else self.models.critics
        for t in range(self.models.num_tasks):
            mask = task_ids == t
            if torch.any(mask):
                q1, q2 = critics[t](obs[mask], action[mask])
                values[mask] = torch.minimum(q1, q2)
        return values

    def update_selected_task(
        self,
        batch: dict[str, torch.Tensor],
        selected_task: int,
        actor_task_ids: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Run one selected-task critic update + shared actor update.

        Args:
            batch: Sampled replay batch.
            selected_task: Intention index used for critic target/reward selection.
            actor_task_ids: Optional per-sample task IDs for actor update. If omitted,
                uses sampled ``batch["task_id"]``.
        """
        obs = batch["obs"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        action = batch["action"].to(self.device)
        done = batch["done"].to(self.device).view(-1, 1)
        reward_terms = batch["reward_terms"].to(self.device)
        sample_task_ids = batch["task_id"].to(self.device).long().view(-1)

        if selected_task < 0 or selected_task >= self.models.num_tasks:
            raise ValueError(f"selected_task out of range: {selected_task}")
        reward = reward_terms[:, selected_task].view(-1, 1)

        # ---- selected-task critic update ----
        critic = self.models.critics[selected_task]
        target_critic = self.models.target_critics[selected_task]

        with torch.no_grad():
            next_task_ids = torch.full((next_obs.shape[0],), selected_task, dtype=torch.long, device=self.device)
            next_action, next_log_prob = self.models.actor.sample(next_obs, next_task_ids)
            target_q1, target_q2 = target_critic(next_obs, next_action)
            target_min_q = torch.minimum(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1.0 - done) * self.gamma * target_min_q

        q1, q2 = critic(obs, action)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        critic_optim = self.critic_optimizers[selected_task]
        critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optim.step()

        self._polyak_update(selected_task)

        # ---- shared actor update ----
        if actor_task_ids is None:
            actor_task_ids = sample_task_ids
        actor_task_ids = actor_task_ids.to(self.device).long().view(-1)

        for c in self.models.critics:
            c.requires_grad_(False)

        pi_action, log_prob = self.models.actor.sample(obs, actor_task_ids)
        min_q = self._min_q_by_task(obs, pi_action, actor_task_ids, target=False)
        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        for c in self.models.critics:
            c.requires_grad_(True)

        return {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "selected_task": float(selected_task),
        }

