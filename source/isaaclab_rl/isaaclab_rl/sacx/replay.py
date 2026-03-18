# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal replay buffer for SAC-X skeleton training."""

from __future__ import annotations

from typing import cast

import torch


class SACXReplayBuffer:
    """Simple uniform replay buffer.

    Stored fields are:
    - obs
    - next_obs
    - action
    - done
    - task_id
    - reward_terms
    """

    def __init__(self, capacity: int, device: str | torch.device):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = int(capacity)
        self.device = torch.device(device)

        self._obs: torch.Tensor | None = None
        self._next_obs: torch.Tensor | None = None
        self._action: torch.Tensor | None = None
        self._done: torch.Tensor | None = None
        self._task_id: torch.Tensor | None = None
        self._reward_terms: torch.Tensor | None = None

        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _lazy_init(self, obs: torch.Tensor, action: torch.Tensor, reward_terms: torch.Tensor):
        obs_shape = tuple(obs.shape[1:])
        action_shape = tuple(action.shape[1:])
        reward_terms_shape = tuple(reward_terms.shape[1:])

        self._obs = torch.empty((self.capacity, *obs_shape), dtype=obs.dtype, device=self.device)
        self._next_obs = torch.empty((self.capacity, *obs_shape), dtype=obs.dtype, device=self.device)
        self._action = torch.empty((self.capacity, *action_shape), dtype=action.dtype, device=self.device)
        self._done = torch.empty((self.capacity,), dtype=torch.float32, device=self.device)
        self._task_id = torch.empty((self.capacity,), dtype=torch.long, device=self.device)
        self._reward_terms = torch.empty((self.capacity, *reward_terms_shape), dtype=reward_terms.dtype, device=self.device)

    def add_batch(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        done: torch.Tensor,
        task_id: torch.Tensor,
        reward_terms: torch.Tensor,
    ):
        """Add a vectorized batch of transitions."""
        batch = obs.shape[0]
        if batch == 0:
            return
        if self._obs is None:
            self._lazy_init(obs, action, reward_terms)
        obs_buf, next_obs_buf, action_buf, done_buf, task_buf, reward_terms_buf = self._storage()

        obs = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        action = action.to(self.device)
        done = done.to(self.device).float().view(-1)
        task_id = task_id.to(self.device).long().view(-1)
        reward_terms = reward_terms.to(self.device)

        if batch >= self.capacity:
            obs = obs[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            action = action[-self.capacity :]
            done = done[-self.capacity :]
            task_id = task_id[-self.capacity :]
            reward_terms = reward_terms[-self.capacity :]
            batch = self.capacity

        end = self._pos + batch
        if end <= self.capacity:
            sl = slice(self._pos, end)
            obs_buf[sl] = obs
            next_obs_buf[sl] = next_obs
            action_buf[sl] = action
            done_buf[sl] = done
            task_buf[sl] = task_id
            reward_terms_buf[sl] = reward_terms
        else:
            first = self.capacity - self._pos
            second = batch - first
            obs_buf[self._pos :] = obs[:first]
            obs_buf[:second] = obs[first:]
            next_obs_buf[self._pos :] = next_obs[:first]
            next_obs_buf[:second] = next_obs[first:]
            action_buf[self._pos :] = action[:first]
            action_buf[:second] = action[first:]
            done_buf[self._pos :] = done[:first]
            done_buf[:second] = done[first:]
            task_buf[self._pos :] = task_id[:first]
            task_buf[:second] = task_id[first:]
            reward_terms_buf[self._pos :] = reward_terms[:first]
            reward_terms_buf[:second] = reward_terms[first:]

        self._pos = (self._pos + batch) % self.capacity
        self._size = min(self.capacity, self._size + batch)

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._obs is None
            or self._next_obs is None
            or self._action is None
            or self._done is None
            or self._task_id is None
            or self._reward_terms is None
        ):
            raise RuntimeError("Replay buffer is not initialized yet")
        return (
            cast(torch.Tensor, self._obs),
            cast(torch.Tensor, self._next_obs),
            cast(torch.Tensor, self._action),
            cast(torch.Tensor, self._done),
            cast(torch.Tensor, self._task_id),
            cast(torch.Tensor, self._reward_terms),
        )

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Uniformly sample transitions."""
        if not self.can_sample(batch_size):
            raise RuntimeError(f"Not enough samples: size={self._size}, batch_size={batch_size}")
        obs_buf, next_obs_buf, action_buf, done_buf, task_buf, reward_terms_buf = self._storage()
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            "obs": obs_buf[idx],
            "next_obs": next_obs_buf[idx],
            "action": action_buf[idx],
            "done": done_buf[idx],
            "task_id": task_buf[idx],
            "reward_terms": reward_terms_buf[idx],
        }
