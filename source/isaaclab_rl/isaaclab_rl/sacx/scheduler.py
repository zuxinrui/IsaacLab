# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal SAC-X intention scheduler(s).

This module intentionally provides only a tiny baseline scheduler implementation.
"""

from __future__ import annotations

import torch


class UniformIntentionScheduler:
    """Uniform random intention scheduler.

    This is a skeleton scheduler that samples intentions independently and
    uniformly from ``[0, num_tasks)``.
    """

    def __init__(self, num_tasks: int, device: str | torch.device = "cpu"):
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
        self.num_tasks = int(num_tasks)
        self.device = torch.device(device)

    def sample(self, batch_size: int, device: str | torch.device | None = None) -> torch.Tensor:
        """Sample task IDs.

        Args:
            batch_size: Number of IDs to sample.
            device: Optional output device override.

        Returns:
            Tensor of shape ``(batch_size,)`` with dtype ``torch.long``.
        """
        out_device = torch.device(device) if device is not None else self.device
        return torch.randint(0, self.num_tasks, (batch_size,), dtype=torch.long, device=out_device)

