# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal SAC-X skeleton trainer for Isaac Lab tasks.

Notes:
- This script is intentionally lightweight and meant as a runnable skeleton.
- It consumes per-term rewards from env extras keys:
  - ``sacx/reward_terms``
  - ``sacx/reward_term_names``
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
import sys
from typing import Any, cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a minimal SAC-X skeleton (uniform intention scheduler).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--total_steps", type=int, default=50_000, help="Total environment steps.")
parser.add_argument("--device", type=str, default="cuda:0", help="Torch/simulation device.")

# optional skeleton tuning knobs
parser.add_argument("--replay_size", type=int, default=200_000, help="Replay buffer capacity.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for SAC-X updates.")
parser.add_argument("--learning_starts", type=int, default=2_000, help="Number of steps before updates start.")
parser.add_argument("--updates_per_step", type=int, default=1, help="Gradient updates per env step.")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for actor/critic MLPs.")
parser.add_argument("--log_interval", type=int, default=1_000, help="Console log interval in steps.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from isaaclab_rl.sacx import SACXAgent, SACXModels, SACXReplayBuffer, UniformIntentionScheduler

import isaaclab_tasks  # noqa: F401


def _to_policy_obs(obs: torch.Tensor | dict) -> torch.Tensor:
    """Extract policy observation tensor for common Isaac Lab formats."""
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs and isinstance(obs["policy"], torch.Tensor):
            return obs["policy"]
        if "obs" in obs and isinstance(obs["obs"], torch.Tensor):
            return obs["obs"]
    raise TypeError(f"Unsupported observation format for SAC-X skeleton: {type(obs)}")


def _flatten_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 2:
        return obs
    return obs.flatten(start_dim=1)


def _as_action_tensor(action_space, action: torch.Tensor) -> torch.Tensor:
    """Clip to Box bounds when finite, otherwise keep tanh-bounded action."""
    if not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        return action
    low = torch.as_tensor(action_space.low, device=action.device, dtype=action.dtype)
    high = torch.as_tensor(action_space.high, device=action.device, dtype=action.dtype)
    finite = torch.isfinite(low) & torch.isfinite(high)
    if torch.any(finite):
        action = torch.where(action < low, low, action)
        action = torch.where(action > high, high, action)
    return action


@hydra_task_config(args_cli.task, "skrl_sac_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict | None):
    """Run a minimal SAC-X collection/update loop."""
    # basic overrides
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env_unwrapped = cast(Any, env.unwrapped)
        obs, _ = env.reset()
        obs_t = _flatten_obs(_to_policy_obs(obs))

        single_action_space = env_unwrapped.single_action_space
        if not hasattr(single_action_space, "shape"):
            raise RuntimeError("SAC-X skeleton currently expects a continuous Box action space.")

        obs_dim = int(obs_t.shape[-1])
        action_dim = int(single_action_space.shape[0])

        # discover reward term count from first transition
        warmup_action = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)
        next_obs, _, terminated, truncated, extras = env.step(warmup_action)
        reward_terms = extras.get("sacx/reward_terms", None)
        reward_term_names = extras.get("sacx/reward_term_names", None)
        if reward_terms is None:
            raise RuntimeError("Missing extras['sacx/reward_terms']; env must expose SAC-X reward terms.")

        num_tasks = int(reward_terms.shape[-1])
        print(f"[SAC-X] reward terms ({num_tasks}): {reward_term_names}")

        hidden = (args_cli.hidden_dim, args_cli.hidden_dim)
        models = SACXModels(obs_dim=obs_dim, action_dim=action_dim, num_tasks=num_tasks, hidden_dims=hidden)
        agent = SACXAgent(models=models, device=args_cli.device)
        replay = SACXReplayBuffer(capacity=args_cli.replay_size, device=args_cli.device)
        scheduler = UniformIntentionScheduler(num_tasks=num_tasks, device=args_cli.device)

        # consume the warmup transition into replay to validate pipeline early
        done = torch.as_tensor(terminated, device=obs_t.device) | torch.as_tensor(truncated, device=obs_t.device)
        done = done.float()
        replay.add_batch(
            obs=obs_t,
            next_obs=_flatten_obs(_to_policy_obs(next_obs)),
            action=warmup_action,
            done=done,
            task_id=torch.zeros((obs_t.shape[0],), dtype=torch.long, device=obs_t.device),
            reward_terms=reward_terms,
        )
        obs_t = _flatten_obs(_to_policy_obs(next_obs))

        for step in range(1, args_cli.total_steps + 1):
            task_ids = scheduler.sample(obs_t.shape[0], device=obs_t.device)

            if step < args_cli.learning_starts:
                action = 2.0 * torch.rand((obs_t.shape[0], action_dim), device=obs_t.device) - 1.0
            else:
                with torch.no_grad():
                    action = agent.select_action(obs_t, task_ids, deterministic=False)

            action = _as_action_tensor(single_action_space, action)
            next_obs, _, terminated, truncated, extras = env.step(action)

            next_obs_t = _flatten_obs(_to_policy_obs(next_obs))
            done = torch.as_tensor(terminated, device=obs_t.device) | torch.as_tensor(truncated, device=obs_t.device)
            done = done.float()
            reward_terms = extras.get("sacx/reward_terms", None)
            if reward_terms is None:
                raise RuntimeError("Missing extras['sacx/reward_terms'] during rollout.")

            replay.add_batch(
                obs=obs_t,
                next_obs=next_obs_t,
                action=action,
                done=done,
                task_id=task_ids,
                reward_terms=reward_terms,
            )
            obs_t = next_obs_t

            last_metrics = None
            if step >= args_cli.learning_starts and replay.can_sample(args_cli.batch_size):
                for _ in range(args_cli.updates_per_step):
                    batch = replay.sample(args_cli.batch_size)
                    selected_task = int(scheduler.sample(1, device="cpu").item())
                    last_metrics = agent.update_selected_task(batch, selected_task=selected_task)

            if step % args_cli.log_interval == 0:
                if last_metrics is None:
                    print(f"[SAC-X][step={step}] replay={len(replay)} (collecting)")
                else:
                    print(
                        "[SAC-X][step={}] replay={} selected_task={} critic_loss={:.6f} actor_loss={:.6f}".format(
                            step,
                            len(replay),
                            int(last_metrics["selected_task"]),
                            last_metrics["critic_loss"],
                            last_metrics["actor_loss"],
                        )
                    )

        print("[SAC-X] training loop completed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
