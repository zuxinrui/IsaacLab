# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lynx push env with per-physics-step observation delay."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from .joint_pos_env_cfg import LynxCubePushEnvCfg, LynxCubePushEnvCfg_PLAY


@configclass
class LynxCubePushObsDelayEnvCfg(LynxCubePushEnvCfg):
    """Lynx push env config with delayed policy observations.

    Delay is configured in seconds and converted to physics-steps internally.
    """

    obs_delay_s: float = 0.08


@configclass
class LynxCubePushObsDelayEnvCfg_PLAY(LynxCubePushEnvCfg_PLAY):
    """Play config for delayed-observation Lynx push env."""

    obs_delay_s: float = 0.08


class LynxPushObsDelayEnv(ManagerBasedRLEnv):
    """Manager-based RL env with policy observation delay using a tensor ring buffer.

    Implementation goals:
    - no IsaacLab core changes,
    - only one policy observation compute per physics step,
    - delayed read at RL step end (minimal overhead).
    """

    cfg: LynxCubePushObsDelayEnvCfg

    def __init__(self, cfg: LynxCubePushObsDelayEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._policy_group_name = "policy"
        if self._policy_group_name not in self.observation_manager.active_terms:
            raise ValueError(
                f"Observation group '{self._policy_group_name}' not found. "
                f"Available groups: {list(self.observation_manager.active_terms.keys())}."
            )

        self.obs_delay_steps = max(0, int(round(self.cfg.obs_delay_s / self.physics_dt)))
        self._obs_history_len = self.obs_delay_steps + 1

        policy_obs_shape = self.observation_manager.group_obs_dim[self._policy_group_name]
        if isinstance(policy_obs_shape, list):
            raise RuntimeError(
                "Policy observation group must be concatenated into one tensor to apply observation delay."
            )

        self._policy_obs_history = torch.zeros(
            (self._obs_history_len, self.num_envs, *policy_obs_shape),
            device=self.device,
            dtype=torch.float32,
        )
        self._policy_obs_hist_idx = 0

        # Fill initial history so first rollout steps don't read zeros.
        self._fill_policy_history_all_envs()

    def _write_policy_obs_history(self):
        """Compute current policy observation and append to ring buffer."""
        policy_obs_now = self.observation_manager.compute_group(self._policy_group_name, update_history=False)
        if not isinstance(policy_obs_now, torch.Tensor):
            raise RuntimeError("Policy observation group must be a concatenated tensor.")
        self._policy_obs_history[self._policy_obs_hist_idx].copy_(policy_obs_now)
        self._policy_obs_hist_idx = (self._policy_obs_hist_idx + 1) % self._obs_history_len

    def _read_delayed_policy_obs(self) -> torch.Tensor:
        """Read delayed policy observation from ring buffer."""
        read_idx = (self._policy_obs_hist_idx - 1 - self.obs_delay_steps) % self._obs_history_len
        return self._policy_obs_history[read_idx]

    def _compute_obs_with_delayed_policy(self, update_history: bool = True) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute observations while replacing the policy group with delayed observations."""
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for group_name in self.observation_manager.active_terms:
            if group_name == self._policy_group_name:
                obs_buffer[group_name] = self._read_delayed_policy_obs()
            else:
                obs_buffer[group_name] = self.observation_manager.compute_group(group_name, update_history=update_history)
        self.observation_manager._obs_buffer = obs_buffer
        return obs_buffer

    def _fill_policy_history_all_envs(self):
        """Fill full history for all envs with current policy observation."""
        policy_obs_now = self.observation_manager.compute_group(self._policy_group_name, update_history=False)
        if not isinstance(policy_obs_now, torch.Tensor):
            raise RuntimeError("Policy observation group must be a concatenated tensor.")
        self._policy_obs_history[:] = policy_obs_now.unsqueeze(0)

    def step(self, action: torch.Tensor):
        """Execute one RL step with delayed policy observations."""
        # process actions
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # physics stepping + per-physics-step policy obs caching
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self.scene.update(dt=self.physics_dt)
            self._write_policy_obs_history()

        # post-step bookkeeping
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self.extras["sacx/reward_terms"] = self.reward_manager.step_reward_contributions
        self.extras["sacx/reward_term_names"] = self.reward_manager.active_terms

        # recorder path (no obs-history update here to keep semantics similar to base env)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self._compute_obs_with_delayed_policy(update_history=False)
            self.recorder_manager.record_post_step()

        # reset done envs
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            reset_env_ids_list = reset_env_ids.tolist()
            self.recorder_manager.record_pre_reset(reset_env_ids_list)
            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids_list)

        # command/event updates
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # final observations (policy group delayed)
        self.obs_buf = self._compute_obs_with_delayed_policy(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor):
        """Reset envs and refill delayed-observation history for reset envs only."""
        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        super()._reset_idx(cast(Sequence[int], env_ids_tensor))

        policy_obs_now = self.observation_manager.compute_group(self._policy_group_name, update_history=False)
        if not isinstance(policy_obs_now, torch.Tensor):
            raise RuntimeError("Policy observation group must be a concatenated tensor.")
        self._policy_obs_history[:, env_ids_tensor] = policy_obs_now[env_ids_tensor].unsqueeze(0)
