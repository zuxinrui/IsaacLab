# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .joint_actions import JointAction
from .actions_cfg import JointActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TrapezoidalJointPositionAction(JointAction):
    """Joint action term that applies a trapezoidal velocity profile to reach the target position.

    This action term interpolates the joint positions from the current position to the target position
    using a trapezoidal velocity profile. The profile is defined by the maximum velocity and
    maximum acceleration.

    The interpolation is performed at each simulation step.
    """

    cfg: TrapezoidalJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: TrapezoidalJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # parse max velocity and acceleration
        self._max_vel = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        if isinstance(cfg.max_velocity, (float, int)):
            self._max_vel[:] = float(cfg.max_velocity)
        elif isinstance(cfg.max_velocity, dict):
            import isaaclab.utils.string as string_utils
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.max_velocity, self._joint_names)
            self._max_vel[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported max_velocity type: {type(cfg.max_velocity)}")

        self._max_acc = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        if isinstance(cfg.max_acceleration, (float, int)):
            self._max_acc[:] = float(cfg.max_acceleration)
        elif isinstance(cfg.max_acceleration, dict):
            import isaaclab.utils.string as string_utils
            index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.max_acceleration, self._joint_names)
            self._max_acc[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported max_acceleration type: {type(cfg.max_acceleration)}")

        # buffers for trapezoidal profile
        self._start_pos = torch.zeros_like(self._raw_actions)
        self._target_pos = torch.zeros_like(self._raw_actions)
        self._current_pos_target = torch.zeros_like(self._raw_actions)
        self._start_time = torch.zeros(self.num_envs, device=self.device)
        
        # Profile parameters
        self._ta = torch.zeros_like(self._raw_actions)  # acceleration time
        self._tv = torch.zeros_like(self._raw_actions)  # constant velocity time
        self._tf = torch.zeros_like(self._raw_actions)  # total time
        self._dist = torch.zeros_like(self._raw_actions)
        self._sign = torch.zeros_like(self._raw_actions)
        self._v_peak = torch.zeros_like(self._raw_actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        
        # Initialize targets to current positions to avoid jumps on start
        current_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._start_pos[env_ids] = current_pos
        self._target_pos[env_ids] = current_pos
        self._current_pos_target[env_ids] = current_pos
        self._start_time[env_ids] = self._env.sim.current_time
        
        self._ta[env_ids] = 0.0
        self._tv[env_ids] = 0.0
        self._tf[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations to get the target position
        new_target_pos = self._raw_actions * self._scale + self._offset
        
        # clip actions
        if self.cfg.clip is not None:
            new_target_pos = torch.clamp(
                new_target_pos, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        # Check which targets have changed significantly to restart the profile
        # Or just always restart when process_actions is called (usually once per decimation)
        # Isaac Lab typically calls process_actions at the start of each RL step.
        
        self._start_pos[:] = self._current_pos_target
        self._target_pos[:] = new_target_pos
        self._start_time[:] = self._env.sim.current_time
        
        # Calculate trapezoidal profile parameters
        self._dist = torch.abs(self._target_pos - self._start_pos)
        self._sign = torch.sign(self._target_pos - self._start_pos)
        
        # Time to reach max velocity: ta = v_max / a_max
        # Distance covered during acceleration/deceleration: d_acc = 0.5 * a_max * ta^2 = 0.5 * v_max^2 / a_max
        
        d_acc = 0.5 * (self._max_vel**2) / self._max_acc
        
        # Case 1: Profile reaches max velocity (dist > 2 * d_acc)
        mask_reaches_max = self._dist >= 2 * d_acc
        
        # For Case 1:
        self._ta[mask_reaches_max] = self._max_vel[mask_reaches_max] / self._max_acc[mask_reaches_max]
        self._tv[mask_reaches_max] = (self._dist[mask_reaches_max] - 2 * d_acc[mask_reaches_max]) / self._max_vel[mask_reaches_max]
        self._v_peak[mask_reaches_max] = self._max_vel[mask_reaches_max]
        
        # Case 2: Profile does not reach max velocity (triangular profile)
        mask_triangular = ~mask_reaches_max
        # d = a * ta^2 => ta = sqrt(d / a) where d is half distance
        self._ta[mask_triangular] = torch.sqrt(self._dist[mask_triangular] / self._max_acc[mask_triangular])
        self._tv[mask_triangular] = 0.0
        self._v_peak[mask_triangular] = self._max_acc[mask_triangular] * self._ta[mask_triangular]
        
        self._tf = 2 * self._ta + self._tv

    def apply_actions(self):
        # Current time relative to start of profile
        t = self._env.sim.current_time - self._start_time.unsqueeze(1)
        
        # Clamp t to [0, tf]
        t_clamped = torch.clamp(t, min=0.0)
        
        # Calculate relative displacement based on trapezoidal profile
        # Phase 1: Acceleration (0 <= t < ta)
        # s = 0.5 * a * t^2
        s_acc = 0.5 * self._max_acc * (torch.clamp(t_clamped, max=self._ta)**2)
        
        # Phase 2: Constant velocity (ta <= t < ta + tv)
        # s = d_acc + v_peak * (t - ta)
        t_v = torch.clamp(t_clamped - self._ta, min=0.0)
        t_v = torch.min(t_v, self._tv)
        d_acc = 0.5 * self._max_acc * (self._ta**2)
        s_vel = d_acc + self._v_peak * t_v
        
        # Phase 3: Deceleration (ta + tv <= t < 2*ta + tv)
        # s = d_acc + d_vel + v_peak * (t - ta - tv) - 0.5 * a * (t - ta - tv)^2
        t_d = torch.clamp(t_clamped - self._ta - self._tv, min=0.0)
        t_d = torch.min(t_d, self._ta)
        d_vel = self._v_peak * self._tv
        s_dec = d_acc + d_vel + self._v_peak * t_d - 0.5 * self._max_acc * (t_d**2)
        
        # Select displacement based on current time
        displacement = torch.where(t_clamped < self._ta, s_acc,
                                   torch.where(t_clamped < self._ta + self._tv, s_vel, s_dec))
        
        # Final clamping for safety (t > tf)
        displacement = torch.where(t_clamped >= self._tf, self._dist, displacement)
        
        self._current_pos_target = self._start_pos + self._sign * displacement
        
        # set position targets
        self._asset.set_joint_position_target(self._current_pos_target, joint_ids=self._joint_ids)


@configclass
class TrapezoidalJointPositionActionCfg(JointActionCfg):
    """Configuration for the trapezoidal joint position action term.

    See :class:`TrapezoidalJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = TrapezoidalJointPositionAction

    max_velocity: float | dict[str, float] = MISSING
    """Maximum velocity for the trapezoidal profile (rad/s)."""

    max_acceleration: float | dict[str, float] = MISSING
    """Maximum acceleration for the trapezoidal profile (rad/s^2)."""

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.
    """
