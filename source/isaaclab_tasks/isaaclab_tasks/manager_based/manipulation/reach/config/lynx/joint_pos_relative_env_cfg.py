# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from .joint_pos_ee_env_cfg import LynxReachEnvCfg


@configclass
class LynxReachMujocoEnvCfg(LynxReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set control frequency to 5 Hz
        # sim.dt is 1/60, so decimation = 60 / 5 = 12
        self.decimation = 12
        self.sim.render_interval = self.decimation

        # Override actions to relative joint position control
        # Scale is [-10, 10] degrees = [-pi/18, pi/18] radians
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", 
            joint_names=["joint_.*"], 
            scale=math.radians(10.0), 
            use_zero_offset=True
        )
