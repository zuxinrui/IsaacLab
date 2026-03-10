# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.trapezoidal_joint_pos_action import TrapezoidalJointPositionAction, TrapezoidalJointPositionActionCfg

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_assets import LYNX_CFG_TCP

from .joint_pos_ee_env_cfg import ReachEnvCfg


@configclass
class LynxReachTrapezoidalEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to lynx
        self.scene.robot = LYNX_CFG_TCP.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # add contact sensor
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0.0, history_length=3, debug_vis=False
        )

        # simulation settings
        # sim.dt = 1/120 and decimation = 24 (to achieve 5Hz policy frequency with 120Hz physics)
        self.sim.dt = 1.0 / 120.0
        self.decimation = 24
        self.sim.render_interval = self.decimation

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["tcp"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["tcp"]

        # add contact penalty reward
        self.rewards.contact_penalty = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
        )

        # override actions to use TrapezoidalJointPositionActionCfg
        self.actions.arm_action = TrapezoidalJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_.*"],
            scale=1.0,
            use_default_offset=True,
            max_velocity=math.radians(20.0),
            max_acceleration=math.radians(100.0),
        )
        self.actions.arm_action.class_type = TrapezoidalJointPositionAction

        # override command generator body
        self.commands.ee_pose.body_name = "tcp"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class LynxReachTrapezoidalEnvCfg_PLAY(LynxReachTrapezoidalEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
