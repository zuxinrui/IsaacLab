# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.lynx_reach.mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.lynx import LynxArmCfg # isort:skip


##
# Scene definition
##


@configclass
class LynxReachSceneCfg(InteractiveSceneCfg):
    """Configuration for a Lynx reach scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Lynx arm
    robot: ArticulationCfg = LynxArmCfg(prim_path="{ENV_REGEX_NS}/Robot")

    # target sphere
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), opacity=0.3),
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"], scale=0.5)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        end_effector_pos = ObsTerm(func=mdp.end_effector_pos_w)
        target_pos = ObsTerm(func=mdp.target_pos_w)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.01, 0.01),
        },
    )

    reset_target_position = EventTerm(
        func=mdp.reset_root_state_uniform_sphere,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "center": (0.0, 0.0, 1.6),
            "radius": 0.3,
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: reach target
    end_effector_distance = RewTerm(
        func=mdp.l2_distance_from_target,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link"]), "target_cfg": SceneEntityCfg("target")},
    )
    # (4) Penalize joint limits
    joint_limit_penalty = RewTerm(
        func=mdp.joint_pos_limit_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # (5) Penalize excessive joint velocities
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Target reached
    target_reached = DoneTerm(
        func=mdp.l2_distance_from_target_less_than,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["tool_link"]), "target_cfg": SceneEntityCfg("target"), "threshold": 0.05},
    )
    # (3) Arm out of bounds (joint limits)
    joint_pos_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


##
# Environment configuration
##


@configclass
class LynxReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Lynx reach environment."""

    # Scene settings
    scene: LynxReachSceneCfg = LynxReachSceneCfg(num_envs=64, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation