# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Push task environment configuration.

This task involves pushing a cube on a table to a target position and orientation
using the robot's end-effector. The reward design is inspired by the MuJoCo Playground
push_cube.py implementation (DeepMind Technologies Limited, Apache License 2.0).
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class PushTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the push scene with a robot, a cube to push, and a target marker.

    This is the abstract base implementation. The exact scene is defined in the derived
    classes which need to set the robot, end-effector frame, cube object, and target marker.
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Cube object to push: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Target marker (visual only, no physics): will be populated by agent env cfg
    target: RigidObjectCfg = MISSING

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    #     spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    # )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.0, 0.8, 0.018),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.018 / 2)),
    )

    table_leg_1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg1",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.045, 0.045, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5 - 0.045 / 2, 0.4 - 0.045 / 2, -0.6 / 2 - 0.018)),
    )

    table_leg_2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg2",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.045, 0.045, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5 + 0.045 / 2, 0.4 - 0.045 / 2, -0.6 / 2 - 0.018)),
    )

    table_leg_3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg3",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.045, 0.045, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5 - 0.045 / 2, -0.4 + 0.045 / 2, -0.6 / 2 - 0.018)),
    )

    table_leg_4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLeg4",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.045, 0.045, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.5 + 0.045 / 2, -0.4 + 0.045 / 2, -0.6 / 2 - 0.018)),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.6 - 0.018)),
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # End-effector state
        ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        ee_quat = ObsTerm(
            func=mdp.ee_orientation_in_world_frame,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame")},
        )

        # Object (cube) state
        object_pos = ObsTerm(
            func=mdp.object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=mdp.object_orientation_in_world_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )

        # Target state
        target_pos = ObsTerm(
            func=mdp.target_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )
        target_quat = ObsTerm(
            func=mdp.target_orientation_in_world_frame,
            params={"target_cfg": SceneEntityCfg("target")},
        )

        # Relative positions
        ee_to_object = ObsTerm(
            func=mdp.ee_to_object_position,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )
        object_to_target = ObsTerm(
            func=mdp.object_to_target_position,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset all scene entities to their default state
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomize the cube position on the table surface
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # Randomize the target position on the table surface
    reset_target_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    The reward design follows the MuJoCo Playground push_cube.py implementation:
    1. Reach reward: encourage end-effector to approach the cube
    2. Push reward: encourage cube to move toward target position
    3. Orientation reward: encourage cube to match target orientation
    4. Success bonus: large reward when cube reaches target
    """

    # Reward for end-effector approaching the cube (gripper_box in MuJoCo reference)
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=2.0,
    )

    # Reward for cube approaching the target position (box_target in MuJoCo reference)
    pushing_object = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=8.0,
    )

    # Reward for cube matching target orientation (box_orientation in MuJoCo reference)
    object_orientation = RewTerm(
        func=mdp.object_goal_orientation,
        params={
            "std": 0.5,
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=6.0,
    )

    # Large bonus when cube reaches target (success_reward in MuJoCo reference)
    success_bonus = RewTerm(
        func=mdp.object_at_goal,
        params={
            "pos_threshold": 0.03,
            "ori_threshold": 0.1745,  # ~10 degrees
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=50.0,
    )

    # Action regularization penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class PushGoalRewardsCfg:
    """Reward terms for the goal-based push task, following MuJoCo structure."""

    # 1. Distance between end-effector and box (approach)
    reaching_object_dist = RewTerm(
        func=mdp.object_ee_distance_raw,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-0.2,
    )

    # 2. Fine approach reward (approach_fine)
    reaching_object_fine = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.1,
    )

    # 3. Forward push reward (box displacement towards target)
    pushing_object = RewTerm(
        func=mdp.object_goal_distance_raw,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=-0.3,
    )

    # # 4. Push velocity reward (change in displacement)
    # push_velocity = RewTerm(
    #     func=mdp.object_goal_distance_velocity,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "target_cfg": SceneEntityCfg("target"),
    #     },
    #     weight=1.0,
    # )

    # 5. Smoothness penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.1)

    # Add joint_vel back to avoid curriculum errors
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 6. Success bonus
    success_bonus = RewTerm(
        func=mdp.object_at_goal,
        params={
            "pos_threshold": 0.03,
            "ori_threshold": 0.1745,
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=10.0,
    )

    # 7. Undesired robot contacts penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_robot_contacts,
        params={
            "threshold": 1.0,
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_link_cfg": SceneEntityCfg("robot", body_names="ee_cylinder"),
        },
        weight=-5.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the cube falls off the table
    # object_dropping = DoneTerm(
    #     func=mdp.object_out_of_bounds,
    #     params={
    #         "minimum_height": -0.05,
    #         "object_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class PushEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the push environment.

    The robot must push a cube on a table to a target position and orientation.
    """

    # Scene settings
    scene: PushTableSceneCfg = PushTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # No commands needed (target is set via events)
    commands = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 12
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.016666666  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class PushGoalEnvCfg(PushEnvCfg):
    """Configuration for the push environment with a goal-reaching reward.

    This configuration uses a reward function structure similar to the MuJoCo implementation
    provided by the user, adapted for reaching a specific target position.
    """

    # Update rewards to use the new goal-based configuration
    rewards: PushGoalRewardsCfg = PushGoalRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Randomize positions more broadly if needed
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.2, 0.2),
            "z": (0.0, 0.0),
            "yaw": (-3.14159, 3.14159),
        }
        self.events.reset_target_position.params["pose_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.2, 0.2),
            "z": (0.0, 0.0),
            "yaw": (-3.14159, 3.14159),
        }
