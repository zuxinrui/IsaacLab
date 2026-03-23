# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lynx robot push cube environment configuration using joint position control."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CuboidCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.push import mdp
from isaaclab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg, PushGoalEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor  # isort: skip


def _make_lynx_robot_cfg() -> LynxRobotCfg:
    """Create and configure the Lynx robot configuration."""
    robot_cfg = LynxRobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        num_joints=6,
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1,
        rotation_angles=[180.0, 0.0, 0.0, -180.0, 0.0, 0.0],
        l1_end_point_pos=(0.0, 0.0, 0.2),
        l1_end_point_theta=0.0,
        l2_end_point_pos=(0.0, 0.0, 0.2805),
        l2_end_point_theta=0.0,
        l3_end_point_pos=(0.0, 0.0, 0.2),
        l3_end_point_theta=0.0,
        l4_end_point_pos=(0.0, 0.0, 0.2805),
        l4_end_point_theta=0.0,
        l5_end_point_pos=(0.0, 0.0, 0.2),
        l5_end_point_theta=0.0,
        joint_velocity_limit_rad_s=0.3490658503988659,  # 20 deg/s
        joint_acceleration_limit_rad_s2=1.7453292519943295,  # 100 deg/s^2
    )
    # Performance: reduce procedural tube tessellation to lower USD prim count,
    # collision complexity and broadphase pressure in large batched scenes.
    robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    )
    robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    # Enable contact sensors for the robot
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.func = LynxUsdConstructor.spawn
    robot_cfg.spawn.robot_cfg = {
        "genotype_tube": robot_cfg.genotype_tube,
        "genotype_joints": robot_cfg.genotype_joints,
        "rotation_angles": robot_cfg.rotation_angles,
        "bspline_num_segments": robot_cfg.bspline_num_segments,
        "bspline_dual_point_distance": robot_cfg.bspline_dual_point_distance,
        "tube_radiuses": robot_cfg.tube_radiuses,
        "clamp_stl": robot_cfg.clamp_stl,
        "ee_stl": robot_cfg.ee_stl,
        "actuators": robot_cfg.actuators,
        "init_state": robot_cfg.init_state,
        "rigid_props": robot_cfg.spawn.rigid_props,
        "articulation_props": robot_cfg.spawn.articulation_props,
        "activate_contact_sensors": robot_cfg.spawn.activate_contact_sensors,
        "l1_end_point_pos": robot_cfg.l1_end_point_pos,
        "l1_end_point_theta": robot_cfg.l1_end_point_theta,
        "l2_end_point_pos": robot_cfg.l2_end_point_pos,
        "l2_end_point_theta": robot_cfg.l2_end_point_theta,
        "l3_end_point_pos": robot_cfg.l3_end_point_pos,
        "l3_end_point_theta": robot_cfg.l3_end_point_theta,
        "l4_end_point_pos": robot_cfg.l4_end_point_pos,
        "l4_end_point_theta": robot_cfg.l4_end_point_theta,
        "l5_end_point_pos": robot_cfg.l5_end_point_pos,
        "l5_end_point_theta": robot_cfg.l5_end_point_theta,
        "joint_velocity_limit_rad_s": robot_cfg.joint_velocity_limit_rad_s,
        "joint_acceleration_limit_rad_s2": robot_cfg.joint_acceleration_limit_rad_s2,
    }
    return robot_cfg


@configclass
class LynxCubePushEnvCfg(PushGoalEnvCfg):
    """Configuration for the Lynx robot push cube environment.

    The Lynx robot arm pushes a cube on a table to a target position and orientation.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        self.sim.dt = 1.0 / 30.0
        self.decimation = 6
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 256

        # Set Lynx as robot
        self.scene.robot = _make_lynx_robot_cfg()

        # Set actions for the Lynx robot (joint position control)
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,  # 10 degrees in radians
            clip={".*": (-1.0, 1.0)},
        )

        # Set Cube as the object to push
        # The cube is placed on the table surface (table height ~0.0m in env frame)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.3, 0.0, 0.11],
                rot=[1, 0, 0, 0],
            ),
            spawn=CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
        )

        # Set Target marker (visual only, no physics interaction)
        # The target is a semi-transparent cube showing where the object should go
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.3, 0.0, 0.1],
                rot=[1, 0, 0, 0],
            ),
            spawn=CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=1,
                    solver_velocity_iteration_count=0,
                    max_angular_velocity=0.0,
                    max_linear_velocity=0.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=True,
                    kinematic_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.1),
                    # opacity=0.9,
                ),
            ),
        )

        # Set up end-effector frame transformer
        # The Lynx robot's end-effector link is named "ee"
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.0],
                    ),
                ),
            ],
        )

        # Set up contact sensor for the robot
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
        )


@configclass
class LynxCubePushEnvCfg_PLAY(LynxCubePushEnvCfg):
    """Play configuration for the Lynx robot push cube environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Aggressive play-time throughput tuning.
        # Reduce physics substeps per action significantly for faster rollout.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        # Decouple render cadence from control cadence.
        # With decimation=12-style setups, render_interval=decimation can look like "卡" purely due to low viewport FPS.
        self.sim.render_interval = 1

        # Lighter PhysX solver budgets for interactive playback.
        self.sim.physx.bounce_threshold_velocity = 0.3
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 64
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 32

        # Keep object rigid-body properties from task config (already lightweight in this task).
        # make a smaller scene for play
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # # Add orientation visualizer for play
        # from isaaclab.markers import VisualizationMarkersCfg
        # from isaaclab.managers import SceneEntityCfg
        # import isaaclab.sim as sim_utils
        # import copy

        # # Define the visualizer config
        # visualizer_cfg = VisualizationMarkersCfg(
        #     prim_path="{ENV_REGEX_NS}/Visuals/OrientationVisualizer",
        #     markers={
        #         "object": copy.copy(RED_ARROW_X_MARKER_CFG.markers["arrow"]),
        #         "target": copy.copy(BLUE_ARROW_X_MARKER_CFG.markers["arrow"]),
        #     },
        # )
        # # Adjust scale of arrows
        # visualizer_cfg.markers["object"].scale = (0.2, 0.02, 0.02)  # type: ignore
        # visualizer_cfg.markers["target"].scale = (0.2, 0.02, 0.02)  # type: ignore

        # # Add visualization term to observations
        # from isaaclab.managers import ObservationTermCfg as ObsTerm
        # self.observations.policy.object_orientation_vis = ObsTerm(  # type: ignore
        #     func=mdp.visualize_object_orientation,
        #     params={
        #         "object_cfg": SceneEntityCfg("object"),
        #         "target_cfg": SceneEntityCfg("target"),
        #         "visualizer_cfg": visualizer_cfg,
        #     },
        # )
