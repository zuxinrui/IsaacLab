# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lynx ball-in-a-cup environment configuration using joint position control."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_assets.robots.lynx_ball_in_cup import LynxBallInCupConstructor, LynxBallInCupRobotCfg
from isaaclab_tasks.manager_based.manipulation.ball_in_cup.ball_in_cup_env_cfg import (
    BallInCupEnvCfg,
    BallInCupEnvCfg_PLAY,
    BallInCupEnvCfg_V0Legacy,
    BallInCupEnvCfg_V0Legacy_PLAY,
    BallInCupEnvCfg_V1,
    BallInCupEnvCfg_V1_PLAY,
    BallInCupEnvCfg_V2,
    BallInCupEnvCfg_V2_PLAY,
)
from isaaclab_tasks.manager_based.manipulation.ball_in_cup import mdp


def _make_lynx_ball_in_cup_cfg(string_num_segments: int = 10) -> LynxBallInCupRobotCfg:
    """Create and configure the Lynx ball-in-a-cup robot configuration.

    ``string_num_segments`` defaults to the current 10-segment rope. Pass 12 to
    reproduce the legacy geometry that the pre-56783a8b 106-dim checkpoints were
    trained against (those checkpoints expect 45 joint DOFs: 6 arm + 12*3 string
    spherical + 3 ball spherical).

    2026-06-27 (path-1 morph swap): if `$LYNX_MORPH_YAML` points at a yaml
    file shaped like MorphBench's `models/morphbench20/morphologies/morph_*.yaml`
    (top-level key MorphConfig.robot_description_dict), its values override the
    hard-coded defaults below. Unset / non-existent → ancient default geometry
    (cup_radius=0.05, cup_height=0.08, ball_radius=0.02, 10-segment 0.4 m rope).
    Only the robot-cfg-level fields are read from yaml; reward thresholds,
    physics, etc. stay at the ancient values that converged model_400.pt.
    """
    import os, yaml  # local import — no extra module-load cost for ancient default
    overrides: dict = {}
    yaml_path = os.environ.get("LYNX_MORPH_YAML") or os.environ.get(
        "LYNX_MORPH_YAML_BALL_IN_CUP"
    )
    if yaml_path and os.path.isfile(yaml_path):
        with open(yaml_path) as _f:
            _y = yaml.safe_load(_f) or {}
        # MorphBench yaml has top-level "MorphConfig.robot_description_dict";
        # tolerate flat dicts too (test fixtures).
        _root = _y.get("MorphConfig", _y)
        overrides = _root.get("robot_description_dict", _root) or {}
        # tuples in the Cfg are written as lists in yaml — re-tupleify so
        # dataclass equality + downstream consumers stay happy.
        for _k in ("l1_end_point_pos", "l2_end_point_pos", "l3_end_point_pos",
                   "l4_end_point_pos", "l5_end_point_pos"):
            if _k in overrides and isinstance(overrides[_k], list):
                overrides[_k] = tuple(overrides[_k])
        print(f"[ball_in_cup] morph overrides from {yaml_path}: "
              f"{sorted(overrides)}")

    def _ov(key, default):
        return overrides.get(key, default)

    robot_cfg = LynxBallInCupRobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        num_joints=_ov("num_joints", 6),
        genotype_tube=_ov("genotype_tube", [0, 1, 0, 1, 0]),
        genotype_joints=_ov("genotype_joints", 1),
        rotation_angles=_ov("rotation_angles",
                            [180.0, 0.0, 0.0, -180.0, 0.0, 90.0]),
        l1_end_point_pos=_ov("l1_end_point_pos", (0.0, 0.0, 0.2)),
        l1_end_point_theta=_ov("l1_end_point_theta", 0.0),
        l2_end_point_pos=_ov("l2_end_point_pos", (0.0, 0.0, 0.2805)),
        l2_end_point_theta=_ov("l2_end_point_theta", 0.0),
        l3_end_point_pos=_ov("l3_end_point_pos", (0.0, 0.0, 0.2)),
        l3_end_point_theta=_ov("l3_end_point_theta", 0.0),
        l4_end_point_pos=_ov("l4_end_point_pos", (0.0, 0.0, 0.2805)),
        l4_end_point_theta=_ov("l4_end_point_theta", 0.0),
        l5_end_point_pos=_ov("l5_end_point_pos", (0.0, 0.0, 0.2)),
        l5_end_point_theta=_ov("l5_end_point_theta", 0.0),
        cup_radius=_ov("cup_radius", 0.05),
        cup_height=_ov("cup_height", 0.08),
        ball_radius=_ov("ball_radius", 0.02),
        string_length=_ov("string_length", 0.4),
        string_radius=_ov("string_radius", 0.0005),
        string_num_segments=_ov("string_num_segments", string_num_segments),
        joint_velocity_limit_rad_s=_ov("joint_velocity_limit_rad_s",
                                       1.7453292519943295),
        joint_acceleration_limit_rad_s2=_ov("joint_acceleration_limit_rad_s2",
                                            1.7453292519943295),
    )

    robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=32,
        solver_velocity_iteration_count=4,
    )
    robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.func = LynxBallInCupConstructor.spawn
    robot_cfg.spawn.robot_cfg = {
        "num_joints": robot_cfg.num_joints,
        "genotype_tube": robot_cfg.genotype_tube,
        "genotype_joints": robot_cfg.genotype_joints,
        "rotation_angles": robot_cfg.rotation_angles,
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
        "bspline_num_segments": robot_cfg.bspline_num_segments,
        "bspline_dual_point_distance": robot_cfg.bspline_dual_point_distance,
        "tube_radiuses": robot_cfg.tube_radiuses,
        "clamp_stl": robot_cfg.clamp_stl,
        "ee_stl": robot_cfg.ee_stl,
        "actuators": robot_cfg.actuators,
        "init_state": robot_cfg.init_state,
        "cup_radius": robot_cfg.cup_radius,
        "cup_height": robot_cfg.cup_height,
        "ball_radius": robot_cfg.ball_radius,
        "string_length": robot_cfg.string_length,
        "string_radius": robot_cfg.string_radius,
        "string_num_segments": robot_cfg.string_num_segments,
        "collision_mode": robot_cfg.collision_mode,
        "articulation_props": robot_cfg.spawn.articulation_props,
        "rigid_props": robot_cfg.spawn.rigid_props,
        "activate_contact_sensors": robot_cfg.spawn.activate_contact_sensors,
    }
    return robot_cfg


@configclass
class LynxBallInCupEnvCfg(BallInCupEnvCfg):
    """Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        # for the "ball in a cup" task, we need higher sim frequency:
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_PLAY(BallInCupEnvCfg_PLAY):
    """Play-time Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 256
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 256

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V1(BallInCupEnvCfg_V1):
    """Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        # for the "ball in a cup" task, we need higher sim frequency:
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V1_PLAY(BallInCupEnvCfg_V1_PLAY):
    """Play-time Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 256
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 256

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V2(BallInCupEnvCfg_V2):
    """Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        # for the "ball in a cup" task, we need higher sim frequency:
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V2_PLAY(BallInCupEnvCfg_V2_PLAY):
    """Play-time Lynx ball-in-a-cup environment config."""

    def __post_init__(self):
        super().__post_init__()

        # Performance-oriented simulation setup for play/inference:
        # keep 5Hz control while reducing expensive physics sub-steps.
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        # Relax global solver settings for throughput (sufficient for push task stability).
        self.sim.physx.bounce_threshold_velocity = 0.2
        # 16k-env broadphase requires larger aggregate pair buffers.
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 256
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 256

        self.scene.robot = _make_lynx_ball_in_cup_cfg()

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V0Legacy(BallInCupEnvCfg_V0Legacy):
    """Lynx ball-in-a-cup config with the legacy 106-dim observation space.

    Intended purely for loading and visualizing the earliest PPO checkpoints that
    were trained before the observation slimming (commit 56783a8b) and before the
    string was shortened from 12 to 10 segments (commit 5467554a).

    Total obs: joint_pos_rel(45) + joint_vel_rel(45) + cup_ball_features(10) +
    last_action(6) = 106.
    """

    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048

        self.scene.robot = _make_lynx_ball_in_cup_cfg(string_num_segments=12)

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )


@configclass
class LynxBallInCupEnvCfg_V0Legacy_PLAY(BallInCupEnvCfg_V0Legacy_PLAY):
    """Play-time Lynx ball-in-a-cup config with the legacy 106-dim observation space."""

    def __post_init__(self):
        super().__post_init__()

        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 256
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 256

        self.scene.robot = _make_lynx_ball_in_cup_cfg(string_num_segments=12)

        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )
