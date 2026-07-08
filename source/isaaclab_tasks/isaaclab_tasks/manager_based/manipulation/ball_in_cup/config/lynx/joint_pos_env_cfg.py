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
        # MorphBench encodes `genotype_joints` as a per-link list (e.g.
        # [1,0,0,0,0] = 1 joint active at link-1), while ancient
        # LynxBallInCupRobotCfg expects a scalar int (the active-joint count).
        # Bridge by summing the per-link 1s — matches the semantics morph_012
        # (1 active joint) / morph_015 (2 active) / morph_018 (2 active).
        if isinstance(overrides.get("genotype_joints"), list):
            overrides["genotype_joints"] = int(sum(overrides["genotype_joints"]))
        # Same schema drift: drop any keys the ancient Cfg doesn't accept.
        # (Add to this set if a future yaml introduces another MorphBench-
        # only key that doesn't map onto an ancient field.)
        _ANCIENT_ACCEPTED = {
            "num_joints", "genotype_tube", "genotype_joints", "rotation_angles",
            "l1_end_point_pos", "l1_end_point_theta",
            "l2_end_point_pos", "l2_end_point_theta",
            "l3_end_point_pos", "l3_end_point_theta",
            "l4_end_point_pos", "l4_end_point_theta",
            "l5_end_point_pos", "l5_end_point_theta",
            "cup_radius", "cup_height", "ball_radius",
            "string_length", "string_radius", "string_num_segments",
            "joint_velocity_limit_rad_s", "joint_acceleration_limit_rad_s2",
        }
        dropped = sorted(k for k in overrides if k not in _ANCIENT_ACCEPTED)
        overrides = {k: v for k, v in overrides.items() if k in _ANCIENT_ACCEPTED}
        print(f"[ball_in_cup] morph overrides from {yaml_path}: "
              f"applied={sorted(overrides)}  dropped={dropped}")

    # 2026-07-08 (path-2 morph-swap fix): per-morph home joint pose from
    # bic_home_joint_pos_deg in the sibling index.csv. Without this, ALL mb20
    # morphs start at all-zeros → cup starts at wrong orientation for most →
    # RL cannot discover catch behaviour (only 4/20 morphs caught in the
    # 2026-07-06 sweep because they happened to be near-vertical at zeros).
    # Mirrors the MB-native env_isaac wiring (docs/2026-06-16_ball_in_cup_
    # per_morph_home_pose.md L57-70), inlined here so the IsaacLab container
    # doesn't need MorphBench python on its path.
    per_morph_home_deg = None
    if yaml_path and os.path.isfile(yaml_path):
        import csv as _csv, ast as _ast
        index_csv = os.path.join(os.path.dirname(yaml_path), "index.csv")
        if os.path.isfile(index_csv):
            target_real = os.path.realpath(yaml_path)
            target_base = os.path.basename(yaml_path)
            try:
                with open(index_csv, newline="") as _f:
                    for _row in _csv.DictReader(_f):
                        _home = (_row.get("bic_home_joint_pos_deg") or "").strip()
                        _my = (_row.get("morph_yaml") or "").strip()
                        if not _home or not _my:
                            continue
                        # Match on realpath (rare) or basename (mb20 index.csv
                        # stores an alien absolute path; basename is stable).
                        if os.path.realpath(_my) == target_real or \
                           os.path.basename(_my) == target_base:
                            try:
                                per_morph_home_deg = list(_ast.literal_eval(_home))
                                print(f"[ball_in_cup] per-morph home from {index_csv}: "
                                      f"{per_morph_home_deg} deg")
                            except (ValueError, SyntaxError):
                                pass
                            break
            except (OSError, _csv.Error):
                pass

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

    # Apply per-morph home pose loaded above. LynxRobotCfg.__post_init__
    # already filtered init_state.joint_pos by num_joints, so we set a fresh
    # dict (only joints 1..num_joints) with the per-morph radians. This is
    # the reset target that reset_joints_by_scale randomises around AND the
    # frozen-joint locks — the authoritative arm spawn pose.
    if per_morph_home_deg is not None:
        import math as _math
        from isaaclab.assets import ArticulationCfg as _ArticulationCfg
        _nj = robot_cfg.num_joints
        if len(per_morph_home_deg) < _nj:
            print(f"[ball_in_cup] WARN: per_morph_home_deg has {len(per_morph_home_deg)} "
                  f"values but num_joints={_nj}; pad-with-zeros to length {_nj}.")
            per_morph_home_deg = list(per_morph_home_deg) + [0.0] * (_nj - len(per_morph_home_deg))
        robot_cfg.init_state = _ArticulationCfg.InitialStateCfg(
            joint_pos={
                f"joint_{i+1}": float(_math.radians(deg))
                for i, deg in enumerate(per_morph_home_deg[:_nj])
            },
        )
        print(f"[ball_in_cup] init_state.joint_pos (rad): {robot_cfg.init_state.joint_pos}")

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

        # Optional render camera — enabled via BIC_ENABLE_RENDER_CAM=1 env var.
        # Adds a side-view CameraCfg inside __post_init__ so the render product
        # is properly initialized at env creation (mirrors MorphBench's proven
        # pattern in lynx/rl/ball_in_cup/env_isaac/config/lynx/joint_pos_env_cfg.py).
        # Skipped in training runs (env var unset) so no impact on morphpack.
        import os as _os
        if _os.environ.get("BIC_ENABLE_RENDER_CAM"):
            from isaaclab.sensors import CameraCfg
            self.scene.view_1 = CameraCfg(
                prim_path="{ENV_REGEX_NS}/view_1",
                update_period=0.0,
                height=96,
                width=96,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=18.428,       # MorphBench: vfov=42° at 96×96
                    clipping_range=(0.01, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(2.0, 0.0, 0.382),            # az=-180 el=0 dist=2 lookat=(0,0,1) +Z_offset
                    rot=(0.5, 0.5, 0.5, 0.5),         # quat_xyzw for OpenGL camera
                    convention="opengl",
                ),
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
