# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ball-in-a-cup task environment configuration.

This manager-based task is intentionally minimal and state-based. It is designed to expose
clear multi-intention reward terms for SAC-X style training:

- lift
- above_rim
- near_opening
- downward_entry
- catch_sparse

TODO: domain randomization, contact forces
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class BallInCupSceneCfg(InteractiveSceneCfg):
    """Ball-in-a-cup scene with a single robot articulation."""

    robot: ArticulationCfg = MISSING  # type: ignore[assignment]

    # 2026-07-08: rope↔arm contact sensor. `prim_path` targets every rope
    # segment; `filter_prim_paths_expr` restricts recorded net_forces to
    # only forces from arm-link / EE-cylinder contacts (rope-rope tension
    # is excluded — that IS the whole point). Consumed by
    # RewardsCfg.rope_arm_contact via mdp.rope_arm_contact_penalty to
    # discourage wrap-the-rope-around-the-arm catch strategies that the
    # 2026-07-06 sweep found for 3/4 successful morphs.
    rope_arm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/string_seg_.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/link_.*",
            "{ENV_REGEX_NS}/Robot/ee_cylinder",
        ],
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg | mdp.RelativeJointPositionActionCfg = MISSING  # type: ignore[assignment]


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # robot proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 6-dim
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # task-centric geometric features
        cup_ball_features = ObsTerm(
            func=mdp.cup_ball_features_short,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
                "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            },
        )

        # previous action helps with smoothness under low-rate control
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ObservationsCfg_V0Legacy:
    """Legacy 106-dim observation layout used by the earliest ball-in-a-cup PPO checkpoint.

    Layout (pre commit 56783a8b, before the "optimize speed" obs slimming):
    - joint_pos_rel        : 45
    - joint_vel_rel        : 45
    - cup_ball_features    : 10  (full version with v_parallel)
    - last_action          :  6
    - total                : 106
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        cup_ball_features = ObsTerm(
            func=mdp.cup_ball_features,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
                "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            },
        )

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset/randomization events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class EventCfg_V1:
    """Configuration for reset/randomization events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Domain Randomization
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ball", "cup"]),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    randomize_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ball", "cup"]),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        interval_range_s=(10.0, 10.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]),
            "operation": "add",
        },
    )

    joint_noise = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for a clear 5-intention state-based ball-in-a-cup task."""

    # Penalty: Undesired robot contacts
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_robot_contacts,
    #     params={
    #         "threshold": 1.0,
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #     },
    #     weight=-5.0,
    # )

    # Intention 1: Swing-up (early curriculum)
    swing_up = RewTerm(
        func=mdp.reward_swing_up,
        weight=0.2,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "target_height": 0.01,
        },
    )

    # Intention 2: Above base with upright gate
    above_base_upright = RewTerm(
        func=mdp.reward_above_base_upright,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "h_base": 0.0,
            "dh": 1.0,
            "upright_cos_min": 0.5,
        },
    )

    # Intention 3: Above rim with upright gate
    above_rim_upright = RewTerm(
        func=mdp.reward_above_rim_upright,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "h_rim": 0.08,
            "dh": 1.0,
            "upright_cos_min": 0.5,
        },
    )

    # Intention 4: Near opening with upright gate
    near_opening_upright = RewTerm(
        func=mdp.reward_near_opening_upright,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "h_rim": 0.08,
            "dh": 1.0,
            "sigma_d": 0.035,
            "upright_cos_min": 0.5,
        },
    )

    # Intention 5: Downward entry with upright gate
    # downward_entry_upright = RewTerm(
    #     func=mdp.reward_downward_entry_upright,
    #     weight=1.0,
    #     params={
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
    #         "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
    #         "h_rim": 0.08,
    #         "dh_entry": 1.0,
    #         "sigma_d": 0.03,
    #         "v_scale": 1.0,
    #         "upright_cos_min": 0.5,
    #     },
    # )

    # Intention 6: Catch success with upright gate
    catch_success_upright = RewTerm(
        func=mdp.reward_catch_success_upright,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "cup_cfg": SceneEntityCfg("robot", body_names=["cup"]),
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "h_low": 0.01,
            "h_high": 0.09,
            "d_inner": 0.03,
            "rel_speed_max": 0.5,
            "upright_cos_min": 0.75,
        },
    )

    # Penalty: Action magnitude
    action_magnitude = RewTerm(func=mdp.regularization_penalty, weight=0.0001)

    # Penalty: Rope↔arm contact (2026-07-08 user directive). Discourages
    # wrap-the-rope-around-the-arm catch strategies. Weight is env-var
    # tunable so ancient-repro runs can disable via
    # `BIC_ROPE_ARM_CONTACT_WEIGHT=0`. Default -0.1 · per-segment-in-contact
    # / step ≈ -0.5 when 5 segments wrap, small vs +2.0 catch reward but
    # sustained wrap accumulates over the 300-step episode.
    rope_arm_contact = RewTerm(
        func=mdp.rope_arm_contact_penalty,
        weight=float(__import__("os").environ.get("BIC_ROPE_ARM_CONTACT_WEIGHT", "-0.1")),
        params={
            "threshold": float(__import__("os").environ.get("BIC_ROPE_ARM_CONTACT_THRESHOLD", "0.05")),
            "sensor_cfg": SceneEntityCfg("rope_arm_contact"),
        },
    )

    # Penalty: Joint height
    # joint_height_penalty = RewTerm(
    #     func=mdp.joint_height_penalty,
    #     weight=-0.2,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["link_3", "link_4", "link_5", "link_6", "ee_cylinder"]),
    #         "height_threshold": 0.1,
    #         "soft_margin": 0.05,
    #     },
    # )

    # Penalty: Undesired robot contacts
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_robot_contacts,
    #     params={
    #         "threshold": 1.0,
    #         "robot_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces"),
    #     },
    #     weight=-5.0,
    # )


@configclass
class RewardsCfg_V1(RewardsCfg):
    """Reward terms for a clear 5-intention state-based ball-in-a-cup task."""

    # Additional penalty: Undesired robot contacts (heavier weight for v1)
    undesired_contacts = RewTerm(
        func=mdp.undesired_robot_contacts,
        params={
            "threshold": 1.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_out_of_workspace = DoneTerm(
        func=mdp.ball_out_of_workspace,
        params={
            "ball_cfg": SceneEntityCfg("robot", body_names=["ball"]),
            "z_min": -0.25,
            "radial_max": 5.0,
        },
    )


@configclass
class BallInCupEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the state-based ball-in-a-cup task."""

    scene: BallInCupSceneCfg = BallInCupSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 10.0

        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 1

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class BallInCupEnvCfg_PLAY(BallInCupEnvCfg):
    """Play configuration for the ball-in-a-cup task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.sim.render_interval = 1


@configclass
class BallInCupEnvCfg_V1(BallInCupEnvCfg):
    """Configuration for the ball-in-a-cup task (v1).

    Improvements:
    - Added contact sensor for collision detection.
    - Added collision penalty for undesired robot contacts.
    - Added domain randomization:
        - Object mass randomization.
        - Friction coefficient randomization.
        - Gravity randomization.
        - Joint noise/perturbation.
    - Added initial joint position randomization.
    """
    rewards: RewardsCfg_V1 = RewardsCfg_V1()
    events: EventCfg_V1 = EventCfg_V1()

    def __post_init__(self):
        super().__post_init__()

        # Domain Randomization Tuning
        # 1. Object mass randomization
        self.events.randomize_mass.params["mass_distribution_params"] = (0.5, 1.5)

        # 2. Friction coefficient randomization
        self.events.randomize_friction.params["static_friction_range"] = (0.5, 1.5)
        self.events.randomize_friction.params["dynamic_friction_range"] = (0.5, 1.5)

        # 3. Gravity randomization
        self.events.randomize_gravity.params["gravity_distribution_params"] = ([0.0, 0.0, -1.0], [0.0, 0.0, 1.0])

        # 4. Joint noise/perturbation
        self.events.joint_noise.params["position_range"] = (-0.05, 0.05)
        self.events.joint_noise.params["velocity_range"] = (-0.0, 0.0)

        # 5. Initial joint position randomization
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)


@configclass
class BallInCupEnvCfg_V1_PLAY(BallInCupEnvCfg_V1):
    """Play configuration for the ball-in-a-cup task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.sim.render_interval = 1


@configclass
class BallInCupEnvCfg_V2(BallInCupEnvCfg):
    """Configuration for the ball-in-a-cup task (v1).

    Improvements:
    - Added contact sensor for collision detection.
    - Added collision penalty for undesired robot contacts.
    - Added domain randomization:
        - Object mass randomization.
        - Friction coefficient randomization.
        - Gravity randomization.
        - Joint noise/perturbation.
    - Added initial joint position randomization.
    """
    rewards: RewardsCfg_V1 = RewardsCfg_V1()  # including undesired contact penalty
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class BallInCupEnvCfg_V2_PLAY(BallInCupEnvCfg_V2):
    """Play configuration for the ball-in-a-cup task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.sim.render_interval = 1


@configclass
class BallInCupEnvCfg_V0Legacy(BallInCupEnvCfg):
    """Legacy ball-in-a-cup config exposing the original 106-dim observation space.

    Matches the earliest checkpoints (pre-commit 56783a8b) so those policies can be
    loaded and visualized without re-training. Everything else (rewards, events,
    sim settings) follows V0.
    """

    observations: ObservationsCfg_V0Legacy = ObservationsCfg_V0Legacy()

    def __post_init__(self):
        super().__post_init__()


@configclass
class BallInCupEnvCfg_V0Legacy_PLAY(BallInCupEnvCfg_V0Legacy):
    """Play configuration for the legacy 106-dim ball-in-a-cup task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.sim.render_interval = 1
