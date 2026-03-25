# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lynx robot push-deformable-cube environment configuration.

This is a variant of the standard push task where the rigid cube is replaced
with a soft, jelly-like deformable cube simulated via PhysX FEM (MeshCuboidCfg +
DeformableBodyMaterialCfg).  All reward / observation terms that normally rely on
``RigidObject.data.root_pos_w`` are replaced with deformable-specific counterparts
that compute the object centre as the mean of all nodal positions.

**Multi-env workaround**
PhysX FEM soft bodies inside a GridCloner env hierarchy are only registered for
env_0 — PhysX ignores the copies in env_1..env_N-1 regardless of whether they
have ``PhysxDeformableBodyAPI`` applied.  This is a PhysX/GridCloner interaction
bug in Isaac Sim 5.x.

The fix: spawn the deformable objects **outside** the GridCloner hierarchy, at
plain Xform prims under ``/World/_SoftObjects/cube_i`` (positioned at each env's
world origin).  Because these prims are NOT GridCloner children, PhysX cooks and
registers all N FEM bodies independently.

1. ``scene.replicate_physics = False`` — so env xform prims exist before the
   spawner runs (needed to read their world transforms).
2. ``_spawn_deformable_cuboid_per_env`` creates parent Xforms at each env origin
   **outside** ``/World/envs/env_*``, then calls ``spawn_mesh_cuboid.__wrapped__``
   once per env to PhysX-cook each body independently.
3. ``DeformableObjectCfg.prim_path`` is set to ``/World/_SoftObjects/cube_.*/Object``
   so ``DeformableObject._initialize_impl`` builds a SoftBodyView that finds all N
   bodies and returns ``count == num_envs``.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import DeformableObjectCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.push import mdp
from isaaclab_tasks.manager_based.manipulation.push.push_env_cfg import PushEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor  # isort: skip
from .joint_pos_env_cfg import _make_lynx_robot_cfg  # reuse robot factory


# ---------------------------------------------------------------------------
# Custom per-env deformable spawner  —  places objects OUTSIDE the env hierarchy
# ---------------------------------------------------------------------------

# Deformable cubes live here, NOT under /World/envs/env_*
_SOFT_OBJECT_ROOT = "/World/_SoftObjects"

def _spawn_deformable_cuboid_per_env(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn N independent deformable cubes outside the GridCloner env hierarchy.

    PhysX FEM bodies placed inside ``/World/envs/env_*`` (GridCloner's scope)
    are only registered for env_0 regardless of spawning approach.  Placing
    parents via plain ``create_prim`` outside that scope fixes this.

    Critically, the actual mesh spawn uses the **standard** ``spawn_mesh_cuboid``
    (with its ``@clone`` decorator intact) so that ``Cloner.clone`` is called
    from cube_0 to all other cubes — exactly like the passing
    ``test_deformable_object.py`` test which confirms PhysX registers all N bodies
    when parents are created with ``create_prim`` and the prim is cloned.

    Steps
    -----
    1. Find all ``/World/envs/env_*`` xforms (already created by GridCloner
       because ``scene.replicate_physics = False``).
    2. For each env_i, read its world transform (= env origin).
    3. Create ``/World/_SoftObjects/cube_i`` Xform at that position via
       ``create_prim`` (plain USD, NOT a GridCloner child).
    4. Call ``spawn_mesh_cuboid(regex_path, cfg, ...)`` with the full regex
       path ``/World/_SoftObjects/cube_.*/Object``.  The ``@clone`` decorator
       will (a) find all N cube parents, (b) spawn the FEM body at cube_0,
       then (c) ``Cloner.clone`` it to cube_1 .. cube_N-1 with
       ``replicate_physics=False``.
    """
    import sys
    from pxr import Usd, UsdGeom
    from isaaclab.sim.spawners.meshes.meshes import spawn_mesh_cuboid
    from isaaclab.sim.utils import find_matching_prim_paths, get_current_stage, create_prim

    _, asset_name = prim_path.rsplit("/", 1)
    stage = get_current_stage()

    # Find all env xforms — they exist because replicate_physics=False pre-creates them
    raw_env_paths = find_matching_prim_paths("/World/envs/env_.*")
    if not raw_env_paths:
        raise RuntimeError(
            "[DeformableSpawner] No prims matched '/World/envs/env_.*'. "
            "Ensure scene.replicate_physics=False is set so env xforms exist before spawning."
        )

    # Sort numerically so cube_i always corresponds to env_i
    env_paths = sorted(raw_env_paths, key=lambda p: int(p.rsplit("_", 1)[-1]))
    n_envs = len(env_paths)

    print(
        f"\n[DeformableSpawner] Creating {n_envs} parent Xforms outside env hierarchy",
        flush=True,
    )

    # Step 1: create all parent Xforms at env origins (plain create_prim, not GridCloner)
    for i, env_path in enumerate(env_paths):
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim.IsValid():
            raise RuntimeError(f"[DeformableSpawner] env xform {env_path!r} is invalid")

        # World transform of the env xform = env origin
        xformable = UsdGeom.Xformable(env_prim)
        world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        env_origin = world_xform.ExtractTranslation()  # pxr.Gf.Vec3d

        parent_path = f"{_SOFT_OBJECT_ROOT}/cube_{i}"
        create_prim(
            parent_path,
            "Xform",
            translation=(float(env_origin[0]), float(env_origin[1]), float(env_origin[2])),
            stage=stage,
        )
        if i == 0 or i == n_envs - 1:
            print(
                f"[DeformableSpawner]  cube_{i} parent at ({env_origin[0]:.2f},{env_origin[1]:.2f},{env_origin[2]:.2f})",
                flush=True,
            )

    # Step 2: call the STANDARD spawn_mesh_cuboid (with @clone) using the full regex path.
    # @clone will (a) find all N cube_* parents, (b) spawn FEM body at cube_0,
    # (c) Cloner.clone to cube_1..cube_N-1 with replicate_physics=False.
    # PhysX Fabric OOB (DirectGpuHelper.cpp) is avoided by setting sim.use_fabric=False
    # in the env config, which disables the Fabric GPU kernels entirely.
    regex_path = f"{_SOFT_OBJECT_ROOT}/cube_.*/{asset_name}"
    print(f"[DeformableSpawner] Calling spawn_mesh_cuboid(@clone) with regex_path={regex_path!r}", flush=True)
    prim = spawn_mesh_cuboid(regex_path, cfg, translation=translation, orientation=orientation)
    print(f"[DeformableSpawner] Done: {n_envs} cubes spawned via @clone", flush=True)
    sys.stdout.flush()
    return prim


# ---------------------------------------------------------------------------
# Deformable-object reward config (position-only, no orientation)
# ---------------------------------------------------------------------------

@configclass
class DeformablePushGoalRewardsCfg:
    """Rewards for the deformable-cube push task.

    Orientation terms are omitted because a deformable body has no single
    well-defined orientation frame.
    """

    # 1. Distance between end-effector and object centre (approach)
    reaching_object_dist = RewTerm(
        func=mdp.deformable_object_ee_distance_raw,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-0.2,
    )

    # 2. Fine approach reward
    reaching_object_fine = RewTerm(
        func=mdp.deformable_object_ee_distance,
        params={
            "std": 0.05,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=0.1,
    )

    # 3. Push reward (object centre → target, XY plane)
    pushing_object = RewTerm(
        func=mdp.deformable_object_goal_distance_raw,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=-0.3,
    )

    # 4. Success bonus (position only)
    success_bonus = RewTerm(
        func=mdp.deformable_object_at_goal,
        params={
            "pos_threshold": 0.05,
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target"),
        },
        weight=20.0,
    )

    # 5. Smoothness penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 6. Undesired robot contacts (robot body hits the table / object)
    undesired_contacts = RewTerm(
        func=mdp.undesired_robot_contacts,
        params={
            "threshold": 1.0,
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_link_cfg": SceneEntityCfg("robot", body_names=["ee", "ee_cylinder", "link_6"]),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
        weight=-5.0,
    )


# ---------------------------------------------------------------------------
# Deformable-object observation config (position-only)
# ---------------------------------------------------------------------------

@configclass
class DeformablePushObservationsCfg:
    """Observations for the deformable-cube push task."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Robot state [6]
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # End-effector pos [3]
        ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # Deformable object centre pos [3]
        object_pos = ObsTerm(
            func=mdp.deformable_object_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # Target pos [3]
        target_pos = ObsTerm(
            func=mdp.target_position_in_robot_root_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Target orientation [4]
        target_quat = ObsTerm(
            func=mdp.target_orientation_in_world_frame,
            params={"target_cfg": SceneEntityCfg("target")},
        )

        # EE → object [3]
        ee_to_object = ObsTerm(
            func=mdp.deformable_ee_to_object_position,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            },
        )

        # Object → target [3]
        object_to_target = ObsTerm(
            func=mdp.deformable_object_to_target_position,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Last action [6]
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _reset_scene_with_deformable_diag(env, env_ids):
    """Wrapper around reset_scene_to_default that prints SoftBodyView.count once.

    If count < num_envs, the deformable reset is clamped to the available instances
    so that training can continue (with incorrect physics for the missing envs) and
    the full diagnostic is still printed.  This lets the environment survive the
    first reset so the user can at least see what shape the data has.
    """
    if not getattr(env, "_deformable_diag_done", False):
        for name, obj in env.scene.deformable_objects.items():
            count = obj.root_physx_view.count
            n_inst = obj.num_instances
            expected = env.num_envs
            ok = count == expected
            print(
                f"\n[DeformableDiag] *** scene['{name}'] ***\n"
                f"  SoftBodyView.count = {count}\n"
                f"  num_instances      = {n_inst}\n"
                f"  expected           = {expected}\n"
                f"  nodal_state shape  = {obj.data.default_nodal_state_w.shape}\n"
                f"  STATUS: {'OK' if ok else 'MISMATCH — PhysX only registered env_0!'}\n"
            )
        env._deformable_diag_done = True  # type: ignore[attr-defined]

    # --- safe reset: clamp env_ids to the registered count to avoid CUDA OOB ---
    safe_ids = None
    for obj in env.scene.deformable_objects.values():
        count = obj.root_physx_view.count
        if count < env.num_envs:
            safe_ids = env_ids[env_ids < count]
            break

    if safe_ids is not None:
        # Partial reset — only the registered bodies can be reset
        for rigid_object in env.scene.rigid_objects.values():
            default_root_state = rigid_object.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
            rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        for art in env.scene.articulations.values():
            default_root_state = art.data.default_root_state[env_ids].clone()
            default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
            art.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
            art.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
            art.write_joint_state_to_sim(
                art.data.default_joint_pos[env_ids].clone(),
                art.data.default_joint_vel[env_ids].clone(),
                env_ids=env_ids,
            )
        for obj in env.scene.deformable_objects.values():
            if safe_ids.numel() > 0:
                nodal_state = obj.data.default_nodal_state_w[safe_ids].clone()
                obj.write_nodal_state_to_sim(nodal_state, env_ids=safe_ids)
    else:
        # reset_scene_to_default handles rigid objects, articulations, AND deformable
        # nodal state (as of Isaac Lab 2.x) — no extra write needed.
        mdp.reset_scene_to_default(env, env_ids)


# ---------------------------------------------------------------------------
# Event config without object-position randomisation
# (DeformableObject resets via reset_scene_to_default; root-state API absent)
# ---------------------------------------------------------------------------

@configclass
class DeformablePushEventCfg:
    """Events for the deformable-cube push task."""

    # Reset every scene entity (including deformable nodal positions) to defaults
    reset_all = EventTerm(func=_reset_scene_with_deformable_diag, mode="reset")

    # Randomise target position only (target is still a rigid body)
    reset_target_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------

@configclass
class LynxDeformableCubePushEnvCfg(PushEnvCfg):
    """Push task where the object is a soft, jelly-like deformable cube.

    The deformable cube uses PhysX FEM simulation.  The Young's modulus is set
    very low (~5 000 Pa) and Poisson's ratio is near 0.5 to produce a nearly
    incompressible, jelly-like material response.
    """

    # Swap in deformable-specific reward / observation / event configs
    rewards: DeformablePushGoalRewardsCfg = DeformablePushGoalRewardsCfg()
    observations: DeformablePushObservationsCfg = DeformablePushObservationsCfg()
    events: DeformablePushEventCfg = DeformablePushEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- Solver / timing (same as rigid variant) --
        self.sim.dt = 1.0 / 30.0
        self.decimation = 6
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 2048
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 256

        # Disable PhysX Fabric (/physics/fabricEnabled = False).
        # With deformable FEM bodies outside the GridCloner env hierarchy, PhysX Fabric
        # allocates only 1 per-env GPU slot for the deformable object. This causes
        # DirectGpuHelper.cpp device-side asserts (CUDA OOB) in the Fabric GPU kernels
        # during sim.reset() when it tries to index slots 1..N-1.
        # Disabling Fabric avoids these GPU kernels entirely; PhysX still simulates all
        # N FEM bodies correctly via its own SoftBodyView API (which is Fabric-independent).
        # Trade-off: XFormPrimView pose ops fall back to USD (slightly slower), but
        # articulation / rigid-body / deformable state reads all go through the direct
        # PhysX API which is unaffected.
        self.sim.use_fabric = False

        # Must be False so GridCloner pre-creates all env xform prims BEFORE our spawner
        # runs.  The spawner reads the env xforms' world transforms to position each
        # deformable cube at the correct env origin (outside the env hierarchy).
        self.scene.replicate_physics = False

        # -- Robot --
        self.scene.robot = _make_lynx_robot_cfg()

        # -- Deformable jelly cube (same size/position as the rigid cube) --
        self.scene.object = DeformableObjectCfg(
            prim_path=f"{_SOFT_OBJECT_ROOT}/cube_.*/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.11)),
            spawn=MeshCuboidCfg(
                # Override the default spawner with our per-env version.
                # This ensures each environment gets its own PhysX-cooked FEM body
                # instead of a cloned USD prim (which PhysX cannot activate as FEM).
                func=_spawn_deformable_cuboid_per_env,
                size=(0.2, 0.2, 0.2),
                deformable_props=DeformableBodyPropertiesCfg(
                    # Disable self-collision for performance; enable if needed
                    self_collision=False,
                    # Solver iterations
                    solver_position_iteration_count=16,
                ),
                physics_material=DeformableBodyMaterialCfg(
                    # ~5 000 Pa → very soft jelly / gelatin
                    youngs_modulus=5000.0,
                    # Near 0.5 → nearly incompressible (water / jelly-like)
                    poissons_ratio=0.49,
                    dynamic_friction=0.5,
                    # Low elasticity damping → more jiggly / bouncy
                    elasticity_damping=0.002,
                    damping_scale=0.3,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.8, 0.4),   # Jelly green
                    opacity=0.75,
                ),
            ),
        )

        # -- Rigid target marker (visual only, no physics) --
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.1), rot=(1, 0, 0, 0)),
            spawn=sim_utils.CuboidCfg(
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
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
            ),
        )

        # -- Actions --
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.1745,
            clip={".*": (-1.0, 1.0)},
        )

        # -- EE frame transformer --
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
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        # -- Contact sensor --
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            update_period=0.0,
            history_length=3,
            debug_vis=False,
        )


@configclass
class LynxDeformableCubePushEnvCfg_PLAY(LynxDeformableCubePushEnvCfg):
    """Play configuration for the deformable-cube push task."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 1.0 / 60.0
        self.decimation = 12
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.3
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 64
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 32
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


# ---------------------------------------------------------------------------
# Custom env class: diagnoses SoftBodyView.count before any scene.reset crash
# ---------------------------------------------------------------------------

class LynxDeformablePushEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv subclass that prints deformable SoftBodyView.count before
    every scene.reset so the diagnostic is visible even when a crash follows.

    Also provides a safe-reset path: if ``SoftBodyView.count < num_envs`` (PhysX
    only registered env_0), the deformable reset is skipped to prevent the CUDA
    OOB that would otherwise crash the training run.
    """

    def _reset_idx(self, env_ids):
        # ------------------------------------------------------------------
        # 0.  Verify Fabric is disabled
        # ------------------------------------------------------------------
        if not getattr(self, "_fabric_checked", False):
            import carb
            s = carb.settings.get_settings()
            fabric_enabled = s.get("/physics/fabricEnabled")
            fabric_gpu_interop = s.get("/physics/fabricUseGPUInterop")
            print(
                f"\n[LynxDeformablePushEnv] /physics/fabricEnabled = {fabric_enabled}"
                f"  /physics/fabricUseGPUInterop = {fabric_gpu_interop}",
                flush=True,
            )
            self._fabric_checked = True

        # ------------------------------------------------------------------
        # 1.  Diagnostic: print SoftBodyView.count before touching anything
        # ------------------------------------------------------------------
        if not getattr(self, "_soft_count_logged", False):
            for name, obj in self.scene.deformable_objects.items():
                count = obj.root_physx_view.count
                n_inst = obj.num_instances
                ok = count == self.num_envs
                print(
                    f"\n[LynxDeformablePushEnv._reset_idx] '{name}'\n"
                    f"  SoftBodyView.count = {count}\n"
                    f"  num_instances      = {n_inst}\n"
                    f"  expected           = {self.num_envs}\n"
                    f"  STATUS: {'OK — all envs registered' if ok else 'MISMATCH — PhysX only registered env_0!'}\n",
                    flush=True,
                )
            self._soft_count_logged = True  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # 2.  If count < num_envs, PhysX would crash inside scene.reset when
        #     it tries to write physics state for envs 1..N-1 using GPU buffers
        #     sized for 1 body.  Skip normal scene.reset and do a manual reset
        #     of articulations and rigid objects only.
        # ------------------------------------------------------------------
        soft_count_ok = all(
            obj.root_physx_view.count == self.num_envs
            for obj in self.scene.deformable_objects.values()
        )

        if not soft_count_ok:
            # Manual reset: bypass scene.reset to avoid PhysX GPU OOB
            self.curriculum_manager.compute(env_ids=env_ids)
            for art in self.scene.articulations.values():
                art.reset(env_ids)
            for rigid in self.scene.rigid_objects.values():
                rigid.reset(env_ids)
            for sensor in self.scene.sensors.values():
                sensor.reset(env_ids)
            # Apply reset events manually
            if "reset" in self.event_manager.available_modes:
                env_step_count = self._sim_step_counter // self.cfg.decimation
                self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
            import torch
            self.extras["log"] = dict()
            for mgr_name in ("observation_manager", "action_manager", "reward_manager", "curriculum_manager"):
                mgr = getattr(self, mgr_name)
                info = mgr.reset(env_ids)
                self.extras["log"].update(info)
        else:
            super()._reset_idx(env_ids)
