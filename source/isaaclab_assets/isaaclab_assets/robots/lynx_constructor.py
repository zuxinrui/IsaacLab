# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from typing import List, Optional, Any

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, Gf

@configclass
class LynxUsdConstructorSpawnerCfg(sim_utils.RigidObjectSpawnerCfg):
    """Configuration for the Lynx USD constructor spawner."""
    func: Any = None # Will be set to LynxUsdConstructor.spawn
    robot_cfg: Optional[Any] = None
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    )

@configclass
class LynxRobotCfg(ArticulationCfg):
    """Configuration for the procedurally generated Lynx robot."""
    
    # Genotypes
    num_joints: int = 6
    genotype_tube: List[int] = [0, 1, 0, 1, 0]
    genotype_joints: int = 1 # [0-5], default: 1
    rotation_angles: List[float] = [180.0, 0.0, 0.0, -180.0, 0.0, 0.0]
    
    # Tube 1:
    l1_end_point_pos: tuple = (0.0, 0.0, 0.2)
    l1_end_point_theta: float = 0.0
    # Tube 2:
    l2_end_point_pos: tuple = (0.0, 0.0, 0.2805)
    l2_end_point_theta: float = 0.0
    # Tube 3:
    l3_end_point_pos: tuple = (0.0, 0.0, 0.2)
    l3_end_point_theta: float = 0.0
    # Tube 4:
    l4_end_point_pos: tuple = (0.0, 0.0, 0.2805)
    l4_end_point_theta: float = 0.0
    # Tube 5:
    l5_end_point_pos: tuple = (0.0, 0.0, 0.2)
    l5_end_point_theta: float = 0.0

    # Other tube parameters:
    bspline_dual_point_distance: float = 0.05
    mounting_length_start: float = 0.0359  # clamp_1217.stl
    mounting_length_end: float = 0.0359
    
    # Geometric Parameters
    tube_radiuses: List[float] = [0.0396] * 5
    # clamp_length: float = 0.051
    # B-Spline Tube Parameters
    bspline_num_segments: int = 50
    # Collision mode for procedurally generated geometry.
    # - "ee_only": only end-effector geometries are collidable (fastest for RL throughput)
    # - "full": all generated robot body geometries are collidable
    collision_mode: str = "full"
    # Paths to STLs (assuming they will be provided or are in a known location)
    clamp_stl: str = "source/isaaclab_assets/data/Robots/Lynx/models/clamp_0226.stl"
    ee_stl: Optional[str] = None
    
    # Physics/Actuation
    joint_position_limit_deg: float = 180.0
    joint_velocity_limit_rad_s: float = 0.3490658503988659  # 20 deg/s
    joint_acceleration_limit_rad_s2: float = 1.7453292519943295  # 100 deg/s^2

    # NOTE: ImplicitActuatorCfg is the authoritative source for stiffness/damping.
    # IsaacLab will write these values to the USD DriveAPI "angular" channel at runtime.
    actuators: dict[str, ImplicitActuatorCfg] = {
        "lynx_arm_mega": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-2]"],
            effort_limit_sim=130.0,
            velocity_limit_sim=0.3490658503988659,
            stiffness=800.0,
            damping=40.0,
        ),
        "lynx_arm_standard": ImplicitActuatorCfg(
            joint_names_expr=["joint_[3-4]"],
            effort_limit_sim=54.0,
            velocity_limit_sim=0.3490658503988659,
            stiffness=800.0,
            damping=40.0,
        ),
        "lynx_arm_lite": ImplicitActuatorCfg(
            joint_names_expr=["joint_[5-6]"],
            effort_limit_sim=19.0,
            velocity_limit_sim=0.3490658503988659,
            stiffness=800.0,
            damping=40.0,
        ),
    }

    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_1": 0.0,
            "joint_2": -0.45,
            "joint_3": 0.9,
            "joint_4": 0.0,
            "joint_5": 0.6,
            "joint_6": 0.0,
        },
    )

    def __post_init__(self):
        """Post-initialization to filter actuators and init_state based on num_joints."""
        # Filter actuators
        new_actuators = {}
        for name, actuator in self.actuators.items():
            # Filter joint_names_expr to only include joints that exist
            filtered_expr = []
            for expr in actuator.joint_names_expr:
                # This is a simple heuristic: if it's a range like joint_[1-2], 
                # we check if any joint in that range exists.
                # For simplicity, we just check if the joint names match the current num_joints.
                if "joint_" in expr:
                    import re
                    match = re.search(r"\[(\d+)-(\d+)\]", expr)
                    if match:
                        start, end = int(match.group(1)), int(match.group(2))
                        if start <= self.num_joints:
                            actual_end = min(end, self.num_joints)
                            if start == actual_end:
                                filtered_expr.append(f"joint_{start}")
                            else:
                                filtered_expr.append(f"joint_[{start}-{actual_end}]")
                    else:
                        # Single joint name or other expression
                        match_single = re.search(r"joint_(\d+)", expr)
                        if match_single:
                            if int(match_single.group(1)) <= self.num_joints:
                                filtered_expr.append(expr)
                        else:
                            filtered_expr.append(expr)
            
            if filtered_expr:
                actuator.joint_names_expr = filtered_expr
                new_actuators[name] = actuator
        self.actuators = new_actuators

        # Filter init_state
        if self.init_state.joint_pos:
            self.init_state.joint_pos = {
                k: v for k, v in self.init_state.joint_pos.items() 
                if "joint_" in k and int(k.split("_")[1]) <= self.num_joints
            }
    
    # Spawning logic
    spawn: LynxUsdConstructorSpawnerCfg = LynxUsdConstructorSpawnerCfg()

class LynxUsdConstructor:
    """Procedural constructor for the Lynx manipulator in USD."""

    @staticmethod
    def spawn(prim_path: str, cfg: LynxUsdConstructorSpawnerCfg, translation: Optional[tuple] = None, orientation: Optional[tuple] = None):
        """Spawns the robot at the specified prim path.
        
        Args:
            prim_path: The path to spawn the robot at.
            cfg: The configuration for the robot.
            translation: The translation of the robot.
            orientation: The orientation of the robot.
        """
        # If prim_path is empty or None, use a default path
        if not prim_path:
            prim_path = "/World/Robot"
        
        # Handle regex paths by taking the first matching path or a default if none match yet
        if "{ENV_REGEX_NS}" in prim_path:
            # This is a template path, we should probably let the caller handle it
            # but if we are here, it means the spawner was called with the template
            # In Isaac Lab, the InteractiveScene usually resolves this before calling spawn
            # If it didn't, we might be in a single-env setup or something similar
            prim_path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        
        # Fix for the "Ill-formed SdfPath" error which causes DefinePrim to fail with empty path
        if ".*" in prim_path:
            # Replace regex with a concrete index for spawning
            import re
            prim_path = re.sub(r"env_\.\*", "env_0", prim_path)
            
        print(f"[DEBUG] LynxUsdConstructor.spawn called with prim_path: '{prim_path}'")
            
        # Create the root Xform
        sim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
        
        # Apply ArticulationRootAPI to the root Xform.
        # The root Xform is NOT a rigid body - it is the articulation root anchor.
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        
        # Apply articulation properties if provided
        if hasattr(cfg, "articulation_props") and cfg.articulation_props is not None:
            sim_utils.define_articulation_root_properties(prim_path, cfg.articulation_props)
        
        # Build the robot chain
        if hasattr(cfg, "robot_cfg") and cfg.robot_cfg is not None:
            robot_cfg = cfg.robot_cfg
            if isinstance(robot_cfg, dict):
                # Convert dict back to LynxRobotCfg if needed, or just use it
                # The constructor expects LynxRobotCfg
                robot_cfg_obj = LynxRobotCfg()
                for k, v in robot_cfg.items():
                    if k != "prim_path":
                        setattr(robot_cfg_obj, k, v)
                # Manually set spawn properties from dict if they exist
                if "rigid_props" in robot_cfg:
                    robot_cfg_obj.spawn.rigid_props = robot_cfg["rigid_props"]
                if "articulation_props" in robot_cfg:
                    robot_cfg_obj.spawn.articulation_props = robot_cfg["articulation_props"]
                if "activate_contact_sensors" in robot_cfg:
                    robot_cfg_obj.spawn.activate_contact_sensors = robot_cfg["activate_contact_sensors"]
                robot_cfg = robot_cfg_obj
            constructor = LynxUsdConstructor(robot_cfg)
        elif isinstance(cfg, LynxRobotCfg):
            constructor = LynxUsdConstructor(cfg)
        else:
            # Fallback for other cases
            # If cfg is LynxUsdConstructorSpawnerCfg but doesn't have robot_cfg,
            # we might be in trouble, but let's try to use it as is or fallback to default
            constructor = LynxUsdConstructor(LynxRobotCfg())
        constructor._build_chain(prim_path)
        
        return root_prim

    def __init__(self, cfg: LynxRobotCfg):
        self.cfg = cfg

    def _quat_to_tuple(self, q: Gf.Quatf):
        """Helper to convert Gf.Quatf to tuple (w, x, y, z)."""
        return (q.GetReal(), *q.GetImaginary())

    def _apply_mass_properties(
        self,
        prim,
        mass: float,
        diagonal_inertia: tuple[float, float, float],
        center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
        principal_axes: Gf.Quatf | None = None,
    ):
        """Apply explicit mass, center of mass and inertia to a rigid body prim."""
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*diagonal_inertia))
        mass_api.CreatePrincipalAxesAttr().Set(principal_axes if principal_axes is not None else Gf.Quatf(1, 0, 0, 0))

    def _cylinder_inertia(self, mass: float, radius: float, length: float, axis: str = "z") -> tuple[float, float, float]:
        """Approximate solid-cylinder inertia around its center of mass."""
        i_axial = 0.5 * mass * radius * radius
        i_radial = (1.0 / 12.0) * mass * (3.0 * radius * radius + length * length)
        if axis == "x":
            return (i_axial, i_radial, i_radial)
        if axis == "y":
            return (i_radial, i_axial, i_radial)
        return (i_radial, i_radial, i_axial)

    def _sphere_inertia(self, mass: float, radius: float) -> tuple[float, float, float]:
        """Approximate solid-sphere inertia around its center of mass."""
        i = 0.4 * mass * radius * radius
        return (i, i, i)

    def _box_inertia(self, mass: float, size: tuple[float, float, float]) -> tuple[float, float, float]:
        """Approximate cuboid inertia around its center of mass."""
        sx, sy, sz = size
        return (
            (1.0 / 12.0) * mass * (sy * sy + sz * sz),
            (1.0 / 12.0) * mass * (sx * sx + sz * sz),
            (1.0 / 12.0) * mass * (sx * sx + sy * sy),
        )

    def _parallel_axis(self, inertia: tuple[float, float, float], mass: float, offset: Gf.Vec3f) -> tuple[float, float, float]:
        """Shift diagonal inertia with the parallel-axis theorem."""
        ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
        return (
            inertia[0] + mass * (oy * oy + oz * oz),
            inertia[1] + mass * (ox * ox + oz * oz),
            inertia[2] + mass * (ox * ox + oy * oy),
        )

    def _combine_bodies(
        self,
        components: list[tuple[float, tuple[float, float, float], Gf.Vec3f]],
    ) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
        """Combine mass properties of multiple aligned primitive approximations."""
        total_mass = sum(component[0] for component in components)
        if total_mass <= 0.0:
            return 0.0, (1e-6, 1e-6, 1e-6), (0.0, 0.0, 0.0)

        com = Gf.Vec3f(0.0, 0.0, 0.0)
        for mass, _, offset in components:
            com += offset * (mass / total_mass)

        total_inertia = [0.0, 0.0, 0.0]
        for mass, inertia, offset in components:
            shifted = self._parallel_axis(inertia, mass, offset - com)
            for idx in range(3):
                total_inertia[idx] += shifted[idx]

        return total_mass, (float(total_inertia[0]), float(total_inertia[1]), float(total_inertia[2])), (float(com[0]), float(com[1]), float(com[2]))

    def _set_joint_limits(self, joint, lower_deg: float, upper_deg: float):
        """Apply conservative rotational joint limits in degrees."""
        joint.CreateLowerLimitAttr(lower_deg)
        joint.CreateUpperLimitAttr(upper_deg)

    def _set_joint_drive_and_damping(self, joint_prim, joint_index: int):
        """Apply drive plus passive joint properties.
        
        IMPORTANT: For revolute joints in USD/PhysX, the DriveAPI instance name must be
        "angular" (not "revolute"). IsaacLab's ImplicitActuatorCfg will override the
        stiffness/damping values at runtime, but we author the drive schema here so
        PhysX recognises the drive channel exists.
        """
        # Select actuator config based on joint index
        if joint_index < 2:
            actuator_cfg = self.cfg.actuators["lynx_arm_mega"]
        elif joint_index < 4:
            actuator_cfg = self.cfg.actuators["lynx_arm_standard"]
        else:
            actuator_cfg = self.cfg.actuators["lynx_arm_lite"]

        # FIX: Use "angular" as the drive token for revolute joints.
        # "revolute" is NOT a valid PhysX drive instance name and will be silently ignored.
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        # Do NOT call CreateTypeAttr("position") - that attribute does not exist on DriveAPI
        # and calling it with "position" creates a spurious attribute that confuses PhysX.
        # The drive mode (position vs velocity) is controlled by whether you set a
        # position target or velocity target at runtime via IsaacLab's actuator system.

        if drive.GetStiffnessAttr().HasAuthoredValueOpinion():
            drive.GetStiffnessAttr().Set(actuator_cfg.stiffness)
        else:
            drive.CreateStiffnessAttr(actuator_cfg.stiffness)
        if drive.GetDampingAttr().HasAuthoredValueOpinion():
            drive.GetDampingAttr().Set(actuator_cfg.damping)
        else:
            drive.CreateDampingAttr(actuator_cfg.damping)
        if actuator_cfg.effort_limit_sim is not None:
            if drive.GetMaxForceAttr().HasAuthoredValueOpinion():
                drive.GetMaxForceAttr().Set(actuator_cfg.effort_limit_sim)
            else:
                drive.CreateMaxForceAttr(actuator_cfg.effort_limit_sim)

        joint_physx_api = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        armature_value = 0.4 if joint_index < 3 else 0.12
        if joint_physx_api.GetArmatureAttr().HasAuthoredValueOpinion():
            joint_physx_api.GetArmatureAttr().Set(armature_value)
        else:
            joint_physx_api.CreateArmatureAttr(armature_value)
        if joint_physx_api.GetJointFrictionAttr().HasAuthoredValueOpinion():
            joint_physx_api.GetJointFrictionAttr().Set(0.0)
        else:
            joint_physx_api.CreateJointFrictionAttr(0.0)
        if actuator_cfg.velocity_limit_sim is not None:
            if joint_physx_api.GetMaxJointVelocityAttr().HasAuthoredValueOpinion():
                joint_physx_api.GetMaxJointVelocityAttr().Set(self.cfg.joint_velocity_limit_rad_s)
            else:
                joint_physx_api.CreateMaxJointVelocityAttr(self.cfg.joint_velocity_limit_rad_s)

        # Set acceleration limit when supported by the installed PhysX schema.
        # Target: 100 deg/s^2 == 1.7453292519943295 rad/s^2.
        get_acc_attr = getattr(joint_physx_api, "GetMaxJointAccelerationAttr", None)
        create_acc_attr = getattr(joint_physx_api, "CreateMaxJointAccelerationAttr", None)
        if callable(get_acc_attr) and callable(create_acc_attr):
            acc_attr = get_acc_attr()
            if hasattr(acc_attr, "HasAuthoredValueOpinion") and hasattr(acc_attr, "Set"):
                if acc_attr.HasAuthoredValueOpinion():
                    acc_attr.Set(self.cfg.joint_acceleration_limit_rad_s2)
                else:
                    create_acc_attr(self.cfg.joint_acceleration_limit_rad_s2)

        return drive, joint_physx_api

    def _build_chain(self, root_path: str):
        """Internal logic to assemble the links and joints.
        
        Coordinate convention
        ---------------------
        All link prims are created at the world origin (no authored translation/rotation).
        Joint frames are expressed in the *parent body's local frame* via localPos0/localRot0,
        and in the *child body's local frame* via localPos1/localRot1.

        Because every link prim sits at the world origin, the child-body local frame IS the
        world frame, so localPos1 = (0,0,0) and localRot1 = identity is always correct.

        curr_pos  : position of the next joint attachment point expressed in the current
                    parent body's local frame (which equals the world frame since all
                    link prims are at the origin).
        curr_quat : orientation of the joint frame expressed in the current parent body's
                    local frame.  For the first joint this is identity; after each inline
                    joint it becomes the child's joint-frame orientation so that the next
                    joint's axis is expressed correctly.
        """
        stage = sim_utils.get_current_stage()
        non_colliding_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
        collidable_cfg = sim_utils.CollisionPropertiesCfg()
        body_collision_cfg = collidable_cfg if self.cfg.collision_mode == "full" else non_colliding_cfg

        # Pre-define materials to avoid hitting the 64K PhysX material limit in large batched scenes.
        # By using the same config object, Isaac Lab's spawner will reuse the material prim if it's already created.
        black_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
        grey_material = sim_utils.spawners.materials.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.2, 0.2),
            metallic=1.0,
            roughness=0.2,
        )
        dark_grey_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1))
        ee_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15))
        red_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))
        shared_physics_material = sim_utils.spawners.materials.RigidBodyMaterialCfg(
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.0,
        )
        shared_physics_material_path = "/World/PhysicsMaterials/lynx_shared"

        # Pro Arm 900 Physical Parameters
        pro_arm_900_params = {
            "base": {
                "mass": 0.1,
                "com": (0, 0, 0.012005),
                "inertia": (3.009792e-04, 3.0097905e-04, 0.522597e-03)
            },
            "links": [
                {
                    "mass": 4.5,
                    "com": (0.007687, 0.104424, 0),
                    "inertia": (2.053456e-03, 3.997604e-04, 2.077309e-03)
                },
                {
                    "mass": 6.0,
                    "com": (0.092725, 0.145334, 0.000038),
                    "inertia": (5.3897178e-03, 1.3406068e-03, 5.9820744e-03)
                },
                {
                    "mass": 1.52,
                    "com": (-0.021209, -0.000108, 0.009662),
                    "inertia": (1.032334e-03, 6.154116e-04, 1.076372e-03)
                },
                {
                    "mass": 1.52,
                    "com": (0.003206, 0.206689, 0.000108),
                    "inertia": (2.5807422e-03, 1.0628242e-03, 2.9577733e-03)
                },
                {
                    "mass": 1.1,
                    "com": (0.045224, -0.003419, 0.000110),
                    "inertia": (9.4404410e-04, 5.9232063e-04, 9.9105550e-04)
                },
                {
                    "mass": 1.1, # Assuming link 6 matches link 5 as per previous pattern
                    "com": (0.045224, -0.003419, 0.000110),
                    "inertia": (9.4404410e-04, 5.9232063e-04, 9.9105550e-04)
                }
            ]
        }
        
        # ------------------------------------------------------------------ #
        # 1. Base Link
        # ------------------------------------------------------------------ #
        base_path = f"{root_path}/base"
        # FIX: Place base at world origin (no translation offset).
        # Previously base had translation=(0,0,0.02) but the fixed joint anchors
        # were both at (0,0,0), creating an inconsistency that confused the solver.
        sim_utils.create_prim(base_path, prim_type="Xform", translation=(0.0, 0.0, 0.0))
        base_prim = stage.GetPrimAtPath(base_path)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        # Apply rigid body properties if provided
        if hasattr(self.cfg.spawn, "rigid_props") and self.cfg.spawn.rigid_props is not None:
            sim_utils.define_rigid_body_properties(base_path, self.cfg.spawn.rigid_props)
        
        # Apply contact reporter if requested
        if getattr(self.cfg.spawn, "activate_contact_sensors", False):
            if not base_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(base_prim)

        # FIX: Fix the base to the WORLD by leaving body0 empty (no targets).
        # Pointing body0 at a plain Xform (non-rigid-body) is not valid in PhysX
        # and can cause the articulation tree to be parsed incorrectly.
        # An empty body0 means "fixed to the world frame".
        fixed_base_joint = UsdPhysics.FixedJoint.Define(stage, f"{base_path}/root_fixed_joint")
        # body0 intentionally left empty → fixed to world
        fixed_base_joint.CreateBody1Rel().SetTargets([base_path])
        # The joint frame in world space: base sits at world origin, so both anchors are zero.
        fixed_base_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_base_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fixed_base_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_base_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        
        # Base dimensions from LynxRobotics/lynx/robots/lynx_manipulator/constructor.py
        base_length1 = 0.001
        base_radius1 = 0.08
        base_length2 = 0.017 + 0.001
        base_radius2 = 0.08
        base_height = base_length1 + base_length2
        base_radius = max(base_radius1, base_radius2)
        
        self._apply_mass_properties(
            base_prim,
            mass=pro_arm_900_params["base"]["mass"],
            diagonal_inertia=pro_arm_900_params["base"]["inertia"],
            center_of_mass=pro_arm_900_params["base"]["com"],
        )
        
        # Use collision-only geometry on the same prim names as visuals to avoid duplicate flashing meshes.
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{base_path}/visual_1",
            sim_utils.spawners.MeshCylinderCfg(
                radius=base_radius1,
                height=base_length1,
                collision_props=non_colliding_cfg,
                physics_material_path=shared_physics_material_path,
                physics_material=shared_physics_material,
                visual_material=black_material,
            ),
            translation=(0, 0, base_length1 / 2),
        )
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{base_path}/visual_2",
            sim_utils.spawners.MeshCylinderCfg(
                radius=base_radius2,
                height=base_length2,
                collision_props=non_colliding_cfg,
                physics_material_path=shared_physics_material_path,
                physics_material=shared_physics_material,
                visual_material=black_material,
            ),
            translation=(0, 0, base_length1 + base_length2 / 2),
        )

        curr_parent_path = base_path
        # Attachment point for base is at (0, 0, base_height) in the base body's local frame.
        # Since base is at world origin, this is also the world-space position.
        curr_pos = Gf.Vec3f(0, 0, base_length1 + base_length2)
        # The joint frame orientation starts as identity (Z-axis is the rotation axis for joint_1).
        curr_quat = Gf.Quatf(1, 0, 0, 0)

        num_joints = self.cfg.num_joints
        
        # Determine joint types
        joints_inline = [True, True, True, True, True]
        joints_inline[self.cfg.genotype_joints:] = [False] * (len(joints_inline) - self.cfg.genotype_joints)
        # joint_types[0] is for joint 2. Joint 1 is always inline.
        joint_types = ["inline"] + ["inline" if inline else "orthogonal" for inline in joints_inline]

        # Joint parameters from LynxRobotics
        joint_params = [
            # Joint 1
            {"l0": 0.036, "r0": 0.0425, "l1": 0.128, "r1": 0.062, "l2": 0.013, "r2": 0.062},
            # Joint 2
            {"l0": 0.036, "r0": 0.0425, "l1": 0.128, "r1": 0.062, "l2": 0.013, "r2": 0.062},
            # Joint 3
            {"l0": 0.029, "r0": 0.031, "l1": 0.092, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 4
            {"l0": 0.029, "r0": 0.031, "l1": 0.092, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 5
            {"l0": 0.029, "r0": 0.031, "l1": 0.096, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 6
            {"l0": 0.029, "r0": 0.031, "l1": 0.096, "r1": 0.042, "l2": 0.008, "r2": 0.042},
        ]

        joint_limits_deg = [
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
            (-self.cfg.joint_position_limit_deg, self.cfg.joint_position_limit_deg),
        ]
        
        # Legacy target masses for procedural approximations (tubes)
        tube_target_masses = [0.0, 0.35, 0.0, 0.35, 0.0]

        genotype_tube = self.cfg.genotype_tube[:num_joints-1]
        tube_idx = 0
        # final_link_name = f"link_{num_joints}"
        for i in range(num_joints):
            # 1. Check if we need to insert a tube BEFORE this joint (except joint 1)
            if i > 0 and tube_idx < len(genotype_tube) and genotype_tube[tube_idx] == 1:
                tube_name = f"tube_{tube_idx+1}"
                tube_path = f"{root_path}/{tube_name}"
                sim_utils.create_prim(tube_path, prim_type="Xform")
                tube_prim = stage.GetPrimAtPath(tube_path)
                UsdPhysics.RigidBodyAPI.Apply(tube_prim)
                # Apply contact reporter if requested
                if getattr(self.cfg.spawn, "activate_contact_sensors", False):
                    if not tube_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                        PhysxSchema.PhysxContactReportAPI.Apply(tube_prim)
                # Add explicit mass/inertia to the tube after geometry is known
                
                tube_radius = self.cfg.tube_radiuses[tube_idx]
                
                # B-Spline Tube Logic
                num_segs = self.cfg.bspline_num_segments
                
                # Get tube-specific parameters
                if tube_idx == 0:
                    end_pos = Gf.Vec3f(*self.cfg.l1_end_point_pos)
                    theta = np.radians(self.cfg.l1_end_point_theta)
                elif tube_idx == 1:
                    end_pos = Gf.Vec3f(*self.cfg.l2_end_point_pos)
                    theta = np.radians(self.cfg.l2_end_point_theta)
                elif tube_idx == 2:
                    end_pos = Gf.Vec3f(*self.cfg.l3_end_point_pos)
                    theta = np.radians(self.cfg.l3_end_point_theta)
                elif tube_idx == 3:
                    end_pos = Gf.Vec3f(*self.cfg.l4_end_point_pos)
                    theta = np.radians(self.cfg.l4_end_point_theta)
                else:
                    end_pos = Gf.Vec3f(*self.cfg.l5_end_point_pos)
                    theta = np.radians(self.cfg.l5_end_point_theta)

                d = self.cfg.bspline_dual_point_distance
                mounting_l_start = self.cfg.mounting_length_start
                mounting_l_end = self.cfg.mounting_length_end
                
                # Calculate control points (clamped cubic)
                p0 = Gf.Vec3f(0, 0, 0)
                p1 = p0 + Gf.Vec3f(0, 0, d)
                
                # End direction from theta (rotate +Z toward +Y in YZ plane)
                s, c = np.sin(theta), np.cos(theta)
                t_dir = Gf.Vec3f(0, s, c).GetNormalized()
                
                # Offset end point by mounting lengths only (no joint radius offset)
                # p3 should also canlculate the start:
                z_hat = Gf.Vec3f(0, 0, 1)
                
                p3 = end_pos - z_hat * mounting_l_start - t_dir * mounting_l_end
                p2 = p3 - t_dir * d
                
                # Sample points along the curve (Cubic Bezier)
                curve_points = []
                for j in range(num_segs + 1):
                    t = j / num_segs
                    pt = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                    curve_points.append(pt)
                
                # Calculate start direction for mounting offset
                start_direction = (curve_points[1] - curve_points[0]).GetNormalized()
                # first segment will be extended backwards along the curve's initial tangent
                start_offset_vec = start_direction * (mounting_l_start - (curve_points[1] - curve_points[0]).GetLength())

                # Create segments
                last_seg_quat = Gf.Quatf(1, 0, 0, 0)
                actual_end_pos = p3
                for j in range(num_segs):
                    p_start = curve_points[j]
                    p_end = curve_points[j+1]
                    direction = (p_end - p_start).GetNormalized()
                    auto_length = (p_end - p_start).GetLength()
                    
                    is_first = (j == 0)
                    is_last = (j == num_segs - 1)
                    
                    if is_first:
                        seg_len = mounting_l_start
                        seg_center = p_start + direction * (seg_len / 2)
                    elif is_last:
                        seg_len = mounting_l_end
                        shifted_start = p_start + start_offset_vec
                        seg_center = shifted_start + direction * (seg_len / 2)
                    else:
                        seg_len = auto_length
                        shifted_start = p_start + start_offset_vec
                        shifted_end = p_end + start_offset_vec
                        seg_center = (shifted_start + shifted_end) * 0.5
                    
                    # Rotation from Z to direction
                    z_axis = Gf.Vec3f(0, 0, 1)
                    dot = z_axis * direction
                    if abs(dot) > 0.9999:
                        seg_quat = Gf.Quatf(1, 0, 0, 0) if dot > 0 else Gf.Quatf(0, 1, 0, 0)
                    else:
                        axis = (z_axis ^ direction).GetNormalized()
                        angle = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
                        seg_quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(axis), angle).GetQuat())
                    
                    last_seg_quat = seg_quat
                    if is_last:
                        shifted_start = p_start + start_offset_vec
                        actual_end_pos = shifted_start + direction * seg_len

                    sim_utils.spawners.meshes.spawn_mesh_cylinder(
                        f"{tube_path}/visual_{j}",
                        sim_utils.spawners.MeshCylinderCfg(
                            radius=tube_radius,
                            height=seg_len,
                            collision_props=body_collision_cfg,
                            physics_material_path=shared_physics_material_path,
                            physics_material=shared_physics_material,
                            visual_material=dark_grey_material,
                        ),
                        translation=tuple(seg_center),
                        orientation=self._quat_to_tuple(seg_quat)
                    )
                
                self._create_fixed_joint(f"{tube_path}/fixed_joint", curr_parent_path, tube_path, curr_pos, curr_quat)
                approx_length = max((actual_end_pos - Gf.Vec3f(0.0, 0.0, 0.0)).GetLength(), 0.08)
                tube_mass = tube_target_masses[tube_idx]
                self._apply_mass_properties(
                    tube_prim,
                    mass=tube_mass,
                    diagonal_inertia=self._cylinder_inertia(tube_mass, tube_radius, approx_length, axis="z"),
                    center_of_mass=(0.0, 0.0, approx_length / 2.0),
                )
                curr_parent_path = tube_path
                curr_pos = actual_end_pos
                curr_quat = last_seg_quat
                
                tube_idx += 1
            elif i > 0:
                tube_idx += 1

            # 2. Spawn the Joint and its Link
            joint_name = f"joint_{i+1}"
            link_name = f"link_{i+1}"
            link_path = f"{root_path}/{link_name}"
            # FIX: All link prims are created at the world origin (no authored translation).
            # The joint's localPos0/localRot0 encodes where the joint sits in the parent body's
            # local frame.  Since all bodies are at the world origin, "parent local frame" ==
            # "world frame", so curr_pos/curr_quat (accumulated in world space) are correct
            # values for localPos0/localRot0.
            sim_utils.create_prim(link_path, prim_type="Xform")
            link_prim = stage.GetPrimAtPath(link_path)
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            # Apply contact reporter if requested
            if getattr(self.cfg.spawn, "activate_contact_sensors", False):
                if not link_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
            # Apply rigid body properties if provided
            if hasattr(self.cfg.spawn, "rigid_props") and self.cfg.spawn.rigid_props is not None:
                sim_utils.define_rigid_body_properties(link_path, self.cfg.spawn.rigid_props)
            p = joint_params[i]
            j_type = joint_types[i]
            angle_rad = np.deg2rad(self.cfg.rotation_angles[i])
            
            if j_type == "inline":
                # JointInline logic from joints.py
                # rx rotates the joint frame so that the joint's Z-axis (rotation axis) aligns
                # with the physical rotation axis of the inline joint.
                rx = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat())
                twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(angle_rad)).GetQuat())
                new_relative_quat = rx * twist
                
                # These positions are in the child body's local frame (= world frame since
                # the child prim is at the world origin).
                joint_pos = Gf.Vec3f(0, 0, p["l0"] + p["l1"] / 2)
                cylinder2_pos = joint_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["r1"] + p["l2"] / 2))
                cylinder2_add_pos = joint_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["r1"] / 2))

                # Fixed and moving geometry should each have a single visible collidable mesh.
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/fixed_visual",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r0"], height=p["l0"],
                        collision_props=body_collision_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=grey_material
                    ),
                    translation=(0, 0, p["l0"] / 2)
                )
                
                # Revolute Joint
                joint_path = f"{link_path}/{joint_name}"
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.CreateAxisAttr("Z")
                joint.CreateBody0Rel().SetTargets([curr_parent_path])
                joint.CreateBody1Rel().SetTargets([link_path])
                
                # Flip the joint direction by rotating the joint frame 180 degrees around X.
                # This reverses the positive rotation direction.
                flip_quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat())

                # localPos0/localRot0: joint frame expressed in parent body's local frame.
                # curr_pos is the attachment point in the parent body's local frame.
                # curr_quat is the joint frame orientation in the parent body's local frame.
                joint.CreateLocalPos0Attr().Set(curr_pos)
                joint.CreateLocalRot0Attr().Set(curr_quat * flip_quat)
                # localPos1/localRot1: joint frame expressed in child body's local frame.
                # Child body is at world origin with identity orientation, so the joint
                # attachment in the child frame is at the origin with identity rotation.
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot1Attr().Set(flip_quat)
                joint.CreateCollisionEnabledAttr(False)
                self._set_joint_limits(joint, *joint_limits_deg[i])
                sim_utils.safe_set_attribute_on_usd_prim(stage.GetPrimAtPath(joint_path), "physics:jointEnabled", True, camel_case=True)
                
                # Moving part visuals
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_main",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r1"], height=p["l1"],
                        collision_props=body_collision_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(joint_pos)
                )
                
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_attach_add",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r2"], height=p["r1"],
                        collision_props=non_colliding_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(cylinder2_add_pos),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )

                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_attach",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r2"], height=p["l2"],
                        collision_props=non_colliding_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(cylinder2_pos),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )
                
                # Update curr for next iteration.
                # The next joint's attachment point is at the tip of cylinder2, expressed in
                # the current child body's local frame (= world frame).
                curr_parent_path = link_path
                curr_pos = cylinder2_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["l2"] / 2))
                # FIX: Propagate the joint frame orientation so the next joint's axis is
                # expressed correctly in the next parent body's local frame.
                # For inline joints the child body's "output" frame has the same orientation
                # as new_relative_quat (the joint frame rotated by rx*twist).
                curr_quat = new_relative_quat

                # Apply Pro Arm 900 physical parameters
                link_mass = pro_arm_900_params["links"][i]["mass"]
                link_inertia = pro_arm_900_params["links"][i]["inertia"]
                link_com = pro_arm_900_params["links"][i]["com"]
                
                self._apply_mass_properties(link_prim, link_mass, link_inertia, link_com)

            else:
                # JointOrthogonal logic from joints.py
                l2_orig, r2_orig = p["l0"], p["r0"]
                l1, r1 = p["l1"], p["r1"]
                l0_orig, r0_orig = p["l2"], p["r2"]
                
                rx = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat())
                twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(angle_rad)).GetQuat())
                new_relative_quat = twist * rx
                # MuJoCo JointOrthogonal hinge axis is rotated by `new_relative_quat`:
                #     axis_child = new_relative_quat * [0, 0, -1].
                # USD revolute uses principal-axis token (X/Y/Z), so we encode this by rotating
                # both joint frames with the same axis-frame rotation:
                #   - localRot1: axis frame in child body local
                #   - localRot0: same frame expressed in parent "accumulated" frame
                orth_axis_frame_parent = curr_quat * new_relative_quat
                
                cylinder0_pos = Gf.Vec3f(0, 0, l0_orig / 2)
                cylinder0_add_pos = Gf.Vec3f(0, 0, l0_orig + r1 / 2)
                joint_pos_rel = Gf.Vec3f(0, 0, l0_orig + r1)
                cylinder2_pos_rel = joint_pos_rel + new_relative_quat.Transform(
                    Gf.Vec3f(0, 0, l1 / 2 + l2_orig / 2)
                )

                # Parent-frame joint axis frame and parent-side fixed-geometry poses.
                fixed_cylinder0_pos_parent = curr_pos + curr_quat.Transform(cylinder0_pos - joint_pos_rel)
                fixed_cylinder0_add_pos_parent = curr_pos + curr_quat.Transform(cylinder0_add_pos - joint_pos_rel)
                # Uniform module shift in world-up. Applying exactly the same shift to fixed,
                # joint anchors, moving part, and next curr_pos preserves relative geometry.
                # IMPORTANT: avoid cumulative over-shift only for truly consecutive
                # orthogonal joints. If there is a tube between joints, treat as non-consecutive.
                prev_has_tube = i > 0 and (i - 1) < len(self.cfg.genotype_tube) and bool(self.cfg.genotype_tube[i - 1])
                prev_is_orthogonal = i > 0 and joint_types[i - 1] == "orthogonal" and not prev_has_tube
                orth_module_shift = Gf.Vec3f(0, -(l0_orig + r1), 0.0) if not prev_has_tube else curr_quat.Transform(Gf.Vec3f(0, 0, l0_orig + r1))

                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{curr_parent_path}/{joint_name}_fixed_visual_0",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r0_orig, height=l0_orig,
                        collision_props=non_colliding_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(fixed_cylinder0_pos_parent + orth_module_shift),
                    orientation=self._quat_to_tuple(curr_quat)
                )
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{curr_parent_path}/{joint_name}_fixed_visual_add",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r0_orig, height=r1,
                        collision_props=non_colliding_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(fixed_cylinder0_add_pos_parent + orth_module_shift),
                    orientation=self._quat_to_tuple(curr_quat)
                )
                
                # Joint shell (fixed)
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{curr_parent_path}/{joint_name}_fixed_visual_shell",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r1, height=l1,
                        collision_props=body_collision_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=black_material
                    ),
                    translation=tuple(curr_pos + orth_module_shift),
                    orientation=self._quat_to_tuple(orth_axis_frame_parent)
                )
                
                # Revolute Joint
                joint_path = f"{link_path}/{joint_name}"
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.CreateAxisAttr("Z") 
                joint.CreateBody0Rel().SetTargets([curr_parent_path])
                joint.CreateBody1Rel().SetTargets([link_path])
                
                # Flip the joint direction by rotating the joint frame 180 degrees around X or Y.
                # This reverses the positive rotation direction.
                flip_quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat())
                
                joint.CreateLocalPos0Attr().Set(curr_pos + orth_module_shift)
                joint.CreateLocalRot0Attr().Set(orth_axis_frame_parent * flip_quat)
                # Place the hinge at the center of fixed_visual_shell.
                joint.CreateLocalPos1Attr().Set(joint_pos_rel + orth_module_shift)
                joint.CreateLocalRot1Attr().Set(new_relative_quat * flip_quat)
                joint.CreateCollisionEnabledAttr(False)
                self._set_joint_limits(joint, *joint_limits_deg[i])
                sim_utils.safe_set_attribute_on_usd_prim(stage.GetPrimAtPath(joint_path), "physics:jointEnabled", True, camel_case=True)
                
                # Moving part visual
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r2_orig, height=l2_orig,
                        collision_props=body_collision_cfg,
                        physics_material_path=shared_physics_material_path,
                        physics_material=shared_physics_material,
                        visual_material=grey_material
                    ),
                    translation=tuple(cylinder2_pos_rel + orth_module_shift),
                    # Keep the moving visual aligned with the same joint frame used for the
                    # orthogonal revolute axis, matching MuJoCo behavior.
                    orientation=self._quat_to_tuple(new_relative_quat)
                )
                
                # Update curr for next iteration.
                curr_parent_path = link_path
                curr_pos = cylinder2_pos_rel + orth_module_shift + new_relative_quat.Transform(
                    Gf.Vec3f(0, 0, l2_orig / 2)
                )
                # Propagate the orthogonal joint-frame orientation so next joints are expressed
                # in the same accumulated frame convention as MuJoCo.
                curr_quat = new_relative_quat

                # Apply Pro Arm 900 physical parameters
                link_mass = pro_arm_900_params["links"][i]["mass"]
                link_inertia = pro_arm_900_params["links"][i]["inertia"]
                link_com = pro_arm_900_params["links"][i]["com"]
                
                self._apply_mass_properties(link_prim, link_mass, link_inertia, link_com)

            # Apply active/passive joint properties
            self._set_joint_drive_and_damping(stage.GetPrimAtPath(f"{link_path}/{joint_name}"), i)

        # 3. End Effector
        ee_cyl_length = 0.07
        ee_cyl_radius = 0.01
        ee_cyl_path = f"{root_path}/ee_cylinder"
        sim_utils.create_prim(ee_cyl_path, prim_type="Xform")
        ee_cyl_prim = stage.GetPrimAtPath(ee_cyl_path)
        UsdPhysics.RigidBodyAPI.Apply(ee_cyl_prim)
        # Apply contact reporter if requested
        if getattr(self.cfg.spawn, "activate_contact_sensors", False):
            if not ee_cyl_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(ee_cyl_prim)
        ee_cyl_mass = 0.35
        self._apply_mass_properties(
            ee_cyl_prim,
            mass=ee_cyl_mass,
            diagonal_inertia=self._cylinder_inertia(ee_cyl_mass, ee_cyl_radius, ee_cyl_length, axis="z"),
            center_of_mass=(0.0, 0.0, ee_cyl_length / 2.0),
        )
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{ee_cyl_path}/visual",
            sim_utils.spawners.MeshCylinderCfg(
                radius=ee_cyl_radius,
                height=ee_cyl_length,
                collision_props=collidable_cfg,
                physics_material_path=shared_physics_material_path,
                physics_material=shared_physics_material,
                visual_material=ee_material
            ),
            translation=(0, 0, ee_cyl_length / 2)
        )
        self._create_fixed_joint(f"{ee_cyl_path}/fixed_joint", curr_parent_path, ee_cyl_path, curr_pos, curr_quat)
        
        # Final EE
        ee_path = f"{root_path}/ee"
        sim_utils.create_prim(ee_path, prim_type="Xform")
        ee_prim = stage.GetPrimAtPath(ee_path)
        UsdPhysics.RigidBodyAPI.Apply(ee_prim)
        # Apply contact reporter if requested
        if getattr(self.cfg.spawn, "activate_contact_sensors", False):
            if not ee_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(ee_prim)
        ee_mass = 0.01
        ee_radius = 0.01
        self._apply_mass_properties(
            ee_prim,
            mass=ee_mass,
            diagonal_inertia=self._sphere_inertia(ee_mass, ee_radius),
            center_of_mass=(0.0, 0.0, 0.0),
        )
        sim_utils.spawners.meshes.spawn_mesh_sphere(
            f"{ee_path}/visual",
            sim_utils.spawners.MeshSphereCfg(
                radius=ee_radius,
                collision_props=non_colliding_cfg,
                physics_material_path=shared_physics_material_path,
                physics_material=shared_physics_material,
                visual_material=red_material
            )
        )
        self._create_fixed_joint(f"{ee_path}/fixed_joint", ee_cyl_path, ee_path, Gf.Vec3f(0, 0, ee_cyl_length), Gf.Quatf(1, 0, 0, 0))

    def _create_fixed_joint(self, path, body0, body1, pos, quat):
        stage = sim_utils.get_current_stage()
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([body0])
        joint.CreateBody1Rel().SetTargets([body1])
        joint.CreateLocalPos0Attr().Set(pos)
        joint.CreateLocalRot0Attr().Set(quat)
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

# Register the spawn function
LynxUsdConstructorSpawnerCfg.func = LynxUsdConstructor.spawn

# Expose a module-level alias for Hydra/OmegaConf round-trip serialization.
# When callables are serialized through `callable_to_string`, static methods are
# represented as `<module>:<function_name>` (without class qualifier), so we
# provide `spawn` on the module to make deserialization resolve correctly.
spawn = LynxUsdConstructor.spawn
