# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from pxr import UsdPhysics, Gf, PhysxSchema

from .lynx_constructor import LynxUsdConstructor, LynxRobotCfg, LynxUsdConstructorSpawnerCfg

@configclass
class LynxBallInCupRobotCfg(LynxRobotCfg):
    """Configuration for the Lynx robot with a ball and a cup."""
    cup_radius: float = 0.04
    cup_height: float = 0.08
    ball_radius: float = 0.02
    string_length: float = 0.4
    string_radius: float = 0.0005

class LynxBallInCupConstructor(LynxUsdConstructor):
    """Constructor for the Lynx robot with a ball and a cup."""

    def __init__(self, cfg: LynxBallInCupRobotCfg):
        if isinstance(cfg, dict):
            from .lynx_constructor import LynxRobotCfg
            # Create a proper config object from the dict
            new_cfg = LynxBallInCupRobotCfg()
            for key, value in cfg.items():
                if hasattr(new_cfg, key):
                    setattr(new_cfg, key, value)
            cfg = new_cfg
        super().__init__(cfg)
        self.cfg = cfg # Ensure type hint and access

    def _build_chain(self, root_path: str):
        super()._build_chain(root_path)

        stage = sim_utils.get_current_stage()
        ee_cyl_path = f"{root_path}/ee_cylinder"
        ee_cyl_length = 0.07

        grey_material = sim_utils.spawners.materials.PreviewSurfaceCfg(
            diffuse_color=(0.25, 0.25, 0.25),
            metallic=1.0,
            roughness=0.2,
        )
        white_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0))
        red_material = sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2))

        # Cup geometry convention:
        # - Cup rigid-body origin is kept at the EE-side contact point.
        # - Cup visual center is shifted along +X by cup_radius.
        # - Cup cylinder axis is rotated from local Z to local X.
        # - Cup bottom anchor in cup-local frame is x = (cup_radius - cup_height/2).
        cup_side_to_bottom_x = self.cfg.cup_radius - self.cfg.cup_height / 2 + 0.015

        # 1) Cup (rigid)
        cup_path = f"{root_path}/cup"
        sim_utils.create_prim(cup_path, prim_type="Xform")
        cup_prim = stage.GetPrimAtPath(cup_path)
        UsdPhysics.RigidBodyAPI.Apply(cup_prim)
        # Enable contact reporting for the cup
        PhysxSchema.PhysxContactReportAPI.Apply(cup_prim)
        
        cup_mass = 0.1
        self._apply_mass_properties(
            cup_prim,
            mass=cup_mass,
            diagonal_inertia=self._cylinder_inertia(cup_mass, self.cfg.cup_radius, self.cfg.cup_height, axis="x"),
            center_of_mass=(self.cfg.cup_radius, 0.0, 0.0),
        )

        cup_rotation = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90).GetQuat())
        
        # Hollow Cup Implementation:
        # We use a thin cylinder for the bottom and multiple cuboids for the walls.
        wall_thickness = 0.005
        num_wall_segments = 12
        import math

        # Shared physics material for cup, string, and ball
        shared_physics_material = sim_utils.spawners.materials.RigidBodyMaterialCfg(
            static_friction=0.7,
            dynamic_friction=0.7,
            restitution=0.1,
        )

        # Bottom plate
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{cup_path}/bottom",
            sim_utils.spawners.MeshCylinderCfg(
                radius=self.cfg.cup_radius,
                height=wall_thickness,
                visual_material=grey_material,
                physics_material=shared_physics_material,
                collision_props=sim_utils.CollisionPropertiesCfg()
            ),
            translation=(cup_side_to_bottom_x + wall_thickness/2, 0.0, 0.0),
            orientation=self._quat_to_tuple(cup_rotation)
        )

        # Wall segments
        wall_height = self.cfg.cup_height - wall_thickness
        wall_center_x = cup_side_to_bottom_x + wall_thickness + wall_height / 2 + 0.001  # +0.001 to ensure some overlap for stability
        
        # The width of each segment to cover the circumference
        segment_width = 2 * math.pi * self.cfg.cup_radius / num_wall_segments * 0.95  # 0.95 to add some gap between segments for stability
        for i in range(num_wall_segments):
            angle_rad = 2 * math.pi * i / num_wall_segments
            y = self.cfg.cup_radius * math.cos(angle_rad)
            z = self.cfg.cup_radius * math.sin(angle_rad)
            
            # Each segment needs to be rotated around the X-axis (cup axis) to form the ring
            # The cuboid's local X is the cup's X. We rotate around X by the segment's angle.
            seg_rot = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(angle_rad)).GetQuat())
            
            sim_utils.spawners.meshes.spawn_mesh_cuboid(
                f"{cup_path}/wall_{i}",
                sim_utils.spawners.MeshCuboidCfg(
                    size=(wall_height, wall_thickness, segment_width),
                    visual_material=grey_material,
                    physics_material=shared_physics_material,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                ),
                translation=(wall_center_x, y, z),
                orientation=self._quat_to_tuple(seg_rot)
            )

        # Fix cup to EE tip.
        self._create_fixed_joint(
            f"{cup_path}/fixed_joint",
            ee_cyl_path,
            cup_path,
            Gf.Vec3f(0, 0, ee_cyl_length),
            Gf.Quatf(1, 0, 0, 0)
        )

        # 2) Multi-segment String
        num_segments = 12
        segment_length = self.cfg.string_length / num_segments
        segment_radius = self.cfg.string_radius
        segment_mass = 0.0005
        
        prev_link_path = cup_path
        # Initial anchor on cup
        prev_anchor_pos = Gf.Vec3f(-0.015, 0.0, 0.0)
        prev_anchor_rot = cup_rotation

        for i in range(num_segments):
            seg_path = f"{root_path}/string_seg_{i}"
            sim_utils.create_prim(seg_path, prim_type="Xform")
            seg_prim = stage.GetPrimAtPath(seg_path)
            UsdPhysics.RigidBodyAPI.Apply(seg_prim)
            
            # Apply mass properties
            self._apply_mass_properties(
                seg_prim,
                mass=segment_mass,
                diagonal_inertia=self._cylinder_inertia(segment_mass, segment_radius, segment_length, axis="z"),
                center_of_mass=(0.0, 0.0, -segment_length / 2.0),
            )

            # Visual
            sim_utils.spawners.meshes.spawn_mesh_cylinder(
                f"{seg_path}/visual",
                sim_utils.spawners.MeshCylinderCfg(
                    radius=segment_radius,
                    height=segment_length,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    physics_material=shared_physics_material,
                    visual_material=white_material
                ),
                translation=(0.0, 0.0, -segment_length / 2.0),
            )

            # Joint from previous link to this segment
            joint_path = f"{seg_path}/joint"
            joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
            joint.CreateBody0Rel().SetTargets([prev_link_path])
            joint.CreateBody1Rel().SetTargets([seg_path])
            joint.CreateLocalPos0Attr().Set(prev_anchor_pos)
            joint.CreateLocalRot0Attr().Set(prev_anchor_rot)
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
            
            # Enable contact reporting for the segment
            PhysxSchema.PhysxContactReportAPI.Apply(seg_prim)
            
            # --- PhysX Joint Parameters for Stability ---
            physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            
            # 1. Joint Friction: Adds constant resistance to motion.
            # Higher values reduce high-frequency jitter but can make the string "stiff".
            physx_joint.CreateJointFrictionAttr(0.0005)
            
            # 2. Max Joint Velocity: Limits how fast the joint can rotate.
            # Useful for preventing "explosive" reactions to step inputs.
            physx_joint.CreateMaxJointVelocityAttr(100.0) 

            # 3. Armature: Adds virtual inertia to the joint.
            # Increasing this makes the joint feel "heavier" and less prone to sudden oscillations.
            physx_joint.CreateArmatureAttr(0.0001)

            # 4. Solver Iteration Counts: Can be set per-joint, but usually better at articulation level.
            # ---------------------------------------------

            # Update for next segment
            prev_link_path = seg_path
            prev_anchor_pos = Gf.Vec3f(0, 0, -segment_length)
            prev_anchor_rot = Gf.Quatf(1, 0, 0, 0)

        # 3) Ball (rigid)
        ball_path = f"{root_path}/ball"
        sim_utils.create_prim(ball_path, prim_type="Xform")
        ball_prim = stage.GetPrimAtPath(ball_path)
        UsdPhysics.RigidBodyAPI.Apply(ball_prim)
        ball_mass = 0.05
        self._apply_mass_properties(
            ball_prim,
            mass=ball_mass,
            diagonal_inertia=self._sphere_inertia(ball_mass, self.cfg.ball_radius),
            center_of_mass=(0.0, 0.0, -self.cfg.ball_radius),
        )

        # Ball visual
        sim_utils.spawners.meshes.spawn_mesh_sphere(
            f"{ball_path}/visual",
            sim_utils.spawners.MeshSphereCfg(
                radius=self.cfg.ball_radius,
                visual_material=red_material,
                physics_material=shared_physics_material,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
            translation=(0.0, 0.0, -self.cfg.ball_radius)
        )

        # Enable contact reporting for the ball
        PhysxSchema.PhysxContactReportAPI.Apply(ball_prim)

        # Joint from last string segment to ball
        ball_joint = UsdPhysics.SphericalJoint.Define(stage, f"{ball_path}/spherical_joint")
        ball_joint.CreateBody0Rel().SetTargets([prev_link_path])
        ball_joint.CreateBody1Rel().SetTargets([ball_path])
        ball_joint.CreateLocalPos0Attr().Set(prev_anchor_pos)
        ball_joint.CreateLocalRot0Attr().Set(prev_anchor_rot)
        ball_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        ball_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        
        ball_physx_joint = PhysxSchema.PhysxJointAPI.Apply(ball_joint.GetPrim())
        ball_physx_joint.CreateJointFrictionAttr(0.0005)

    @staticmethod
    def spawn(prim_path: str, cfg: LynxUsdConstructorSpawnerCfg, translation: Optional[tuple] = None, orientation: Optional[tuple] = None):
        if not prim_path:
            prim_path = "/World/Robot"

        # Keep path handling consistent with the base Lynx constructor.
        # InteractiveScene may pass template/regex paths before expansion.
        if "{ENV_REGEX_NS}" in prim_path:
            prim_path = prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

        if ".*" in prim_path:
            import re

            prim_path = re.sub(r"env_\.\*", "env_0", prim_path)

        # Final guard against ill-formed/empty path.
        if not prim_path or not str(prim_path).startswith("/"):
            prim_path = "/World/Robot"
        
        sim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        
        # Override articulation properties to enable self-collisions
        if hasattr(cfg, "articulation_props") and cfg.articulation_props is not None:
            cfg.articulation_props.enabled_self_collisions = True
            sim_utils.define_articulation_root_properties(prim_path, cfg.articulation_props)
        else:
            # If no props provided, create default with self-collisions enabled
            from .lynx_constructor import LynxUsdConstructorSpawnerCfg
            default_props = LynxUsdConstructorSpawnerCfg().articulation_props
            default_props.enabled_self_collisions = True
            sim_utils.define_articulation_root_properties(prim_path, default_props)
        
        if isinstance(cfg, LynxBallInCupRobotCfg):
            robot_cfg = cfg
        elif hasattr(cfg, "robot_cfg") and cfg.robot_cfg is not None:
            robot_cfg = cfg.robot_cfg
        else:
            robot_cfg = LynxBallInCupRobotCfg()
            
        constructor = LynxBallInCupConstructor(robot_cfg)
        constructor._build_chain(prim_path)
        
        return root_prim

# Register the spawn function
LynxUsdConstructorSpawnerCfg.func = LynxBallInCupConstructor.spawn
spawn = LynxBallInCupConstructor.spawn
