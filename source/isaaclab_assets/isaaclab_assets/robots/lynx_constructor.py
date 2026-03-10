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

from pxr import Usd, UsdGeom, UsdPhysics, Gf

@configclass
class LynxUsdConstructorSpawnerCfg(sim_utils.SpawnerCfg):
    """Configuration for the Lynx USD constructor spawner."""
    func: Any = None # Will be set to LynxUsdConstructor.spawn

@configclass
class LynxRobotCfg(ArticulationCfg):
    """Configuration for the procedurally generated Lynx robot."""
    
    # Genotypes
    genotype_tube: List[int] = [0, 1, 0, 1, 0]
    genotype_joints: int = 1 # 0: all orthogonal, 1: all inline, 2: mixed (from spec)
    rotation_angles: List[float] = [180.0, 0.0, 0.0, -180.0, 0.0, 0.0]
    
    # B-Spline Tube Parameters
    use_bspline: bool = True
    bspline_num_segments: int = 100
    bspline_end_point_pos: tuple = (0.0, 0.1, 0.3)
    bspline_end_point_theta: float = 0.0
    bspline_dual_point_distance: float = 0.07
    
    # Geometric Parameters
    tube_radiuses: List[float] = [0.035] * 5  # [0.0396] * 5
    clamp_length: float = 0.051
    # Paths to STLs (assuming they will be provided or are in a known location)
    clamp_stl: str = "source/isaaclab_assets/data/Robots/Lynx/models/clamp_0226.stl"
    ee_stl: Optional[str] = None
    
    # Physics/Actuation
    actuators: dict[str, ImplicitActuatorCfg] = {
        "lynx_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_.*"],
            stiffness=5000000.0,
            damping=1.0,
        )
    }
    
    # Spawning logic
    spawn: sim_utils.SpawnerCfg = LynxUsdConstructorSpawnerCfg()

class LynxUsdConstructor:
    """Procedural constructor for the Lynx manipulator in USD."""

    @staticmethod
    def spawn(prim_path: str, cfg: LynxRobotCfg, translation: Optional[tuple] = None, orientation: Optional[tuple] = None):
        """Spawns the robot at the specified prim path.
        
        Args:
            prim_path: The path to spawn the robot at.
            cfg: The configuration for the robot.
            translation: The translation of the robot.
            orientation: The orientation of the robot.
        """
        # If orientation is not provided, rotate 90 degrees around X to align with Z axis
        # if orientation is None:
        #     # Rotation of 90 degrees around X-axis: (w=0.707, x=0.707, y=0, z=0)
        #     # This converts the Y-up orientation to Z-up
        #     orientation = (0.7071068, 0.7071068, 0.0, 0.0)
        
        # # If translation is not provided, set a default height to keep it above the horizon
        # if translation is None:
        #     translation = (0.0, 0.0, 0.5)

        # Create the root Xform
        sim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
        
        # Apply ArticulationRootAPI to the root
        stage = sim_utils.get_current_stage()
        root_prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        
        # Build the robot chain
        constructor = LynxUsdConstructor(cfg)
        constructor._build_chain(prim_path)
        
        return root_prim

    def __init__(self, cfg: LynxRobotCfg):
        self.cfg = cfg

    def _quat_to_tuple(self, q: Gf.Quatf):
        """Helper to convert Gf.Quatf to tuple (w, x, y, z)."""
        return (q.GetReal(), *q.GetImaginary())

    def _build_chain(self, root_path: str):
        """Internal logic to assemble the links and joints."""
        stage = sim_utils.get_current_stage()
        
        # 1. Base Link
        base_path = f"{root_path}/base"
        sim_utils.create_prim(base_path, prim_type="Xform")
        base_prim = stage.GetPrimAtPath(base_path)
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        
        # Fix the base to the world (root Xform)
        fixed_base_joint = UsdPhysics.FixedJoint.Define(stage, f"{base_path}/root_fixed_joint")
        fixed_base_joint.CreateBody0Rel().SetTargets([root_path])
        fixed_base_joint.CreateBody1Rel().SetTargets([base_path])
        # No offset needed as base is at (0,0,0) relative to root
        fixed_base_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_base_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
        fixed_base_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
        fixed_base_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        
        # Base dimensions from LynxRobotics/lynx/robots/lynx_manipulator/constructor.py
        base_length1 = 0.001
        base_radius1 = 0.08
        base_length2 = 0.017 + 0.001
        base_radius2 = 0.08
        
        # Add base visuals
        # Base is upright: visual_1 at z=base_length1/2, visual_2 at z=base_length1 + base_length2/2
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{base_path}/visual_1",
            sim_utils.spawners.MeshCylinderCfg(
                radius=base_radius1,
                height=base_length1,
                visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
            ),
            translation=(0, 0, base_length1 / 2)
        )
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{base_path}/visual_2",
            sim_utils.spawners.MeshCylinderCfg(
                radius=base_radius2,
                height=base_length2,
                visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
            ),
            translation=(0, 0, base_length1 + base_length2 / 2)
        )

        curr_parent_path = base_path
        # Attachment point for base is at (0, 0, length)
        curr_pos = Gf.Vec3f(0, 0, base_length1 + base_length2)
        curr_quat = Gf.Quatf(1, 0, 0, 0)

        num_joints = 6
        
        # Determine joint types
        joints_inline = [True, True, True, True, True]
        joints_inline[self.cfg.genotype_joints:] = [False] * (len(joints_inline) - self.cfg.genotype_joints)
        # joint_types[0] is for joint 2. Joint 1 is always inline.
        joint_types = ["inline"] + ["inline" if inline else "orthogonal" for inline in joints_inline]

        # Joint parameters from LynxRobotics
        joint_params = [
            # Joint 1
            {"l0": 0.036, "r0": 0.058, "l1": 0.128, "r1": 0.062, "l2": 0.013, "r2": 0.062},
            # Joint 2
            {"l0": 0.036, "r0": 0.058, "l1": 0.128, "r1": 0.062, "l2": 0.013, "r2": 0.062},
            # Joint 3
            {"l0": 0.029, "r0": 0.04, "l1": 0.092, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 4
            {"l0": 0.029, "r0": 0.04, "l1": 0.092, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 5
            {"l0": 0.029, "r0": 0.04, "l1": 0.096, "r1": 0.042, "l2": 0.008, "r2": 0.042},
            # Joint 6
            {"l0": 0.029, "r0": 0.04, "l1": 0.096, "r1": 0.042, "l2": 0.008, "r2": 0.042},
        ]

        tube_idx = 0
        for i in range(num_joints):
            # 1. Check if we need to insert a tube BEFORE this joint (except joint 1)
            if i > 0 and tube_idx < len(self.cfg.genotype_tube) and self.cfg.genotype_tube[tube_idx] == 1:
                tube_name = f"tube_{tube_idx+1}"
                tube_path = f"{root_path}/{tube_name}"
                sim_utils.create_prim(tube_path, prim_type="Xform")
                tube_prim = stage.GetPrimAtPath(tube_path)
                UsdPhysics.RigidBodyAPI.Apply(tube_prim)
                
                tube_radius = self.cfg.tube_radiuses[tube_idx]
                
                if self.cfg.use_bspline:
                    # B-Spline Tube Logic
                    num_segs = self.cfg.bspline_num_segments
                    end_pos = Gf.Vec3f(*self.cfg.bspline_end_point_pos)
                    theta = self.cfg.bspline_end_point_theta
                    d = self.cfg.bspline_dual_point_distance
                    
                    # Offsets for joint volumes
                    pre_joint_r = joint_params[i-1]["r2"] if i > 0 else 0.08
                    next_joint_r = joint_params[i]["r0"]
                    
                    # Calculate control points (clamped cubic)
                    p0 = Gf.Vec3f(0, 0, 0)
                    p1 = p0 + Gf.Vec3f(0, 0, d)
                    
                    # End direction from theta (rotate +Z toward +Y in YZ plane)
                    s, c = np.sin(theta), np.cos(theta)
                    t_dir = Gf.Vec3f(0, s, c).GetNormalized()
                    
                    # Offset end point by joint volumes
                    p3 = end_pos - Gf.Vec3f(0, 0, pre_joint_r) - t_dir * next_joint_r
                    p2 = p3 - t_dir * d
                    
                    # Sample points along the curve (Cubic Bezier)
                    curve_points = []
                    for j in range(num_segs + 1):
                        t = j / num_segs
                        pt = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                        curve_points.append(pt)
                    
                    # Create segments
                    last_seg_quat = Gf.Quatf(1, 0, 0, 0)
                    for j in range(num_segs):
                        seg_start = curve_points[j]
                        seg_end = curve_points[j+1]
                        seg_dir = (seg_end - seg_start)
                        seg_len = seg_dir.GetLength()
                        if seg_len < 1e-6: continue
                        seg_dir /= seg_len
                        
                        seg_center = (seg_start + seg_end) * 0.5
                        
                        # Rotation from Z to seg_dir
                        z_axis = Gf.Vec3f(0, 0, 1)
                        dot = z_axis * seg_dir
                        if abs(dot) > 0.9999:
                            seg_quat = Gf.Quatf(1, 0, 0, 0) if dot > 0 else Gf.Quatf(0, 1, 0, 0)
                        else:
                            axis = (z_axis ^ seg_dir).GetNormalized()
                            angle = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
                            seg_quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(axis), angle).GetQuat())
                        
                        last_seg_quat = seg_quat
                        
                        is_first = (j == 0)
                        is_last = (j == num_segs - 1)
                        
                        if is_first or is_last:
                            # Use STL for first and last segments (clamps)
                            mesh_pos = seg_center
                            mesh_quat = seg_quat
                            if is_last:
                                # Rotate 180 around X for "upside down" clamp
                                mesh_quat = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat()) * seg_quat
                            
                            sim_utils.create_prim(
                                f"{tube_path}/visual_{j}",
                                prim_type="Xform",
                                translation=tuple(mesh_pos),
                                orientation=self._quat_to_tuple(mesh_quat)
                            )
                            # Use create_prim with usd_path to reference the STL
                            sim_utils.create_prim(
                                f"{tube_path}/visual_{j}/mesh",
                                usd_path=self.cfg.clamp_stl,
                                scale=(0.001, 0.001, 0.001)
                            )
                        else:
                            sim_utils.spawners.meshes.spawn_mesh_cylinder(
                                f"{tube_path}/visual_{j}",
                                sim_utils.spawners.MeshCylinderCfg(
                                    radius=tube_radius,
                                    height=seg_len,
                                    visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1))
                                ),
                                translation=tuple(seg_center),
                                orientation=self._quat_to_tuple(seg_quat)
                            )
                    
                    self._create_fixed_joint(f"{tube_path}/fixed_joint", curr_parent_path, tube_path, curr_pos, curr_quat)
                    curr_parent_path = tube_path
                    curr_pos = p3
                    curr_quat = last_seg_quat
                    
                else:
                    # Straight Tube Logic
                    tube_length = 0.2805 # Default
                    sim_utils.spawners.meshes.spawn_mesh_cylinder(
                        f"{tube_path}/visual",
                        sim_utils.spawners.MeshCylinderCfg(
                            radius=tube_radius,
                            height=tube_length,
                            visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1))
                        ),
                        translation=(0, 0, tube_length / 2)
                    )
                    self._create_fixed_joint(f"{tube_path}/fixed_joint", curr_parent_path, tube_path, curr_pos, curr_quat)
                    curr_parent_path = tube_path
                    curr_pos = Gf.Vec3f(0, 0, tube_length)
                    curr_quat = Gf.Quatf(1, 0, 0, 0)
                
                tube_idx += 1
            elif i > 0:
                tube_idx += 1

            # 2. Spawn the Joint and its Link
            joint_name = f"joint_{i+1}"
            link_name = f"link_{i+1}"
            link_path = f"{root_path}/{link_name}"
            sim_utils.create_prim(link_path, prim_type="Xform")
            link_prim = stage.GetPrimAtPath(link_path)
            UsdPhysics.RigidBodyAPI.Apply(link_prim)
            
            p = joint_params[i]
            j_type = joint_types[i]
            angle_rad = np.deg2rad(self.cfg.rotation_angles[i])
            
            if j_type == "inline":
                # JointInline logic from joints.py
                rx = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat())
                twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(angle_rad)).GetQuat())
                new_relative_quat = rx * twist
                
                joint_pos = Gf.Vec3f(0, 0, p["l0"] + p["l1"] / 2)
                cylinder2_pos = joint_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["r1"] + p["l2"] / 2))
                cylinder2_add_pos = joint_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["r1"] / 2))

                # Fixed part visual
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/fixed_visual",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r0"], height=p["l0"],
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
                    ),
                    translation=(0, 0, p["l0"] / 2)
                )
                
                # Revolute Joint
                joint_path = f"{link_path}/{joint_name}"
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.CreateAxisAttr("Z")
                joint.CreateBody0Rel().SetTargets([curr_parent_path])
                joint.CreateBody1Rel().SetTargets([link_path])
                
                joint.CreateLocalPos0Attr().Set(curr_pos)
                joint.CreateLocalRot0Attr().Set(curr_quat)
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                
                # Moving part visuals
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_main",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r1"], height=p["l1"],
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(joint_pos)
                )
                
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_attach_add",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r2"], height=p["r1"],
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(cylinder2_add_pos),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )

                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual_attach",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=p["r2"], height=p["l2"],
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(cylinder2_pos),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )
                
                # Update curr for next iteration
                curr_parent_path = link_path
                curr_pos = cylinder2_pos + new_relative_quat.Transform(Gf.Vec3f(0, 0, p["l2"] / 2))
                # Fix: Use new_relative_quat * curr_quat to accumulate rotation correctly
                curr_quat = new_relative_quat

            else:
                # JointOrthogonal logic from joints.py
                l2_orig, r2_orig = p["l0"], p["r0"]
                l1, r1 = p["l1"], p["r1"]
                l0_orig, r0_orig = p["l2"], p["r2"]
                
                rx = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90).GetQuat())
                twist = Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), np.rad2deg(angle_rad)).GetQuat())
                new_relative_quat = twist * rx
                
                cylinder0_pos = Gf.Vec3f(0, 0, l0_orig / 2)
                cylinder0_add_pos = Gf.Vec3f(0, 0, l0_orig + r1 / 2)
                joint_pos_rel = Gf.Vec3f(0, 0, l0_orig + r1)
                cylinder2_pos_rel = joint_pos_rel + new_relative_quat.Transform(Gf.Vec3f(0, 0, l1 / 2 + l2_orig / 2))

                # Fixed part visuals
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/fixed_visual_0",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r0_orig, height=l0_orig,
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(cylinder0_pos)
                )
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/fixed_visual_add",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r0_orig, height=r1,
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(cylinder0_add_pos)
                )
                
                # Joint shell (fixed)
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/fixed_visual_shell",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r1, height=l1,
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
                    ),
                    translation=tuple(joint_pos_rel),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )
                
                # Revolute Joint
                joint_path = f"{link_path}/{joint_name}"
                joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
                joint.CreateAxisAttr("Z") 
                joint.CreateBody0Rel().SetTargets([curr_parent_path])
                joint.CreateBody1Rel().SetTargets([link_path])
                
                joint.CreateLocalPos0Attr().Set(curr_pos)
                joint.CreateLocalRot0Attr().Set(curr_quat) 
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
                joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
                
                # Moving part visual
                sim_utils.spawners.meshes.spawn_mesh_cylinder(
                    f"{link_path}/moving_visual",
                    sim_utils.spawners.MeshCylinderCfg(
                        radius=r2_orig, height=l2_orig,
                        visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3))
                    ),
                    translation=tuple(cylinder2_pos_rel),
                    orientation=self._quat_to_tuple(new_relative_quat)
                )
                
                # Update curr for next iteration
                curr_parent_path = link_path
                curr_pos = cylinder2_pos_rel + new_relative_quat.Transform(Gf.Vec3f(0, 0, l2_orig / 2))
                # Fix: Use new_relative_quat * curr_quat to accumulate rotation correctly
                curr_quat = new_relative_quat

            # Apply Drive API
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{link_path}/{joint_name}"), "revolute")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(self.cfg.actuators["lynx_arm"].stiffness)
            drive.CreateDampingAttr(self.cfg.actuators["lynx_arm"].damping)

        # 3. End Effector
        ee_cyl_length = 0.07
        ee_cyl_radius = 0.010
        ee_cyl_path = f"{root_path}/ee_cylinder"
        sim_utils.create_prim(ee_cyl_path, prim_type="Xform")
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(ee_cyl_path))
        
        sim_utils.spawners.meshes.spawn_mesh_cylinder(
            f"{ee_cyl_path}/visual",
            sim_utils.spawners.MeshCylinderCfg(
                radius=ee_cyl_radius, height=ee_cyl_length,
                visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15))
            ),
            translation=(0, 0, ee_cyl_length / 2)
        )
        self._create_fixed_joint(f"{ee_cyl_path}/fixed_joint", curr_parent_path, ee_cyl_path, curr_pos, curr_quat)
        
        # Final EE
        ee_path = f"{root_path}/ee"
        sim_utils.create_prim(ee_path, prim_type="Xform")
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(ee_path))
        
        sim_utils.spawners.meshes.spawn_mesh_cuboid(
            f"{ee_path}/visual",
            sim_utils.spawners.MeshCuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.spawners.materials.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2))
            )
        )
        self._create_fixed_joint(f"{ee_path}/fixed_joint", ee_cyl_path, ee_path, Gf.Vec3f(0, 0, ee_cyl_length / 2), Gf.Quatf(1, 0, 0, 0))

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
