# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script provides an interactive GUI to control the Lynx robot for the Push task."""

import argparse
import torch
import numpy as np
import re

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Interactive Joint Control for Lynx Push Task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import omni.ui as ui
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import CuboidCfg, MeshCuboidCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor

class LynxInteractiveGui:
    """GUI window for controlling robot joints."""
    def __init__(self, robot: Articulation):
        self.robot = robot
        self.num_joints = robot.num_joints
        self.joint_names = robot.data.joint_names
        
        # Identify actuated joints (those with stiffness > 0)
        self.actuated_joint_indices = []
        
        for i in range(self.num_joints):
            joint_name = self.joint_names[i]
            is_actuated = False
            
            # Check if it matches any actuator expression
            for actuator_name, actuator_cfg in robot.cfg.actuators.items():
                for expr in actuator_cfg.joint_names_expr:
                    if re.search(expr, joint_name):
                        # Check stiffness
                        stiffness = actuator_cfg.stiffness
                        if isinstance(stiffness, dict):
                            val = stiffness.get(joint_name, 0.0)
                            if val > 0:
                                is_actuated = True
                        elif stiffness is not None and stiffness > 0:
                            is_actuated = True
                if is_actuated:
                    break
            
            if is_actuated:
                self.actuated_joint_indices.append(i)

        # Create a window
        self.window = ui.Window(
            "Lynx Joint Control",
            width=400,
            height=400,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM
        )
        self.window.deferred_dock_in("Property")
        
        # Target positions (initialized to current)
        self.target_joint_pos = robot.data.joint_pos[0].clone()
        
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("Adjust Actuated Joint Positions (Radians)", alignment=ui.Alignment.CENTER)
                ui.Spacer(height=10)

                def on_reset():
                    for idx, slider in self.sliders.items():
                        slider.model.set_value(0.0)
                        self.target_joint_pos[idx] = 0.0
                
                ui.Button("Reset Actuated to Zero", clicked_fn=on_reset)
                ui.Spacer(height=10)
                
                self.sliders = {}
                for i in self.actuated_joint_indices:
                    with ui.HStack():
                        ui.Label(f"{self.joint_names[i]}:", width=120)
                        slider = ui.FloatSlider(min=-3.14, max=3.14)
                        slider.model.set_value(float(self.target_joint_pos[i]))
                        
                        # Update target pos when slider changes
                        def on_change(model, idx=i):
                            self.target_joint_pos[idx] = model.get_value_as_float()
                        
                        slider.model.add_value_changed_fn(on_change)
                        self.sliders[i] = slider

    def get_target_pos(self):
        return self.target_joint_pos

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view((1.5, 0.0, 1.2), (0.0, 0.0, 0.5))

    # Design scene
    # Lights
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Table (from push_env_cfg.py)
    table_cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 0.8, 0.018),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, -0.018 / 2))

    # Table Legs
    leg_positions = [
        (0.5 - 0.045 / 2, 0.4 - 0.045 / 2, -0.6 / 2 - 0.018),
        (-0.5 + 0.045 / 2, 0.4 - 0.045 / 2, -0.6 / 2 - 0.018),
        (0.5 - 0.045 / 2, -0.4 + 0.045 / 2, -0.6 / 2 - 0.018),
        (-0.5 + 0.045 / 2, -0.4 + 0.045 / 2, -0.6 / 2 - 0.018),
    ]
    for i, pos in enumerate(leg_positions):
        leg_cfg = sim_utils.MeshCuboidCfg(
            size=(0.045, 0.045, 0.6),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        leg_cfg.func(f"/World/TableLeg{i+1}", leg_cfg, translation=pos)

    # Ground plane
    plane_cfg = sim_utils.GroundPlaneCfg()
    plane_cfg.func("/World/defaultGroundPlane", plane_cfg, translation=(0.0, 0.0, -0.6 - 0.018))

    # Create Lynx robot configuration (from joint_pos_env_cfg.py)
    robot_cfg = LynxRobotCfg(
        prim_path="/World/Robot",
        num_joints=6,
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1,
        rotation_angles=[180.0, 60.0, 60.0, -180.0, 60.0, 90.0],
        l1_end_point_pos=(0.0, 0.0, 0.2),
        l1_end_point_theta=0.0,
        l2_end_point_pos=(0.0, 0.05, 0.2),
        l2_end_point_theta=30.0,
        l3_end_point_pos=(0.0, 0.0, 0.2),
        l3_end_point_theta=0.0,
        l4_end_point_pos=(0.0, 0.05, 0.2),
        l4_end_point_theta=30.0,
        l5_end_point_pos=(0.0, 0.0, 0.2),
        l5_end_point_theta=0.0,
    )
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
    }
    
    # Wrap in Articulation
    robot = Articulation(robot_cfg)

    # Create Cube (from joint_pos_env_cfg.py)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.09),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=CuboidCfg(
            size=(0.15, 0.15, 0.15),
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
    cube = RigidObject(cube_cfg)

    # Create Target (from joint_pos_env_cfg.py)
    target_cfg = RigidObjectCfg(
        prim_path="/World/Target",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.075),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=CuboidCfg(
            size=(0.15, 0.15, 0.15),
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
            ),
        ),
    )
    target = RigidObject(target_cfg)

    # Play the simulator
    sim.reset()
    
    # Initialize GUI
    gui = LynxInteractiveGui(robot)

    # Simulate
    while simulation_app.is_running():
        # Get target from GUI
        target_pos = gui.get_target_pos().unsqueeze(0) # (1, num_joints)
        
        # Apply joint positions directly
        robot.set_joint_position_target(target_pos)
        
        # Write data to sim
        robot.write_data_to_sim()
        
        # perform step
        sim.step(render=True)
        
        # Update robot data
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        target.update(sim.get_physics_dt())

    print("[INFO]: Simulation finished.")

if __name__ == "__main__":
    main()
    simulation_app.close()
