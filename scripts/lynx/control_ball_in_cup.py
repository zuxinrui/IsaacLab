# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script provides an interactive GUI to control the Lynx Ball-in-a-Cup robot joints."""

import argparse
import torch
import numpy as np
import re

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Interactive Joint Control for Lynx Ball-in-a-Cup.")
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
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab_assets.robots.lynx_ball_in_cup import LynxBallInCupRobotCfg, LynxBallInCupConstructor

class LynxInteractiveGui:
    """GUI window for controlling robot joints."""
    def __init__(self, robot: Articulation):
        self.robot = robot
        self.num_joints = robot.num_joints
        self.joint_names = robot.data.joint_names
        
        # Identify actuated joints (those with stiffness > 0)
        self.actuated_joint_indices = []
        
        # Heuristic: Lynx arm joints are named joint_1 to joint_6
        # We can also check the actuator config
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
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create Lynx robot configuration
    robot_cfg = LynxBallInCupRobotCfg(
        prim_path="/World/Lynx",
        num_joints=6,
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1,
        rotation_angles=[180.0, 0.0, 0.0, -180.0, 0.0, 90.0],
        l1_end_point_pos=(0.0, 0.0, 0.2),
        l1_end_point_theta=0.0,
        l2_end_point_pos=(0.0, 0.0, 0.2),
        l2_end_point_theta=0.0,
        l3_end_point_pos=(0.0, 0.0, 0.2),
        l3_end_point_theta=0.0,
        l4_end_point_pos=(0.0, 0.05, 0.2),
        l4_end_point_theta=30.0,
        l5_end_point_pos=(0.0, 0.0, 0.2),
        l5_end_point_theta=0.0,
        # Ball in cup specific
        cup_radius=0.05,
        cup_height=0.08,
        ball_radius=0.02,
        string_length=0.4
    )
    
    # Set the spawn function
    robot_cfg.spawn.func = LynxBallInCupConstructor.spawn
    
    # Ensure the spawn function has the robot_cfg object.
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
        "collision_mode": "full",  # "full" or "ee_only" / robot_cfg.collision_mode
        "articulation_props": robot_cfg.spawn.articulation_props,
        "rigid_props": robot_cfg.spawn.rigid_props,
    }
    
    # We need to wrap it in Articulation for easy control
    robot = Articulation(robot_cfg)

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

    print("[INFO]: Simulation finished.")

if __name__ == "__main__":
    main()
    simulation_app.close()
