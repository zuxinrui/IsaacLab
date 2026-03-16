# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script implements a 5Hz motor control for the Lynx robot with velocity clipping."""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="5Hz Motor Control for Lynx with Velocity Clipping.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor

def dummy_obs(env):
    return torch.zeros((env.num_envs, 1), device=env.device)

@configclass
class LynxSceneCfg(InteractiveSceneCfg):
    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    
    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    # Robot configuration
    robot: LynxRobotCfg = LynxRobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    
    robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, 
        solver_position_iteration_count=32, 
        solver_velocity_iteration_count=4
    )
    
    # Ensure the spawn function is set and has the robot_cfg
    robot.spawn.func = LynxUsdConstructor.spawn
    robot.spawn.robot_cfg = {
        "genotype_tube": robot.genotype_tube,
        "genotype_joints": robot.genotype_joints,
        "rotation_angles": robot.rotation_angles,
        "bspline_num_segments": robot.bspline_num_segments,
        "bspline_dual_point_distance": robot.bspline_dual_point_distance,
        "tube_radiuses": robot.tube_radiuses,
        "clamp_stl": robot.clamp_stl,
        "ee_stl": robot.ee_stl,
        "actuators": robot.actuators,
        "init_state": robot.init_state,
        "rigid_props": robot.spawn.rigid_props,
        "articulation_props": robot.spawn.articulation_props,
    }

@configclass
class LynxActionsCfg:
    arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class LynxObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_obs)
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class Lynx5HzControlEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = LynxSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        # Physics runs at 100Hz (dt=0.01)
        self.sim.dt = 1.0 / 60.0
        
        # Control frequency is 5Hz (dt=0.2s)
        # Decimation = Control DT / Sim DT = 0.2 / 0.01 = 20
        self.decimation = 12
        
        self.sim.render_interval = 1

        # Action settings
        self.actions = LynxActionsCfg()
        
        # Observation settings
        self.observations = LynxObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = Lynx5HzControlEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Get number of joints from the robot asset
    num_joints = env.scene["robot"].num_joints

    # Velocity limits (rad/s)
    # Mega M1 (Joints 1-2): np.radians(20.0)
    # Standard S1 (Joints 3-4): np.radians(20.0)
    # Lite L1 (Joints 5-6): np.radians(20.0)
    velocity_limits = torch.tensor([np.radians(20.0)] * num_joints, device=env.device)
    
    # Control DT (5Hz)
    control_dt = 0.2
    
    # Max delta position per control step
    max_delta_pos = velocity_limits * control_dt

    print(f"[INFO]: Control Frequency: {1.0/control_dt} Hz (dt={control_dt}s)")
    print(f"[INFO]: Velocity Limits: {velocity_limits}")
    print(f"[INFO]: Max Delta Pos per step: {max_delta_pos}")

    # Initialize target position (absolute)
    # We start at the default joint positions
    current_joint_pos = env.scene["robot"].data.joint_pos.clone()
    
    # Simulate
    step_idx = 0
    while simulation_app.is_running():
        # Generate a random desired position (absolute)
        # For testing, let's oscillate between -0.5 and 0.5 rad
        desired_pos = 10 * torch.sin(torch.ones((env.num_envs, num_joints), device=env.device) * step_idx * 3)
        
        # Get current position from sim
        current_pos = env.scene["robot"].data.joint_pos
        
        # Calculate required delta
        delta_pos = desired_pos - current_pos
        
        # Clip delta based on velocity limits
        clipped_delta = torch.clamp(delta_pos, -max_delta_pos, max_delta_pos)
        
        # New target position
        target_pos = current_pos + clipped_delta
        
        # The JointPositionAction expects targets relative to default or absolute depending on config.
        # In LynxActionsCfg, use_default_offset=True, so env.step(action) adds action to default_joint_pos.
        # We want to provide (target_pos - default_joint_pos) as the action.
        action = target_pos - env.scene["robot"].data.default_joint_pos
        
        # Step environment
        env.step(action)
        
        # Logging
        if step_idx % 5 == 0: # Every 1 second at 5Hz
            applied_efforts = env.scene["robot"].data.applied_torque
            print(f"--- Step {step_idx} ---")
            print(f"Desired Pos: {desired_pos[0].cpu().numpy()}")
            print(f"Current Pos: {current_pos[0].cpu().numpy()}")
            print(f"Target Pos (clipped): {target_pos[0].cpu().numpy()}")
            print(f"Applied Torque: {applied_efforts[0].cpu().numpy()}")
        
        step_idx += 1

    print("[INFO]: Simulation finished.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
