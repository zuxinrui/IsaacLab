# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script tests the rotation direction of each joint of the Lynx robot."""

import argparse
import torch
import numpy as np
import time

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Test joint rotation directions for Lynx.")
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
class LynxJointTestEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = LynxSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        self.sim.dt = 1.0 / 60.0
        self.decimation = 4 # 15Hz control for smoother testing
        
        self.sim.render_interval = 1

        # Action settings
        self.actions = LynxActionsCfg()
        
        # Observation settings
        self.observations = LynxObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = LynxJointTestEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Get joint names and number of joints
    joint_names = env.scene["robot"].data.joint_names
    num_joints = env.scene["robot"].num_joints

    print(f"[INFO]: Testing {num_joints} joints: {joint_names}")
    
    # Test parameters
    test_amplitude = 0.5  # radians
    test_duration_per_joint = 4.0  # seconds (longer duration)
    pause_duration = 1.0 # seconds pause between joints
    control_dt = env_cfg.sim.dt * env_cfg.decimation
    steps_per_joint = int(test_duration_per_joint / control_dt)
    steps_per_pause = int(pause_duration / control_dt)
    total_steps_per_cycle = steps_per_joint + steps_per_pause

    # Initialize
    step_idx = 0
    
    print(f"\n[INFO]: Starting joint direction test...")
    print(f"[INFO]: Each joint will move to +{test_amplitude} rad and then back to 0.")
    print(f"[INFO]: Test duration per joint: {test_duration_per_joint}s + {pause_duration}s pause.")

    while simulation_app.is_running():
        # Reset action
        action = torch.zeros((env.num_envs, num_joints), device=env.device)
        
        # Calculate which joint to move
        current_joint_idx = (step_idx // total_steps_per_cycle) % num_joints
        
        # Step within the current joint's cycle
        cycle_step = step_idx % total_steps_per_cycle
        
        if cycle_step < steps_per_joint:
            # Calculate phase within the movement period (0 to 1)
            phase = cycle_step / steps_per_joint
            
            # Simple movement: 0 -> +amplitude -> 0
            if phase < 0.5:
                # Move to positive amplitude
                target_val = test_amplitude * (phase / 0.5)
            else:
                # Move back to 0
                target_val = test_amplitude * (1.0 - (phase - 0.5) / 0.5)
                
            action[0, current_joint_idx] = target_val
            
            # Log when starting a new joint movement
            if cycle_step == 0:
                print(f"\n>>> Testing Joint {current_joint_idx}: {joint_names[current_joint_idx]}")
                print(f"    Moving: 0 -> +{test_amplitude} -> 0")
        else:
            # Pause period: keep all joints at 0
            if cycle_step == steps_per_joint:
                print(f"    Finished {joint_names[current_joint_idx]}. Pausing...")
        
        # Step environment
        env.step(action)
        
        step_idx += 1
        
        # Optional: stop after one full cycle of all joints
        if step_idx >= num_joints * total_steps_per_cycle:
            print("\n[INFO]: Completed one full cycle of all joint tests.")
            # Break or loop
            # break 

    print("[INFO]: Simulation finished.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
