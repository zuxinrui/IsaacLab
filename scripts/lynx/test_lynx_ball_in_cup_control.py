# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script implements a 5Hz motor control for the Lynx Ball-in-a-Cup robot with velocity clipping."""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="5Hz Motor Control for Lynx Ball-in-a-Cup with Velocity Clipping.")
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

from isaaclab_assets.robots.lynx_ball_in_cup import LynxBallInCupRobotCfg, LynxBallInCupConstructor

def dummy_obs(env):
    return torch.zeros((env.num_envs, 1), device=env.device)

@configclass
class LynxBallInCupSceneCfg(InteractiveSceneCfg):
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
    robot: LynxBallInCupRobotCfg = LynxBallInCupRobotCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
    
    # We must set the spawn function and its properties correctly.
    # The error "Path must be an absolute path: <>" usually happens when the spawner
    # is called without a valid prim_path, or when the configuration is misaligned.
    robot.spawn.func = LynxBallInCupConstructor.spawn
    robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, 
        solver_position_iteration_count=32, 
        solver_velocity_iteration_count=4
    )
    
    # Ensure the spawn function has a non-recursive robot_cfg object.
    # We use a dictionary to avoid recursion error in config validation.
    robot.spawn.robot_cfg = {
        "num_joints": robot.num_joints,
        "genotype_tube": robot.genotype_tube,
        "genotype_joints": robot.genotype_joints,
        "rotation_angles": robot.rotation_angles,
        "l1_end_point_pos": robot.l1_end_point_pos,
        "l1_end_point_theta": robot.l1_end_point_theta,
        "l2_end_point_pos": robot.l2_end_point_pos,
        "l2_end_point_theta": robot.l2_end_point_theta,
        "l3_end_point_pos": robot.l3_end_point_pos,
        "l3_end_point_theta": robot.l3_end_point_theta,
        "l4_end_point_pos": robot.l4_end_point_pos,
        "l4_end_point_theta": robot.l4_end_point_theta,
        "l5_end_point_pos": robot.l5_end_point_pos,
        "l5_end_point_theta": robot.l5_end_point_theta,
        "bspline_num_segments": robot.bspline_num_segments,
        "bspline_dual_point_distance": robot.bspline_dual_point_distance,
        "tube_radiuses": robot.tube_radiuses,
        "clamp_stl": robot.clamp_stl,
        "ee_stl": robot.ee_stl,
        "actuators": robot.actuators,
        "init_state": robot.init_state,
        "cup_radius": robot.cup_radius,
        "cup_height": robot.cup_height,
        "ball_radius": robot.ball_radius,
        "string_length": robot.string_length,
        "string_radius": robot.string_radius,
        "collision_mode": robot.collision_mode,
        "articulation_props": robot.spawn.articulation_props,
        "rigid_props": robot.spawn.rigid_props,
    }

@configclass
class LynxActionsCfg:
    arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
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
class LynxBallInCup5HzControlEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = LynxBallInCupSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        # Physics runs at 60Hz (dt=1/60)
        self.sim.dt = 1.0 / 60.0
        
        # Control frequency is 5Hz (dt=0.2s)
        # Decimation = Control DT / Sim DT = 0.2 / (1/60) = 12
        self.decimation = 12
        
        self.sim.render_interval = 1

        # Action settings
        self.actions = LynxActionsCfg()
        
        # Observation settings
        self.observations = LynxObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = LynxBallInCup5HzControlEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Get number of joints from the robot asset
    num_joints = env.scene["robot"].num_joints

    # Velocity limits (rad/s)
    velocity_limits = torch.tensor([np.radians(20.0)] * num_joints, device=env.device)
    
    # Control DT (5Hz)
    control_dt = 0.2
    
    # Max delta position per control step
    max_delta_pos = velocity_limits * control_dt

    print(f"[INFO]: Control Frequency: {1.0/control_dt} Hz (dt={control_dt}s)")
    print(f"[INFO]: Velocity Limits: {velocity_limits}")
    print(f"[INFO]: Max Delta Pos per step: {max_delta_pos}")

    # Simulate
    step_idx = 0
    while simulation_app.is_running():
        # Get current position from sim
        current_pos = env.scene["robot"].data.joint_pos
        
        # Generate a random desired position (absolute) for the first 6 joints
        # For testing, let's oscillate between -0.5 and 0.5 rad
        desired_pos_arm = 0.5 * torch.sin(torch.ones((env.num_envs, 6), device=env.device) * step_idx * 0.5)
        
        # Current position of the arm joints
        current_pos_arm = current_pos[:, :6]
        
        # Calculate required delta for arm joints
        delta_pos_arm = desired_pos_arm - current_pos_arm
        
        # Clip delta based on velocity limits
        clipped_delta_arm = torch.clamp(delta_pos_arm, -max_delta_pos[:6], max_delta_pos[:6])
        
        # New target position for arm joints
        target_pos_arm = current_pos_arm + clipped_delta_arm
        
        # The JointPositionAction expects targets relative to default or absolute depending on config.
        # In LynxActionsCfg, use_default_offset=True, so env.step(action) adds action to default_joint_pos.
        action = target_pos_arm - env.scene["robot"].data.default_joint_pos[:, :6]
        
        # Step environment
        env.step(action)
        
        # Logging
        if step_idx % 5 == 0: # Every 1 second at 5Hz
            applied_efforts = env.scene["robot"].data.applied_torque
            print(f"--- Step {step_idx} ---")
            print(f"Desired Pos (Arm): {desired_pos_arm[0].cpu().numpy()}")
            print(f"Current Pos (All): {current_pos[0].cpu().numpy()}")
            print(f"Target Pos (Arm, clipped): {target_pos_arm[0].cpu().numpy()}")
            print(f"Applied Torque: {applied_efforts[0].cpu().numpy()}")
        
        step_idx += 1

    print("[INFO]: Simulation finished.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
