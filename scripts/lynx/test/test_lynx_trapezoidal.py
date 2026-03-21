# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the TrapezoidalJointPositionAction with the modular Lynx robot."""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Test script for Lynx Trapezoidal Interpolation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import ActionTermCfg, SceneEntityCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.trapezoidal_joint_pos_action import TrapezoidalJointPositionAction, TrapezoidalJointPositionActionCfg

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
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1
    )
    # Increase stiffness and damping for more rigid behavior
    # robot.actuators["lynx_arm"].stiffness = 5000000.0
    # robot.actuators["lynx_arm"].damping = 1000000.0
    
    # Ensure the spawn function is set and has the robot_cfg
    robot.spawn.func = LynxUsdConstructor.spawn
    # Use a dictionary to avoid recursion in configclass validation
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
    }

@configclass
class LynxActionsCfg:
    arm_action: TrapezoidalJointPositionActionCfg = TrapezoidalJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_.*"],
        scale=1.0,
        use_default_offset=True,
        max_velocity=math.radians(20.0),
        max_acceleration=math.radians(100.0),
    )

@configclass
class LynxObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_obs)
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class LynxTrapezoidalTestEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = LynxSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        self.sim.dt = 1.0 / 120.0  # 120 Hz
        # self.sim.gravity = (0.0, 0.0, -9.81)
        self.decimation = 24       # 5 Hz policy frequency
        self.sim.render_interval = 2 # 60 Hz rendering/control update (approx)

        # Action settings
        self.actions = LynxActionsCfg()
        
        # Observation settings
        self.observations = LynxObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = LynxTrapezoidalTestEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Play the simulator
    print("[INFO]: Simulation setup complete. Running trapezoidal interpolation test...")

    # Simulation loop
    step_count = 0
    steps_per_target = 100 # 100 * (1/5) = 20 seconds per target
    num_joints = 6

    # Simulate
    while simulation_app.is_running():
        range_rad = math.radians(10.0)
        current_target = (torch.rand((env.num_envs, num_joints), device=env.device) * 2.0 - 1.0) * range_rad
        print(f"[INFO]: Commanding new random target (degrees): {torch.rad2deg(current_target)}")
        
        # Step environment (this calls process_action and apply_action)
        env.step(current_target)
        
        step_count += 1

    print("[INFO]: Simulation finished.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
