# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the JointPositionAction with the Franka Panda robot."""

import argparse
import torch
import copy

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Test script for Franka Panda Position Control.")
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

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

def dummy_obs(env):
    return torch.zeros((env.num_envs, 1), device=env.device)

@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
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
    robot = copy.deepcopy(FRANKA_PANDA_CFG)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"

@configclass
class FrankaActionsCfg:
    arm_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=1.0,
        use_default_offset=True,
    )

@configclass
class FrankaObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        dummy = ObservationTermCfg(func=dummy_obs)
    
    policy: PolicyCfg = PolicyCfg()

@configclass
class FrankaPosControlTestEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = FrankaSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = 1 
        self.decimation = 1

        # Action settings
        self.actions = FrankaActionsCfg()
        
        # Observation settings
        self.observations = FrankaObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = FrankaPosControlTestEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # Play the simulator
    print("[INFO]: Simulation setup complete. Running Franka position control test with random actions...")

    # Simulation loop
    num_joints = env.action_manager.total_action_dim
    print(f"[INFO]: Number of joints in action: {num_joints}")
    
    # Initialize target position
    target_pos = torch.zeros((env.num_envs, num_joints), device=env.device)

    # Simulate
    step_idx = 0
    while simulation_app.is_running():
        # Periodically update target position with random values
        if step_idx % 120 == 0:
            # Generate random joint positions within a small range (-0.5 to 0.5 radians)
            target_pos = (torch.rand((env.num_envs, num_joints), device=env.device) - 0.5) * 1.0
            print(f"Step: {step_idx}, New Target Joint Pos (relative): {target_pos}")

        # Step environment
        env.step(target_pos)
        
        # Optional: print current joint positions to see if they reach the target
        if step_idx % 60 == 0:
            current_pos = env.scene["robot"].data.joint_pos
            print(f"Step: {step_idx}, Current Joint Pos: {current_pos}")
        
        step_idx += 1

    print("[INFO]: Simulation finished.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
