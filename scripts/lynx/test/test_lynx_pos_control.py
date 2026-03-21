# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the JointPositionAction with the modular Lynx robot."""

import argparse
import math
import torch

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Test script for Lynx Simple Position Control.")
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
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1,
    )
    # Set stiffness and damping for the joints
    # These are the Kp and Kd values for the joint position control
    # robot.actuators["lynx_arm"].stiffness = 80.0
    # robot.actuators["lynx_arm"].damping = 4.0
    # robot.actuators["lynx_arm"].effort_limit = 87.0
    # robot.actuators["lynx_arm"].friction = 0.0
    
    # Strictly match Franka's articulation properties
    robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, 
        solver_position_iteration_count=32, 
        solver_velocity_iteration_count=4
    )
    robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
        linear_damping=0.5,
        angular_damping=0.5,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
    )
    
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
class LynxPosControlTestEnvCfg(ManagerBasedEnvCfg):
    def __post_init__(self):
        # Scene settings
        self.scene = LynxSceneCfg(num_envs=1, env_spacing=2.5)

        # Simulation settings
        self.sim.dt = 1.0 / 60.0  # 120 Hz
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.2,
            restitution=0.0,
        )
        # self.sim.gravity = (0.0, 0.0, -9.81)
        self.decimation = 1        # 120 Hz policy frequency for direct control
        self.sim.render_interval = 1 

        # Action settings
        self.actions = LynxActionsCfg()
        
        # Observation settings
        self.observations = LynxObservationsCfg()

def main():
    """Main function."""
    # Create environment configuration
    env_cfg = LynxPosControlTestEnvCfg()
    
    # Create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    print(f"[INFO]: Resolved action dimension: {env.action_manager.total_action_dim}")
    print(f"[INFO]: Default joint positions: {env.scene['robot'].data.default_joint_pos}")

    # Play the simulator
    print("[INFO]: Simulation setup complete. Running staged Lynx position-control validation...")

    # Simulation loop
    num_joints = env.action_manager.total_action_dim
    
    # Initialize target position
    target_pos = torch.zeros((env.num_envs, num_joints), device=env.device)

    # Simulate
    step_idx = 0
    while simulation_app.is_running():
        # Staged test: hold, then per-joint bounded excitations, then mild random commands.
        phase = (step_idx // 240) % (num_joints + 2)
        if phase == 0:
            target_pos.zero_()
        elif 1 <= phase <= num_joints:
            target_pos.zero_()
            joint_id = phase - 1
            target_pos[:, joint_id] = 0.02 * math.sin(step_idx * env_cfg.sim.dt)
        else:
            target_pos = (torch.rand((env.num_envs, num_joints), device=env.device) - 0.5) * 0.02

        if step_idx % 120 == 0:
            print(f"Step: {step_idx}, Target Joint Pos (relative): {target_pos}")

        # Step environment
        env.step(target_pos)
        
        # Optional: print current joint positions to see if they reach the target
        if step_idx % 60 == 0:
            current_pos = env.scene["robot"].data.joint_pos
            current_target = env.scene["robot"].data.joint_pos_target
            print(f"Step: {step_idx}, Current Joint Pos: {current_pos}")
            print(f"Step: {step_idx}, Current Joint Pos Target: {current_target}")
        
        step_idx += 1

    print("[INFO]: Simulation finished.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
