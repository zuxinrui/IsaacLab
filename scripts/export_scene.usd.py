# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to export an Isaac Lab environment scene to a USD file.
This allows editing the scene in Isaac Sim and then using it back in Isaac Lab.
"""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Export an Isaac Lab environment scene to a USD file.")
parser.add_argument("--task", type=str, default="Isaac-Push-Cube-Lynx-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--output", type=str, default="exported_scene.usd", help="Path to the output USD file.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

def main():
    # create environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, device="cuda")
    
    # reset environment to ensure everything is spawned
    env.reset()
    
    # Get the stage
    import omni.usd
    from omni.isaac.core.utils.stage import save_stage
    stage = omni.usd.get_context().get_stage()
    
    # Export the stage to a USD file
    output_path = os.path.abspath(args_cli.output)
    print(f"[INFO] Exporting scene to: {output_path}")
    
    # We want to save the current stage. 
    # Note: In Isaac Lab, the environments are usually under /World/envs/env_0, etc.
    # If you want to edit the whole scene, saving the whole stage is fine.
    save_stage(output_path)
    
    print(f"[INFO] Successfully exported scene to {output_path}")
    
    # close the environment
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
