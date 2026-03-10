# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the LynxUsdConstructor to spawn a robot."""

import argparse

from isaaclab.app import AppLauncher

# add argparser
parser = argparse.ArgumentParser(description="Test script for LynxUsdConstructor.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of the imports."""
import torch

import isaaclab.sim as sim_utils
from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view((2.5, 2.5, 2.5), (0.0, 0.0, 0.0))

    # Design scene
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create Lynx robot configuration
    robot_cfg = LynxRobotCfg(
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1
    )

    # Spawn the robot
    print("[INFO]: Spawning Lynx robot...")
    LynxUsdConstructor.spawn("/World/Lynx", robot_cfg)
    print("[INFO]: Robot spawned successfully.")

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Simulation setup complete. Running for visual inspection...")

    # Simulate
    while simulation_app.is_running():
        # perform step
        # sim.step(render=True)
        # Just render to keep it static
        sim.render()

    print("[INFO]: Simulation finished.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
