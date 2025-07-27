# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the Lynx robot with random joint control.

It illustrates how to setup a simple scene with the Lynx robot and apply random joint position commands
to its actuators.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from omni.isaac.kit import SimulationApp

import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg, ImplicitActuatorCfg
from isaaclab.actuators import actuator_pd
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
# isort: off
from isaaclab_assets import (
    FRANKA_PANDA_CFG,
    UR10_CFG,
    KINOVA_JACO2_N7S300_CFG,
    KINOVA_JACO2_N6S300_CFG,
    KINOVA_GEN3_N7_CFG,
    SAWYER_CFG,
)

##
# Scene configuration
##

class LynxSceneCfg(InteractiveSceneCfg):
    """Configuration for the Lynx robot scene."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # ground plane
        self.terrain = TerrainImporterCfg(
            prim_path="/World/defaultGroundPlane",
            terrain_type="plane",
            physics_material=RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        )

        # robot
        self.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",  # This prim_path must point to a prim with ArticulationRootAPI applied.
                                      # For the Lynx robot, ensure 'source/isaaclab_assets/data/Robots/Lynx/lynx.usd'
                                      # has ArticulationRootAPI applied to its root prim.  {ENV_REGEX_NS}
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    "joint_1": 0.0,
                    "joint_2": 0.0,
                    "joint_3": 0.0,
                    "joint_4": 0.0,
                    "joint_5": 0.0,
                    "joint_6": 0.0,
                },
                # pos=(0.0, 0.0, 0.8),
                ),
            spawn=UsdFileCfg(
                usd_path="source/isaaclab_assets/data/Robots/Lynx/lynx-isaacsim2-urdf.usd",  # {ISAAC_NUCLEUS_DIR} / source/isaaclab_assets/data/
                scale=(1.0, 1.0, 1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                ),
                # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #     enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
                # ),
                activate_contact_sensors=False,
            ),
            actuators={
                "lynx_arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint_[1-6]"],
                    velocity_limit=100.0,
                    effort_limit=87.0,
                    stiffness=800.0,
                    damping=40.0,
                ),
                # "joint2_joint": ImplicitActuatorCfg(
                #     joint_names_expr=["joint2_joint"],
                #     velocity_limit=100.0,
                #     effort_limit=87.0,
                #     stiffness=800.0,
                #     damping=40.0,
                # ),
                # "joint3_joint": ImplicitActuatorCfg(
                #     joint_names_expr=["joint3_joint"],
                #     velocity_limit=100.0,
                #     effort_limit=87.0,
                #     stiffness=800.0,
                #     damping=40.0,
                # ),
            },
        )

        # self.robot = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
        # self.robot.init_state.pos = (0.0, 0.0, 1.05)

        # lights
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )


##
# Main
##

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):

    # Acquire the robot
    robot = scene.articulations["robot"]

    while simulation_app.is_running():
        # If simulation is paused, then skip.
        if not sim.is_playing():
            continue
        # Else, perform simulation step
        # generate random joint position targets
        joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
        # clamp the targets within soft joint limits
        joint_pos_target.clamp_(robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1])
        # set the joint position targets
        robot.set_joint_position_target(joint_pos_target)
        # write data to simulation
        robot.write_data_to_sim()
        # step the simulation
        sim.step()


if __name__ == "__main__":

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # Load scene
    scene = InteractiveScene(LynxSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene)

    # Close the simulator
    simulation_app.close()