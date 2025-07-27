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

import omni.usd
from pxr import UsdPhysics
import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def get_joint_names_from_usd(usd_path):
    stage_utils.open_stage(usd_path)
    # Reinitialize the simulation
    app = omni.kit.app.get_app_interface()
    # Run simulation
    with contextlib.suppress(KeyboardInterrupt):
        while app.is_running():
            # perform step
            app.update()

if __name__ == "__main__":
    usd_file_path = "source/isaaclab_assets/data/Robots/Lynx/lynx-urdf.usd"
    ur_usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd"
    jet_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd"
    get_joint_names_from_usd(jet_path)
    simulation_app.close()