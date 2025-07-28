# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

##
# USD Reference
##
_USD_PATH = "source/isaaclab_assets/data/Robots/Lynx/lynx-isaacsim-materials-urdf.usd"


##
# Configuration for the Lynx arm.
##


@configclass
class LynxArmCfg(ArticulationCfg):
    """Configuration for the Lynx arm."""

    spawn: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    )

    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Base position from lynx_orth.xml
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
        },
    )

    actuators: dict[str, ImplicitActuatorCfg] = {
        "lynx_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
            ],
            stiffness=100000.0,
            damping=10000.0,
        )
    }

    # end-effector link name
    end_effector_link_name: str = "ee_cylinder"
    # end-effector site name
    end_effector_site_name: str = "end_effector"

LYNX_CFG = LynxArmCfg()
LYNX_HD_CFG = LynxArmCfg()
