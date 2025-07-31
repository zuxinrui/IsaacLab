<<<<<<< HEAD
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
=======
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
>>>>>>> 9cd69dd5d80 (Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab (#2968))
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
<<<<<<< HEAD
=======
from .forge_env import ForgeEnv
from .forge_env_cfg import ForgeTaskGearMeshCfg, ForgeTaskNutThreadCfg, ForgeTaskPegInsertCfg
>>>>>>> 9cd69dd5d80 (Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab (#2968))

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Forge-PegInsert-Direct-v0",
<<<<<<< HEAD
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskPegInsertCfg",
=======
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskPegInsertCfg,
>>>>>>> 9cd69dd5d80 (Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab (#2968))
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-GearMesh-Direct-v0",
<<<<<<< HEAD
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskGearMeshCfg",
=======
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskGearMeshCfg,
>>>>>>> 9cd69dd5d80 (Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab (#2968))
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Forge-NutThread-Direct-v0",
<<<<<<< HEAD
    entry_point=f"{__name__}.forge_env:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.forge_env_cfg:ForgeTaskNutThreadCfg",
=======
    entry_point="isaaclab_tasks.direct.forge:ForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ForgeTaskNutThreadCfg,
>>>>>>> 9cd69dd5d80 (Adds FORGE tasks for contact-rich manipulation with force sensing to IsaacLab (#2968))
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg_nut_thread.yaml",
    },
)
