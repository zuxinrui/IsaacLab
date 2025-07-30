import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Reach-Lynx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:LynxReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LynxReachRslRlOnPolicyRunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Lynx-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_reduced_obs_env_cfg:ReducedLynxReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LynxReachRslRlOnPolicyRunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-Lynx-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_reduced_obs_env_cfg:ReducedLoopLynxReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LynxReachRslRlOnPolicyRunnerCfg",
    },
)