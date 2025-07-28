import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Reach-Lynx-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lynx_reach_env_cfg:LynxReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LynxReachRslRlOnPolicyRunnerCfg",
    },
)