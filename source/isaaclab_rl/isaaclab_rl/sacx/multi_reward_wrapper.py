import gymnasium as gym
import torch
from typing import Dict, List, Any

class MultiRewardWrapper(gym.Wrapper):
    """Wrapper to extract multiple rewards from the environment and store them in the info dict.
    
    This wrapper assumes the environment is an Isaac Lab ManagerBasedRLEnv which has a 
    reward_manager that computes individual reward terms.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Check if it's a manager-based env to access reward terms
        if not hasattr(self.unwrapped, "reward_manager"):
            raise ValueError("MultiRewardWrapper requires an environment with a 'reward_manager'.")
        
        self.reward_terms = self.unwrapped.reward_manager._reward_terms
        self.reward_names = list(self.reward_terms.keys())
        print(f"[MultiRewardWrapper] Detected reward terms: {self.reward_names}")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract individual reward terms from the reward manager
        # In Isaac Lab, reward_manager.compute() returns the total reward, 
        # but the individual terms are stored in reward_manager._episode_sums or can be recomputed.
        # A more direct way is to look at reward_manager.get_term_values() if available,
        # or access the computed values from the last step.
        
        rewards_dict = {}
        for name, term in self.reward_terms.items():
            # term.value contains the computed reward for the last step (batched)
            rewards_dict[name] = term.value
            
        info["reward_dict"] = rewards_dict
        return obs, reward, terminated, truncated, info
