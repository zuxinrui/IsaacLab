import torch
import numpy as np
from typing import Dict, Any, List, Optional

class SACXReplayBuffer:
    """Replay buffer for SAC-X that stores multiple rewards and intention IDs.
    
    This buffer is designed for batched environments (Isaac Lab).
    """
    def __init__(self, capacity: int, num_envs: int, obs_dim: int, action_dim: int, device: str = "cuda"):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        
        # Pre-allocate buffers
        self.obs = torch.zeros((capacity, num_envs, obs_dim), device=device)
        self.actions = torch.zeros((capacity, num_envs, action_dim), device=device)
        self.next_obs = torch.zeros((capacity, num_envs, obs_dim), device=device)
        self.dones = torch.zeros((capacity, num_envs, 1), device=device)
        self.z_exec = torch.zeros((capacity, num_envs, 1), dtype=torch.long, device=device)
        
        # Reward dict will be stored as a tensor of shape (capacity, num_envs, num_tasks)
        self.reward_tensor: Optional[torch.Tensor] = None
        self.task_names: Optional[List[str]] = None
        
        self.ptr = 0
        self.size = 0
        self.full = False

    def add(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, 
            dones: torch.Tensor, z_exec: torch.Tensor, reward_dict: Dict[str, torch.Tensor]):
        
        # Initialize reward tensor if not already done
        if self.reward_tensor is None or self.task_names is None:
            self.task_names = sorted(list(reward_dict.keys()))
            num_tasks = len(self.task_names)
            self.reward_tensor = torch.zeros((self.capacity, self.num_envs, num_tasks), device=self.device)
            print(f"[SACXReplayBuffer] Initialized with tasks: {self.task_names}")

        # Store transitions
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = dones.unsqueeze(-1) if dones.dim() == 1 else dones
        self.z_exec[self.ptr] = z_exec.unsqueeze(-1) if z_exec.dim() == 1 else z_exec
        
        # Store rewards for all tasks
        for i, name in enumerate(self.task_names):
            self.reward_tensor[self.ptr, :, i] = reward_dict[name]

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        """Sample a batch of transitions.
        
        Returns a dictionary of tensors.
        """
        if self.reward_tensor is None:
            raise ValueError("Replay buffer is empty or not initialized.")
            
        # Flatten the buffer for sampling
        # (size * num_envs) total transitions
        total_transitions = self.size * self.num_envs
        indices = torch.randint(0, total_transitions, (batch_size,), device=self.device)
        
        # Convert flat indices to (step, env) indices
        step_indices = indices // self.num_envs
        env_indices = indices % self.num_envs
        
        batch = {
            "obs": self.obs[step_indices, env_indices],
            "actions": self.actions[step_indices, env_indices],
            "next_obs": self.next_obs[step_indices, env_indices],
            "dones": self.dones[step_indices, env_indices],
            "z_exec": self.z_exec[step_indices, env_indices],
            "rewards": self.reward_tensor[step_indices, env_indices]
        }
        
        return batch
