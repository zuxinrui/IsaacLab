import torch
import numpy as np

class SACXScheduler:
    """Scheduler for SAC-X that decides which intention to execute.
    
    Supports uniform random sampling and segment-based switching.
    """
    def __init__(self, num_envs: int, num_tasks: int, segment_length: int = 32, device: str = "cuda"):
        self.num_envs = num_envs
        self.num_tasks = num_tasks
        self.segment_length = segment_length
        self.device = device
        
        # Current intention for each environment
        self.current_z = torch.randint(0, num_tasks, (num_envs,), device=device)
        # Steps since last switch for each environment
        self.steps_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def sample(self) -> torch.Tensor:
        """Returns the current intention for each environment."""
        return self.current_z

    def update(self, dones: torch.Tensor):
        """Updates the scheduler state.
        
        Switches intention if segment length is reached or if environment is reset.
        """
        self.steps_count += 1
        
        # Switch if segment length reached OR environment reset
        # Ensure dones is a 1D tensor
        if dones.dim() > 1:
            dones = dones.squeeze(-1)
            
        switch_mask = (self.steps_count >= self.segment_length) | dones.bool()
        
        if switch_mask.any():
            num_switches = int(switch_mask.sum().item())
            new_z = torch.randint(0, self.num_tasks, (num_switches,), device=self.device)
            self.current_z[switch_mask] = new_z
            self.steps_count[switch_mask] = 0
            
        return self.current_z
