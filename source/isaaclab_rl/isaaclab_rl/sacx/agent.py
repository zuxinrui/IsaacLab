import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256]):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class SACXAgent:
    """SAC-X Agent with multi-critic and task-conditioned actor.
    
    This implementation uses a shared actor trunk and multiple critic heads.
    """
    def __init__(self, obs_dim: int, action_dim: int, num_tasks: int, 
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, 
                 device: str = "cuda"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_tasks = num_tasks
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Actor: pi(a | s, z)
        # Input: obs + task_one_hot
        self.actor = MLP(obs_dim + num_tasks, action_dim * 2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critics: Q_k(s, a) for each task k
        # We use 2 critics per task for clipped double-Q learning
        self.critics_1 = nn.ModuleList([MLP(obs_dim + action_dim, 1) for _ in range(num_tasks)]).to(device)
        self.critics_2 = nn.ModuleList([MLP(obs_dim + action_dim, 1) for _ in range(num_tasks)]).to(device)
        
        self.target_critics_1 = nn.ModuleList([MLP(obs_dim + action_dim, 1) for _ in range(num_tasks)]).to(device)
        self.target_critics_2 = nn.ModuleList([MLP(obs_dim + action_dim, 1) for _ in range(num_tasks)]).to(device)
        
        for i in range(num_tasks):
            self.target_critics_1[i].load_state_dict(self.critics_1[i].state_dict())
            self.target_critics_2[i].load_state_dict(self.critics_2[i].state_dict())
            
        self.critic_optimizer = optim.Adam(list(self.critics_1.parameters()) + list(self.critics_2.parameters()), lr=lr)
        
        # Entropy temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim

    def get_action(self, obs: torch.Tensor, z: torch.Tensor, sample: bool = True):
        # z is (num_envs,)
        z_one_hot = torch.nn.functional.one_hot(z, num_classes=self.num_tasks).float()
        actor_input = torch.cat([obs, z_one_hot], dim=-1)
        
        mu_logstd = self.actor(actor_input)
        mu, log_std = mu_logstd.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mu, std)
        if sample:
            u = dist.rsample()
        else:
            u = mu
            
        action = torch.tanh(u)
        
        # Log prob for entropy
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob

    def update(self, batch: Dict[str, torch.Tensor]):
        obs = batch["obs"]
        actions = batch["actions"]
        next_obs = batch["next_obs"]
        rewards = batch["rewards"] # (batch_size, num_tasks)
        dones = batch["dones"]
        z_exec = batch["z_exec"] # (batch_size, 1)
        
        batch_size = obs.shape[0]
        
        # 1. Update Critics
        with torch.no_grad():
            # Sample next actions for all tasks (using current policy)
            # For simplicity, we use the same next action for all task updates in this batch
            next_z = torch.randint(0, self.num_tasks, (batch_size,), device=self.device)
            next_actions, next_log_probs = self.get_action(next_obs, next_z)
            
            alpha = self.log_alpha.exp()
            
            target_q_values = []
            for k in range(self.num_tasks):
                q1_target = self.target_critics_1[k](torch.cat([next_obs, next_actions], dim=-1))
                q2_target = self.target_critics_2[k](torch.cat([next_obs, next_actions], dim=-1))
                q_target = torch.min(q1_target, q2_target) - alpha * next_log_probs
                
                # r_k is rewards[:, k]
                target_q = rewards[:, k:k+1] + (1 - dones) * self.gamma * q_target
                target_q_values.append(target_q)
        
        # Compute current Q values and loss for all tasks
        critic_loss = torch.tensor(0.0, device=self.device)
        for k in range(self.num_tasks):
            q1 = self.critics_1[k](torch.cat([obs, actions], dim=-1))
            q2 = self.critics_2[k](torch.cat([obs, actions], dim=-1))
            critic_loss += torch.nn.functional.mse_loss(q1, target_q_values[k])
            critic_loss += torch.nn.functional.mse_loss(q2, target_q_values[k])
            
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 2. Update Actor
        # Randomly pick a task to optimize the actor for this batch
        task_to_opt = np.random.randint(0, self.num_tasks)
        
        # Sample new actions
        z_opt = torch.full((batch_size,), task_to_opt, dtype=torch.long, device=self.device)
        new_actions, log_probs = self.get_action(obs, z_opt)
        
        q1_new = self.critics_1[task_to_opt](torch.cat([obs, new_actions], dim=-1))
        q2_new = self.critics_2[task_to_opt](torch.cat([obs, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (alpha.detach() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 3. Update Alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # 4. Soft Update Targets
        for k in range(self.num_tasks):
            for param, target_param in zip(self.critics_1[k].parameters(), self.target_critics_1[k].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critics_2[k].parameters(), self.target_critics_2[k].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item()
        }
