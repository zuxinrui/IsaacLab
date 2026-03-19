import argparse
import os
import torch
import numpy as np
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train SAC-X on Isaac Lab environments.")
parser.add_argument("--task", type=str, default="Isaac-Push-Cube-Lynx-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the experiment.")
parser.add_argument("--total_steps", type=int, default=1000000, help="Total number of steps to train.")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training.")
parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size.")
parser.add_argument("--segment_length", type=int, default=32, help="SAC-X segment length.")
parser.add_argument("--update_every", type=int, default=1, help="Update every N steps.")
parser.add_argument("--num_updates", type=int, default=1, help="Number of updates per step.")
parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps before training.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv

from isaaclab_rl.sacx.multi_reward_wrapper import MultiRewardWrapper
from isaaclab_rl.sacx.replay import SACXReplayBuffer
from isaaclab_rl.sacx.scheduler import SACXScheduler
from isaaclab_rl.sacx.agent import SACXAgent

def train():
    # Create environment
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, render_mode="rgb_array" if args_cli.headless else "human")
    
    # Wrap with multi-reward extractor
    env = MultiRewardWrapper(env)
    
    # Isaac Lab environments have (num_envs, obs_dim)
    obs_dim = env.observation_space.shape[1]
    action_dim = env.action_space.shape[1]
    num_tasks = len(env.reward_names)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Obs dim: {obs_dim}, Action dim: {action_dim}, Num tasks: {num_tasks}")
    
    # Initialize SAC-X components
    agent = SACXAgent(obs_dim, action_dim, num_tasks, device=device)
    replay_buffer = SACXReplayBuffer(args_cli.buffer_size, args_cli.num_envs, obs_dim, action_dim, device=device)
    scheduler = SACXScheduler(args_cli.num_envs, num_tasks, segment_length=args_cli.segment_length, device=device)
    
    obs, _ = env.reset()
    
    for step in range(args_cli.total_steps):
        # 1. Sample intention from scheduler
        z = scheduler.sample()
        
        # 2. Get action from agent
        with torch.no_grad():
            action, _ = agent.get_action(obs, z)
            
        # 3. Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        # In Isaac Lab, terminated and truncated are tensors of shape (num_envs,)
        dones = terminated | truncated
        
        # 4. Add to replay buffer
        # info["reward_dict"] contains rewards for all tasks
        replay_buffer.add(obs, action, next_obs, dones, z, info["reward_dict"])
        
        obs = next_obs
        
        # 5. Update scheduler
        scheduler.update(dones)
        
        # 6. Training update
        if step > args_cli.warmup_steps and step % args_cli.update_every == 0:
            for _ in range(args_cli.num_updates):
                batch = replay_buffer.sample(args_cli.batch_size)
                train_info = agent.update(batch)
                
                if step % 100 == 0:
                    print(f"Step {step}: Critic Loss: {train_info['critic_loss']:.4f}, Actor Loss: {train_info['actor_loss']:.4f}")

    env.close()

if __name__ == "__main__":
    train()
    simulation_app.close()
