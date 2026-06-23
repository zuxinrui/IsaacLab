"""Collect expert demonstrations from IsaacLab best policy.

Runs the trained policy in IsaacLab's physics simulation and saves
(obs_54dim, action_6dim, reward, done) tuples for behavior cloning.

Usage (from IsaacLab-2.0 directory):
    python scripts/lynx/collect_expert_demos.py \
        --num_envs 8 \
        --n_episodes 100 \
        --device cuda:1
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect expert demonstrations.")
parser.add_argument("--task", type=str, default="Lynx-Ball-In-Cup-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--n_episodes", type=int, default=100,
                    help="Total episodes to collect")
parser.add_argument("--checkpoint", type=str,
                    default="/home/zuxinrui/IsaacLab-2.0/logs/rsl_rl/lynx_ball_in_cup/"
                            "2026-03-23_23-10-54/best_model_1021_reward_20_10.pt")
parser.add_argument("--output_dir", type=str,
                    default="/home/zuxinrui/LynxRobotics/.results/ball_in_cup_experiments/"
                            "isaaclab_expert_demos")
parser.add_argument("--seed", type=int, default=42)

# AppLauncher args
AppLauncher.add_app_launcher_args(parser)
# parse known args, pass rest to Hydra
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- After AppLauncher init ----
import os
import json
import numpy as np
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlBaseRunnerCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Configure env
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=args_cli.device
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=args_cli.device)

    num_envs = args_cli.num_envs
    n_episodes_target = args_cli.n_episodes

    # Storage
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []

    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    episodes_collected = 0

    # Per-env episode tracking
    env_ep_rewards = [0.0] * num_envs
    env_ep_lengths = [0] * num_envs

    print(f"\n[INFO] Collecting {n_episodes_target} episodes with {num_envs} envs...")

    obs = env.get_observations()  # dict with "policy" key -> (N, 54)

    with torch.inference_mode():
        while episodes_collected < n_episodes_target and simulation_app.is_running():
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs

            # Get actions from policy
            actions = policy(obs_tensor)

            # Step environment
            next_obs, rewards, dones, infos = env.step(actions)

            # Store data (move to CPU numpy)
            obs_np = obs_tensor.cpu().numpy()
            act_np = actions.cpu().numpy()
            rew_np = rewards.cpu().numpy()
            done_np = dones.cpu().numpy()

            all_obs.append(obs_np)
            all_actions.append(act_np)
            all_rewards.append(rew_np)
            all_dones.append(done_np)

            # Track per-env episodes
            for i in range(num_envs):
                env_ep_rewards[i] += rew_np[i]
                env_ep_lengths[i] += 1

                if done_np[i]:
                    episode_rewards.append(env_ep_rewards[i])
                    episode_lengths.append(env_ep_lengths[i])
                    # Check success from info
                    if isinstance(infos, dict) and "log" in infos:
                        success = infos["log"].get("is_success", {})
                        if hasattr(success, '__getitem__'):
                            try:
                                episode_successes.append(float(success[i]))
                            except (IndexError, TypeError):
                                episode_successes.append(0.0)
                        else:
                            episode_successes.append(0.0)
                    else:
                        episode_successes.append(0.0)

                    episodes_collected += 1
                    env_ep_rewards[i] = 0.0
                    env_ep_lengths[i] = 0

                    if episodes_collected % 10 == 0:
                        recent_rewards = episode_rewards[-10:]
                        print(f"  Episodes: {episodes_collected}/{n_episodes_target}, "
                              f"recent avg reward: {np.mean(recent_rewards):.2f}")

            # Reset policy for done envs
            policy.reset(dones)
            obs = next_obs

    # Concatenate all data
    all_obs = np.concatenate(all_obs, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_dones = np.concatenate(all_dones, axis=0)

    # Save
    os.makedirs(args_cli.output_dir, exist_ok=True)

    np.save(os.path.join(args_cli.output_dir, "observations.npy"), all_obs)
    np.save(os.path.join(args_cli.output_dir, "actions.npy"), all_actions)
    np.save(os.path.join(args_cli.output_dir, "rewards.npy"), all_rewards)
    np.save(os.path.join(args_cli.output_dir, "dones.npy"), all_dones)

    # Save metadata
    metadata = {
        "n_episodes": episodes_collected,
        "n_transitions": len(all_obs),
        "obs_dim": int(all_obs.shape[1]),
        "action_dim": int(all_actions.shape[1]),
        "action_scale": 0.1745,  # IsaacLab RelativeJointPositionAction scale
        "control_dt": 0.2,       # 12 decimation * 1/60 sim_dt
        "max_episode_steps": 50,
        "avg_episode_reward": float(np.mean(episode_rewards)),
        "std_episode_reward": float(np.std(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(episode_successes)) if episode_successes else 0.0,
        "checkpoint": args_cli.checkpoint,
        "task": args_cli.task,
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
    }

    with open(os.path.join(args_cli.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Collected {episodes_collected} episodes ({len(all_obs)} transitions)")
    print(f"Obs shape: {all_obs.shape}, Action shape: {all_actions.shape}")
    print(f"Avg reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Avg length: {np.mean(episode_lengths):.1f}")
    print(f"Success rate: {np.mean(episode_successes)*100:.1f}%")
    print(f"Saved to: {args_cli.output_dir}")
    print(f"{'='*60}")

    env.close()


if __name__ == "__main__":
    main()
