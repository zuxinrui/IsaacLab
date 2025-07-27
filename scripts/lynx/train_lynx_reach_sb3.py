import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--log_interval", type=int, default=100_000, help="Log data every n timesteps.")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--keep_all_info",
    action="store_true",
    default=False,
    help="Use a slower SB3 wrapper but keep all the extra training info.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """
    A small helper to stop training and
    cleanup progress bar properly on ctrl+c
    """
    import gc

    tqdm_objects = [obj for obj in gc.get_objects() if "tqdm" in type(obj).__name__]
    for tqdm_object in tqdm_objects:
        if "tqdm_rich" in type(tqdm_object).__name__:
            tqdm_object.close()
    raise KeyboardInterrupt


# disable KeyboardInterrupt override
signal.signal(signal.SIGINT, cleanup_pbar)


import gymnasium as gym
from stable_baselines3 import PPO
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from isaaclab_tasks.manager_based.classic.lynx_reach.lynx_reach_env_cfg import LynxReachEnvCfg

def main():
    """Main function to train the Lynx Reach SB3 agent."""

    # Initialize the simulation context
    # Instantiate the environment
    env_cfg = LynxReachEnvCfg()
    env_cfg.sim = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device) # Pass the simulation configuration to the environment configuration
    env = ManagerBasedRLEnv(env_cfg)

    # Create a vectorized environment for SB3
    vec_env = Sb3VecEnvWrapper(env)

    # Define the SB3 model (PPO)
    model = PPO("MlpPolicy", vec_env, verbose=1, device=args_cli.device)

    # Train the agent
    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    # Save the trained model
    model.save("lynx_reach_ppo")

    # Close the environment
    vec_env.close()

if __name__ == "__main__":
    main()