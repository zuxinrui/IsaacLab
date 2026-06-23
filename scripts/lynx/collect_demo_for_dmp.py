"""Collect demonstration data from a legacy 106-dim ball-in-cup policy in IsaacLab 2.0.

Mirror of ``LynxLab/scripts/collect_demo_for_dmp.py`` but targeting the legacy
106-dim observation layout (``Isaac-Ball-In-Cup-Lynx-Play-V0Legacy``) so the
earliest PPO checkpoints can be replayed on the original 12-segment string
geometry they were trained against.

The recorded PD targets (``joint_pos_before + clip(raw_action * scale, -1, 1)``)
match what ``RelativeJointPositionAction`` actually sends to the physics
controller, so downstream DMP + PoWER fitting sees the exact command trajectory.

Usage:
    # Headless collection
    python scripts/lynx/collect_demo_for_dmp.py --headless

    # With video
    python scripts/lynx/collect_demo_for_dmp.py --headless --video

    # Custom checkpoint (must be a pre-56783a8b / 106-dim checkpoint)
    python scripts/lynx/collect_demo_for_dmp.py --headless \
        --checkpoint /home/zuxinrui/IsaacLab-2.0/logs/rsl_rl/lynx_ball_in_cup/\
2026-03-21_10-42-55/model_700.pt
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

# -- argparse (BEFORE AppLauncher so hydra args pass through) --
parser = argparse.ArgumentParser(description="Collect legacy 106-dim demo data for DMP+PoWER.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record video of the policy running.")
parser.add_argument("--video_length", type=int, default=200,
                    help="Video length in env steps (>= full_episode_steps to show whole episode).")
parser.add_argument("--task", type=str, default="Isaac-Ball-In-Cup-Lynx-Play-V0Legacy",
                    help="IsaacLab 2.0 legacy 106-dim task id.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", type=str,
                    default="/home/zuxinrui/IsaacLab-2.0/logs/rsl_rl/lynx_ball_in_cup/"
                            "2026-03-21_10-42-55/model_700.pt",
                    help="Path to a legacy 106-dim .pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel envs (1 for a clean demo trace).")
parser.add_argument("--n_episodes", type=int, default=10,
                    help="Number of episodes to roll out; the best (highest total reward) is picked.")
parser.add_argument("--demo_steps", type=int, default=15,
                    help="Leading steps kept as the DMP demo (15 = 3s at 5Hz).")
parser.add_argument("--full_episode_steps", type=int, default=50,
                    help="Steps to actually run per episode so reward ranking is meaningful.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--output_dir", type=str,
    default="/home/zuxinrui/LynxRobotics/.results/ball_in_cup_experiments/dmp_demo_data_legacy",
    help="Output directory for demo data.",
)
parser.add_argument(
    "--camera_zoom", type=float, default=1.0,
    help="Multiplier for camera-to-target distance (1.0 = default, 0.33 = 3x zoom in).",
)
parser.add_argument(
    "--camera_eye", type=float, nargs=3, default=None,
    help="Override viewer camera eye position (x y z, in meters).",
)
parser.add_argument(
    "--camera_lookat", type=float, nargs=3, default=None,
    help="Override viewer camera lookat target (x y z, in meters).",
)
parser.add_argument(
    "--action_repeat", type=int, default=1,
    help="Repeat each policy action over N env.steps. Use with auto-scaled "
         "decimation to render video faster than the trained 5Hz policy. "
         "E.g. action_repeat=6 -> decimation 12->2 -> 30 FPS video while the "
         "policy still decides at 5Hz.",
)
parser.add_argument(
    "--log_high_freq", action="store_true", default=False,
    help="In addition to the 5Hz (per-policy-step) joint_pos log, also capture "
         "joint_pos after every env.step inside the action_repeat inner loop. "
         "With action_repeat=6 that yields 30 Hz samples (demo_steps*6 rows). "
         "Saved as demo_joint_pos_30hz.npy (best episode) and "
         "demo_joint_pos_30hz_all.npy (all episodes).",
)

# AppLauncher args (must be attached before parse_known_args)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras if recording video
if args_cli.video:
    args_cli.enable_cameras = True

# Keep hydra-style "key=value" overrides, drop shell line-continuation junk
hydra_args = [arg for arg in hydra_args if arg.strip() not in ("\\", "")]
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports ---
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401 -- registers all tasks including Ball-In-Cup-Lynx-V0Legacy
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata
installed_version = metadata.version("rsl-rl-lib")


# RelativeJointPositionAction scale used by the V0Legacy env cfg. Keep in sync
# with LynxBallInCupEnvCfg_V0Legacy_PLAY.actions.arm_action.scale.
ACTION_SCALE = 0.1745


def _unwrap_dictlike(x):
    """If ``x`` is a dict / TensorDict with a 'policy' key, return that value.

    We avoid ``isinstance(x, dict)`` because ``tensordict.TensorDict`` is **not**
    a subclass of ``dict``; duck-typing on ``.keys()`` is the robust check.
    """
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "keys") and callable(x.keys):
        try:
            keys = list(x.keys())
        except Exception:
            return x
        if "policy" in keys:
            return x["policy"]
        if keys:
            return x[keys[0]]
    return x


def _extract_vec(x) -> np.ndarray:
    """Coerce a per-step env return value into a 1-D numpy vector.

    Handles the half-dozen shapes that rsl-rl / IsaacLab wrappers return for
    num_envs=1:
    - ``torch.Tensor`` of shape ``(1, D)``         → take row 0
    - ``torch.Tensor`` of shape ``(D,)``           → use as-is
    - ``dict`` / ``TensorDict`` with 'policy' key  → recurse on that value
    - ``(obs, extras)`` tuple/list                 → recurse on first element

    The TensorDict case is what made the earlier version raise
    ``RuntimeError: generator raised StopIteration`` from ``np.asarray``.
    """
    x = _unwrap_dictlike(x)
    if isinstance(x, (tuple, list)):
        x = _unwrap_dictlike(x[0])
    if isinstance(x, torch.Tensor):
        if x.dim() >= 2:
            x = x[0]
        return x.detach().cpu().numpy().copy()
    # Final fallback — only reachable for plain numpy / python sequences.
    return np.asarray(x).reshape(-1).copy()


def _resolve_arm_joint_indices(env_unwrapped) -> list[int]:
    """Return the articulation DOF indices that correspond to arm joints 1..6.

    The legacy env has 45 DOFs (6 arm + 12*3 string spherical + 1*3 ball
    spherical); the arm order in ``robot.data.joint_names`` is what we need.
    """
    robot = env_unwrapped.scene["robot"]
    names = list(robot.data.joint_names)
    arm_names = [f"joint_{i}" for i in range(1, 7)]
    missing = [n for n in arm_names if n not in names]
    if missing:
        # Fallback: first 6 joints (matches LynxLab collect_demo_for_dmp.py assumption).
        print(f"[WARN] arm joints {missing} not found in {names[:10]}..., falling back to [0:6]")
        return list(range(6))
    return [names.index(n) for n in arm_names]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Collect demonstration data from a legacy 106-dim IsaacLab 2.0 policy."""
    # Overrides
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # Action repeat: keep policy at trained 5Hz but step the env at action_repeat * 5 Hz
    # by shrinking decimation. This raises the rendered video FPS without changing
    # the physics work per policy decision (decimation_new * action_repeat == decimation_old).
    action_repeat = max(int(args_cli.action_repeat), 1)
    if action_repeat > 1:
        orig_dec = int(env_cfg.decimation)
        new_dec = max(int(round(orig_dec / action_repeat)), 1)
        if new_dec * action_repeat != orig_dec:
            print(f"[WARN] action_repeat={action_repeat} does not evenly divide "
                  f"decimation={orig_dec}; using decimation={new_dec} "
                  f"(physics steps per policy step = {new_dec * action_repeat} "
                  f"vs original {orig_dec})")
        env_cfg.decimation = new_dec
        if hasattr(env_cfg.sim, "render_interval"):
            env_cfg.sim.render_interval = 1
        effective_fps = 1.0 / (env_cfg.sim.dt * new_dec)
        print(f"[INFO] action_repeat={action_repeat}, decimation {orig_dec} -> {new_dec}, "
              f"effective video FPS = {effective_fps:.1f}")

    # Camera viewer overrides (applied before gym.make so RecordVideo sees them)
    if args_cli.camera_lookat is not None:
        env_cfg.viewer.lookat = tuple(args_cli.camera_lookat)
    if args_cli.camera_eye is not None:
        env_cfg.viewer.eye = tuple(args_cli.camera_eye)
    elif args_cli.camera_zoom != 1.0:
        eye = np.array(env_cfg.viewer.eye, dtype=float)
        target = np.array(env_cfg.viewer.lookat, dtype=float)
        new_eye = target + (eye - target) * args_cli.camera_zoom
        env_cfg.viewer.eye = tuple(new_eye.tolist())
        print(f"[INFO] Camera zoom={args_cli.camera_zoom} -> eye={env_cfg.viewer.eye}, "
              f"lookat={env_cfg.viewer.lookat}")

    # Resolve checkpoint
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Create environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Video recording wrapper (wraps the gym env before RSL-RL vec wrapper)
    if args_cli.video:
        video_folder = os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        # Match video length to the SAVED demo length, not the full rollout —
        # the user wants "as much video as data collected".
        video_length = args_cli.demo_steps * action_repeat
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # RSL-RL vec wrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]
    arm_idx = _resolve_arm_joint_indices(unwrapped_env)
    arm_idx_t = torch.as_tensor(arm_idx, device=env.unwrapped.device)
    print(f"[INFO] Arm joint DOF indices: {arm_idx}")

    # ================================================================
    # Collect episodes
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Task:           {args_cli.task}")
    print(f"  Checkpoint:     {resume_path}")
    print(f"  Episodes:       {args_cli.n_episodes}")
    print(f"  Demo length:    {args_cli.demo_steps} steps ({args_cli.demo_steps * 0.2:.1f}s @ 5Hz)")
    print(f"  Episode length: {args_cli.full_episode_steps} steps "
          f"({args_cli.full_episode_steps * 0.2:.1f}s @ 5Hz)")
    print(f"{'=' * 60}\n")

    def _get_arm_joint_pos() -> np.ndarray:
        jp = robot.data.joint_pos
        return jp.index_select(1, arm_idx_t)[0].cpu().numpy().copy()

    def _get_arm_joint_vel() -> np.ndarray:
        jv = robot.data.joint_vel
        return jv.index_select(1, arm_idx_t)[0].cpu().numpy().copy()

    all_episodes = []

    for ep in range(args_cli.n_episodes):
        # Full reset so each collection episode starts from the default state.
        unwrapped_env.reset()
        obs = env.get_observations()

        ep_joint_pos = []
        ep_joint_pos_before = []
        ep_pd_targets = []
        ep_joint_vel = []
        ep_observations = []
        ep_actions = []
        ep_rewards = []
        ep_joint_pos_high_freq = []  # filled only when args_cli.log_high_freq

        for t in range(args_cli.full_episode_steps):
            joint_pos_before = _get_arm_joint_pos()
            ep_joint_pos_before.append(joint_pos_before)

            with torch.inference_mode():
                actions = policy(obs)

            # PD target replica: raw_action * scale, clip to +-1 rad, add to pos_before.
            # (Matches RelativeJointPositionAction.process_actions under the V0Legacy cfg
            # arm_action: scale=0.1745, clip=(-1.0, 1.0).)
            act_np = _extract_vec(actions)
            processed = act_np * ACTION_SCALE
            clipped_processed = np.clip(processed, -1.0, 1.0)
            pd_target = joint_pos_before + clipped_processed
            ep_pd_targets.append(pd_target)

            # Hold the same policy action for `action_repeat` env.steps so the
            # video renders at action_repeat * 5 = 30 FPS while the policy still
            # decides at the trained 5Hz.
            done_flag = False
            reward_val = 0.0
            for _ in range(action_repeat):
                obs, _, dones, _ = env.step(actions)
                if args_cli.log_high_freq:
                    # Read joint_pos straight after each physics env.step so the
                    # trace lives at (5Hz * action_repeat) = 30Hz when
                    # action_repeat=6 with decimation=2 sim_dt=1/60.
                    ep_joint_pos_high_freq.append(_get_arm_joint_pos())
                if hasattr(unwrapped_env, "reward_buf"):
                    reward_val += float(unwrapped_env.reward_buf[0].item())
                if dones.any():
                    done_flag = True
                    break

            joint_pos = _get_arm_joint_pos()
            joint_vel = _get_arm_joint_vel()
            obs_np = _extract_vec(obs)

            ep_joint_pos.append(joint_pos)
            ep_joint_vel.append(joint_vel)
            ep_observations.append(obs_np)
            ep_actions.append(act_np)
            ep_rewards.append(reward_val)

            if done_flag:
                break

        ep_data = {
            "joint_pos": np.array(ep_joint_pos),
            "joint_pos_before": np.array(ep_joint_pos_before),
            "pd_targets": np.array(ep_pd_targets),
            "joint_vel": np.array(ep_joint_vel),
            "observations": np.array(ep_observations),
            "actions": np.array(ep_actions),
            "rewards": np.array(ep_rewards),
            "total_reward": float(np.sum(ep_rewards)),
            "n_steps": len(ep_joint_pos),
        }
        if args_cli.log_high_freq:
            ep_data["joint_pos_high_freq"] = np.array(ep_joint_pos_high_freq)
        all_episodes.append(ep_data)

        print(f"  Episode {ep + 1}/{args_cli.n_episodes}: "
              f"steps={ep_data['n_steps']}, "
              f"total_reward={ep_data['total_reward']:.2f}")

    # ================================================================
    # Pick best episode and slice to demo length
    # ================================================================
    episode_rewards = [ep["total_reward"] for ep in all_episodes]
    best_idx = int(np.argmax(episode_rewards))
    best_ep = all_episodes[best_idx]
    print(f"\n[INFO] Best episode: #{best_idx + 1} (reward={best_ep['total_reward']:.2f})")

    demo_steps = min(args_cli.demo_steps, best_ep["n_steps"])
    demo_data = {
        "joint_pos": best_ep["joint_pos"][:demo_steps],
        "joint_pos_before": best_ep["joint_pos_before"][:demo_steps],
        "pd_targets": best_ep["pd_targets"][:demo_steps],
        "joint_vel": best_ep["joint_vel"][:demo_steps],
        "observations": best_ep["observations"][:demo_steps],
        "actions": best_ep["actions"][:demo_steps],
        "rewards": best_ep["rewards"][:demo_steps],
    }

    # ================================================================
    # Save
    # ================================================================
    os.makedirs(args_cli.output_dir, exist_ok=True)

    for key, arr in demo_data.items():
        np.save(os.path.join(args_cli.output_dir, f"demo_{key}.npy"), arr)

    np.save(
        os.path.join(args_cli.output_dir, "all_episode_rewards.npy"),
        np.array(episode_rewards),
    )

    # High-frequency joint_pos log (30Hz when action_repeat=6). Saved only when
    # --log_high_freq was passed. Sliced to demo_steps * action_repeat rows to
    # align with the 5Hz demo_joint_pos.npy schema (every 6th row of the 30Hz
    # trace should match demo_joint_pos.npy within numerical noise).
    demo_joint_pos_30hz_shape = None
    if args_cli.log_high_freq:
        demo_rows_hf = demo_steps * action_repeat
        # Best-episode slice
        best_hf_full = best_ep.get("joint_pos_high_freq")
        if best_hf_full is not None and best_hf_full.size > 0:
            best_hf = best_hf_full[:demo_rows_hf]
            np.save(os.path.join(args_cli.output_dir, "demo_joint_pos_30hz.npy"), best_hf)
            demo_joint_pos_30hz_shape = list(best_hf.shape)
            # All episodes, truncated to a common length so it stacks cleanly
            all_hf = [ep.get("joint_pos_high_freq") for ep in all_episodes]
            if all([a is not None for a in all_hf]) and len(all_hf) > 0:
                min_rows = min(a.shape[0] for a in all_hf)
                rows_to_keep = min(min_rows, demo_rows_hf)
                all_hf_stack = np.stack([a[:rows_to_keep] for a in all_hf], axis=0)
                np.save(
                    os.path.join(args_cli.output_dir, "demo_joint_pos_30hz_all.npy"),
                    all_hf_stack,
                )
            print(f"[INFO] Saved high-freq joint_pos: shape={demo_joint_pos_30hz_shape} "
                  f"(effective {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.1f} Hz)")
        else:
            print("[WARN] --log_high_freq was set but best episode has no high-freq buffer.")

    # ================================================================
    # Plot demo trajectory: joint angles, raw policy actions, PD targets
    # ================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        joint_pos_arr = demo_data["joint_pos"]          # (T, 6)  current angles after step
        actions_arr = demo_data["actions"]              # (T, 6)  raw policy output
        pd_targets_arr = demo_data["pd_targets"]        # (T, 6)  joint_pos_before + clip(action*scale, -1, 1)
        T = joint_pos_arr.shape[0]
        n_joints = joint_pos_arr.shape[1]
        t_axis = np.arange(T) * 0.2  # 5Hz policy step -> 0.2s

        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        joint_labels = [f"j{i+1}" for i in range(n_joints)]

        for j in range(n_joints):
            axes[0].plot(t_axis, joint_pos_arr[:, j], label=joint_labels[j], linewidth=1.4)
            axes[1].plot(t_axis, actions_arr[:, j],   label=joint_labels[j], linewidth=1.4)
            axes[2].plot(t_axis, pd_targets_arr[:, j], label=joint_labels[j], linewidth=1.4)

        axes[0].set_ylabel("joint angle (rad)")
        axes[0].set_title(f"Demo trajectory — best ep #{best_idx + 1} "
                          f"(reward={best_ep['total_reward']:.2f}), "
                          f"{T} steps @ 5Hz ({T * 0.2:.1f}s)")
        axes[1].set_ylabel("raw policy action\n(pre-scale, pre-clip)")
        axes[1].axhline( 1.0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].axhline(-1.0, color="gray", linewidth=0.5, linestyle="--")
        axes[2].set_ylabel(f"PD target (rad)\nbefore + clip(act*{ACTION_SCALE:g}, -1, 1)")
        axes[2].set_xlabel("time (s)")

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", ncol=n_joints, fontsize=8)

        fig.tight_layout()
        plot_path = os.path.join(args_cli.output_dir, "demo_trajectory.png")
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"[INFO] Saved trajectory plot: {plot_path}")
    except Exception as e:
        print(f"[WARN] Failed to render trajectory plot: {e}")

    meta = {
        "checkpoint_path": str(resume_path),
        "task": args_cli.task,
        "n_episodes_collected": args_cli.n_episodes,
        "demo_steps": demo_steps,
        "full_episode_steps": args_cli.full_episode_steps,
        "dt_action": 0.2,
        "control_frequency_hz": 5,
        "sim_dt": float(env_cfg.sim.dt),
        "decimation": int(env_cfg.decimation),
        "action_repeat": action_repeat,
        "video_fps_effective": float(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        "action_scale": ACTION_SCALE,
        "n_joints": 6,
        "best_episode_idx": best_idx,
        "best_episode_reward": float(best_ep["total_reward"]),
        "all_episode_rewards": [float(r) for r in episode_rewards],
        "demo_joint_pos_shape": list(demo_data["joint_pos"].shape),
        "log_high_freq": bool(args_cli.log_high_freq),
        "demo_joint_pos_30hz_shape": demo_joint_pos_30hz_shape,
        "obs_dim": int(demo_data["observations"].shape[-1]) if demo_data["observations"].ndim >= 2 else -1,
        "arm_joint_indices": list(arm_idx),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "seed": args_cli.seed,
        "source_repo": "IsaacLab-2.0 (legacy 106-dim Ball-In-Cup)",
    }
    with open(os.path.join(args_cli.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Demo Data Saved")
    print(f"  Output dir:    {args_cli.output_dir}")
    print(f"  Demo shape:    {demo_data['joint_pos'].shape} (steps, joints)")
    print(f"  Demo duration: {demo_steps * 0.2:.1f}s ({demo_steps} steps @ 5Hz)")
    obs_arr = demo_data["observations"]
    obs_dim_str = str(obs_arr.shape[-1]) if obs_arr.ndim >= 2 else f"?? (ndim={obs_arr.ndim}, shape={obs_arr.shape})"
    print(f"  Obs dim:       {obs_dim_str} (should be 106)")
    print(f"  Best episode:  #{best_idx + 1} (reward={best_ep['total_reward']:.2f})")
    print(f"  All rewards:   {[f'{r:.1f}' for r in episode_rewards]}")
    print(f"{'=' * 60}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
