"""Test Script A: Validate the current deformable push architecture.

This script tests the existing ``Isaac-Push-DeformableCube-Lynx-v0`` task
with a small number of environments (4) to diagnose why loading gets stuck.

Key diagnostics:
1. Prints timing at each stage (scene construction, sim.reset, first step)
2. Checks SoftBodyView.count vs num_envs
3. Verifies env_origins alignment between GridCloner and deformable objects
4. Tests one full reset + step cycle

Usage:
    python scripts/lynx/test_deformable_push_A.py --num_envs 4 --headless
"""

import argparse
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Script A: diagnose deformable push loading.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments (keep small!).")
parser.add_argument("--task", type=str, default="Isaac-Push-DeformableCube-Lynx-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401 — registers gym envs


def main():
    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"[TestA] Starting: task={args_cli.task}  num_envs={args_cli.num_envs}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------ #
    # 1. Build the environment config (without creating the env yet)
    # ------------------------------------------------------------------ #
    print("[TestA] Step 1: Resolving env config ...", flush=True)
    t1 = time.time()

    # We need to get the env_cfg to override num_envs before gym.make
    from isaaclab.envs import ManagerBasedRLEnvCfg
    spec = gym.spec(args_cli.task)
    env_cfg_entry = spec.kwargs.get("env_cfg_entry_point")
    if env_cfg_entry is None:
        raise RuntimeError(f"No env_cfg_entry_point for {args_cli.task}")

    # Import the config class
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    EnvCfgClass = getattr(mod, class_name)
    env_cfg = EnvCfgClass()

    # Force small num_envs
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"[TestA]   num_envs = {env_cfg.scene.num_envs}", flush=True)
    print(f"[TestA]   replicate_physics = {env_cfg.scene.replicate_physics}", flush=True)
    print(f"[TestA]   use_fabric = {env_cfg.sim.use_fabric}", flush=True)
    print(f"[TestA]   dt = {env_cfg.sim.dt}", flush=True)
    print(f"[TestA]   Step 1 done in {time.time()-t1:.2f}s\n", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Create the environment (this triggers scene construction + spawning)
    # ------------------------------------------------------------------ #
    print("[TestA] Step 2: gym.make() — scene construction + spawning ...", flush=True)
    t2 = time.time()

    env = gym.make(args_cli.task, cfg=env_cfg)
    dt_make = time.time() - t2
    print(f"[TestA]   gym.make() completed in {dt_make:.2f}s\n", flush=True)

    # ------------------------------------------------------------------ #
    # 3. Inspect the scene
    # ------------------------------------------------------------------ #
    print("[TestA] Step 3: Inspecting scene ...", flush=True)
    unwrapped = env.unwrapped
    scene = unwrapped.scene

    print(f"[TestA]   Articulations: {list(scene.articulations.keys())}")
    print(f"[TestA]   Rigid objects: {list(scene.rigid_objects.keys())}")
    print(f"[TestA]   Deformable objects: {list(scene.deformable_objects.keys())}")
    print(f"[TestA]   Sensors: {list(scene.sensors.keys())}")
    print(f"[TestA]   env_origins shape: {scene.env_origins.shape}")
    print(f"[TestA]   env_origins:\n{scene.env_origins}\n")

    # Check deformable object details
    for name, obj in scene.deformable_objects.items():
        view = obj.root_physx_view
        count = view.count
        n_inst = obj.num_instances
        print(f"[TestA]   Deformable '{name}':")
        print(f"[TestA]     SoftBodyView.count   = {count}")
        print(f"[TestA]     num_instances         = {n_inst}")
        print(f"[TestA]     expected              = {unwrapped.num_envs}")
        print(f"[TestA]     nodal_state shape     = {obj.data.default_nodal_state_w.shape}")
        print(f"[TestA]     nodal_pos_w shape     = {obj.data.nodal_pos_w.shape}")

        # Check if the deformable object centers are near the env origins
        centers = obj.data.nodal_pos_w.mean(dim=1)  # (num_instances, 3)
        print(f"[TestA]     object centers (world):")
        for i in range(min(count, 8)):
            c = centers[i]
            print(f"[TestA]       env {i}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

        if count != unwrapped.num_envs:
            print(f"\n[TestA] *** CRITICAL: SoftBodyView.count ({count}) != num_envs ({unwrapped.num_envs}) ***")
            print(f"[TestA] *** PhysX only registered {count} FEM bodies — the GridCloner workaround FAILED ***\n")
        else:
            print(f"[TestA]     STATUS: OK — all {count} FEM bodies registered\n")

    # ------------------------------------------------------------------ #
    # 4. Try stepping the environment
    # ------------------------------------------------------------------ #
    print("[TestA] Step 4: Attempting env.reset() + 10 steps ...", flush=True)
    t4 = time.time()

    try:
        obs, info = env.reset()
        print(f"[TestA]   reset() succeeded. obs shape = {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
    except Exception as e:
        print(f"[TestA]   *** reset() FAILED: {e}")
        import traceback
        traceback.print_exc()
        env.close()
        return

    for step in range(10):
        try:
            action = torch.zeros(unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device)
            obs, reward, terminated, truncated, info = env.step(action)
            if step == 0 or step == 9:
                print(f"[TestA]   step {step}: reward mean={reward.mean():.4f}")
        except Exception as e:
            print(f"[TestA]   *** step {step} FAILED: {e}")
            import traceback
            traceback.print_exc()
            break

    dt_step = time.time() - t4
    print(f"[TestA]   Steps completed in {dt_step:.2f}s\n", flush=True)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    total = time.time() - t0
    print(f"{'='*70}")
    print(f"[TestA] SUMMARY")
    print(f"  Total time:          {total:.2f}s")
    print(f"  gym.make() time:     {dt_make:.2f}s")
    print(f"  reset+steps time:    {dt_step:.2f}s")
    for name, obj in scene.deformable_objects.items():
        count = obj.root_physx_view.count
        ok = count == unwrapped.num_envs
        print(f"  SoftBodyView.count:  {count} / {unwrapped.num_envs}  {'OK' if ok else 'FAILED'}")
    print(f"{'='*70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
