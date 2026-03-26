"""Test Script B: Standalone deformable cube + Lynx robot simulation.

This script bypasses ManagerBasedRLEnv / InteractiveScene entirely and builds
the scene manually, following the **proven** pattern from:
  - scripts/tutorials/01_assets/run_deformable_object.py
  - source/isaaclab/test/assets/test_deformable_object.py

The idea: if this script works, we know that the PhysX FEM + Lynx combination
is fundamentally sound, and the problem lies in the InteractiveScene / GridCloner
integration.  If this also fails, the issue is deeper (PhysX FEM + contact, GPU
memory, etc.).

Key differences from the current architecture:
  1. NO GridCloner, no InteractiveScene  — just plain create_prim + DeformableObject
  2. NO use_fabric toggling needed (no GridCloner = no env hierarchy conflict)
  3. Deformable cubes use the SAME pattern as the working tutorial
  4. Lynx robot is spawned via LynxUsdConstructor.spawn directly

Usage:
    python scripts/lynx/test_deformable_push_B.py --headless --num_envs 4
    python scripts/lynx/test_deformable_push_B.py --num_envs 2   # with viewer
"""

import argparse
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Script B: standalone deformable + Lynx.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Spacing between environments.")
parser.add_argument("--num_steps", type=int, default=500, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg
from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor


# --------------------------------------------------------------------------- #
# Scene setup helpers
# --------------------------------------------------------------------------- #

def _compute_env_origins(num_envs: int, spacing: float) -> list[tuple[float, float, float]]:
    """Compute grid origins for N environments (same layout as GridCloner)."""
    import math
    cols = int(math.ceil(math.sqrt(num_envs)))
    origins = []
    for i in range(num_envs):
        row = i // cols
        col = i % cols
        x = col * spacing
        y = row * spacing
        origins.append((x, y, 0.0))
    return origins


def _spawn_lynx_robot(prim_path: str):
    """Spawn a Lynx robot at the given prim path (no regex)."""
    robot_cfg = LynxRobotCfg(
        prim_path=prim_path,
        num_joints=6,
        genotype_tube=[0, 1, 0, 1, 0],
        genotype_joints=1,
        rotation_angles=[180.0, 0.0, 0.0, -180.0, 0.0, 0.0],
        l1_end_point_pos=(0.0, 0.0, 0.2),
        l1_end_point_theta=0.0,
        l2_end_point_pos=(0.0, 0.0, 0.2805),
        l2_end_point_theta=0.0,
        l3_end_point_pos=(0.0, 0.0, 0.2),
        l3_end_point_theta=0.0,
        l4_end_point_pos=(0.0, 0.0, 0.2805),
        l4_end_point_theta=0.0,
        l5_end_point_pos=(0.0, 0.0, 0.2),
        l5_end_point_theta=0.0,
        joint_velocity_limit_rad_s=0.3490658503988659,
        joint_acceleration_limit_rad_s2=1.7453292519943295,
    )
    robot_cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=0,
    )
    robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.func = LynxUsdConstructor.spawn
    robot_cfg.spawn.robot_cfg = {
        "genotype_tube": robot_cfg.genotype_tube,
        "genotype_joints": robot_cfg.genotype_joints,
        "rotation_angles": robot_cfg.rotation_angles,
        "bspline_num_segments": robot_cfg.bspline_num_segments,
        "bspline_dual_point_distance": robot_cfg.bspline_dual_point_distance,
        "tube_radiuses": robot_cfg.tube_radiuses,
        "clamp_stl": robot_cfg.clamp_stl,
        "ee_stl": robot_cfg.ee_stl,
        "actuators": robot_cfg.actuators,
        "init_state": robot_cfg.init_state,
        "rigid_props": robot_cfg.spawn.rigid_props,
        "articulation_props": robot_cfg.spawn.articulation_props,
        "activate_contact_sensors": robot_cfg.spawn.activate_contact_sensors,
        "l1_end_point_pos": robot_cfg.l1_end_point_pos,
        "l1_end_point_theta": robot_cfg.l1_end_point_theta,
        "l2_end_point_pos": robot_cfg.l2_end_point_pos,
        "l2_end_point_theta": robot_cfg.l2_end_point_theta,
        "l3_end_point_pos": robot_cfg.l3_end_point_pos,
        "l3_end_point_theta": robot_cfg.l3_end_point_theta,
        "l4_end_point_pos": robot_cfg.l4_end_point_pos,
        "l4_end_point_theta": robot_cfg.l4_end_point_theta,
        "l5_end_point_pos": robot_cfg.l5_end_point_pos,
        "l5_end_point_theta": robot_cfg.l5_end_point_theta,
        "joint_velocity_limit_rad_s": robot_cfg.joint_velocity_limit_rad_s,
        "joint_acceleration_limit_rad_s2": robot_cfg.joint_acceleration_limit_rad_s2,
    }
    return robot_cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def design_scene(num_envs: int, spacing: float):
    """Build the scene: ground + lights + N × (Lynx robot + deformable cube + table)."""

    t0 = time.time()
    origins = _compute_env_origins(num_envs, spacing)
    print(f"[TestB] Env origins: {origins}", flush=True)

    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/light", cfg)

    # ------------------------------------------------------------------ #
    # Per-env parent Xforms (plain USD, no GridCloner)
    # ------------------------------------------------------------------ #
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Env{i}", "Xform", translation=origin)

    print(f"[TestB] Created {num_envs} env Xforms in {time.time()-t0:.2f}s", flush=True)

    # ------------------------------------------------------------------ #
    # Spawn Lynx robots — one per env, via the constructor
    # ------------------------------------------------------------------ #
    t1 = time.time()
    robot_cfgs = []
    for i in range(num_envs):
        prim_path = f"/World/Env{i}/Robot"
        rcfg = _spawn_lynx_robot(prim_path)
        # Manually call the spawner for each env
        rcfg.spawn.func(prim_path, rcfg.spawn)
        robot_cfgs.append(rcfg)
    print(f"[TestB] Spawned {num_envs} Lynx robots in {time.time()-t1:.2f}s", flush=True)

    # Create Articulation view over all robots via regex
    from isaaclab.assets import ArticulationCfg
    all_robot_cfg = ArticulationCfg(
        prim_path="/World/Env.*/Robot",
        spawn=None,  # already spawned
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={"joint_[1-6]": 0.0},
        ),
        actuators={
            "lynx_arm": sim_utils.ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=800.0,
                damping=75.0,
            ),
        },
    )
    robot = Articulation(cfg=all_robot_cfg)
    print(f"[TestB] Articulation view created for regex '/World/Env.*/Robot'", flush=True)

    # ------------------------------------------------------------------ #
    # Spawn tables — simple collision cuboid per env
    # ------------------------------------------------------------------ #
    t2 = time.time()
    for i in range(num_envs):
        sim_utils.create_prim(f"/World/Env{i}/Table", "Xform")
    table_cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 0.8, 0.018),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    # Spawn table at env 0, clone to others
    table_cfg.func(
        f"/World/Env.*/Table/Mesh",
        table_cfg,
        translation=(0.0, 0.0, -0.018 / 2),
    )
    print(f"[TestB] Spawned tables in {time.time()-t2:.2f}s", flush=True)

    # ------------------------------------------------------------------ #
    # Spawn deformable cubes — the PROVEN tutorial pattern
    # ------------------------------------------------------------------ #
    t3 = time.time()

    # Parent Xforms for deformable cubes (same hierarchy as Env, NOT separate)
    # Key insight: since we are NOT using GridCloner, there's no GridCloner bug!
    # We can put deformable cubes directly under /World/Env{i}/
    deformable_cfg = DeformableObjectCfg(
        prim_path="/World/Env.*/Cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.2, 0.2, 0.2),
            deformable_props=DeformableBodyPropertiesCfg(
                self_collision=False,
                solver_position_iteration_count=16,
            ),
            physics_material=DeformableBodyMaterialCfg(
                youngs_modulus=5000.0,
                poissons_ratio=0.49,
                dynamic_friction=0.5,
                elasticity_damping=0.002,
                damping_scale=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.4),
                opacity=0.75,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.11)),
        debug_vis=False,
    )
    cube_object = DeformableObject(cfg=deformable_cfg)
    print(f"[TestB] Spawned deformable cubes in {time.time()-t3:.2f}s", flush=True)

    # ------------------------------------------------------------------ #
    # Spawn target markers (rigid, kinematic, no collision)
    # ------------------------------------------------------------------ #
    target_cfg = RigidObjectCfg(
        prim_path="/World/Env.*/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.1), rot=(1, 0, 0, 0)),
    )
    target_object = RigidObject(cfg=target_cfg)
    print(f"[TestB] Spawned target markers", flush=True)

    entities = {
        "robot": robot,
        "cube": cube_object,
        "target": target_object,
    }
    return entities, origins


def run_simulation(sim: SimulationContext, entities: dict, origins_list: list, num_steps: int):
    """Run the simulation loop with basic diagnostics."""

    robot: Articulation = entities["robot"]
    cube: DeformableObject = entities["cube"]
    target: RigidObject = entities["target"]

    origins = torch.tensor(origins_list, device=sim.device, dtype=torch.float32)
    sim_dt = sim.get_physics_dt()
    num_envs = len(origins_list)

    # ------------------------------------------------------------------ #
    # Diagnostic: check SoftBodyView
    # ------------------------------------------------------------------ #
    view_count = cube.root_physx_view.count
    print(f"\n[TestB] === Post-reset Diagnostics ===")
    print(f"[TestB]   SoftBodyView.count = {view_count}")
    print(f"[TestB]   num_instances       = {cube.num_instances}")
    print(f"[TestB]   expected            = {num_envs}")
    print(f"[TestB]   nodal_pos_w shape   = {cube.data.nodal_pos_w.shape}")
    print(f"[TestB]   robot num_instances  = {robot.num_instances}")
    print(f"[TestB]   target num_instances = {target.num_instances}")

    if view_count != num_envs:
        print(f"[TestB]   *** CRITICAL: SoftBodyView.count mismatch! ***")
    else:
        print(f"[TestB]   STATUS: OK — all {view_count} FEM bodies registered")

    # Print cube centers
    centers = cube.data.nodal_pos_w.mean(dim=1)
    for i in range(min(view_count, 8)):
        c = centers[i]
        print(f"[TestB]   cube {i} center: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")

    print(f"\n[TestB] Running {num_steps} simulation steps ...", flush=True)

    # Prepare kinematic target for deformable
    nodal_kinematic_target = cube.data.nodal_kinematic_target.clone()

    count = 0
    t_start = time.time()

    while simulation_app.is_running() and count < num_steps:
        # Reset every 200 steps
        if count % 200 == 0:
            # Reset deformable cubes to default state
            nodal_state = cube.data.default_nodal_state_w.clone()
            cube.write_nodal_state_to_sim(nodal_state)

            # Free all vertices
            nodal_kinematic_target[..., :3] = nodal_state[..., :3]
            nodal_kinematic_target[..., 3] = 1.0
            cube.write_nodal_kinematic_target_to_sim(nodal_kinematic_target)
            cube.reset()

            # Reset robot joints to zero
            joint_pos = torch.zeros(num_envs, robot.num_joints, device=sim.device)
            joint_vel = torch.zeros(num_envs, robot.num_joints, device=sim.device)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            target.reset()

            if count == 0:
                print(f"[TestB]   Initial reset done", flush=True)

        # Write data and step
        cube.write_data_to_sim()
        robot.write_data_to_sim()
        target.write_data_to_sim()
        sim.step()
        count += 1

        # Update buffers
        cube.update(sim_dt)
        robot.update(sim_dt)
        target.update(sim_dt)

        # Print progress
        if count % 100 == 0:
            centers = cube.data.nodal_pos_w.mean(dim=1)
            elapsed = time.time() - t_start
            fps = count / elapsed if elapsed > 0 else 0
            print(
                f"[TestB]   step {count}/{num_steps}  "
                f"cube[0] center=({centers[0,0]:.3f},{centers[0,1]:.3f},{centers[0,2]:.3f})  "
                f"fps={fps:.1f}",
                flush=True,
            )

    total_time = time.time() - t_start
    print(f"\n[TestB] Simulation done: {count} steps in {total_time:.2f}s  ({count/total_time:.1f} fps)")


def main():
    t0 = time.time()
    num_envs = args_cli.num_envs
    spacing = args_cli.env_spacing

    print(f"\n{'='*70}")
    print(f"[TestB] Standalone Deformable + Lynx Test")
    print(f"[TestB]   num_envs={num_envs}  spacing={spacing}")
    print(f"[TestB]   NO GridCloner, NO InteractiveScene, NO ManagerBasedRLEnv")
    print(f"{'='*70}\n")

    # Simulation context — use_fabric=True is fine here (no GridCloner conflict)
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device if args_cli.device else "cuda:0",
        dt=1.0 / 30.0,
        # Note: we can keep use_fabric=True because there's no GridCloner!
        use_fabric=True,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[3.0, 0.0, 1.0], target=[0.0, 0.0, 0.5])

    # Design scene
    print("[TestB] Designing scene ...", flush=True)
    entities, origins = design_scene(num_envs, spacing)

    # Start simulation
    print("[TestB] Calling sim.reset() ...", flush=True)
    t_reset = time.time()
    sim.reset()
    print(f"[TestB] sim.reset() completed in {time.time()-t_reset:.2f}s\n", flush=True)

    # Run
    run_simulation(sim, entities, origins, args_cli.num_steps)

    total = time.time() - t0
    print(f"\n[TestB] Total elapsed: {total:.2f}s")


if __name__ == "__main__":
    main()
    simulation_app.close()
