"""Deformable cube push environment using DirectRLEnv (no GridCloner for deformable).

This bypasses the GridCloner/InteractiveScene CUDA crash by:
1. Only putting the robot + target in InteractiveScene (GridCloner handles these fine)
2. Manually spawning deformable cubes outside the env hierarchy (like test_deformable_push_B)
3. Registering deformable cubes with the scene dict so they are auto-updated

Key insight from test_deformable_push_B.py: deformable + Lynx works perfectly with manual
spawning and use_fabric=True. The crash only happens in the ManagerBasedRLEnv path with
use_fabric=False + GridCloner.
"""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_assets.robots.lynx_constructor import LynxRobotCfg, LynxUsdConstructor  # isort: skip
from .joint_pos_env_cfg import _make_lynx_robot_cfg  # isort: skip

# Root path for deformable objects (outside GridCloner env hierarchy)
_SOFT_ROOT = "/World/_SoftObjects"


# ---------------------------------------------------------------------------
# Scene config — robot + target only (deformable cube spawned manually)
# ---------------------------------------------------------------------------

@configclass
class _DeformableSceneCfg(InteractiveSceneCfg):
    """Scene with Lynx robot and rigid target. Deformable cube is added in _setup_scene."""

    robot: LynxRobotCfg = _make_lynx_robot_cfg()

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.1), rot=(1, 0, 0, 0)),
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
    )

    # table is spawned inline in _setup_scene, not via scene config


# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

@configclass
class DeformablePushDirectEnvCfg(DirectRLEnvCfg):
    """Config for the deformable push DirectRLEnv."""

    # Sim
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 30.0,
        render_interval=1,
        use_fabric=True,  # use_fabric=True works (proven by test B)
        physx=sim_utils.PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 2048,
            gpu_total_aggregate_pairs_capacity=1024 * 2048,
            gpu_max_rigid_patch_count=1024 * 256,
        ),
    )

    # Scene — replicate_physics=True so GridCloner properly clones robot + target
    scene: _DeformableSceneCfg = _DeformableSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    # Episode
    decimation: int = 6
    episode_length_s: float = 20.0

    # Spaces: obs=31, action=6
    # obs: joint_pos_rel[6] + ee_pos[3] + obj_pos[3] + target_pos[3] + target_quat[4]
    #      + ee_to_obj[3] + obj_to_target[3] + last_action[6]
    observation_space: int = 31
    action_space: int = 6

    state_space: int = 0

    # Action scale (relative joint position, ~10 deg)
    action_scale: float = 0.1745


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DeformablePushDirectEnv(DirectRLEnv):
    """DirectRLEnv for pushing a deformable cube with a Lynx robot.

    Deformable cubes are spawned outside the GridCloner env hierarchy and
    registered with the InteractiveScene manually, avoiding the PhysX FEM +
    GridCloner CUDA crash.
    """

    cfg: DeformablePushDirectEnvCfg

    def _setup_scene(self):
        """Spawn deformable cubes outside env hierarchy and register with scene."""

        env_regex = self.scene.env_regex_ns  # e.g. "/World/envs/env_.*"

        # -- Table (static mesh, per env, via GridCloner regex)
        table_cfg = sim_utils.MeshCuboidCfg(
            size=(1.0, 0.8, 0.018),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.5, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        table_cfg.func(f"{env_regex}/Table", table_cfg, translation=(0.0, 0.0, -0.018 / 2))

        # -- Ground plane (lowered by table height so it doesn't overlap the table surface)
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg, translation=(0.0, 0.0, -0.6-0.018))
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/light", light_cfg)

        # -- Spawn deformable cubes OUTSIDE env hierarchy (proven pattern from test B)
        # env_origins are available after clone_environments() (replicate_physics=True)
        from isaaclab.sim.utils import create_prim

        env_origins = self.scene.env_origins  # (num_envs, 3) tensor
        num_envs = self.scene.cfg.num_envs

        for i in range(num_envs):
            origin = env_origins[i]
            create_prim(
                f"{_SOFT_ROOT}/cube_{i}",
                "Xform",
                translation=(float(origin[0]), float(origin[1]), float(origin[2])),
            )

        # Use standard DeformableObjectCfg with regex across all cube parents
        cube_cfg = DeformableObjectCfg(
            prim_path=f"{_SOFT_ROOT}/cube_.*/Object",
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
        self._cube = DeformableObject(cfg=cube_cfg)

        # Register with the scene so write_data_to_sim / update / reset are auto-called
        self.scene._deformable_objects["cube"] = self._cube

    # ------------------------------------------------------------------ #
    # DirectRLEnv abstract methods
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor):
        """Compute joint position targets from relative actions."""
        self._actions = actions.clone().clamp(-1.0, 1.0)
        robot = self.scene["robot"]
        # Relative joint position control: target = current_target + action * scale
        self._joint_pos_target = robot.data.joint_pos + self._actions * self.cfg.action_scale

    def _apply_action(self):
        """Write joint targets to the robot."""
        robot = self.scene["robot"]
        robot.set_joint_position_target(self._joint_pos_target)

    def _get_observations(self) -> dict:
        """Compute observation vector (31-dim)."""
        robot = self.scene["robot"]
        cube = self._cube
        target: RigidObject = self.scene["target"]

        # EE body index (cached on first call)
        if not hasattr(self, "_ee_body_idx"):
            self._ee_body_idx = robot.body_names.index("ee")

        # Robot state
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos  # (N, 6)

        # EE position in robot root frame
        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx]  # (N, 3)
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
        )

        # Deformable object center (mean of all nodal positions)
        obj_pos_w = cube.data.nodal_pos_w.mean(dim=1)  # (N, 3)
        obj_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, obj_pos_w
        )

        # Target pos in robot root frame
        target_pos_w = target.data.root_pos_w[:, :3]  # (N, 3)
        target_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w
        )

        # Target orientation (world frame)
        target_quat = target.data.root_quat_w  # (N, 4)

        # Relative vectors (world frame)
        ee_to_obj = obj_pos_w - ee_pos_w  # (N, 3)
        obj_to_target = target_pos_w - obj_pos_w  # (N, 3)

        # Last action
        last_action = self._actions if hasattr(self, "_actions") else torch.zeros(
            self.num_envs, 6, device=self.device
        )

        obs = torch.cat([
            joint_pos_rel,   # 6
            ee_pos_b,        # 3
            obj_pos_b,       # 3
            target_pos_b,    # 3
            target_quat,     # 4
            ee_to_obj,       # 3
            obj_to_target,   # 3
            last_action,     # 6
        ], dim=-1)  # total: 31

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward."""
        robot = self.scene["robot"]
        cube = self._cube
        target: RigidObject = self.scene["target"]

        if not hasattr(self, "_ee_body_idx"):
            self._ee_body_idx = robot.body_names.index("ee")

        ee_pos_w = robot.data.body_pos_w[:, self._ee_body_idx]
        obj_pos_w = cube.data.nodal_pos_w.mean(dim=1)
        target_pos_w = target.data.root_pos_w[:, :3]

        # 1. EE → object distance (approach)
        ee_obj_dist = torch.norm(obj_pos_w - ee_pos_w, dim=1)
        reaching_raw = -0.2 * ee_obj_dist
        reaching_fine = 0.1 * (1.0 - torch.tanh(ee_obj_dist / 0.05))

        # 2. Object → target distance (push)
        obj_target_dist = torch.norm(obj_pos_w - target_pos_w, dim=1)
        pushing_raw = -0.3 * obj_target_dist

        # 3. Success bonus (XY distance < 0.05)
        obj_target_dist_xy = torch.norm(obj_pos_w[:, :2] - target_pos_w[:, :2], dim=1)
        success = 20.0 * (obj_target_dist_xy < 0.05).float()

        # 4. Smoothness penalties
        actions = self._actions if hasattr(self, "_actions") else torch.zeros(
            self.num_envs, 6, device=self.device
        )
        action_l2 = -0.001 * torch.sum(actions ** 2, dim=1)

        # Action rate (difference from previous action)
        if not hasattr(self, "_prev_actions"):
            self._prev_actions = torch.zeros_like(actions)
        action_rate = -0.001 * torch.sum((actions - self._prev_actions) ** 2, dim=1)
        self._prev_actions = actions.clone()

        # Joint velocity penalty
        joint_vel = robot.data.joint_vel
        joint_vel_penalty = -1e-4 * torch.sum(joint_vel ** 2, dim=1)

        reward = reaching_raw + reaching_fine + pushing_raw + success + action_l2 + action_rate + joint_vel_penalty
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        super()._reset_idx(env_ids)

        robot = self.scene["robot"]
        cube = self._cube
        target: RigidObject = self.scene["target"]

        # Reset robot joints
        default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        # Reset robot root state
        default_root = robot.data.default_root_state[env_ids].clone()
        default_root[:, 0:3] += self.scene.env_origins[env_ids]
        robot.write_root_pose_to_sim(default_root[:, :7], env_ids=env_ids)
        robot.write_root_velocity_to_sim(default_root[:, 7:], env_ids=env_ids)

        # Reset deformable cube to default nodal state
        nodal_state = cube.data.default_nodal_state_w[env_ids].clone()
        cube.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)

        # Randomize target position (XY ± 0.1/0.25 around default)
        target_default = target.data.default_root_state[env_ids].clone()
        target_default[:, 0:3] += self.scene.env_origins[env_ids]
        n = len(env_ids)
        target_default[:, 0] += torch.empty(n, device=self.device).uniform_(-0.1, 0.1)
        target_default[:, 1] += torch.empty(n, device=self.device).uniform_(-0.25, 0.25)
        # Random yaw
        yaw = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
        half_yaw = yaw * 0.5
        target_default[:, 3] = torch.cos(half_yaw)  # w
        target_default[:, 4] = 0.0  # x
        target_default[:, 5] = 0.0  # y
        target_default[:, 6] = torch.sin(half_yaw)  # z
        target.write_root_pose_to_sim(target_default[:, :7], env_ids=env_ids)
        target.write_root_velocity_to_sim(target_default[:, 7:], env_ids=env_ids)

        # Reset action buffers
        if hasattr(self, "_actions"):
            self._actions[env_ids] = 0.0
        if hasattr(self, "_prev_actions"):
            self._prev_actions[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Play variant
# ---------------------------------------------------------------------------

@configclass
class DeformablePushDirectEnvCfg_PLAY(DeformablePushDirectEnvCfg):
    """Play/inference config with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
