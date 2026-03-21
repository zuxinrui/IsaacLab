./isaaclab.sh -p scripts/tools/convert_urdf.py /home/zuxinrui/adaptive_robot_planning/real_robot/robot_models/URDF_description/urdf/lynx_ses900_optimized.urdf source/isaaclab_assets/data/Robots/Lynx/lynx-urdf.usd --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none

./isaaclab.sh -p scripts/tools/convert_mjcf.py ./scripts/lynx/lynx_orth.xml source/isaaclab_assets/data/Robots/Lynx/lynx-fixed.usd --fix-base --import-sites --make-instanceable

# THe USD file needs to be manually imported into the original isaacsim app and imported as a new asset, not the reference asset:
# 1. Open the original isaacsim app:
# 2. Import the USD file as a new asset.
# 3. Change the articulation root from root_joint to the root prim (just the name of the robot)
# 4. Save the asset.

# the final usd file for lynx urdf version: lynx-isaacsim2-urdf.usd

# Distributed training via rsl_rl:
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v0 --distributed --headless --num_envs 16384
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v2 --distributed --headless --num_envs 16384
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Open-Drawer-Franka-v0 --distributed --headless --num_envs 16384
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Push-Cube-Lynx-v0 --distributed --headless --num_envs 16384
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Push-Cube-Lynx-ObsDelay-v0 --distributed --headless --num_envs 16384
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ball-In-Cup-Lynx-v0 --distributed --headless --num_envs 8192



# also works for 4070 ti super (16GB VRAM), the observation excludes joint velocities:
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v1 --distributed --headless --num_envs 16384

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v2 --distributed --headless --num_envs 16384

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v3 --headless --num_envs 16384

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v4 --num_envs 1024
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v4 --num_envs 16384 --headless

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-Mujoco-v0 --num_envs 1024  # Isaac-Lift-Cube-OpenArm-v0; Isaac-Stack-Cube-Franka-v0; Isaac-Lift-Cube-Franka-v0
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Lift-Cube-Franka-v0 --num_envs 16384 --headless

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Push-Cube-Lynx-v0 --num_envs 1024
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Push-Cube-Lynx-ObsDelay-v0 --num_envs 32 --headless  # Isaac-Push-Cube-Lynx-ObsDelay-v0 | Isaac-Push-Cube-Lynx-v0


python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-Trapezoidal-v0 --num_envs 1024 --headless

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ball-In-Cup-Lynx-v1 --headless --num_envs 32
python scripts/reinforcement_learning/skrl/train_sacx.py --task Isaac-Ball-In-Cup-Lynx-v0 --num_envs 1024 --total_steps 200000 --headless


# play the trained model:
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-28_17-51-38 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-28_17-51-38/model_4000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v2 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-31_17-19-35 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-31_17-19-35/model_1000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Open-Drawer-Franka-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/franka_open_drawer/2025-07-31_12-54-00 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/franka_open_drawer/2025-07-31_12-54-00/model_50.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v3 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-08-05_12-11-18 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-08-05_12-11-18/model_1000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-Trapezoidal-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2026-03-10_16-19-00 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2026-03-10_16-19-00/model_500.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Push-Cube-Lynx-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-15_01-25-19 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-15_01-25-19/model_1499.pt

# Recommended high-throughput play command (reduced render + optimized PLAY env cfg):
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Push-Cube-Lynx-Play-v0 --num_envs 32 --headless --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-15_01-25-19 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-15_01-25-19/model_1499.pt

# Interactive debug command (GUI, fewer envs for smooth viewport):
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Push-Cube-Lynx-Play-v0 --num_envs 8 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-18_23-07-04 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_push/2026-03-18_23-07-04/model_1499.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Ball-In-Cup-Lynx-Play-v0 --num_envs 8 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_ball_in_cup/2026-03-19_13-46-12 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/lynx_ball_in_cup/2026-03-19_13-46-12/model_100.pt


