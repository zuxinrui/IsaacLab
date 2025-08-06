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

# also works for 4070 ti super (16GB VRAM), the observation excludes joint velocities:
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v1 --distributed --headless --num_envs 16384

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v2 --distributed --headless --num_envs 16384

python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Reach-Lynx-v3 --headless --num_envs 16384


# play the trained model:
python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-28_17-51-38 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-28_17-51-38/model_4000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v2 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-31_17-19-35 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-07-31_17-19-35/model_1000.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Open-Drawer-Franka-v0 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/franka_open_drawer/2025-07-31_12-54-00 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/franka_open_drawer/2025-07-31_12-54-00/model_50.pt

python scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Reach-Lynx-v3 --num_envs 32 --load_run /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-08-05_12-11-18 --checkpoint /home/zuxinrui/IsaacLab/logs/rsl_rl/LynxReach/2025-08-05_12-11-18/model_1000.pt
