./isaaclab.sh -p scripts/tools/convert_urdf.py /home/zuxinrui/adaptive_robot_planning/real_robot/robot_models/URDF_description/urdf/lynx_ses900_optimized.urdf source/isaaclab_assets/data/Robots/Lynx/lynx-urdf.usd --merge-joints --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none

./isaaclab.sh -p scripts/tools/convert_mjcf.py ./scripts/lynx/lynx_orth.xml source/isaaclab_assets/data/Robots/Lynx/lynx-fixed.usd --fix-base --import-sites --make-instanceable

