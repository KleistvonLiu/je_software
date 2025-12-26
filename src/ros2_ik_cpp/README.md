ros2_ik_cpp
============

Minimal ROS2 C++ package providing an IK node using Pinocchio.

Usage:
  - Build in colcon workspace: colcon build --packages-select ros2_ik_cpp
  - Run: ros2 run ros2_ik_cpp ik_node --ros-args -p urdf_path:="/path/to/robot.urdf" -p tip_frame:="Link17"

Notes:
  - Requires pinocchio C++ and headers available to CMake.
  - May need to adjust include/link names depending on your Pinocchio packaging.
