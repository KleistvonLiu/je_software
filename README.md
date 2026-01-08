colcon build --symlink-install
source install/setup.bash

conda activate ros2-humble-py310
source ~/ros2_ws/install/setup.bash

ros2 launch je_software agilex_robot.launch.py joint_pub_topic:=/joint_states_right can_port:=can_right joint_sub_topic:=/joint_cmd_right
ros2 launch je_software agilex_robot.launch.py joint_pub_topic:=/joint_states_left can_port:=can_left joint_sub_topic:=/joint_cmd_left
ros2 launch orbbec_camera 3_cameras.launch.py
ros2 launch je_software manager.launch.py episode_idx:=50 save_dir:/home/test/jemotor/jedata/temp/

ros2 launch je_software replay_node.launch.py jsonl:=/home/test/jemotor/jedata/meta.jsonl rate:=30.0 loop:=false


ros2 launch je_software replay_node.launch.py jsonl:=/home/test/jemotor/jedata/action.csv rate:=30.0 loop:=false

ros2 launch je_software manager.launch.py episode_idx:=0 save_dir:=/home/test/jemotor/temp_data/ overwrite:=true

ros2 run orbbec_camera list_devices_node

source install/share/ros2_ik_cpp/local_setup.bash

colcon build --merge-install --symlink-install


## ik
install pinocchio
https://stack-of-tasks.github.io/pinocchio/download.html

source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ros2_ik_cpp ik_node --ros-args --params-file src/ros2_ik_cpp/config/planning_module.yaml
ros2 launch je_software mujoco_sim.launch.py
python test/test_ik.py

ros2 topic pub /target_end_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'base_link'}, pose: {position: {x: 0.145938, y: -0.321009, z: 0.636514}, orientation: {x: 0.737535, y: -0.664778, z: -0.0404043, w: 0.111800}}}" -1