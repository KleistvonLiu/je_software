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