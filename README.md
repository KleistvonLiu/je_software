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
ros2 launch je_software manager.launch.py episode_idx:=0 save_dir:=/home/test/jemotor/temp_data/ overwrite:=true

ros2 run orbbec_camera list_devices_node


ros2 launch je_software manager.launch.py overwrite:=false episode_idx:=1


# 机器人的节点
ros2 launch je_software agilex_robot.launch.py joint_pub_topic:=/joint_states_right can_port:=can_right
ros2 launch je_software agilex_robot.launch.py joint_pub_topic:=/joint_states_left can_port:=can_left
ros2 launch orbbec_camera 3_cameras.launch.py
ros2 launch je_software tactile_sensor.launch.py
ros2 launch je_software manager.launch.py save_dir:=/home/kleist/jemotor/log
ros2 launch je_software je_robot_node.launch.py \
fps:=30 \
dt_init:=5
ros2 run je_software end_effector_cli --ros-args -p hand:=left
ros2 launch je_software jsonl_replayer_node.launch.py\
  jsonl_path:=/home/kleist/Documents/temp/meta.jsonl\
  rate_hz:=30.0\
  loop:=false     \
  send_arm:=left \
  dt_init:=5

# 设置串口低延迟模式
sudo apt install setserial
sudo setserial /dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0 low_latency
# 取消设置
sudo setserial /dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0 ^low_latency

# 确认 low_latency 已生效
setserial -g /dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0
# 只启用左臂
ros2 launch je_software dynamixel_init_joint_state.launch.py   \
left_port:=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0   \
left_enabled:=true \
right_enabled:=false   \
left_models:='["xl330-m288","xl330-m288","xl330-m288","xl330-m077","xl330-m288","xl330-m288","xl330-m288","xl330-m288"]'   \
left_signs:="[1,-1,1,-1,1,1,1,-1]"   \
right_signs:="[1,1,1,1,1,1,1,1]" \
zero_on_start:=true
# 启用左右臂
ros2 launch je_software dynamixel_init_joint_state.launch.py   \
left_port:=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0   \
right_port:=/dev/serial/by-path/pci-0000:00:14.0-usb-0:6:1.0-port0   \
left_enabled:=true \
right_enabled:=true   \
left_models:='["xl330-m288","xl330-m288","xl330-m288","xl330-m077","xl330-m288","xl330-m288","xl330-m288","xl330-m288"]'   \
right_models:='["xl330-m077","xl330-m077","xl330-m077","xl330-m077","xl330-m077","xl330-m077","xl330-m077","xl330-m077"]'   \
left_signs:="[1,-1,1,-1,1,1,1,1]"   \
right_signs:="[-1,1,-1,-1,-1,1,-1,1]" \
positions_log_path:=/home/kleist/jemotor/log/dynamixel_positions.log \
zero_on_start:=false
#plot
ros2 run je_software plot_dynamixel_positions --log /home/kleist/jemotor/log/dynamixel_positions.log --arm right --joint 7

ros2 run je_software joint_rate_monitor --ros-args \
  -p topic:=/joint_cmd_double_arm \
  -p msg_type:=oculus_init_joint_state \
  -p log_period_s:=1.0

orbbec需要在conda环境之外build
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to orbbec_camera