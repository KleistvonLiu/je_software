1. 进入环境
conda activate ros2-humble-py310-new
source ~/ros2_ws/install/setup.bash
source /opt/ros/humble/setup.bash

2. 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select je_software

3. 运行程序
编译完之后需要
source ~/ros2_ws/install/setup.bash

# shm
非launch文件启动的节点需要设置shm
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_shm_only.xml

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
or
left_port:=/dev/serial/by-path/pci-0000:00:14.0-usb-0:8:1.0-port0  \
right_port:=/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0   \
left_enabled:=true \
right_enabled:=false   \
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

4. 上传文件到服务器
bbcp   -f -P 5 -s 64 -w 128M -v -r   /home/kleist/Documents/Database/test_0207/ alice@10.215.247.2:/jedata/jemotor/source/

5. 从服务器下载文件
bbcp   -f -P 5 -s 64 -w 128M -v  alice@10.215.247.2:/jedata/jemotor/model/0207_pi05_test/15000/ /目标路径