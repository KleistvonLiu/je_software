1. 进入环境
conda activate ros2-humble-py310
source ~/ros2_ws/install/setup.bash
source /opt/ros/humble/setup.bash

2. 编译
cd ~/ros2_ws
colcon build --symlink-install --packages-select je_software je_common

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
  jsonl_path:=/media/kleist/NewNTFS/20260225log/episode_000000/meta.jsonl\
  rate_hz:=30.0\
  loop:=false     \
  send_arm:=left \
  dt_init:=5 \
  target_string:=cmd
ros2 run orbbec_camera list_devices_node

# 相机节点
# 多个相机
ros2 launch orbbec_camera 3_cameras.launch.py
# 单个相机
ros2 launch orbbec_camera gemini_330_series.launch.py   usb_port:=1-11   enable_color_auto_white_balance:=false   color_width:=1920 color_height:=1080 color_fps:=8

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

# eye-to-hand 标定（默认左臂）
ros2 run je_software eye_to_hand_calibration --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p endpose_sub_topic:=/endpose_states_double_arm \
  -p arm:=left \
  -p image_is_rectified:=true \
  -p hand_eye_method:=all \
  -p min_samples:=12 \
  -p charuco_config_path:=/path/to/charuco_board.json

# 运行后按键
# s 或空格: 保存当前样本
# c: 执行标定并输出结果到 ~/eye_to_hand_calibration/eye_to_hand_calibration_时间戳/
# r: 清空当前会话已采样样本
# q: 退出

# eye-to-hand 外参验证，一致性验证（采新样本验证 gripper_T_target 是否稳定）
ros2 run je_software eye_to_hand_validator --ros-args \
  -p mode:=consistency \
  -p calibration_result_path:=/path/to/eye_to_hand_result.json \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p endpose_sub_topic:=/endpose_states_double_arm \
  -p arm:=left \
  -p image_is_rectified:=true

# 按键：
# s 或空格: 保存一个新姿态样本
# c: 生成一致性报告
# r: 清空当前会话样本
# q: 退出
# 输出目录：~/eye_to_hand_validation/eye_to_hand_validation_consistency_时间戳/
# 结果包含 consistency_samples.json、consistency_summary.json/.txt、
# consistency_overview.png、consistency_region_heatmap.png 和每个样本的 overlay PNG

# eye-to-hand 外参验证，point check（新样本上比较视觉链和机器人链的板点 base 坐标）
ros2 run je_software eye_to_hand_validator --ros-args \
  -p mode:=point_check \
  -p calibration_result_path:=/path/to/eye_to_hand_result.json \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p endpose_sub_topic:=/endpose_states_double_arm \
  -p arm:=left \
  -p image_is_rectified:=true \
  -p point_set:=center_corners \
  -p point_z_offset_m:=0.0 \
  -p charuco_ids_csv:=1,8,15

# 按键：
# n/p: 切换当前 board 点
# m / s / 空格: 记录一个当前点的 point-check 测量
# c: 生成点位验证报告
# r: 清空当前会话测量
# q: 退出
# 输出目录：~/eye_to_hand_validation/eye_to_hand_validation_point_check_时间戳/
# 结果包含 point_check_measurements.json、point_check_summary.json/.txt、
# point_check_overview.png 和每个测量样本的 overlay PNG
# point_z_offset_m 会把所有候选点的 z 一起平移到同一个值，适合板前方固定高度的工具点

# eye-in-hand 标定（默认左臂）
ros2 run je_software eye_in_hand_calibration --ros-args \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p endpose_sub_topic:=/endpose_states_double_arm \
  -p arm:=left \
  -p gripper_frame:=gripper_link \
  -p image_is_rectified:=true \
  -p hand_eye_method:=all \
  -p min_samples:=12 \
  -p charuco_config_path:=/path/to/charuco_board.json

# 输出结果到 ~/eye_in_hand_calibration/eye_in_hand_calibration_时间戳/
# 结果里会包含 gripper_T_camera、base_T_target 和 static_transform_publisher 命令

# eye-in-hand 外参验证，第一层方程一致性验证
ros2 run je_software eye_in_hand_validator --ros-args \
  -p mode:=consistency \
  -p calibration_result_path:=/path/to/eye_in_hand_result.json \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p oculus_topic:=/oculus_controllers \
  -p arm:=left \
  -p image_is_rectified:=true

# 按键：
# s 或空格: 保存一个固定姿态样本
# c: 生成一致性报告
# r: 清空当前会话样本
# q: 退出
# 输出目录：~/eye_in_hand_validation/eye_in_hand_validation_consistency_时间戳/
# 结果包含 consistency_samples.json、consistency_summary.json/.txt、
# consistency_overview.png、consistency_region_heatmap.png 和每个样本的 overlay PNG

# eye-in-hand 外参验证，第五层真实空间点位验证
ros2 run je_software eye_in_hand_validator --ros-args \
  -p mode:=point_check \
  -p calibration_result_path:=/path/to/eye_in_hand_result.json \
  -p image_topic:=/camera/color/image_raw \
  -p camera_info_topic:=/camera/color/camera_info \
  -p oculus_topic:=/oculus_controllers \
  -p arm:=left \
  -p image_is_rectified:=true \
  -p point_set:=center_corners \
  -p point_z_offset_m:=0.0 \
  -p charuco_ids_csv:=1,8,15

# 按键：
# n/p: 切换当前 board 点
# l: 锁定当前点对应的 base 目标位置
# m: 人工对点后记录一次测量
# c: 生成点位验证报告
# r: 清空当前会话
# q: 退出
# 输出目录：~/eye_in_hand_validation/eye_in_hand_validation_point_check_时间戳/
# 结果包含 point_check_locks.json、point_check_measurements.json、
# point_check_summary.json/.txt、point_check_overview.png 和每个 lock 的 overlay PNG
# point_z_offset_m 会把所有候选点的 z 一起平移到同一个值

orbbec需要在conda环境之外build
colcon build --event-handlers console_direct+ --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to orbbec_camera

4. 上传文件到服务器
bbcp   -f -P 5 -s 64 -w 128M -v -r   /home/kleist/Documents/Database/test_0207/ alice@10.215.247.2:/jedata/jemotor/source/

5. 从服务器下载文件
bbcp   -f -P 5 -s 64 -w 128M -v  -r -z alice@10.215.247.2:/jedata/jemotor/model/0207_pi05_test/15000/ /目标路径
