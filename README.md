colcon build --symlink-install
source install/setup.bash
ros2 launch je_software manager_test.launch.py
