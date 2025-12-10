import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# 启动 Manager node 和 test_publisher node

# --- 新增：Fast DDS 配置文件（默认指向 4GB SHM 配置，可在命令行覆盖） ---
fastdds_profiles = DeclareLaunchArgument(
    'fastdds_profiles_file',
    default_value=os.path.expanduser('~/fastdds_shm_only.xml'),
    description='Fast DDS profiles XML（包含 <type>SHM</type> 与 segment_size=4GiB）'
)

def generate_launch_description():
    return LaunchDescription([
        # 声明 fastdds 配置文件参数（确保 LaunchConfiguration 可用）
        fastdds_profiles,
        # 让 XML 在整个 Launch 会话中生效
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', LaunchConfiguration('fastdds_profiles_file')),

        Node(
            package='je_software',
            executable='manager_node',  # Python 节点
            name='manager',
            output='screen',
            parameters=[
                {'color_topics_csv': 'cam0,cam1,cam2,cam3'},
                {'depth_topics_csv': 'dep0,dep1,dep2,dep3'},
                {'joint_state_topic': 'joint'},
                {'tactile_topic': 'tactile'},
                {'save_dir': '/home/test/jedata/je_dataset/'},
                {'episode_idx': 0},
                {'mode': 1},
            ],
        ),
        Node(
            package='je_software',
            executable='test_publisher',
            name='test_publisher',
            output='screen',
        ),
    ])
