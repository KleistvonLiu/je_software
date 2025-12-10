from launch import LaunchDescription
from launch_ros.actions import Node

# 启动 Manager node 和 test_publisher node

def generate_launch_description():
    return LaunchDescription([
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
