from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('je_software'),
        'config',
        'pcb_process_demo.yaml',
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Parameter file for the fixed PCB process demo.',
    )

    config_file = LaunchConfiguration('config_file')
    moveit_launch = os.path.join(
        get_package_share_directory('je_software'),
        'launch',
        'l_jearm_moveit.launch.py',
    )

    nodes = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(moveit_launch),
        ),
        Node(
            package='je_software',
            executable='terminal_key_signal_node',
            name='terminal_key_signal_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='je_software',
            executable='fixed_line_vision_node',
            name='fixed_line_vision_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='je_software',
            executable='fixed_inspection_node',
            name='fixed_inspection_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='je_software',
            executable='fixed_slot_provider_node',
            name='fixed_slot_provider_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='je_software',
            executable='zmq_motion_backend_node',
            name='zmq_motion_backend_node',
            output='screen',
            parameters=[config_file],
        ),
        Node(
            package='je_software',
            executable='pcb_process_task_manager_node',
            name='pcb_process_task_manager_node',
            output='screen',
            parameters=[config_file],
        ),
    ]

    return LaunchDescription([config_file_arg, *nodes])
