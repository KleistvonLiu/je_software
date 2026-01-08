from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_arg = DeclareLaunchArgument('model_xml', default_value='/home/agx/price/issac-sim/URDF_LAST/LH_JEARM/zongzhuang2.xml')
    use_viewer_arg = DeclareLaunchArgument('use_viewer', default_value='true')
    joint_names_arg = DeclareLaunchArgument('joint_names', default_value='["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]')

    model_xml = LaunchConfiguration('model_xml')
    use_viewer = LaunchConfiguration('use_viewer')
    joint_names = LaunchConfiguration('joint_names')

    node = Node(
        package='je_software',
        executable='mujoco_sim_node',
        name='mujoco_sim_node',
        output='screen',
        parameters=[{
            'state_topic': '/joint_states',
            'model_xml': model_xml,
            'use_viewer': use_viewer,
            'joint_names': joint_names,
            'control_frequency': 50.0,
        }]
    )

    return LaunchDescription([model_arg, use_viewer_arg, joint_names_arg, node])
