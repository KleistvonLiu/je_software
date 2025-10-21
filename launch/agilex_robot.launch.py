# launch/agilex_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 可配置参数（可在命令行覆盖）
    joint_sub_topic_arg = DeclareLaunchArgument('joint_sub_topic', default_value='/joint_states')
    end_pose_topic_arg  = DeclareLaunchArgument('end_pose_topic',  default_value='/end_pose')
    joint_pub_topic_arg = DeclareLaunchArgument('joint_pub_topic', default_value='/joint_state')
    fps_arg             = DeclareLaunchArgument('fps',             default_value='50')          # 字符串即可
    can_port_arg        = DeclareLaunchArgument('can_port',        default_value='can_right')

    node = Node(
        package='je_software',          # <<< 修改为你的包名
        executable='agilex_robot',     # <<< 修改为你的可执行名（console_scripts 暴露的名字）
        name='agilex_robot_node',
        output='screen',
        parameters=[{
            'joint_sub_topic': LaunchConfiguration('joint_sub_topic'),
            'end_pose_topic':  LaunchConfiguration('end_pose_topic'),
            'joint_pub_topic': LaunchConfiguration('joint_pub_topic'),
            'fps':             LaunchConfiguration('fps'),
            'can_port':        LaunchConfiguration('can_port'),
        }],
        # 如你更偏向用 remap 配 topic，也可以用 remappings=[('原名','新名')] 的方式
    )

    return LaunchDescription([
        joint_sub_topic_arg,
        end_pose_topic_arg,
        joint_pub_topic_arg,
        fps_arg,
        can_port_arg,
        node
    ])