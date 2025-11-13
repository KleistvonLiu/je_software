# launch/agilex_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    # 可配置参数（可在命令行覆盖）
    joint_sub_topic_arg = DeclareLaunchArgument('joint_sub_topic', default_value='/joint_cmd')
    end_pose_topic_arg  = DeclareLaunchArgument('end_pose_topic',  default_value='/end_pose')
    joint_pub_topic_arg = DeclareLaunchArgument('joint_pub_topic', default_value='/joint_states')
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

    # --- 新增：Fast DDS 配置文件（默认指向 4GB SHM 配置，可在命令行覆盖） ---
    fastdds_profiles = DeclareLaunchArgument(
        'fastdds_profiles_file',
        default_value=os.path.expanduser('~/fastdds_shm_only.xml'),
        description='Fast DDS profiles XML（包含 <type>SHM</type> 与 segment_size=4GiB）'
    )

    return LaunchDescription([
        # Declare the argument so LaunchConfiguration('fastdds_profiles_file') exists
        fastdds_profiles,
        # 让 XML 在整个 Launch 会话中生效
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', LaunchConfiguration('fastdds_profiles_file')),
        joint_sub_topic_arg,
        end_pose_topic_arg,
        joint_pub_topic_arg,
        fps_arg,
        can_port_arg,
        node
    ])