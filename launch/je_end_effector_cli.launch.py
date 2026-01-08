# launch/je_end_effector_cli.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    # -------------------- ROS params (可命令行覆盖) --------------------
    end_effector_topic_arg = DeclareLaunchArgument(
        'end_effector_topic', default_value='/end_effector_cmd')
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic', default_value='/end_pose')
    frame_id_arg = DeclareLaunchArgument(
        'frame_id', default_value='base_link')
    send_init_pose_arg = DeclareLaunchArgument(
        'send_init_pose_on_start', default_value='false')
    attach_init_pose_arg = DeclareLaunchArgument(
        'attach_init_pose_to_cmd', default_value='false')
    init_pose_arg = DeclareLaunchArgument(
        'init_pose', default_value='[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]')

    node = Node(
        package="je_software",
        executable="end_effector_cli",
        name="end_effector_cli",
        output="screen",
        parameters=[
            {
                "end_effector_topic": LaunchConfiguration("end_effector_topic"),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "frame_id": LaunchConfiguration("frame_id"),
                "send_init_pose_on_start": ParameterValue(
                    LaunchConfiguration("send_init_pose_on_start"), value_type=bool),
                "attach_init_pose_to_cmd": ParameterValue(
                    LaunchConfiguration("attach_init_pose_to_cmd"), value_type=bool),
                "init_pose": ParameterValue(
                    LaunchConfiguration("init_pose"), value_type=str),
            }
        ],
    )

    # --- Fast DDS 配置（沿用 je_robot_node.launch.py 模板，可选） ---
    fastdds_profiles = DeclareLaunchArgument(
        'fastdds_profiles_file',
        default_value=os.path.expanduser('~/fastdds_shm_only.xml'),
        description='Fast DDS profiles XML（包含 SHM 配置等）'
    )

    return LaunchDescription([
        fastdds_profiles,
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        SetEnvironmentVariable('FASTRTPS_LOG_LEVEL', 'DEBUG'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', LaunchConfiguration('fastdds_profiles_file')),

        end_effector_topic_arg,
        pose_topic_arg,
        frame_id_arg,
        send_init_pose_arg,
        attach_init_pose_arg,
        init_pose_arg,

        node
    ])
