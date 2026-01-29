# launch/jsonl_replayer_node.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    # -------------------- Replay params --------------------
    jsonl_path_arg = DeclareLaunchArgument('jsonl_path', default_value='')
    rate_hz_arg = DeclareLaunchArgument('rate_hz', default_value='50.0')
    loop_arg = DeclareLaunchArgument('loop', default_value='false')
    output_type_arg = DeclareLaunchArgument('output_type', default_value='oculus_joint')
    use_file_stamp_arg = DeclareLaunchArgument('use_file_stamp', default_value='true')
    dt_init_arg = DeclareLaunchArgument('dt_init', default_value='5.0')
    init_repeat_count_arg = DeclareLaunchArgument('init_repeat_count', default_value='10')

    # -------------------- Topics --------------------
    oculus_controllers_topic_arg = DeclareLaunchArgument(
        'oculus_controllers_topic', default_value='/oculus_controllers')
    oculus_init_joint_state_topic_arg = DeclareLaunchArgument(
        'oculus_init_joint_state_topic', default_value='/joint_cmd_double_arm')

    # -------------------- Pose options --------------------
    frame_id_arg = DeclareLaunchArgument('frame_id', default_value='base_link')
    pose_field_arg = DeclareLaunchArgument('pose_field', default_value='Cartesian')
    pose_left_valid_arg = DeclareLaunchArgument('pose_left_valid', default_value='true')
    pose_right_valid_arg = DeclareLaunchArgument('pose_right_valid', default_value='true')
    send_arm_arg = DeclareLaunchArgument('send_arm', default_value='both')

    # -------------------- Joint options --------------------
    joint_position_field_arg = DeclareLaunchArgument('joint_position_field', default_value='Joint')
    joint_velocity_field_arg = DeclareLaunchArgument('joint_velocity_field', default_value='')
    joint_effort_field_arg = DeclareLaunchArgument('joint_effort_field', default_value='')
    init_left_valid_arg = DeclareLaunchArgument('init_left_valid', default_value='true')
    init_right_valid_arg = DeclareLaunchArgument('init_right_valid', default_value='true')

    node = Node(
        package="je_software",
        executable="jsonl_replayer_node",
        name="jsonl_replayer_node",
        output="screen",
        parameters=[
            {
                "jsonl_path": LaunchConfiguration("jsonl_path"),
                "rate_hz": ParameterValue(LaunchConfiguration("rate_hz"), value_type=float),
                "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
                "output_type": LaunchConfiguration("output_type"),
                "use_file_stamp": ParameterValue(LaunchConfiguration("use_file_stamp"), value_type=bool),
                "dt_init": ParameterValue(LaunchConfiguration("dt_init"), value_type=float),
                "init_repeat_count": ParameterValue(
                    LaunchConfiguration("init_repeat_count"), value_type=int
                ),
                "oculus_controllers_topic": LaunchConfiguration("oculus_controllers_topic"),
                "oculus_init_joint_state_topic": LaunchConfiguration("oculus_init_joint_state_topic"),
                "frame_id": LaunchConfiguration("frame_id"),
                "pose_field": LaunchConfiguration("pose_field"),
                "pose_left_valid": ParameterValue(LaunchConfiguration("pose_left_valid"), value_type=bool),
                "pose_right_valid": ParameterValue(LaunchConfiguration("pose_right_valid"), value_type=bool),
                "send_arm": LaunchConfiguration("send_arm"),
                "joint_position_field": LaunchConfiguration("joint_position_field"),
                "joint_velocity_field": LaunchConfiguration("joint_velocity_field"),
                "joint_effort_field": LaunchConfiguration("joint_effort_field"),
                "init_left_valid": ParameterValue(LaunchConfiguration("init_left_valid"), value_type=bool),
                "init_right_valid": ParameterValue(LaunchConfiguration("init_right_valid"), value_type=bool),
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

        jsonl_path_arg,
        rate_hz_arg,
        loop_arg,
        output_type_arg,
        use_file_stamp_arg,
    dt_init_arg,
    init_repeat_count_arg,
        oculus_controllers_topic_arg,
        oculus_init_joint_state_topic_arg,
        frame_id_arg,
        pose_field_arg,
        pose_left_valid_arg,
        pose_right_valid_arg,
        send_arm_arg,
        joint_position_field_arg,
        joint_velocity_field_arg,
        joint_effort_field_arg,
        init_left_valid_arg,
        init_right_valid_arg,

        node
    ])
