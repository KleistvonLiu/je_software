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

    # -------------------- Topics --------------------
    oculus_controllers_topic_arg = DeclareLaunchArgument(
        'oculus_controllers_topic', default_value='/oculus_controllers')
    oculus_init_joint_state_topic_arg = DeclareLaunchArgument(
        'oculus_init_joint_state_topic', default_value='/oculus_init_joint_state')

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
    joint_init_flag_arg = DeclareLaunchArgument('joint_init_flag', default_value='false')
    init_left_joint_position_arg = DeclareLaunchArgument(
        'init_left_joint_position', default_value='[-0.6627995426156521,-1.0077295038416256,0.0804381175647535,-0.9015732396299769,-0.8931363452966059,-0.5050631744113485,1.1892425742581567]')
    init_right_joint_position_arg = DeclareLaunchArgument(
        'init_right_joint_position', default_value='[-0.07940747422289282,-1.2996412540863018,-1.0468460139327096,-1.8282414462117706,-1.3567341015354206,0.5511784718471607,2.3845489290070394]')
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
                "joint_init_flag": ParameterValue(LaunchConfiguration("joint_init_flag"), value_type=bool),
                "init_left_joint_position": ParameterValue(
                    LaunchConfiguration("init_left_joint_position"), value_type=str),
                "init_right_joint_position": ParameterValue(
                    LaunchConfiguration("init_right_joint_position"), value_type=str),
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
        joint_init_flag_arg,
        init_left_joint_position_arg,
        init_right_joint_position_arg,
        init_left_valid_arg,
        init_right_valid_arg,

        node
    ])
