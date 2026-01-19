# launch/dynamixel_init_joint_state.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    left_port_arg = DeclareLaunchArgument("left_port", default_value="/dev/ttyUSB0")
    right_port_arg = DeclareLaunchArgument("right_port", default_value="/dev/ttyUSB1")
    left_enabled_arg = DeclareLaunchArgument("left_enabled", default_value="true")
    right_enabled_arg = DeclareLaunchArgument("right_enabled", default_value="true")
    left_baudrate_arg = DeclareLaunchArgument("left_baudrate", default_value="1000000")
    right_baudrate_arg = DeclareLaunchArgument("right_baudrate", default_value="1000000")
    left_ids_arg = DeclareLaunchArgument("left_ids", default_value="[1,2,3,4,5,6,7,8]")
    right_ids_arg = DeclareLaunchArgument("right_ids", default_value="[1,2,3,4,5,6,7,8]")
    left_signs_arg = DeclareLaunchArgument("left_signs", default_value="[1]")
    right_signs_arg = DeclareLaunchArgument("right_signs", default_value="[1]")
    left_models_arg = DeclareLaunchArgument(
        "left_models", default_value='["xl330-m077"]'
    )
    right_models_arg = DeclareLaunchArgument(
        "right_models", default_value='["xl330-m077"]'
    )
    position_scale_arg = DeclareLaunchArgument(
        # "position_scale", default_value="0.087890625" # 360/4096
        "position_scale", default_value="0.001533981" # 2pi/4096
    )
    zero_on_start_arg = DeclareLaunchArgument("zero_on_start", default_value="false")
    zero_file_arg = DeclareLaunchArgument(
        "zero_file",
        default_value=os.path.join(
            os.path.dirname(__file__), "..", "config", "dynamixel_zero_offsets.json"
        ),
    )
    print_positions_arg = DeclareLaunchArgument(
        "print_positions", default_value="true"
    )
    print_period_arg = DeclareLaunchArgument("print_period", default_value="0.1")
    joint_names_arg = DeclareLaunchArgument(
        "joint_names",
        default_value='["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]',
    )
    motor_model_arg = DeclareLaunchArgument("motor_model", default_value="xl330-m077")
    fps_arg = DeclareLaunchArgument("fps", default_value="30.0")
    publish_topic_arg = DeclareLaunchArgument(
        "publish_topic", default_value="/oculus_init_joint_state"
    )
    frame_id_arg = DeclareLaunchArgument("frame_id", default_value="")

    node = Node(
        package="je_software",
        executable="dynamixel_init_joint_state",
        name="dynamixel_init_joint_state",
        output="screen",
        parameters=[
            {
                "left_port": LaunchConfiguration("left_port"),
                "right_port": LaunchConfiguration("right_port"),
                "left_enabled": ParameterValue(
                    LaunchConfiguration("left_enabled"), value_type=bool
                ),
                "right_enabled": ParameterValue(
                    LaunchConfiguration("right_enabled"), value_type=bool
                ),
                "left_baudrate": ParameterValue(
                    LaunchConfiguration("left_baudrate"), value_type=int
                ),
                "right_baudrate": ParameterValue(
                    LaunchConfiguration("right_baudrate"), value_type=int
                ),
                "left_ids": ParameterValue(
                    LaunchConfiguration("left_ids"), value_type=str
                ),
                "right_ids": ParameterValue(
                    LaunchConfiguration("right_ids"), value_type=str
                ),
                "left_signs": ParameterValue(
                    LaunchConfiguration("left_signs"), value_type=str
                ),
                "right_signs": ParameterValue(
                    LaunchConfiguration("right_signs"), value_type=str
                ),
                "left_models": ParameterValue(
                    LaunchConfiguration("left_models"), value_type=str
                ),
                "right_models": ParameterValue(
                    LaunchConfiguration("right_models"), value_type=str
                ),
                "position_scale": ParameterValue(
                    LaunchConfiguration("position_scale"), value_type=float
                ),
                "zero_on_start": ParameterValue(
                    LaunchConfiguration("zero_on_start"), value_type=bool
                ),
                "zero_file": LaunchConfiguration("zero_file"),
                "print_positions": ParameterValue(
                    LaunchConfiguration("print_positions"), value_type=bool
                ),
                "print_period": ParameterValue(
                    LaunchConfiguration("print_period"), value_type=float
                ),
                "joint_names": ParameterValue(
                    LaunchConfiguration("joint_names"), value_type=str
                ),
                "motor_model": LaunchConfiguration("motor_model"),
                "fps": ParameterValue(LaunchConfiguration("fps"), value_type=float),
                "publish_topic": LaunchConfiguration("publish_topic"),
                "frame_id": LaunchConfiguration("frame_id"),
            }
        ],
    )

    fastdds_profiles = DeclareLaunchArgument(
        "fastdds_profiles_file",
        default_value=os.path.expanduser("~/fastdds_shm_only.xml"),
        description="Fast DDS profiles XML（包含 SHM 配置等）",
    )

    return LaunchDescription(
        [
            fastdds_profiles,
            SetEnvironmentVariable("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
            SetEnvironmentVariable("FASTRTPS_LOG_LEVEL", "DEBUG"),
            SetEnvironmentVariable(
                "FASTRTPS_DEFAULT_PROFILES_FILE",
                LaunchConfiguration("fastdds_profiles_file"),
            ),
            left_port_arg,
            right_port_arg,
            left_enabled_arg,
            right_enabled_arg,
            left_baudrate_arg,
            right_baudrate_arg,
            left_ids_arg,
            right_ids_arg,
            left_signs_arg,
            right_signs_arg,
            left_models_arg,
            right_models_arg,
            position_scale_arg,
            zero_on_start_arg,
            zero_file_arg,
            print_positions_arg,
            print_period_arg,
            joint_names_arg,
            motor_model_arg,
            fps_arg,
            publish_topic_arg,
            frame_id_arg,
            node,
        ]
    )
