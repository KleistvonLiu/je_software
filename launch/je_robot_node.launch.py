# launch/je_robot.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os

def generate_launch_description():
    # -------------------- ROS params (可命令行覆盖) --------------------
    joint_sub_topic_arg = DeclareLaunchArgument('joint_sub_topic', default_value='/joint_cmd')
    end_pose_topic_arg  = DeclareLaunchArgument('end_pose_topic',  default_value='/end_pose')
    joint_pub_topic_arg = DeclareLaunchArgument('joint_pub_topic', default_value='/joint_states_double_arm')
    oculus_controllers_topic_arg = DeclareLaunchArgument(
        'oculus_controllers_topic', default_value='/oculus_controllers')
    oculus_init_joint_state_topic_arg = DeclareLaunchArgument(
        'oculus_init_joint_state_topic', default_value='/oculus_init_joint_state')
    gripper_sub_topic_arg = DeclareLaunchArgument(
        'gripper_sub_topic', default_value='/end_effector_cmd_lr')
    fps_arg             = DeclareLaunchArgument('fps',             default_value='50')
    dt_arg              = DeclareLaunchArgument('dt',              default_value='0.014')
    dt_init_arg         = DeclareLaunchArgument('dt_init',         default_value='3.0')
    # per-solve IK console logging
    ik_log_arg = DeclareLaunchArgument('ik_log', default_value='true')

    # -------------------- ZMQ params --------------------
    robot_ip_arg  = DeclareLaunchArgument('robot_ip',  default_value='192.168.0.99')
    pub_port_arg  = DeclareLaunchArgument('pub_port',  default_value='8001')
    sub_port_arg  = DeclareLaunchArgument('sub_port',  default_value='8000')

    # -------------------- IK YAML paths (per-arm) --------------------
    ik_left_yaml_path_arg = DeclareLaunchArgument(
        'ik_left_yaml_path',
        default_value='/home/test/ros2_ws/src/je_software/config/ik_para_left.yaml')
    ik_right_yaml_path_arg = DeclareLaunchArgument(
        'ik_right_yaml_path',
        default_value='/home/test/ros2_ws/src/je_software/config/ik_para_right.yaml')

    # 你的节点（按需改 package / executable）
    node = Node(
        package="je_software",  # <<< 改成你的包名
        executable="je_robot_node",  # <<< 改成你的可执行名（ament_cmake install 后的可执行文件名）
        name="je_robot_node",
        output="screen",
        parameters=[
            {
                "joint_sub_topic": LaunchConfiguration("joint_sub_topic"),
                "end_pose_topic": LaunchConfiguration("end_pose_topic"),
                "joint_pub_topic": LaunchConfiguration("joint_pub_topic"),
                "oculus_controllers_topic": LaunchConfiguration("oculus_controllers_topic"),
                "oculus_init_joint_state_topic": LaunchConfiguration("oculus_init_joint_state_topic"),
                "gripper_sub_topic": LaunchConfiguration("gripper_sub_topic"),
                "fps": ParameterValue(LaunchConfiguration("fps"), value_type=float),
                "dt": ParameterValue(LaunchConfiguration("dt"), value_type=float),
                "dt_init": ParameterValue(LaunchConfiguration("dt_init"), value_type=float),
                "robot_ip": LaunchConfiguration("robot_ip"),
                "pub_port": LaunchConfiguration("pub_port"),
                "sub_port": LaunchConfiguration("sub_port"),
                "ik_log": ParameterValue(LaunchConfiguration("ik_log"), value_type=bool),
                "ik_left_yaml_path": LaunchConfiguration("ik_left_yaml_path"),
                "ik_right_yaml_path": LaunchConfiguration("ik_right_yaml_path"),
            }
        ],
        # 如需 remap，可加：remappings=[('/joint_cmd','/xxx'), ...]
    )

    # --- Fast DDS 配置（沿用你给的模板，可选） ---
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

        joint_sub_topic_arg,
        end_pose_topic_arg,
        joint_pub_topic_arg,
        oculus_controllers_topic_arg,
        oculus_init_joint_state_topic_arg,
        gripper_sub_topic_arg,
        fps_arg,
        dt_arg,
        dt_init_arg,
        ik_log_arg,
        robot_ip_arg,
        pub_port_arg,
        sub_port_arg,
        ik_left_yaml_path_arg,
        ik_right_yaml_path_arg,

        node
    ])
