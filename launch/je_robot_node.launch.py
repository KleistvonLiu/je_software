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

    # -------------------- ZMQ params --------------------
    robot_ip_arg  = DeclareLaunchArgument('robot_ip',  default_value='192.168.0.99')
    pub_port_arg  = DeclareLaunchArgument('pub_port',  default_value='8001')
    sub_port_arg  = DeclareLaunchArgument('sub_port',  default_value='8000')

    # -------------------- IK / solver parameters (per-arm) --------------------
    robot_description_left_arg = DeclareLaunchArgument('robot_description_left', default_value='')
    robot_description_right_arg = DeclareLaunchArgument('robot_description_right', default_value='')
    ik_left_tip_frame_arg = DeclareLaunchArgument('ik_left_tip_frame', default_value='left_ee_link')
    ik_right_tip_frame_arg = DeclareLaunchArgument('ik_right_tip_frame', default_value='right_ee_link')

    # left solver tuning
    ik_left_max_iters_arg = DeclareLaunchArgument('ik_left_max_iters', default_value='200')
    ik_left_eps_arg = DeclareLaunchArgument('ik_left_eps', default_value='1e-4')
    ik_left_pos_weight_arg = DeclareLaunchArgument('ik_left_pos_weight', default_value='1.0')
    ik_left_ang_weight_arg = DeclareLaunchArgument('ik_left_ang_weight', default_value='1.0')
    ik_left_use_numeric_jacobian_arg = DeclareLaunchArgument('ik_left_use_numeric_jacobian', default_value='true')
    ik_left_ik_svd_damping_arg = DeclareLaunchArgument('ik_left_ik_svd_damping', default_value='1e-6')
    ik_left_max_delta_arg = DeclareLaunchArgument('ik_left_max_delta', default_value='0.03')
    ik_left_nullspace_penalty_scale_arg = DeclareLaunchArgument('ik_left_nullspace_penalty_scale', default_value='1e-4')

    # right solver tuning
    ik_right_max_iters_arg = DeclareLaunchArgument('ik_right_max_iters', default_value='200')
    ik_right_eps_arg = DeclareLaunchArgument('ik_right_eps', default_value='1e-4')
    ik_right_pos_weight_arg = DeclareLaunchArgument('ik_right_pos_weight', default_value='1.0')
    ik_right_ang_weight_arg = DeclareLaunchArgument('ik_right_ang_weight', default_value='1.0')
    ik_right_use_numeric_jacobian_arg = DeclareLaunchArgument('ik_right_use_numeric_jacobian', default_value='true')
    ik_right_ik_svd_damping_arg = DeclareLaunchArgument('ik_right_ik_svd_damping', default_value='1e-6')
    ik_right_max_delta_arg = DeclareLaunchArgument('ik_right_max_delta', default_value='0.03')
    ik_right_nullspace_penalty_scale_arg = DeclareLaunchArgument('ik_right_nullspace_penalty_scale', default_value='1e-4')

    # IK solve timeouts (milliseconds)
    ik_left_timeout_ms_arg = DeclareLaunchArgument('ik_left_timeout_ms', default_value='100')
    ik_right_timeout_ms_arg = DeclareLaunchArgument('ik_right_timeout_ms', default_value='100')

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

                # IK parameters
                "robot_description_left": LaunchConfiguration("robot_description_left"),
                "robot_description_right": LaunchConfiguration("robot_description_right"),
                "ik_left_tip_frame": LaunchConfiguration("ik_left_tip_frame"),
                "ik_right_tip_frame": LaunchConfiguration("ik_right_tip_frame"),

                "ik_left_max_iters": ParameterValue(LaunchConfiguration("ik_left_max_iters"), value_type=int),
                "ik_left_eps": ParameterValue(LaunchConfiguration("ik_left_eps"), value_type=float),
                "ik_left_pos_weight": ParameterValue(LaunchConfiguration("ik_left_pos_weight"), value_type=float),
                "ik_left_ang_weight": ParameterValue(LaunchConfiguration("ik_left_ang_weight"), value_type=float),
                "ik_left_use_numeric_jacobian": ParameterValue(LaunchConfiguration("ik_left_use_numeric_jacobian"), value_type=bool),
                "ik_left_ik_svd_damping": ParameterValue(LaunchConfiguration("ik_left_ik_svd_damping"), value_type=float),
                "ik_left_max_delta": ParameterValue(LaunchConfiguration("ik_left_max_delta"), value_type=float),
                "ik_left_nullspace_penalty_scale": ParameterValue(LaunchConfiguration("ik_left_nullspace_penalty_scale"), value_type=float),

                "ik_right_max_iters": ParameterValue(LaunchConfiguration("ik_right_max_iters"), value_type=int),
                "ik_right_eps": ParameterValue(LaunchConfiguration("ik_right_eps"), value_type=float),
                "ik_right_pos_weight": ParameterValue(LaunchConfiguration("ik_right_pos_weight"), value_type=float),
                "ik_right_ang_weight": ParameterValue(LaunchConfiguration("ik_right_ang_weight"), value_type=float),
                "ik_right_use_numeric_jacobian": ParameterValue(LaunchConfiguration("ik_right_use_numeric_jacobian"), value_type=bool),
                "ik_right_ik_svd_damping": ParameterValue(LaunchConfiguration("ik_right_ik_svd_damping"), value_type=float),
                "ik_right_max_delta": ParameterValue(LaunchConfiguration("ik_right_max_delta"), value_type=float),
                "ik_right_nullspace_penalty_scale": ParameterValue(LaunchConfiguration("ik_right_nullspace_penalty_scale"), value_type=float),

                # IK timeouts
                "ik_left_timeout_ms": ParameterValue(LaunchConfiguration("ik_left_timeout_ms"), value_type=int),
                "ik_right_timeout_ms": ParameterValue(LaunchConfiguration("ik_right_timeout_ms"), value_type=int),
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
        robot_ip_arg,
        pub_port_arg,
        sub_port_arg,
        robot_description_left_arg,
        robot_description_right_arg,
        ik_left_tip_frame_arg,
        ik_right_tip_frame_arg,
        ik_left_max_iters_arg,
        ik_left_eps_arg,
        ik_left_pos_weight_arg,
        ik_left_ang_weight_arg,
        ik_left_use_numeric_jacobian_arg,
        ik_left_ik_svd_damping_arg,
        ik_left_max_delta_arg,
        ik_left_nullspace_penalty_scale_arg,
        ik_right_max_iters_arg,
        ik_right_eps_arg,
        ik_right_pos_weight_arg,
        ik_right_ang_weight_arg,
        ik_right_use_numeric_jacobian_arg,
        ik_right_ik_svd_damping_arg,
        ik_right_max_delta_arg,
        ik_right_nullspace_penalty_scale_arg,
        ik_left_timeout_ms_arg,
        ik_right_timeout_ms_arg,

        node
    ])
