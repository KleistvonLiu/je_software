from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def _as_bool(text: str) -> bool:
    return text.strip().lower() in ('1', 'true', 'yes', 'on')


def _make_nodes(context):
    urdf_path = LaunchConfiguration('urdf_path').perform(context)
    jsonl_path = LaunchConfiguration('jsonl_path').perform(context)
    joint_state_topic = LaunchConfiguration('joint_state_topic').perform(context)
    robot_description_topic = LaunchConfiguration('robot_description_topic').perform(context)
    frame_prefix = LaunchConfiguration('frame_prefix').perform(context)
    launch_replayer = _as_bool(LaunchConfiguration('launch_replayer').perform(context))
    rate_hz = float(LaunchConfiguration('rate_hz').perform(context))
    paused = _as_bool(LaunchConfiguration('paused').perform(context))
    use_recorded_timestamps = _as_bool(
        LaunchConfiguration('use_recorded_timestamps').perform(context)
    )
    follow_recorded_timing = _as_bool(
        LaunchConfiguration('follow_recorded_timing').perform(context)
    )

    if launch_replayer and not jsonl_path:
        raise RuntimeError('Launch argument "jsonl_path" must be provided.')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f'URDF file does not exist: {urdf_path}')

    with open(urdf_path, 'r', encoding='utf-8') as handle:
        robot_description = handle.read()

    rviz_config = os.path.join(
        get_package_share_directory('je_software'),
        'rviz',
        'jearm_replay.rviz',
    )

    nodes = [
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='jearm_replay_robot_state_publisher',
            output='screen',
            parameters=[
                {
                    'robot_description': robot_description,
                    'frame_prefix': frame_prefix,
                }
            ],
            remappings=[
                ('joint_states', joint_state_topic),
                ('robot_description', robot_description_topic),
            ],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='jearm_replay_rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ]

    if launch_replayer:
        nodes.insert(
            1,
            Node(
                package='je_software',
                executable='jearm_jsonl_replayer_node',
                name='jearm_jsonl_replayer_node',
                output='screen',
                parameters=[
                    {
                        'jsonl_path': jsonl_path,
                        'joint_state_topic': joint_state_topic,
                        'rate_hz': rate_hz,
                        'paused': paused,
                        'use_recorded_timestamps': use_recorded_timestamps,
                        'follow_recorded_timing': follow_recorded_timing,
                    }
                ],
            ),
        )

    return nodes


def generate_launch_description():
    default_urdf = os.path.join(
        get_package_share_directory('je_software'),
        'urdf',
        'URDF_LAST',
        'LH_JEARM',
        'LH_JEARM.urdf',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'urdf_path',
                default_value=default_urdf,
                description='Path to the JEARM URDF file.',
            ),
            DeclareLaunchArgument(
                'jsonl_path',
                default_value='',
                description='Path to the JSONL replay file.',
            ),
            DeclareLaunchArgument(
                'joint_state_topic',
                default_value='/jearm_replay/joint_states',
                description='Dedicated JointState topic for the replay pipeline.',
            ),
            DeclareLaunchArgument(
                'launch_replayer',
                default_value='true',
                description='Whether to start the JSONL replayer node. Set false to consume an external JointState topic.',
            ),
            DeclareLaunchArgument(
                'robot_description_topic',
                default_value='/jearm_replay/robot_description',
                description='Dedicated robot_description topic for the replay pipeline.',
            ),
            DeclareLaunchArgument(
                'frame_prefix',
                default_value='jearm_replay/',
                description='TF frame prefix used to isolate replay frames from other robots.',
            ),
            DeclareLaunchArgument(
                'rate_hz',
                default_value='120.0',
                description='JointState output rate in Hz.',
            ),
            DeclareLaunchArgument(
                'paused',
                default_value='false',
                description='Start replay in paused mode.',
            ),
            DeclareLaunchArgument(
                'use_recorded_timestamps',
                default_value='false',
                description='Replay JointState header stamps from the log instead of current wall time.',
            ),
            DeclareLaunchArgument(
                'follow_recorded_timing',
                default_value='true',
                description='Follow the log timing and interpolate between frames for smoother playback.',
            ),
            OpaqueFunction(function=_make_nodes),
        ]
    )
