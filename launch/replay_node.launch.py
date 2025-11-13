from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def _make_execute(context):
    # Run the replay script as a Python module to avoid launch appending --ros-args
    jsonl = LaunchConfiguration('jsonl').perform(context)
    rate = LaunchConfiguration('rate').perform(context)
    loop = LaunchConfiguration('loop').perform(context).lower()

    cmd = ['python3', '-m', 'je_software.replay_node', jsonl, '--rate', rate]
    if loop in ('false', '0', 'no'):
        cmd.append('--no-loop')

    return [ExecuteProcess(cmd=cmd, output='screen')]

def generate_launch_description():
    jsonl_arg = DeclareLaunchArgument(
        'jsonl',
        default_value=os.path.join(get_package_share_directory('je_software'), '..', 'log', 'episode_000000', 'meta.jsonl'),
        description='Path to the .jsonl file'
    )
    rate_arg = DeclareLaunchArgument('rate', default_value='30.0', description='Publish rate in Hz')
    loop_arg = DeclareLaunchArgument('loop', default_value='true', description='Loop playback (true/false)')

    # --- 新增：Fast DDS 配置文件（默认指向 4GB SHM 配置，可在命令行覆盖） ---
    fastdds_profiles = DeclareLaunchArgument(
        'fastdds_profiles_file',
        default_value='~/fastdds_shm_4g.xml',
        description='Fast DDS profiles XML（包含 <type>SHM</type> 与 segment_size=4GiB）'
    )

    return LaunchDescription([
        # 让 XML 在整个 Launch 会话中生效
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', LaunchConfiguration('fastdds_profiles_file')),
        jsonl_arg,
        rate_arg,
        loop_arg,
        OpaqueFunction(function=_make_execute)
    ])