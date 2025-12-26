from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port',       default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('topic',      default_value='/tactile_data'),
        DeclareLaunchArgument('baudrate',   default_value='460800'),
        DeclareLaunchArgument('timeout',    default_value='0.5'),
        DeclareLaunchArgument('frame_size', default_value='70'),
        DeclareLaunchArgument('header_hex', default_value='FF 84'),

        Node(
            package='je_software',              # ← 改成你的包名
            executable='tactile_sensor',     # ← setup.py 里 console_scripts 的可执行名
            name='tactile_sensor',
            parameters=[{
                'port':       LaunchConfiguration('port'),
                'topic':      LaunchConfiguration('topic'),
                'baudrate':   LaunchConfiguration('baudrate'),
                'timeout':    LaunchConfiguration('timeout'),
                'frame_size': LaunchConfiguration('frame_size'),
                'header_hex': LaunchConfiguration('header_hex'),
            }],
            output='screen',
        )
    ])
