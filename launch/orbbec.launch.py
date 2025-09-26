from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 允许通过命名空间与外部参数文件覆盖
    namespace = LaunchConfiguration('namespace')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='ROS namespace for the node'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('je_software'),  # 改成你的包名
                'config',
                'orbbec.yaml'
            ]),
            description='Full path to the YAML parameters file'
        ),

        Node(
            package='je_software',          # 改成你的包名
            executable='camera_node',       # 对应 setup.py 里 console_scripts 的可执行名
            name='orbbec_publisher',        # 对应节点名称
            namespace=namespace,
            parameters=[params_file],
            output='screen',
            emulate_tty=True,
            respawn=False,
        )
    ])
