from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # ---- 环境变量（Fast DDS 4GB SHM + 可选 RMW）----
    fastdds_profiles_arg = DeclareLaunchArgument(
        'fastdds_profiles_file',
        default_value=os.path.expanduser('~/fastdds_shm_only.xml'),
        description='Fast DDS profiles XML（包含 <type>SHM</type> 与 segment_size=4GiB）'
    )
    rmw_arg = DeclareLaunchArgument(
        'rmw',
        default_value='rmw_fastrtps_cpp',
        description='可选：指定 RMW 实现（rmw_fastrtps_cpp / rmw_cyclonedds_cpp ...）'
    )
    set_fastdds_env = SetEnvironmentVariable(
        'FASTRTPS_DEFAULT_PROFILES_FILE',
        LaunchConfiguration('fastdds_profiles_file')
    )
    set_rmw_env = SetEnvironmentVariable(
        'RMW_IMPLEMENTATION',
        LaunchConfiguration('rmw')
    )

    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')
    base_launch = os.path.join(launch_file_dir, 'gemini_330_series.launch.py')

    common_args = {
        # —— 公共参数（3 台相同）——
        'device_num': '4',
        'sync_mode': 'standalone',

        # 彩色
        'enable_color': 'true',
        'color_width': '640',
        'color_height': '480',
        'color_fps': '30',
        'color_format': 'YUYV',  # 注意：常用是 YUYV（不是 YUVY）

        # 深度
        'enable_depth': 'true',
        'depth_width': '640',
        'depth_height': '480',
        'depth_fps': '30',
        'depth_format': 'Y16',

        # 左右红外
        'enable_left_ir': 'true',
        'enable_right_ir': 'true',
        'left_ir_width': '640',
        'left_ir_height': '480',
        'left_ir_fps': '30',
        'left_ir_format': 'Y8',
        'right_ir_width': '640',
        'right_ir_height': '480',
        'right_ir_fps': '30',
        'right_ir_format': 'Y8',
    }

    cam1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={**common_args, 'camera_name': 'camera_01', 'usb_port': '2-4'}.items()
    )
    cam2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={**common_args, 'camera_name': 'camera_02', 'usb_port': '2-8'}.items()
    )
    cam3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={**common_args, 'camera_name': 'camera_03', 'usb_port': '2-1'}.items()
    )
    cam4 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={**common_args, 'camera_name': 'camera_04', 'usb_port': '2-3'}.items()
    )

    return LaunchDescription([
        fastdds_profiles_arg,
        rmw_arg,
        set_fastdds_env,
        set_rmw_env,
        GroupAction([cam1]),
        GroupAction([cam2]),
        GroupAction([cam3]),
        GroupAction([cam4]),
    ])
