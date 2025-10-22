from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _as_bool(s: str, default: bool) -> bool:
    if s is None:
        return default
    s = str(s).strip().lower()
    if s in ('1', 'true', 'yes', 'on'):
        return True
    if s in ('0', 'false', 'no', 'off'):
        return False
    return default


def _parse_list(context, name: str):
    """支持两种写法：'[0,1,2]'（YAML）或 '0,1,2'（CSV）；返回 float 列表。"""
    raw = LaunchConfiguration(name).perform(context) or ''
    if not raw.strip():
        return []
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, (list, tuple)):
            return [float(x) for x in data]
        if isinstance(data, str):
            parts = [p.strip() for p in data.split(',') if p.strip()]
            return [float(x) for x in parts]
    except Exception:
        pass
    # 兜底：给一个空列表，避免类型不匹配
    return []


def _launch_setup(context, *args, **kwargs):
    # 读取简单标量
    pkg = LaunchConfiguration('package').perform(context)
    exe = LaunchConfiguration('executable').perform(context)
    ns = LaunchConfiguration('namespace')
    name = LaunchConfiguration('node_name')

    params = {
        'color_topics_csv': LaunchConfiguration('color_topics_csv').perform(context),
        'depth_topics_csv': LaunchConfiguration('depth_topics_csv').perform(context),
        'joint_state_topic': LaunchConfiguration('joint_state_topic').perform(context),
        'tactile_topic': LaunchConfiguration('tactile_topic').perform(context),
        'rate_hz': float(LaunchConfiguration('rate_hz').perform(context)),
        'image_tolerance_ms': float(LaunchConfiguration('image_tolerance_ms').perform(context)),
        'joint_tolerance_ms': float(LaunchConfiguration('joint_tolerance_ms').perform(context)),
        'tactile_tolerance_ms': float(LaunchConfiguration('tactile_tolerance_ms').perform(context)),
        'queue_seconds': float(LaunchConfiguration('queue_seconds').perform(context)),
        'save_dir': LaunchConfiguration('save_dir').perform(context),
        'session_name': LaunchConfiguration('session_name').perform(context),
        'save_depth': _as_bool(LaunchConfiguration('save_depth').perform(context), True),
        'use_ros_time': _as_bool(LaunchConfiguration('use_ros_time').perform(context), True),
        'color_jpeg_quality': int(float(LaunchConfiguration('color_jpeg_quality').perform(context))),
        'joint_offset_ms': float(LaunchConfiguration('joint_offset_ms').perform(context)),
        'tactile_offset_ms': float(LaunchConfiguration('tactile_offset_ms').perform(context)),
    }

    # 只有非空时再设置列表参数，避免空序列触发 launch 的类型推断问题
    color_offs = _parse_list(context, 'color_offsets_ms')
    depth_offs = _parse_list(context, 'depth_offsets_ms')
    if color_offs:
        params['color_offsets_ms'] = color_offs
    if depth_offs:
        params['depth_offsets_ms'] = depth_offs

    node = Node(
        package=pkg,
        executable=exe,
        namespace=ns,
        name=name,
        output='screen',
        parameters=[params],
    )
    return [node]


def generate_launch_description():
    # --- 运行配置 ---
    pkg_arg = DeclareLaunchArgument('package', default_value='je_software')
    exec_arg = DeclareLaunchArgument('executable', default_value='manager_node')
    ns_arg = DeclareLaunchArgument('namespace', default_value='')
    name_arg = DeclareLaunchArgument('node_name', default_value='manager')

    # --- 话题（CSV） ---
    color_csv = DeclareLaunchArgument(
        'color_topics_csv',
        default_value='/camera_01/color/image_raw,/camera_03/color/image_raw,/camera_04/color/image_raw,'
    )
    depth_csv = DeclareLaunchArgument(
        'depth_topics_csv',
        default_value='/camera_01/depth/image_raw,/camera_03/depth/image_raw,/camera_04/depth/image_raw'
    )
    joint_arg = DeclareLaunchArgument('joint_state_topic', default_value='/joint_states')
    tact_arg = DeclareLaunchArgument('tactile_topic', default_value='/tactile_data')

    # --- 频率/容差/窗口 ---
    rate_arg = DeclareLaunchArgument('rate_hz', default_value='60.0')
    img_tol = DeclareLaunchArgument('image_tolerance_ms', default_value='22.0')  # 33.3/2
    jnt_tol = DeclareLaunchArgument('joint_tolerance_ms', default_value='10.0')  #
    tac_tol = DeclareLaunchArgument('tactile_tolerance_ms', default_value='50.0')  # 50/2
    win_sec = DeclareLaunchArgument('queue_seconds', default_value='2.0')

    # --- 目录/控制 ---
    dir_arg = DeclareLaunchArgument('save_dir', default_value='/home/kleist/Documents/manager_node_temp/')
    sess_arg = DeclareLaunchArgument('session_name', default_value='')
    save_dep = DeclareLaunchArgument('save_depth', default_value='true')
    use_rtime = DeclareLaunchArgument('use_ros_time', default_value='true')
    do_cal_hz = DeclareLaunchArgument('do_calculate_hz', default_value='true')
    jpg_q = DeclareLaunchArgument('color_jpeg_quality', default_value='95')

    # --- 偏置（字符串，YAML/CSV 都行）---
    color_off = DeclareLaunchArgument('color_offsets_ms', default_value='[]', description='如 [0,0,0] 或 0,0,0')
    depth_off = DeclareLaunchArgument('depth_offsets_ms', default_value='[]', description='如 [0,0,0] 或 0,0,0')
    jnt_off = DeclareLaunchArgument('joint_offset_ms', default_value='0.0')
    tac_off = DeclareLaunchArgument('tactile_offset_ms', default_value='0.0')

    episode_idx = DeclareLaunchArgument('episode_idx', default_value='0')
    mode = DeclareLaunchArgument('mode', default_value='1', description='必须指定mode, 1为录制，2为推理')

    return LaunchDescription([
        pkg_arg, exec_arg, ns_arg, name_arg,
        color_csv, depth_csv, joint_arg, tact_arg,
        rate_arg, img_tol, jnt_tol, tac_tol, win_sec,
        dir_arg, sess_arg, save_dep, use_rtime, do_cal_hz, jpg_q,
        color_off, depth_off, jnt_off, tac_off, episode_idx, mode,
        OpaqueFunction(function=_launch_setup),
    ])
