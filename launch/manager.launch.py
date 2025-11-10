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


def _parse_str_list(context, name: str):
    """
    解析字符串列表参数：支持 YAML 列表或 CSV 字符串。
    返回 list[str]；空/无效时返回 []。
    """
    raw = LaunchConfiguration(name).perform(context) or ''
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, (list, tuple)):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, str):
            return [s.strip() for s in data.split(',') if s.strip()]
    except Exception:
        pass
    return []


def _maybe_set_scalar_or_list(params: dict, context, name: str):
    """
    将 name 参数作为“标量或列表”写入 params：
    - 若能解析出非空列表（见 _parse_list），则设列表
    - 否则回落到 float 标量
    """
    lst = _parse_list(context, name)
    if lst:
        params[name] = lst
    else:
        val = LaunchConfiguration(name).perform(context)
        if val is not None and str(val).strip() != '':
            params[name] = float(val)


def _launch_setup(context, *args, **kwargs):
    # 读取简单标量
    pkg = LaunchConfiguration('package').perform(context)
    exe = LaunchConfiguration('executable').perform(context)
    ns = LaunchConfiguration('namespace')
    name = LaunchConfiguration('node_name')

    params = {
        # 颜色/深度（CSV直传）
        'color_topics_csv': LaunchConfiguration('color_topics_csv').perform(context),
        'depth_topics_csv': LaunchConfiguration('depth_topics_csv').perform(context),

        # 单路旧参数（保持兼容；若你传了多路参数，节点内部会优先使用多路）
        'joint_state_topic': LaunchConfiguration('joint_state_topic').perform(context),
        'tactile_topic': LaunchConfiguration('tactile_topic').perform(context),

        # 频率/容差/窗口
        'rate_hz': float(LaunchConfiguration('rate_hz').perform(context)),
        'image_tolerance_ms': float(LaunchConfiguration('image_tolerance_ms').perform(context)),
        'queue_seconds': float(LaunchConfiguration('queue_seconds').perform(context)),

        # 目录/控制
        'save_dir': LaunchConfiguration('save_dir').perform(context),
        'session_name': LaunchConfiguration('session_name').perform(context),
        'save_depth': _as_bool(LaunchConfiguration('save_depth').perform(context), True),
        'overwrite': _as_bool(LaunchConfiguration('overwrite').perform(context), False),
        'use_ros_time': _as_bool(LaunchConfiguration('use_ros_time').perform(context), True),
        'do_calculate_hz': _as_bool(LaunchConfiguration('do_calculate_hz').perform(context), True),
        'color_jpeg_quality': int(float(LaunchConfiguration('color_jpeg_quality').perform(context))),

        # 运行模式/集
        'episode_idx': int(float(LaunchConfiguration('episode_idx').perform(context))),
        'mode': int(float(LaunchConfiguration('mode').perform(context))),
    }

    # --- 多路 joint/tactile 话题（支持 YAML list 或 CSV）---
    jt_list = _parse_str_list(context, 'joint_state_topics')
    jt_csv = LaunchConfiguration('joint_state_topics_csv').perform(context) or ''
    if jt_list:
        params['joint_state_topics'] = jt_list
    if jt_csv.strip():
        params['joint_state_topics_csv'] = jt_csv

    tac_list = _parse_str_list(context, 'tactile_topics')
    tac_csv = LaunchConfiguration('tactile_topics_csv').perform(context) or ''
    if tac_list:
        params['tactile_topics'] = tac_list
    if tac_csv.strip():
        params['tactile_topics_csv'] = tac_csv

    # --- 容差：joint/tactile 支持标量或列表 ---
    _maybe_set_scalar_or_list(params, context, 'joint_tolerance_ms')
    _maybe_set_scalar_or_list(params, context, 'tactile_tolerance_ms')

    # --- 偏置：color/depth 为列表（仅在非空时传），joint/tactile 支持标量或列表 ---
    color_offs = _parse_list(context, 'color_offsets_ms')
    depth_offs = _parse_list(context, 'depth_offsets_ms')
    if color_offs:
        params['color_offsets_ms'] = color_offs
    if depth_offs:
        params['depth_offsets_ms'] = depth_offs

    _maybe_set_scalar_or_list(params, context, 'joint_offset_ms')
    _maybe_set_scalar_or_list(params, context, 'tactile_offset_ms')

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

    # --- 颜色/深度（CSV） ---
    color_csv = DeclareLaunchArgument(
        'color_topics_csv',
        default_value='/camera_01/color/image_raw,/camera_02/color/image_raw,/camera_03/color/image_raw,/camera_04/color/image_raw,/camera_05/color/image_raw,'
    )
    depth_csv = DeclareLaunchArgument(
        'depth_topics_csv',
        default_value='/camera_01/depth/image_raw,/camera_03/depth/image_raw,/camera_04/depth/image_raw'
    )

    # --- joint/tactile：多路与兼容 ---
    joint_csv = DeclareLaunchArgument('joint_state_topics_csv', default_value='/joint_states_right, /joint_states_left', description='CSV: /arm_a/joint_states,/arm_b/joint_states')
    joint_list = DeclareLaunchArgument('joint_state_topics', default_value='[]', description='YAML 列表: [/arm_a/joint_states, /arm_b/joint_states]')
    joint_legacy = DeclareLaunchArgument('joint_state_topic', default_value='', description='单路兼容')

    tactile_csv = DeclareLaunchArgument('tactile_topics_csv', default_value='', description='CSV: /tactile/left,/tactile/right')
    tactile_list = DeclareLaunchArgument('tactile_topics', default_value='[]', description='YAML 列表: [/tactile/left, /tactile/right]')
    tactile_legacy = DeclareLaunchArgument('tactile_topic', default_value='', description='单路兼容')

    # --- 频率/容差/窗口 ---
    rate_arg = DeclareLaunchArgument('rate_hz', default_value='60.0')
    img_tol = DeclareLaunchArgument('image_tolerance_ms', default_value='22.0')   # 33.3/2
    jnt_tol = DeclareLaunchArgument('joint_tolerance_ms', default_value='10.0', description='标量或列表')
    tac_tol = DeclareLaunchArgument('tactile_tolerance_ms', default_value='50.0', description='标量或列表')
    win_sec = DeclareLaunchArgument('queue_seconds', default_value='2.0')

    # --- 目录/控制 ---
    dir_arg = DeclareLaunchArgument('save_dir', default_value='/home/test/jemotor/log/')
    sess_arg = DeclareLaunchArgument('session_name', default_value='')
    save_dep = DeclareLaunchArgument('save_depth', default_value='true')
    overwrite = DeclareLaunchArgument('overwrite', default_value='false')
    use_rtime = DeclareLaunchArgument('use_ros_time', default_value='true')
    do_cal_hz = DeclareLaunchArgument('do_calculate_hz', default_value='true')
    jpg_q = DeclareLaunchArgument('color_jpeg_quality', default_value='95')

    # --- 偏置（字符串，YAML/CSV 都行）---
    color_off = DeclareLaunchArgument('color_offsets_ms', default_value='[]', description='如 [0,0,0] 或 0,0,0')
    depth_off = DeclareLaunchArgument('depth_offsets_ms', default_value='[]', description='如 [0,0,0] 或 0,0,0')
    jnt_off = DeclareLaunchArgument('joint_offset_ms', default_value='0.0', description='标量或列表')
    tac_off = DeclareLaunchArgument('tactile_offset_ms', default_value='0.0', description='标量或列表')

    # --- 运行模式/集 ---
    episode_idx = DeclareLaunchArgument('episode_idx', default_value='0')
    mode = DeclareLaunchArgument('mode', default_value='1', description='必须指定mode, 1为录制，2为推理')

    return LaunchDescription([
        # 基本
        pkg_arg, exec_arg, ns_arg, name_arg,

        # 颜色/深度
        color_csv, depth_csv,

        # joint/tactile 多路 + 兼容
        joint_csv, joint_list, joint_legacy,
        tactile_csv, tactile_list, tactile_legacy,

        # 频率/容差/窗口
        rate_arg, img_tol, jnt_tol, tac_tol, win_sec,

        # 目录/控制
        dir_arg, sess_arg, save_dep, overwrite, use_rtime, do_cal_hz, jpg_q,

        # 偏置
        color_off, depth_off, jnt_off, tac_off,

        # 运行模式/集
        episode_idx, mode,

        OpaqueFunction(function=_launch_setup),
    ])
