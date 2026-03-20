from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f'File does not exist: {path}')
    return path.read_text(encoding='utf-8')


def _load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'File does not exist: {path}')
    with path.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _launch_setup(context, *args, **kwargs):
    share_dir = Path(get_package_share_directory('je_software'))
    urdf_path = Path(LaunchConfiguration('urdf_path').perform(context))
    srdf_path = share_dir / 'config' / 'moveit' / 'lh_jearm.srdf'
    kinematics_path = share_dir / 'config' / 'moveit' / 'kinematics.yaml'
    joint_limits_path = share_dir / 'config' / 'moveit' / 'joint_limits.yaml'
    ompl_path = share_dir / 'config' / 'moveit' / 'ompl_planning.yaml'

    robot_description = {
        'robot_description': _load_text(urdf_path),
    }
    robot_description_semantic = {
        'robot_description_semantic': _load_text(srdf_path),
    }
    robot_description_kinematics = {
        'robot_description_kinematics': _load_yaml(kinematics_path),
    }
    robot_description_planning = {
        'robot_description_planning': _load_yaml(joint_limits_path),
    }
    ompl_planning_pipeline = {
        'planning_pipelines': ['ompl'],
        'default_planning_pipeline': 'ompl',
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/ResolveConstraintFrames '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'
            ),
            'start_state_max_bounds_error': 0.1,
        },
    }
    ompl_planning_pipeline['ompl'].update(_load_yaml(ompl_path))

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'monitor_dynamics': False,
    }
    trajectory_execution = {
        'moveit_manage_controllers': False,
        'allowed_execution_duration_scaling': 1.2,
        'allowed_goal_duration_margin': 0.5,
        'allowed_start_tolerance': 0.01,
    }

    return [
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            namespace='pcb_moveit',
            output='screen',
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                robot_description_planning,
                ompl_planning_pipeline,
                planning_scene_monitor_parameters,
                trajectory_execution,
            ],
        ),
    ]


def generate_launch_description():
    share_dir = Path(get_package_share_directory('je_software'))
    default_urdf = (
        share_dir / 'urdf' / 'URDF_LAST' / 'LH_JEARM' / 'LH_JEARM.urdf'
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'urdf_path',
                default_value=str(default_urdf),
                description='URDF used by MoveIt2 move_group.',
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
