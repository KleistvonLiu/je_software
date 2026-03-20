import copy
from types import MethodType

import pytest
import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import GoalResponse

from je_software.action import RecoverToInitial
from je_software.msg import InspectionResult
from je_software.msg import PcbPresence
from je_software.msg import RobotState
from je_software.pcb_process_common import build_joint_trajectory_sequence
from je_software.pcb_process_common import build_home_sequence
from je_software.pcb_process_common import build_inspection_sequence
from je_software.pcb_process_common import build_pick_sequence
from je_software.pcb_process_common import build_place_sequence
from je_software.pcb_process_common import build_recover_to_initial_sequence
from je_software.pcb_process_common import make_pose_stamped
from je_software.pcb_process_common import MOVEJ
from je_software.pcb_process_common import MOVEL
from je_software.pcb_process_task_manager_node import PcbProcessTaskManagerNode
from je_software.pcb_process_task_manager_node import ProcessState
from je_software.srv import GetAvailableSlot
from je_software.srv import GetPcbPickPose
from je_software.srv import GetRobotState


@pytest.fixture(scope='module', autouse=True)
def rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


class FakeMotionResolver:
    def __init__(self):
        self.calls = []

    def resolve_steps(self, steps):
        self.calls.append([step.name for step in steps])
        resolved_steps = []
        synthetic_index = 0
        for step in steps:
            if step.command_type == MOVEJ and not list(step.joint_target):
                synthetic_index += 1
                resolved_step = copy.deepcopy(step)
                resolved_step.joint_target = [
                    synthetic_index * 0.1 + axis * 0.01
                    for axis in range(7)
                ]
                resolved_steps.append(resolved_step)
                continue

            if step.command_type == MOVEL:
                for point_index in range(2):
                    synthetic_index += 1
                    resolved_step = copy.deepcopy(step)
                    resolved_step.name = f'{step.name}_{point_index + 1:04d}'
                    resolved_step.command_type = MOVEJ
                    resolved_step.joint_target = [
                        synthetic_index * 0.1 + axis * 0.01
                        for axis in range(7)
                    ]
                    resolved_step.dwell_sec = float(step.dwell_sec) / 2.0
                    resolved_steps.append(resolved_step)
                continue

            resolved_steps.append(copy.deepcopy(step))
        return resolved_steps


@pytest.fixture(autouse=True)
def fake_moveit_motion_resolver(monkeypatch):
    created_resolvers = []

    def _create_motion_resolver(self):
        resolver = FakeMotionResolver()
        created_resolvers.append(resolver)
        return resolver

    monkeypatch.setattr(
        PcbProcessTaskManagerNode,
        '_create_motion_resolver',
        _create_motion_resolver,
    )
    return created_resolvers


class DummyFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class DummyAsyncFuture:
    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def add_done_callback(self, callback):
        callback(self)

    def result(self):
        if self._error is not None:
            raise self._error
        return self._value


class DummyRobotStateClient:
    def __init__(self, *, ready=True, response=None, error=None):
        self.ready = ready
        self.response = response
        self.error = error
        self.call_count = 0

    def wait_for_service(self, timeout_sec=0.0):
        return self.ready

    def call_async(self, _request):
        self.call_count += 1
        return DummyAsyncFuture(self.response, self.error)


class DummyMotionWrappedResult:
    def __init__(self, success=True, error_code='', status=GoalStatus.STATUS_SUCCEEDED):
        self.status = status
        self.result = type(
            'MotionResult',
            (),
            {
                'success': success,
                'error_code': error_code,
            },
        )()


class DummyRecoverGoalHandle:
    def __init__(self):
        self.is_cancel_requested = False
        self.succeeded = False
        self.aborted = False

    def succeed(self):
        self.succeeded = True

    def abort(self):
        self.aborted = True


class DummyMotionGoalHandle:
    def __init__(self):
        self.accepted = True

    def get_result_async(self):
        return DummyAsyncFuture(DummyMotionWrappedResult())


class DummyMotionClient:
    def __init__(self):
        self.sent_goal = None

    def wait_for_server(self, timeout_sec=0.0):
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.sent_goal = goal
        return DummyAsyncFuture(DummyMotionGoalHandle())


def test_motion_sequences_match_expected_order():
    frame_id = 'base_link'
    pose = make_pose_stamped([0.4, 0.1, 0.2, 3.14, 0.0, 0.0], frame_id)
    movej = {
        'velocity': 0.3,
        'acceleration': 0.3,
        'blend_radius': 0.0,
        'dwell_sec': 0.3,
    }
    movel = {
        'velocity': 0.1,
        'acceleration': 0.1,
        'blend_radius': 0.0,
        'dwell_sec': 0.2,
    }

    pick_steps = build_pick_sequence(
        pose,
        frame_id,
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        movej,
        movel,
        0.4,
    )
    inspection_steps = build_inspection_sequence(
        frame_id,
        [0.2, 0.2, 0.2, 3.14, 0.0, 0.0],
        [0.2, 0.2, 0.1, 3.14, 0.0, 0.0],
        movej,
        movel,
        0.5,
    )
    place_steps = build_place_sequence(
        pose,
        frame_id,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        movej,
        movel,
        0.4,
    )
    home_steps = build_home_sequence(
        frame_id,
        [0.3, 0.0, 0.25, 3.14, 0.0, 0.0],
        movej,
    )

    all_names = [step.name for step in pick_steps + inspection_steps + place_steps + home_steps]
    assert all_names == [
        'pre_grasp',
        'grasp',
        'close',
        'retreat',
        'inspection_pre',
        'inspection',
        'inspection_wait',
        'pre_place',
        'place',
        'open',
        'retreat',
        'home',
    ]
    assert pick_steps[0].command_type == MOVEJ
    assert pick_steps[1].command_type == MOVEL
    assert place_steps[0].command_type == MOVEJ
    assert place_steps[1].command_type == MOVEL


def test_build_joint_trajectory_sequence_uses_absolute_joint_waypoints():
    movej = {
        'velocity': 0.2,
        'acceleration': 0.2,
        'blend_radius': 0.0,
        'dwell_sec': 0.5,
    }
    steps = build_joint_trajectory_sequence(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ],
        movej,
        step_name_prefix='init_waypoint',
    )

    assert [step.name for step in steps] == [
        'init_waypoint_001',
        'init_waypoint_002',
    ]
    assert all(step.command_type == MOVEJ for step in steps)
    assert list(steps[0].joint_target) == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert list(steps[1].joint_target) == pytest.approx(
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    )


def test_build_recover_to_initial_sequence_reverses_init_path_and_deduplicates_target():
    movej = {
        'velocity': 0.2,
        'acceleration': 0.2,
        'blend_radius': 0.0,
        'dwell_sec': 0.5,
    }
    steps = build_recover_to_initial_sequence(
        'base_link',
        [0.3, 0.0, 0.25, 3.14, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
        ],
        movej,
    )

    assert [step.name for step in steps] == [
        'home',
        'recover_init_001',
        'recover_init_002',
    ]
    assert list(steps[1].joint_target) == pytest.approx(
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1]
    )
    assert list(steps[2].joint_target) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_build_recover_to_initial_sequence_appends_initial_target_when_needed():
    movej = {
        'velocity': 0.2,
        'acceleration': 0.2,
        'blend_radius': 0.0,
        'dwell_sec': 0.5,
    }
    steps = build_recover_to_initial_sequence(
        'base_link',
        [0.3, 0.0, 0.25, 3.14, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            [0.2, 0.1, 0.0, -0.1, 0.3, 0.2, 0.1],
            [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
        ],
        movej,
    )

    assert [step.name for step in steps] == [
        'home',
        'recover_init_001',
        'recover_init_002',
        'recover_initial_check_target',
    ]
    assert list(steps[-1].joint_target) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )


def test_build_recover_to_initial_sequence_rejects_empty_init_path():
    movej = {
        'velocity': 0.2,
        'acceleration': 0.2,
        'blend_radius': 0.0,
        'dwell_sec': 0.5,
    }
    with pytest.raises(ValueError, match='init_trajectory_points_empty'):
        build_recover_to_initial_sequence(
            'base_link',
            [0.3, 0.0, 0.25, 3.14, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [],
            movej,
        )


def test_send_motion_goal_resolves_movej_and_movel_steps_before_dispatch():
    node = PcbProcessTaskManagerNode()
    node.motion_client = DummyMotionClient()

    steps = build_pick_sequence(
        make_pose_stamped([0.45, 0.0, 0.12, 3.14, 0.0, 0.0], 'base_link'),
        'base_link',
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],
        {
            'velocity': 0.3,
            'acceleration': 0.3,
            'blend_radius': 0.0,
            'dwell_sec': 0.3,
        },
        {
            'velocity': 0.1,
            'acceleration': 0.1,
            'blend_radius': 0.0,
            'dwell_sec': 0.2,
        },
        0.4,
    )

    node._send_motion_goal('probe', 'probe_sequence', steps)

    assert node.motion_client.sent_goal is not None
    sent_steps = list(node.motion_client.sent_goal.steps)
    assert [step.name for step in sent_steps] == [
        'pre_grasp',
        'grasp_0001',
        'grasp_0002',
        'close',
        'retreat_0001',
        'retreat_0002',
    ]
    assert sent_steps[0].command_type == MOVEJ
    assert list(sent_steps[0].joint_target)
    assert sent_steps[1].command_type == MOVEJ
    assert list(sent_steps[1].joint_target)
    assert sent_steps[3].command_type == 'GRIPPER'

    node.destroy_node()


def test_task_manager_happy_path_state_flow():
    node = PcbProcessTaskManagerNode()
    recorded_calls = []

    def record_request_pick_pose(self):
        recorded_calls.append('request_pick_pose')

    def record_send_motion_goal(self, purpose, sequence_name, steps):
        recorded_calls.append((purpose, sequence_name, [step.name for step in steps]))

    def record_trigger_inspection(self):
        recorded_calls.append('trigger_inspection')

    def record_request_good_slot(self):
        recorded_calls.append('request_good_slot')

    node._request_pick_pose = MethodType(record_request_pick_pose, node)
    node._send_motion_goal = MethodType(record_send_motion_goal, node)
    node._trigger_inspection = MethodType(record_trigger_inspection, node)
    node._request_good_slot = MethodType(record_request_good_slot, node)

    node._on_startup_timer()
    assert node._state == ProcessState.WAIT_PCB

    presence = PcbPresence()
    presence.ready_for_pick = True
    node._presence_callback(presence)
    assert node._state == ProcessState.MOVE_HOME_BEFORE_PICK
    assert recorded_calls[-1][0] == 'home_before_pick'
    assert node._current_pcb_id is not None

    node._on_motion_result(
        'home_before_pick',
        DummyFuture(DummyMotionWrappedResult()),
    )
    assert node._state == ProcessState.REQUEST_PICK_POSE
    assert recorded_calls[-1] == 'request_pick_pose'

    pick_response = GetPcbPickPose.Response()
    pick_response.success = True
    pick_response.pick_pose_base = make_pose_stamped(
        [0.45, 0.0, 0.12, 3.14, 0.0, 0.0],
        'base_link',
    )
    node._on_pick_pose_response(DummyFuture(pick_response))
    assert node._state == ProcessState.EXECUTE_PICK
    assert recorded_calls[-1][0] == 'pick'

    node._on_motion_result('pick', DummyFuture(DummyMotionWrappedResult()))
    assert node._state == ProcessState.MOVE_TO_INSPECTION
    assert recorded_calls[-1][0] == 'inspection_move'

    node._on_motion_result(
        'inspection_move',
        DummyFuture(DummyMotionWrappedResult()),
    )
    assert node._state == ProcessState.TRIGGER_INSPECTION
    assert recorded_calls[-1] == 'trigger_inspection'

    trigger_response = type(
        'TriggerResponse',
        (),
        {
            'accepted': True,
            'reason': 'accepted',
        },
    )()
    node._on_trigger_inspection_response(DummyFuture(trigger_response))
    assert node._state == ProcessState.WAIT_INSPECTION_RESULT

    result = InspectionResult()
    result.pcb_id = node._current_pcb_id
    result.result = 'good'
    result.valid = True
    node._inspection_result_callback(result)
    assert node._state == ProcessState.REQUEST_GOOD_SLOT
    assert recorded_calls[-1] == 'request_good_slot'

    slot_response = GetAvailableSlot.Response()
    slot_response.success = True
    slot_response.slot_empty = True
    slot_response.slot_pose_base = make_pose_stamped(
        [0.2, -0.2, 0.08, 3.14, 0.0, 0.0],
        'base_link',
    )
    node._on_slot_response(DummyFuture(slot_response))
    assert node._state == ProcessState.EXECUTE_PLACE
    assert recorded_calls[-1][0] == 'place'

    node._on_motion_result('place', DummyFuture(DummyMotionWrappedResult()))
    assert node._state == ProcessState.GO_HOME
    assert recorded_calls[-1][0] == 'home'

    node._on_motion_result('home', DummyFuture(DummyMotionWrappedResult()))
    assert node._state == ProcessState.WAIT_PCB
    assert node._current_pcb_id is None

    node.destroy_node()


def test_recover_to_initial_goal_callback_accepts_only_allowed_states():
    node = PcbProcessTaskManagerNode()

    node._state = ProcessState.WAIT_PCB
    assert (
        node._recover_to_initial_goal_callback(RecoverToInitial.Goal())
        == GoalResponse.ACCEPT
    )

    node._state = ProcessState.EXECUTE_PICK
    assert (
        node._recover_to_initial_goal_callback(RecoverToInitial.Goal())
        == GoalResponse.REJECT
    )

    node._state = ProcessState.ERROR
    node._recovering = True
    assert (
        node._recover_to_initial_goal_callback(RecoverToInitial.Goal())
        == GoalResponse.REJECT
    )

    node.destroy_node()


def test_execute_recover_to_initial_action_sends_recovery_sequence_and_returns_wait_pcb():
    node = PcbProcessTaskManagerNode()
    recorded_calls = []

    def record_send_motion_goal(self, purpose, sequence_name, steps):
        recorded_calls.append((purpose, sequence_name, [step.name for step in steps]))
        self._current_pcb_id = None
        self._transition(
            ProcessState.WAIT_PCB,
            'manual_recovery_sequence_done',
        )
        self._finalize_manual_recovery(True, '')

    node.init_trajectory_points = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1],
    ]
    node.initial_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    node._state = ProcessState.ERROR
    node._send_motion_goal = MethodType(record_send_motion_goal, node)

    goal_handle = DummyRecoverGoalHandle()
    result = node._execute_recover_to_initial(goal_handle)

    assert result.success
    assert result.error_code == ''
    assert goal_handle.succeeded
    assert not goal_handle.aborted
    assert node._state == ProcessState.WAIT_PCB
    assert recorded_calls[-1][0] == 'manual_recover'
    assert recorded_calls[-1][2] == [
        'home',
        'recover_init_001',
        'recover_init_002',
    ]

    node.destroy_node()


def test_manual_recovery_motion_result_failure_enters_error():
    node = PcbProcessTaskManagerNode()
    node._manual_recovery_active = True
    node._state = ProcessState.EXECUTE_MANUAL_RECOVERY

    node._on_motion_result(
        'manual_recover',
        DummyFuture(
            DummyMotionWrappedResult(
                success=False,
                error_code='backend_failed',
            )
        ),
    )

    assert node._state == ProcessState.ERROR
    assert node._manual_recovery_result == (
        False,
        'motion_failed:manual_recover:backend_failed',
    )

    node.destroy_node()


def test_task_manager_runs_initialization_sequence_when_initial_joint_matches():
    node = PcbProcessTaskManagerNode()
    recorded_calls = []

    def record_send_motion_goal(self, purpose, sequence_name, steps):
        recorded_calls.append((purpose, sequence_name, [step for step in steps]))

    node.enable_initialization = True
    node.initial_joint_position = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    node.init_trajectory_points = [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]
    node._send_motion_goal = MethodType(record_send_motion_goal, node)
    response = GetRobotState.Response()
    response.success = True
    response.state = RobotState()
    response.state.valid = True
    response.state.joint_valid = True
    response.state.joint_position = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    node.robot_state_client = DummyRobotStateClient(response=response)

    node._on_startup_timer()
    assert node._state == ProcessState.WAIT_INIT_JOINT_STATE

    node._check_initialization_ready()
    assert node.robot_state_client.call_count == 1
    assert node._state == ProcessState.EXECUTE_INITIALIZATION
    assert recorded_calls[-1][0] == 'initialization'
    assert recorded_calls[-1][1] == 'initialization_sequence'
    assert [step.name for step in recorded_calls[-1][2]] == [
        'init_waypoint_001',
        'init_waypoint_002',
    ]
    assert list(recorded_calls[-1][2][0].joint_target) == pytest.approx(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )

    node._on_motion_result(
        'initialization',
        DummyFuture(DummyMotionWrappedResult()),
    )
    assert node._state == ProcessState.WAIT_PCB

    node.destroy_node()


def test_task_manager_reports_error_when_initial_joint_mismatches():
    node = PcbProcessTaskManagerNode()
    recorded_calls = []

    def record_send_motion_goal(self, purpose, sequence_name, steps):
        recorded_calls.append((purpose, sequence_name, [step.name for step in steps]))

    node.enable_initialization = True
    node.initial_joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    node.initial_joint_tolerance = 0.01
    node.init_trajectory_points = [
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    node._send_motion_goal = MethodType(record_send_motion_goal, node)
    response = GetRobotState.Response()
    response.success = True
    response.state = RobotState()
    response.state.valid = True
    response.state.joint_valid = True
    response.state.joint_position = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    node.robot_state_client = DummyRobotStateClient(response=response)

    node._on_startup_timer()
    node._check_initialization_ready()

    assert node._state == ProcessState.ERROR
    assert recorded_calls == []

    node.destroy_node()


def test_task_manager_reports_error_when_robot_state_times_out():
    node = PcbProcessTaskManagerNode()
    node.enable_initialization = True
    node.initial_joint_state_timeout_sec = 0.0
    node.robot_state_client = DummyRobotStateClient(ready=False)

    node._on_startup_timer()
    node._check_initialization_ready()

    assert node._state == ProcessState.ERROR
    node.destroy_node()
