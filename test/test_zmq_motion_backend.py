import json
import threading

import pytest
import rclpy

from je_software.fixed_inspection_node import FixedInspectionNode
from je_software.fixed_line_vision_node import FixedLineVisionNode
from je_software.fixed_slot_provider_node import FixedSlotProviderNode
from je_software.msg import MotionStep
from je_software.msg import PcbPresence
from je_software.msg import RobotState
from je_software.pcb_process_common import GRIPPER
from je_software.pcb_process_common import GRIPPER_CLOSE
from je_software.pcb_process_common import MOVEJ
from je_software.pcb_process_common import make_motion_step
from je_software.pcb_process_common import make_pose_stamped
from je_software.pcb_process_common import MOVEL
from je_software.srv import GetAvailableSlot
from je_software.srv import GetPcbPickPose
from je_software.srv import GetRobotState
from je_software.zmq_motion_backend_node import build_robot_state_from_json
from je_software.zmq_motion_backend_node import parse_state_message
from je_software.zmq_motion_backend_node import motion_step_to_payload
from je_software.zmq_motion_backend_node import ZmqMotionBackendNode


@pytest.fixture(scope='module', autouse=True)
def rclpy_context():
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


def test_motion_step_to_payload_formats_movea_when_joint_target_is_configured():
    step = make_motion_step(
        'pre_grasp',
        MOVEJ,
        target_pose=make_pose_stamped([0.4, 0.0, 0.2, 3.14, 0.0, 0.0], 'base_link'),
        velocity=0.3,
        acceleration=0.2,
        blend_radius=0.01,
        dwell_sec=0.0,
    )

    payload = motion_step_to_payload(
        step,
        robot_id=0,
        movea_joint_targets={
            'pre_grasp': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
    )
    prefix, json_text = payload.split(' ', 1)
    message = json.loads(json_text)

    assert prefix == 'MoveA'
    assert message['Robot0']['speed'] == pytest.approx(0.3)
    assert message['Robot0']['joint'] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )


def test_motion_step_to_payload_prefers_step_joint_target_for_movea():
    step = make_motion_step(
        'init_waypoint_001',
        MOVEJ,
        joint_target=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        velocity=0.25,
        dwell_sec=0.0,
    )

    payload = motion_step_to_payload(
        step,
        robot_id=0,
        movea_joint_targets={
            'init_waypoint_001': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )
    prefix, json_text = payload.split(' ', 1)
    message = json.loads(json_text)

    assert prefix == 'MoveA'
    assert message['Robot0']['joint'] == pytest.approx(
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )


def test_motion_step_to_payload_rejects_movej_without_joint_target():
    step = make_motion_step(
        'pre_grasp',
        MOVEJ,
        target_pose=make_pose_stamped([0.4, 0.0, 0.2, 3.14, 0.0, 0.0], 'base_link'),
        velocity=0.3,
        acceleration=0.2,
        blend_radius=0.01,
        dwell_sec=0.0,
    )

    with pytest.raises(
        ValueError,
        match='step.joint_target and movea_joint_targets.pre_grasp',
    ):
        motion_step_to_payload(step, robot_id=0)


def test_motion_step_to_payload_formats_movel_for_jeserver():
    step = make_motion_step(
        'inspection',
        MOVEL,
        target_pose=make_pose_stamped([0.3, -0.1, 0.15, 3.14, 0.1, -0.2], 'base_link'),
        velocity=0.12,
        acceleration=0.34,
        dwell_sec=0.0,
    )

    payload = motion_step_to_payload(step, robot_id=1)
    prefix, json_text = payload.split(' ', 1)
    message = json.loads(json_text)

    assert prefix == 'MoveL'
    assert message['Robot1']['tra_speed'] == pytest.approx(0.12)
    assert message['Robot1']['rot_speed'] == pytest.approx(0.34)
    assert len(message['Robot1']['cartesian']) == 6


def test_motion_step_to_payload_formats_gripper():
    step = MotionStep()
    step.name = 'close'
    step.command_type = GRIPPER
    step.gripper_command = GRIPPER_CLOSE

    payload = motion_step_to_payload(step, robot_id=2)
    prefix, json_text = payload.split(' ', 1)
    message = json.loads(json_text)

    assert prefix == 'Gripper'
    assert message == {
        'robot_id': 2,
        'command': 'CLOSE',
    }


def test_parse_state_message_returns_json_for_valid_state_text():
    state_json = parse_state_message(
        'State {"Robot0":{"Joint":[0,1,2,3,4,5,6],"Cartesian":[0,0,0,0,0,0]}}'
    )
    assert state_json is not None
    assert state_json['Robot0']['Joint'][0] == 0


def test_parse_state_message_rejects_invalid_topic_and_json():
    assert parse_state_message('MoveA {"Robot0":{"joint":[0,0,0,0,0,0,0]}}') is None
    assert parse_state_message('State not_json') is None


def test_build_robot_state_from_json_extracts_joint_and_cartesian():
    state = build_robot_state_from_json(
        {
            'Robot0': {
                'Joint': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                'JointVelocity': [1, 2, 3, 4, 5, 6, 7],
                'JointSensorTorque': [7, 6, 5, 4, 3, 2, 1],
                'Cartesian': [0.3, -0.1, 0.15, 3.14, 0.1, -0.2],
            }
        },
        robot_id=0,
    )

    assert state.valid
    assert state.joint_valid
    assert state.cartesian_valid
    assert list(state.joint_position) == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )
    assert list(state.joint_velocity) == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    assert list(state.joint_effort) == pytest.approx([7, 6, 5, 4, 3, 2, 1])
    assert state.tcp_pose.header.frame_id == 'base_link'


def test_build_robot_state_from_json_marks_cartesian_invalid_when_short():
    state = build_robot_state_from_json(
        {
            'Robot0': {
                'Joint': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                'Cartesian': [0.3, -0.1, 0.15],
            }
        },
        robot_id=0,
    )

    assert state.valid
    assert state.joint_valid
    assert not state.cartesian_valid


def test_get_robot_state_service_returns_failure_without_cache():
    node = object.__new__(ZmqMotionBackendNode)
    node._state_lock = threading.Lock()
    node._latest_robot_state = None
    response = node._handle_get_robot_state(
        GetRobotState.Request(),
        GetRobotState.Response(),
    )
    assert not response.success
    assert response.reason == 'robot_state_unavailable'


def test_get_robot_state_service_returns_cached_state():
    node = object.__new__(ZmqMotionBackendNode)
    node._state_lock = threading.Lock()
    node._latest_robot_state = build_robot_state_from_json(
        {
            'Robot0': {
                'Joint': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
        robot_id=0,
    )

    response = node._handle_get_robot_state(
        GetRobotState.Request(),
        GetRobotState.Response(),
    )
    assert response.success
    assert response.state.valid
    assert response.state.joint_valid


def test_fixed_line_vision_requires_ready_signal():
    node = FixedLineVisionNode()
    request = GetPcbPickPose.Request()
    response = node._handle_get_pick_pose(request, GetPcbPickPose.Response())
    assert not response.success

    presence = PcbPresence()
    presence.present = True
    presence.stable = True
    presence.ready_for_pick = True
    node._presence_callback(presence)
    response = node._handle_get_pick_pose(request, GetPcbPickPose.Response())
    assert response.success
    assert response.confidence == pytest.approx(1.0)
    node.destroy_node()


def test_fixed_slot_provider_returns_configured_slot():
    node = FixedSlotProviderNode()
    request = GetAvailableSlot.Request()
    request.box_type = 'good'
    request.require_empty = True
    response = node._handle_slot_request(request, GetAvailableSlot.Response())
    assert response.success
    assert response.slot_id == 0
    assert response.slot_empty
    node.destroy_node()


def test_fixed_inspection_publishes_good_result():
    node = FixedInspectionNode()
    published = []

    class DummyPublisher:
        def __init__(self, sink):
            self._sink = sink

        def publish(self, msg):
            self._sink.append(msg)

    node.response_delay_sec = 0.0
    node.publisher = DummyPublisher(published)

    node._publish_result_after_delay('pcb_000001')

    assert published
    assert published[0].pcb_id == 'pcb_000001'
    assert published[0].result == 'good'
    assert published[0].valid
    node.destroy_node()
