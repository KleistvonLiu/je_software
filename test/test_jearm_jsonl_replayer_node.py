from pathlib import Path

import pytest
from builtin_interfaces.msg import Time as TimeMsg

from je_software.jearm_jsonl_replayer_node import DEFAULT_JOINT_NAMES
from je_software.jearm_jsonl_replayer_node import JointFrame
from je_software.jearm_jsonl_replayer_node import JearmJsonlReplayerNode
from je_software.jearm_jsonl_replayer_node import load_joint_frames_from_jsonl
from je_software.jearm_jsonl_replayer_node import make_joint_state_message


def write_jsonl(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / 'sample.jsonl'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def test_load_joint_frames_from_jsonl_reads_robot0_joint_positions_and_stamp(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            '{"Robot0":{"Joint":[0,1,2,3,4,5,6]},"__ros_stamp_ns":123456789}',
        ],
    )

    frames = load_joint_frames_from_jsonl(str(path))

    assert len(frames) == 1
    assert frames[0].stamp_ns == 123456789
    assert frames[0].positions == pytest.approx([0, 1, 2, 3, 4, 5, 6])


def test_load_joint_frames_from_jsonl_falls_back_to_ros_stamp_sec(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            '{"Robot0":{"Joint":[0,1,2,3,4,5,6]},"__ros_stamp_sec":1.5}',
        ],
    )

    frames = load_joint_frames_from_jsonl(str(path))

    assert len(frames) == 1
    assert frames[0].stamp_ns == 1_500_000_000


def test_load_joint_frames_from_jsonl_skips_invalid_joint_length(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            '{"Robot0":{"Joint":[0,1,2]}}',
            '{"Robot0":{"Joint":[0,1,2,3,4,5,6]}}',
        ],
    )

    frames = load_joint_frames_from_jsonl(str(path))

    assert len(frames) == 1
    assert frames[0].positions == pytest.approx([0, 1, 2, 3, 4, 5, 6])


def test_load_joint_frames_from_jsonl_requires_at_least_one_valid_frame(tmp_path):
    path = write_jsonl(
        tmp_path,
        [
            '{"Robot0":{"Joint":[0,1,2]}}',
            '{"Robot1":{"Joint":[0,1,2,3,4,5,6]}}',
        ],
    )

    with pytest.raises(ValueError, match='No valid joint frames'):
        load_joint_frames_from_jsonl(str(path))


def test_make_joint_state_message_uses_joint_names_and_stamp():
    msg = make_joint_state_message(
        JointFrame(
            stamp_ns=42,
            positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        ),
        DEFAULT_JOINT_NAMES,
    )

    assert list(msg.name) == DEFAULT_JOINT_NAMES
    assert list(msg.position) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert msg.header.stamp.sec == 0
    assert msg.header.stamp.nanosec == 42


def test_handle_key_toggles_pause_state():
    node = object.__new__(JearmJsonlReplayerNode)
    node._paused = False
    node.get_logger = lambda: type('Logger', (), {'info': lambda *args, **kwargs: None})()

    node._handle_key(' ')
    assert node._paused is True

    node._handle_key(' ')
    assert node._paused is False


def test_replay_once_stops_at_last_frame():
    published = []
    node = object.__new__(JearmJsonlReplayerNode)
    node._paused = False
    node._frame_offsets_ns = []
    node._last_tick_ns = 0
    node._frames = [
        JointFrame(stamp_ns=1, positions=[0, 0, 0, 0, 0, 0, 0]),
        JointFrame(stamp_ns=2, positions=[1, 1, 1, 1, 1, 1, 1]),
    ]
    node._current_index = 0
    node._publish_current_frame = lambda: published.append(node._current_index)

    import je_software.jearm_jsonl_replayer_node as replay_module
    original_monotonic_ns = replay_module.time.monotonic_ns
    replay_module.time.monotonic_ns = iter([10, 20]).__next__
    try:
        node._replay_once()
        node._replay_once()
    finally:
        replay_module.time.monotonic_ns = original_monotonic_ns

    assert published == [1]
    assert node._current_index == 1


def test_publish_current_frame_uses_current_time_when_not_using_recorded_timestamps():
    published = []
    node = object.__new__(JearmJsonlReplayerNode)
    node._frames = [
        JointFrame(stamp_ns=123, positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]
    node._current_index = 0
    node.joint_names = DEFAULT_JOINT_NAMES
    node._use_recorded_timestamps = False
    node._frame_offsets_ns = []
    node._playback_elapsed_ns = 0
    node.publisher = type(
        'Publisher',
        (),
        {'publish': lambda self, msg: published.append(msg)},
    )()
    node.get_clock = lambda: type(
        'ClockHolder',
        (),
        {
            'now': lambda self: type(
                'NowHolder',
                (),
                {
                    'to_msg': lambda self: TimeMsg(sec=7, nanosec=89)
                },
            )()
        },
    )()

    node._publish_current_frame()

    assert len(published) == 1
    assert published[0].header.stamp.sec == 7
    assert published[0].header.stamp.nanosec == 89


def test_publish_current_frame_keeps_recorded_time_when_enabled():
    published = []
    node = object.__new__(JearmJsonlReplayerNode)
    node._frames = [
        JointFrame(stamp_ns=123, positions=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ]
    node._current_index = 0
    node.joint_names = DEFAULT_JOINT_NAMES
    node._use_recorded_timestamps = True
    node._frame_offsets_ns = []
    node._playback_elapsed_ns = 0
    node.publisher = type(
        'Publisher',
        (),
        {'publish': lambda self, msg: published.append(msg)},
    )()

    node._publish_current_frame()

    assert len(published) == 1
    assert published[0].header.stamp.sec == 0
    assert published[0].header.stamp.nanosec == 123


def test_sample_positions_interpolates_when_following_recorded_timing():
    node = object.__new__(JearmJsonlReplayerNode)
    node._frames = [
        JointFrame(stamp_ns=100, positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        JointFrame(stamp_ns=200, positions=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ]
    node._frame_offsets_ns = [0, 100]
    node._playback_elapsed_ns = 50
    node._current_index = 0

    positions = node._sample_positions()

    assert positions == pytest.approx([0.5] * 7)
    assert node._current_index == 0
