from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from je_software.ros2bag_parser.parser import POINT_FIELD_FLOAT32
from je_software.ros2bag_parser.parser import POINT_FIELD_UINT32
from je_software.ros2bag_parser.parser import build_pcd_header
from je_software.ros2bag_parser.parser import merge_corrupt_ranges
from je_software.ros2bag_parser.parser import parse_bag
from je_software.ros2bag_parser.parser import query_message_id_bounds
from je_software.ros2bag_parser.parser import sanitize_topic_name
from je_software.ros2bag_parser.parser import select_image_export_kind
from je_software.ros2bag_parser.parser import select_pcd_fields
from je_software.ros2bag_parser.parser import walk_message_id_windows


@dataclass
class FakeStamp:
    sec: int
    nanosec: int


@dataclass
class FakeHeader:
    stamp: FakeStamp
    frame_id: str


@dataclass
class FakeImageMessage:
    header: FakeHeader
    width: int
    height: int
    encoding: str
    step: int
    data: bytes
    is_bigendian: int = 0


@dataclass
class FakeRegionOfInterest:
    x_offset: int = 0
    y_offset: int = 0
    height: int = 0
    width: int = 0
    do_rectify: bool = False


@dataclass
class FakeCameraInfo:
    header: FakeHeader
    height: int
    width: int
    distortion_model: str
    d: list[float]
    k: list[float]
    r: list[float]
    p: list[float]
    binning_x: int = 0
    binning_y: int = 0
    roi: FakeRegionOfInterest = FakeRegionOfInterest()


@dataclass
class FakePointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass
class FakePointCloud2:
    header: FakeHeader
    width: int
    height: int
    fields: list[FakePointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: bytes
    is_dense: bool


class FakeRosRuntime:
    def __init__(self, payloads: dict[bytes, object]) -> None:
        self._payloads = payloads

    def get_message(self, msg_type: str) -> str:
        return msg_type

    def deserialize_message(self, data: bytes, msg_type: str) -> object:
        return self._payloads[data]


def test_sanitize_topic_name_replaces_slashes():
    assert sanitize_topic_name('/camera/color/image_raw') == 'camera_color_image_raw'
    assert sanitize_topic_name('///camera///depth///points///') == 'camera_depth_points'


def test_select_image_export_kind_supports_depth_and_color():
    assert select_image_export_kind('bgr8') == 'png'
    assert select_image_export_kind('16UC1') == 'png16'
    assert select_image_export_kind('32FC1') == 'npy'


def test_select_pcd_fields_prefers_xyz_plus_rgb():
    selected_fields, ignored_fields = select_pcd_fields(
        [
            FakePointField('x', 0, POINT_FIELD_FLOAT32, 1),
            FakePointField('y', 4, POINT_FIELD_FLOAT32, 1),
            FakePointField('z', 8, POINT_FIELD_FLOAT32, 1),
            FakePointField('rgb', 12, POINT_FIELD_UINT32, 1),
            FakePointField('intensity', 16, POINT_FIELD_FLOAT32, 1),
        ]
    )

    assert [field.name for field in selected_fields] == ['x', 'y', 'z', 'rgb']
    assert ignored_fields == ['intensity']

    header = build_pcd_header(
        point_count=2,
        width=2,
        height=1,
        selected_fields=selected_fields,
    )
    assert 'FIELDS x y z rgb' in header
    assert 'TYPE F F F U' in header


def test_walk_message_id_windows_bisects_corrupt_ranges():
    collected_rows = []
    corrupt_ranges = []

    def fetch_rows(start_id: int, end_id: int):
        if start_id <= 3 <= end_id or start_id <= 4 <= end_id:
            raise sqlite3.DatabaseError('bad page')
        return list(range(start_id, end_id + 1))

    walk_message_id_windows(
        first_id=1,
        last_id=5,
        window_size=4,
        fetch_rows=fetch_rows,
        handle_rows=lambda rows: collected_rows.extend(rows),
        handle_corrupt=lambda start_id, end_id, error: corrupt_ranges.append(
            {
                'start_id': start_id,
                'end_id': end_id,
                'error': error,
            }
        ),
    )

    assert collected_rows == [1, 2, 5]
    assert merge_corrupt_ranges(corrupt_ranges) == [
        {
            'start_id': 3,
            'end_id': 4,
            'error': 'bad page',
        }
    ]


def test_query_message_id_bounds_falls_back_to_declared_count_when_desc_scan_fails(monkeypatch):
    class FakeConnection:
        def execute(self, sql: str):
            if 'ORDER BY id ASC' in sql:
                return FakeCursor([{'id': 5}])
            if 'ORDER BY id DESC' in sql:
                raise sqlite3.DatabaseError('database disk image is malformed')
            raise AssertionError(sql)

    first_id, last_id = query_message_id_bounds(
        FakeConnection(),
        declared_message_count=12,
    )

    assert first_id == 5
    assert last_id == 16


class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows[0]


def test_parse_bag_exports_common_topics_with_fake_ros_runtime(tmp_path, monkeypatch):
    bag_dir = tmp_path / 'bag'
    bag_dir.mkdir()
    db_path = bag_dir / 'sample_0.db3'
    payloads = build_fake_bag(db_path)
    write_fake_metadata(bag_dir)

    import je_software.ros2bag_parser.parser as parser_module

    monkeypatch.setattr(
        parser_module,
        '_load_ros_runtime',
        lambda: FakeRosRuntime(payloads),
    )

    progress_messages = []
    manifest = parse_bag(
        bag_path=str(bag_dir),
        output_dir=str(bag_dir / '_parsed'),
        overwrite=False,
        progress_callback=progress_messages.append,
    )

    output_root = Path(manifest['output_dir'])
    assert (output_root / 'manifest.json').is_file()

    color_dir = output_root / 'camera_color_image_raw'
    assert (color_dir / 'index.jsonl').is_file()
    assert (color_dir / '00000001.png').is_file()

    camera_info_dir = output_root / 'camera_color_camera_info'
    assert (camera_info_dir / 'messages.jsonl').is_file()
    assert (camera_info_dir / 'latest.json').is_file()

    point_dir = output_root / 'camera_depth_points'
    assert (point_dir / 'index.jsonl').is_file()
    assert (point_dir / '00000003.pcd').is_file()

    topic_payloads = {item['topic']: item for item in manifest['topics']}
    assert topic_payloads['/camera/color/image_raw']['exported_count'] == 1
    assert topic_payloads['/camera/color/camera_info']['exported_count'] == 1
    assert topic_payloads['/camera/depth/points']['exported_count'] == 1
    assert any(message.startswith('Starting parse:') for message in progress_messages)
    assert any(message.startswith('Progress [sample_0.db3]: 100%') for message in progress_messages)
    assert any(message.startswith('Finished parse:') for message in progress_messages)


def build_fake_bag(db_path: Path) -> dict[bytes, object]:
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute(
            (
                'CREATE TABLE topics ('
                'id INTEGER PRIMARY KEY, '
                'name TEXT NOT NULL, '
                'type TEXT NOT NULL, '
                'serialization_format TEXT NOT NULL, '
                'offered_qos_profiles TEXT NOT NULL'
                ')'
            )
        )
        connection.execute(
            (
                'CREATE TABLE messages ('
                'id INTEGER PRIMARY KEY, '
                'topic_id INTEGER NOT NULL, '
                'timestamp INTEGER NOT NULL, '
                'data BLOB NOT NULL'
                ')'
            )
        )
        connection.executemany(
            'INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) VALUES (?, ?, ?, ?, ?)',
            [
                (1, '/camera/color/image_raw', 'sensor_msgs/msg/Image', 'cdr', ''),
                (2, '/camera/color/camera_info', 'sensor_msgs/msg/CameraInfo', 'cdr', ''),
                (3, '/camera/depth/points', 'sensor_msgs/msg/PointCloud2', 'cdr', ''),
            ],
        )

        color_blob = b'color-image'
        camera_info_blob = b'camera-info'
        pointcloud_blob = b'pointcloud'
        connection.executemany(
            'INSERT INTO messages (id, topic_id, timestamp, data) VALUES (?, ?, ?, ?)',
            [
                (1, 1, 100, sqlite3.Binary(color_blob)),
                (2, 2, 101, sqlite3.Binary(camera_info_blob)),
                (3, 3, 102, sqlite3.Binary(pointcloud_blob)),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    point_dtype = np.dtype(
        [
            ('x', '<f4'),
            ('y', '<f4'),
            ('z', '<f4'),
            ('rgb', '<u4'),
        ]
    )
    point_array = np.array(
        [
            (0.1, 0.2, 0.3, 0x00112233),
            (1.1, 1.2, 1.3, 0x00445566),
        ],
        dtype=point_dtype,
    )

    return {
        color_blob: FakeImageMessage(
            header=FakeHeader(FakeStamp(0, 100), 'camera_color_optical_frame'),
            width=2,
            height=1,
            encoding='bgr8',
            step=6,
            data=bytes([0, 0, 255, 0, 255, 0]),
        ),
        camera_info_blob: FakeCameraInfo(
            header=FakeHeader(FakeStamp(0, 101), 'camera_color_optical_frame'),
            height=1,
            width=2,
            distortion_model='plumb_bob',
            d=[0.0] * 5,
            k=[1.0] * 9,
            r=[1.0] * 9,
            p=[1.0] * 12,
        ),
        pointcloud_blob: FakePointCloud2(
            header=FakeHeader(FakeStamp(0, 102), 'camera_depth_optical_frame'),
            width=2,
            height=1,
            fields=[
                FakePointField('x', 0, POINT_FIELD_FLOAT32, 1),
                FakePointField('y', 4, POINT_FIELD_FLOAT32, 1),
                FakePointField('z', 8, POINT_FIELD_FLOAT32, 1),
                FakePointField('rgb', 12, POINT_FIELD_UINT32, 1),
            ],
            is_bigendian=False,
            point_step=16,
            row_step=32,
            data=point_array.tobytes(),
            is_dense=True,
        ),
    }


def write_fake_metadata(bag_dir: Path) -> None:
    payload = {
        'rosbag2_bagfile_information': {
            'storage_identifier': 'sqlite3',
            'duration': {'nanoseconds': 2},
            'starting_time': {'nanoseconds_since_epoch': 100},
            'message_count': 3,
            'relative_file_paths': ['sample_0.db3'],
            'topics_with_message_count': [
                {
                    'topic_metadata': {
                        'name': '/camera/color/image_raw',
                        'type': 'sensor_msgs/msg/Image',
                        'serialization_format': 'cdr',
                        'offered_qos_profiles': '',
                    },
                    'message_count': 1,
                },
                {
                    'topic_metadata': {
                        'name': '/camera/color/camera_info',
                        'type': 'sensor_msgs/msg/CameraInfo',
                        'serialization_format': 'cdr',
                        'offered_qos_profiles': '',
                    },
                    'message_count': 1,
                },
                {
                    'topic_metadata': {
                        'name': '/camera/depth/points',
                        'type': 'sensor_msgs/msg/PointCloud2',
                        'serialization_format': 'cdr',
                        'offered_qos_profiles': '',
                    },
                    'message_count': 1,
                },
            ],
        }
    }
    with (bag_dir / 'metadata.yaml').open('w', encoding='utf-8') as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)
