#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml


SUPPORTED_MESSAGE_TYPES = {
    'sensor_msgs/msg/Image',
    'sensor_msgs/msg/CameraInfo',
    'sensor_msgs/msg/PointCloud2',
}

DEFAULT_WINDOW_SIZE = 16

POINT_FIELD_INT8 = 1
POINT_FIELD_UINT8 = 2
POINT_FIELD_INT16 = 3
POINT_FIELD_UINT16 = 4
POINT_FIELD_INT32 = 5
POINT_FIELD_UINT32 = 6
POINT_FIELD_FLOAT32 = 7
POINT_FIELD_FLOAT64 = 8


@dataclass(frozen=True)
class TopicInfo:
    topic_id: int
    name: str
    msg_type: str
    serialization_format: str
    offered_qos_profiles: str
    declared_count: int
    sanitized_name: str


@dataclass
class TopicStats:
    info: TopicInfo
    output_dir: Path
    read_rows: int = 0
    decoded_count: int = 0
    exported_count: int = 0
    skipped_decode_count: int = 0
    unsupported_count: int = 0

    @property
    def skipped_corrupt_count(self) -> int:
        return max(int(self.info.declared_count) - int(self.read_rows), 0)

    def to_json(self) -> dict[str, Any]:
        return {
            'topic': self.info.name,
            'type': self.info.msg_type,
            'serialization_format': self.info.serialization_format,
            'declared_count': int(self.info.declared_count),
            'read_rows': int(self.read_rows),
            'decoded_count': int(self.decoded_count),
            'exported_count': int(self.exported_count),
            'skipped_corrupt_count': int(self.skipped_corrupt_count),
            'skipped_decode_count': int(self.skipped_decode_count),
            'unsupported_count': int(self.unsupported_count),
            'output_dir': str(self.output_dir),
        }


@dataclass(frozen=True)
class SelectedPointField:
    name: str
    offset: int
    datatype: int
    count: int


@dataclass(frozen=True)
class RosRuntime:
    get_message: Callable[[str], Any]
    deserialize_message: Callable[[bytes, Any], Any]


class BaseExporter:
    def __init__(self, topic_info: TopicInfo, output_root: Path) -> None:
        self.topic_info = topic_info
        self.topic_dir = output_root / topic_info.sanitized_name
        self.topic_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.topic_dir / 'index.jsonl'
        self._index_file = self.index_path.open('w', encoding='utf-8')

    def export(
        self,
        message: Any,
        *,
        message_id: int,
        timestamp_ns: int,
        output_root: Path,
    ) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self._index_file.close()

    def finalize(self) -> dict[str, str]:
        return {
            'index_jsonl': _relative_path(self.index_path, self.topic_dir.parent),
        }

    def _write_index_record(self, payload: dict[str, Any]) -> None:
        self._index_file.write(json.dumps(payload, ensure_ascii=True) + '\n')


class ImageExporter(BaseExporter):
    def export(
        self,
        message: Any,
        *,
        message_id: int,
        timestamp_ns: int,
        output_root: Path,
    ) -> None:
        encoding = str(getattr(message, 'encoding', ''))
        export_kind = select_image_export_kind(encoding)
        image_array = image_message_to_numpy(message)
        file_suffix = '.png' if export_kind in ('png', 'png16') else '.npy'
        file_path = self.topic_dir / f'{message_id:08d}{file_suffix}'
        if export_kind in ('png', 'png16'):
            image_to_write = convert_image_for_png(image_array, encoding)
            if not cv2.imwrite(str(file_path), image_to_write):
                raise RuntimeError(f'Failed to write image file: {file_path}')
        else:
            with file_path.open('wb') as file_obj:
                np.save(file_obj, image_array, allow_pickle=False)

        header_stamp_ns = _header_stamp_ns(message)
        self._write_index_record(
            {
                'topic': self.topic_info.name,
                'message_id': int(message_id),
                'timestamp_ns': int(timestamp_ns),
                'header_stamp_ns': header_stamp_ns,
                'frame_id': _frame_id(message),
                'encoding': encoding,
                'width': int(getattr(message, 'width', 0)),
                'height': int(getattr(message, 'height', 0)),
                'step': int(getattr(message, 'step', 0)),
                'path': _relative_path(file_path, output_root),
            }
        )


class CameraInfoExporter(BaseExporter):
    def __init__(self, topic_info: TopicInfo, output_root: Path) -> None:
        super().__init__(topic_info, output_root)
        self.messages_path = self.topic_dir / 'messages.jsonl'
        self.latest_path = self.topic_dir / 'latest.json'
        self._messages_file = self.messages_path.open('w', encoding='utf-8')
        self._latest_payload: dict[str, Any] | None = None

    def export(
        self,
        message: Any,
        *,
        message_id: int,
        timestamp_ns: int,
        output_root: Path,
    ) -> None:
        payload = camera_info_to_json(message)
        payload['topic'] = self.topic_info.name
        payload['message_id'] = int(message_id)
        payload['timestamp_ns'] = int(timestamp_ns)
        self._messages_file.write(json.dumps(payload, ensure_ascii=True) + '\n')
        self._write_index_record(payload)
        self._latest_payload = payload

    def close(self) -> None:
        self._messages_file.close()
        super().close()

    def finalize(self) -> dict[str, str]:
        if self._latest_payload is not None:
            with self.latest_path.open('w', encoding='utf-8') as file_obj:
                json.dump(self._latest_payload, file_obj, ensure_ascii=True, indent=2)
        result = super().finalize()
        result['messages_jsonl'] = _relative_path(self.messages_path, self.topic_dir.parent)
        if self._latest_payload is not None:
            result['latest_json'] = _relative_path(self.latest_path, self.topic_dir.parent)
        return result


class PointCloudExporter(BaseExporter):
    def export(
        self,
        message: Any,
        *,
        message_id: int,
        timestamp_ns: int,
        output_root: Path,
    ) -> None:
        structured_array, selected_fields, ignored_fields = pointcloud_message_to_structured_array(
            message
        )
        file_path = self.topic_dir / f'{message_id:08d}.pcd'
        write_binary_pcd(
            file_path=file_path,
            point_count=int(structured_array.shape[0]),
            width=int(getattr(message, 'width', structured_array.shape[0])),
            height=int(getattr(message, 'height', 1)),
            structured_array=structured_array,
            selected_fields=selected_fields,
        )
        self._write_index_record(
            {
                'topic': self.topic_info.name,
                'message_id': int(message_id),
                'timestamp_ns': int(timestamp_ns),
                'header_stamp_ns': _header_stamp_ns(message),
                'frame_id': _frame_id(message),
                'width': int(getattr(message, 'width', 0)),
                'height': int(getattr(message, 'height', 0)),
                'is_dense': bool(getattr(message, 'is_dense', False)),
                'point_step': int(getattr(message, 'point_step', 0)),
                'row_step': int(getattr(message, 'row_step', 0)),
                'fields': point_fields_to_json(getattr(message, 'fields', [])),
                'selected_fields': [field.name for field in selected_fields],
                'ignored_fields': list(ignored_fields),
                'path': _relative_path(file_path, output_root),
            }
        )


def sanitize_topic_name(topic_name: str) -> str:
    stripped = str(topic_name).strip().strip('/')
    if not stripped:
        return 'root'
    parts = [part for part in stripped.split('/') if part]
    sanitized = '_'.join(parts)
    return sanitized or 'root'


def select_image_export_kind(encoding: str) -> str:
    normalized = str(encoding).strip().lower()
    if normalized in {'bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8', '8uc1'}:
        return 'png'
    if normalized in {'16uc1', 'mono16'}:
        return 'png16'
    return 'npy'


def merge_corrupt_ranges(
    corrupt_ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not corrupt_ranges:
        return []
    merged: list[dict[str, Any]] = []
    for item in sorted(corrupt_ranges, key=lambda value: (value['start_id'], value['end_id'])):
        if not merged:
            merged.append(dict(item))
            continue
        previous = merged[-1]
        if (
            item['start_id'] <= previous['end_id'] + 1
            and item.get('error', '') == previous.get('error', '')
        ):
            previous['end_id'] = max(previous['end_id'], item['end_id'])
            continue
        merged.append(dict(item))
    return merged


def walk_message_id_windows(
    *,
    first_id: int | None,
    last_id: int | None,
    window_size: int,
    fetch_rows: Callable[[int, int], list[Any]],
    handle_rows: Callable[[list[Any]], None],
    handle_corrupt: Callable[[int, int, str], None],
) -> None:
    if first_id is None or last_id is None:
        return
    if int(window_size) <= 0:
        raise ValueError('window_size must be positive.')
    current_id = int(first_id)
    while current_id <= int(last_id):
        window_end = min(current_id + int(window_size) - 1, int(last_id))
        _walk_range_best_effort(
            start_id=current_id,
            end_id=window_end,
            fetch_rows=fetch_rows,
            handle_rows=handle_rows,
            handle_corrupt=handle_corrupt,
        )
        current_id = window_end + 1


def _walk_range_best_effort(
    *,
    start_id: int,
    end_id: int,
    fetch_rows: Callable[[int, int], list[Any]],
    handle_rows: Callable[[list[Any]], None],
    handle_corrupt: Callable[[int, int, str], None],
) -> None:
    try:
        rows = fetch_rows(start_id, end_id)
    except sqlite3.DatabaseError as exc:
        if start_id == end_id:
            handle_corrupt(start_id, end_id, str(exc))
            return
        middle_id = (start_id + end_id) // 2
        _walk_range_best_effort(
            start_id=start_id,
            end_id=middle_id,
            fetch_rows=fetch_rows,
            handle_rows=handle_rows,
            handle_corrupt=handle_corrupt,
        )
        _walk_range_best_effort(
            start_id=middle_id + 1,
            end_id=end_id,
            fetch_rows=fetch_rows,
            handle_rows=handle_rows,
            handle_corrupt=handle_corrupt,
        )
        return
    handle_rows(rows)


def parse_bag(
    bag_path: str,
    output_dir: str,
    overwrite: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    bag_dir = Path(bag_path).expanduser().resolve()
    if not bag_dir.is_dir():
        raise FileNotFoundError(f'Bag directory does not exist: {bag_dir}')

    metadata_path = bag_dir / 'metadata.yaml'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'metadata.yaml not found: {metadata_path}')

    metadata = load_bag_metadata(metadata_path)
    storage_identifier = str(
        metadata.get('rosbag2_bagfile_information', {}).get('storage_identifier', '')
    )
    if storage_identifier != 'sqlite3':
        raise RuntimeError(
            f'Only sqlite3 rosbag2 storage is supported, got: {storage_identifier}'
        )

    db3_paths = find_sqlite_bag_files(bag_dir, metadata)
    if not db3_paths:
        raise FileNotFoundError(f'No sqlite3 bag files found in: {bag_dir}')

    output_root = Path(output_dir).expanduser().resolve()
    prepare_output_directory(output_root, overwrite=bool(overwrite))

    ros_runtime = _load_ros_runtime()
    declared_counts = declared_topic_message_counts(metadata)

    topics: dict[int, TopicInfo] = {}
    topic_stats: dict[int, TopicStats] = {}
    exporters: dict[int, BaseExporter] = {}
    corrupt_ranges: list[dict[str, Any]] = []
    topic_files: dict[int, dict[str, str]] = {}
    total_read_rows = 0
    total_decoded_count = 0
    total_skipped_decode_count = 0
    total_unsupported_count = 0
    message_classes: dict[str, Any] = {}

    if progress_callback is not None:
        progress_callback(
            f'Starting parse: bag={bag_dir}, output={output_root}'
        )

    try:
        for db3_path in db3_paths:
            connection = sqlite3.connect(str(db3_path))
            connection.row_factory = sqlite3.Row
            try:
                if not topics:
                    topics = load_topics(connection, declared_counts)
                    topic_stats = {
                        topic_id: TopicStats(
                            info=topic_info,
                            output_dir=output_root / topic_info.sanitized_name,
                        )
                        for topic_id, topic_info in topics.items()
                    }
                    exporters = {
                        topic_id: make_topic_exporter(topic_info, output_root)
                        for topic_id, topic_info in topics.items()
                        if topic_info.msg_type in SUPPORTED_MESSAGE_TYPES
                    }
                    if progress_callback is not None:
                        progress_callback(
                            'Loaded topics: '
                            f'{len(topics)} total, '
                            f'{len(exporters)} supported, '
                            f'declared_messages='
                            f'{int(metadata.get("rosbag2_bagfile_information", {}).get("message_count", 0))}'
                        )

                first_id, last_id = query_message_id_bounds(
                    connection,
                    declared_message_count=int(
                        metadata.get('rosbag2_bagfile_information', {}).get(
                            'message_count',
                            0,
                        )
                    ),
                )
                if first_id is None or last_id is None:
                    continue

                if progress_callback is not None:
                    progress_callback(
                        f'Parsing file {db3_path.name}: '
                        f'id_range={first_id}..{last_id}'
                    )

                last_reported_percent = -1

                def emit_progress(current_id: int, *, force: bool = False) -> None:
                    nonlocal last_reported_percent
                    if progress_callback is None:
                        return
                    total_ids = max(int(last_id) - int(first_id) + 1, 1)
                    completed_ids = max(
                        0,
                        min(int(current_id), int(last_id)) - int(first_id) + 1,
                    )
                    percent = int((completed_ids * 100) / total_ids)
                    if not force and percent <= last_reported_percent:
                        return
                    last_reported_percent = percent
                    progress_callback(
                        f'Progress [{db3_path.name}]: '
                        f'{percent}% '
                        f'({completed_ids}/{total_ids} ids), '
                        f'read_rows={total_read_rows}, '
                        f'decoded={total_decoded_count}, '
                        f'decode_skips={total_skipped_decode_count}, '
                        f'corrupt_ranges={len(corrupt_ranges)}'
                    )

                def fetch_rows(start_id: int, end_id: int) -> list[sqlite3.Row]:
                    cursor = connection.execute(
                        (
                            'SELECT id, topic_id, timestamp, data '
                            'FROM messages '
                            'WHERE id BETWEEN ? AND ? '
                            'ORDER BY id'
                        ),
                        (int(start_id), int(end_id)),
                    )
                    return list(cursor.fetchall())

                def handle_rows(rows: list[sqlite3.Row]) -> None:
                    nonlocal total_read_rows
                    nonlocal total_decoded_count
                    nonlocal total_skipped_decode_count
                    nonlocal total_unsupported_count

                    for row in rows:
                        row_topic_id = int(row['topic_id'])
                        if row_topic_id not in topics:
                            continue
                        topic_info = topics[row_topic_id]
                        stats = topic_stats[row_topic_id]
                        stats.read_rows += 1
                        total_read_rows += 1

                        if topic_info.msg_type not in SUPPORTED_MESSAGE_TYPES:
                            stats.unsupported_count += 1
                            total_unsupported_count += 1
                            continue

                        try:
                            message_class = message_classes.get(topic_info.msg_type)
                            if message_class is None:
                                message_class = ros_runtime.get_message(topic_info.msg_type)
                                message_classes[topic_info.msg_type] = message_class
                            message = ros_runtime.deserialize_message(
                                row['data'],
                                message_class,
                            )
                        except Exception:
                            stats.skipped_decode_count += 1
                            total_skipped_decode_count += 1
                            continue

                        stats.decoded_count += 1
                        total_decoded_count += 1
                        exporters[row_topic_id].export(
                            message,
                            message_id=int(row['id']),
                            timestamp_ns=int(row['timestamp']),
                            output_root=output_root,
                        )
                        stats.exported_count += 1

                    if rows:
                        emit_progress(int(rows[-1]['id']))

                def handle_corrupt(start_id: int, end_id: int, error_text: str) -> None:
                    corrupt_ranges.append(
                        {
                            'file': db3_path.name,
                            'start_id': int(start_id),
                            'end_id': int(end_id),
                            'error': str(error_text),
                        }
                    )
                    if progress_callback is not None:
                        progress_callback(
                            f'Skipping corrupt id range '
                            f'{start_id}..{end_id} in {db3_path.name}: '
                            f'{error_text}'
                        )
                    emit_progress(int(end_id))

                walk_message_id_windows(
                    first_id=first_id,
                    last_id=last_id,
                    window_size=DEFAULT_WINDOW_SIZE,
                    fetch_rows=fetch_rows,
                    handle_rows=handle_rows,
                    handle_corrupt=handle_corrupt,
                )
                emit_progress(int(last_id), force=True)
            finally:
                connection.close()

        for topic_id, exporter in exporters.items():
            topic_files[topic_id] = exporter.finalize()

        merged_corrupt_ranges = merge_corrupt_ranges(corrupt_ranges)
        manifest = build_manifest(
            bag_dir=bag_dir,
            output_root=output_root,
            metadata=metadata,
            db3_paths=db3_paths,
            topics=topics,
            topic_stats=topic_stats,
            topic_files=topic_files,
            corrupt_ranges=merged_corrupt_ranges,
            total_read_rows=total_read_rows,
            total_decoded_count=total_decoded_count,
            total_skipped_decode_count=total_skipped_decode_count,
            total_unsupported_count=total_unsupported_count,
        )
        manifest_path = output_root / 'manifest.json'
        with manifest_path.open('w', encoding='utf-8') as file_obj:
            json.dump(manifest, file_obj, ensure_ascii=True, indent=2)
        manifest['manifest_path'] = str(manifest_path)
        if progress_callback is not None:
            progress_callback(
                'Finished parse: '
                f'read_rows={total_read_rows}, '
                f'decoded={total_decoded_count}, '
                f'skipped_corrupt={manifest["skipped_corrupt_count"]}, '
                f'skipped_decode={total_skipped_decode_count}, '
                f'manifest={manifest_path}'
            )
        return manifest
    finally:
        for exporter in exporters.values():
            exporter.close()


def load_bag_metadata(metadata_path: Path) -> dict[str, Any]:
    with metadata_path.open('r', encoding='utf-8') as file_obj:
        return yaml.safe_load(file_obj) or {}


def declared_topic_message_counts(metadata: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    topics = metadata.get('rosbag2_bagfile_information', {}).get(
        'topics_with_message_count',
        [],
    )
    for item in topics:
        topic_metadata = item.get('topic_metadata', {})
        name = str(topic_metadata.get('name', ''))
        if not name:
            continue
        result[name] = int(item.get('message_count', 0))
    return result


def find_sqlite_bag_files(bag_dir: Path, metadata: dict[str, Any]) -> list[Path]:
    relative_paths = metadata.get('rosbag2_bagfile_information', {}).get(
        'relative_file_paths',
        [],
    )
    db3_paths = [bag_dir / str(path) for path in relative_paths if str(path).endswith('.db3')]
    if db3_paths:
        return db3_paths
    return sorted(bag_dir.glob('*.db3'))


def prepare_output_directory(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f'Output directory already exists: {output_root}. '
                'Use --overwrite to remove it first.'
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def load_topics(
    connection: sqlite3.Connection,
    declared_counts: dict[str, int],
) -> dict[int, TopicInfo]:
    rows = connection.execute(
        (
            'SELECT id, name, type, serialization_format, offered_qos_profiles '
            'FROM topics '
            'ORDER BY id'
        )
    ).fetchall()
    topics: dict[int, TopicInfo] = {}
    for row in rows:
        topic_id = int(row['id'])
        name = str(row['name'])
        topics[topic_id] = TopicInfo(
            topic_id=topic_id,
            name=name,
            msg_type=str(row['type']),
            serialization_format=str(row['serialization_format']),
            offered_qos_profiles=str(row['offered_qos_profiles']),
            declared_count=int(declared_counts.get(name, 0)),
            sanitized_name=sanitize_topic_name(name),
        )
    return topics


def query_message_id_bounds(
    connection: sqlite3.Connection,
    *,
    declared_message_count: int = 0,
) -> tuple[int | None, int | None]:
    """Return a best-effort message id range for a possibly corrupted bag.

    `MIN/MAX(id)` can force a scan over damaged pages and fail immediately on
    malformed sqlite bags. Prefer indexed primary-key lookups first. If the
    descending lookup still hits corruption, fall back to the metadata-declared
    message count, assuming rosbag2's message ids are contiguous and 1-based.
    """
    try:
        first_row = connection.execute(
            'SELECT id FROM messages ORDER BY id ASC LIMIT 1'
        ).fetchone()
    except sqlite3.DatabaseError:
        first_row = None

    if first_row is None:
        return None, None

    first_id = int(first_row['id'])

    try:
        last_row = connection.execute(
            'SELECT id FROM messages ORDER BY id DESC LIMIT 1'
        ).fetchone()
    except sqlite3.DatabaseError:
        last_row = None

    if last_row is not None:
        return first_id, int(last_row['id'])

    if int(declared_message_count) > 0:
        return first_id, first_id + int(declared_message_count) - 1

    return first_id, first_id


def make_topic_exporter(topic_info: TopicInfo, output_root: Path) -> BaseExporter:
    if topic_info.msg_type == 'sensor_msgs/msg/Image':
        return ImageExporter(topic_info, output_root)
    if topic_info.msg_type == 'sensor_msgs/msg/CameraInfo':
        return CameraInfoExporter(topic_info, output_root)
    if topic_info.msg_type == 'sensor_msgs/msg/PointCloud2':
        return PointCloudExporter(topic_info, output_root)
    raise ValueError(f'Unsupported exporter type: {topic_info.msg_type}')


def build_manifest(
    *,
    bag_dir: Path,
    output_root: Path,
    metadata: dict[str, Any],
    db3_paths: list[Path],
    topics: dict[int, TopicInfo],
    topic_stats: dict[int, TopicStats],
    topic_files: dict[int, dict[str, str]],
    corrupt_ranges: list[dict[str, Any]],
    total_read_rows: int,
    total_decoded_count: int,
    total_skipped_decode_count: int,
    total_unsupported_count: int,
) -> dict[str, Any]:
    bag_info = metadata.get('rosbag2_bagfile_information', {})
    topics_payload = []
    for topic_id in sorted(topics):
        topic_info = topics[topic_id]
        payload = topic_stats[topic_id].to_json()
        if topic_id in topic_files:
            payload['files'] = dict(topic_files[topic_id])
        topics_payload.append(payload)

    declared_message_count = int(bag_info.get('message_count', 0))
    skipped_corrupt_count = max(declared_message_count - total_read_rows, 0)
    return {
        'bag_path': str(bag_dir),
        'output_dir': str(output_root),
        'storage_identifier': bag_info.get('storage_identifier', ''),
        'duration_ns': int(bag_info.get('duration', {}).get('nanoseconds', 0)),
        'starting_time_ns': int(
            bag_info.get('starting_time', {}).get('nanoseconds_since_epoch', 0)
        ),
        'declared_message_count': declared_message_count,
        'read_rows': int(total_read_rows),
        'decoded_count': int(total_decoded_count),
        'skipped_corrupt_count': int(skipped_corrupt_count),
        'skipped_decode_count': int(total_skipped_decode_count),
        'unsupported_count': int(total_unsupported_count),
        'db3_files': [str(path) for path in db3_paths],
        'corrupt_ranges': list(corrupt_ranges),
        'topics': topics_payload,
    }


def _load_ros_runtime() -> RosRuntime:
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'ROS2 Python interfaces are unavailable. '
            'Please source your ROS2 environment before running ros2bag_parser.'
        ) from exc
    return RosRuntime(
        get_message=get_message,
        deserialize_message=deserialize_message,
    )


def _header_stamp_ns(message: Any) -> int | None:
    header = getattr(message, 'header', None)
    if header is None:
        return None
    stamp = getattr(header, 'stamp', None)
    if stamp is None:
        return None
    sec = int(getattr(stamp, 'sec', 0))
    nanosec = int(getattr(stamp, 'nanosec', 0))
    return sec * 1_000_000_000 + nanosec


def _frame_id(message: Any) -> str:
    header = getattr(message, 'header', None)
    if header is None:
        return ''
    return str(getattr(header, 'frame_id', ''))


def camera_info_to_json(message: Any) -> dict[str, Any]:
    roi = getattr(message, 'roi', None)
    return {
        'header_stamp_ns': _header_stamp_ns(message),
        'frame_id': _frame_id(message),
        'height': int(getattr(message, 'height', 0)),
        'width': int(getattr(message, 'width', 0)),
        'distortion_model': str(getattr(message, 'distortion_model', '')),
        'd': [float(value) for value in getattr(message, 'd', [])],
        'k': [float(value) for value in getattr(message, 'k', [])],
        'r': [float(value) for value in getattr(message, 'r', [])],
        'p': [float(value) for value in getattr(message, 'p', [])],
        'binning_x': int(getattr(message, 'binning_x', 0)),
        'binning_y': int(getattr(message, 'binning_y', 0)),
        'roi': {
            'x_offset': int(getattr(roi, 'x_offset', 0)),
            'y_offset': int(getattr(roi, 'y_offset', 0)),
            'height': int(getattr(roi, 'height', 0)),
            'width': int(getattr(roi, 'width', 0)),
            'do_rectify': bool(getattr(roi, 'do_rectify', False)),
        },
    }


def image_message_to_numpy(message: Any) -> np.ndarray:
    encoding = str(getattr(message, 'encoding', '')).strip().lower()
    width = int(getattr(message, 'width', 0))
    height = int(getattr(message, 'height', 0))
    step = int(getattr(message, 'step', 0))
    is_bigendian = bool(getattr(message, 'is_bigendian', 0))
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError('Image message width/height/step must be positive.')

    base_dtype, channel_count = resolve_image_encoding(encoding)
    bytes_per_pixel = np.dtype(base_dtype).itemsize * channel_count
    if step < width * bytes_per_pixel:
        raise ValueError(
            f'Image step {step} is smaller than width*bytes_per_pixel {width * bytes_per_pixel}.'
        )

    row_data = np.frombuffer(memoryview(getattr(message, 'data')), dtype=np.uint8)
    expected_bytes = step * height
    if row_data.size < expected_bytes:
        raise ValueError(
            f'Image data is truncated: {row_data.size} bytes < expected {expected_bytes}.'
        )

    rows = row_data[:expected_bytes].reshape(height, step)
    used = rows[:, : width * bytes_per_pixel].copy()
    endian_char = '>' if is_bigendian and np.dtype(base_dtype).itemsize > 1 else '<'
    typed_dtype = np.dtype(base_dtype).newbyteorder(endian_char)
    array = used.view(typed_dtype)
    if channel_count == 1:
        array = array.reshape(height, width)
    else:
        array = array.reshape(height, width, channel_count)
    if np.dtype(base_dtype).itemsize > 1 and typed_dtype.byteorder not in ('=', '|'):
        array = array.astype(base_dtype, copy=False)
    return array


def resolve_image_encoding(encoding: str) -> tuple[np.dtype, int]:
    normalized = str(encoding).strip().lower()
    if normalized == 'bgr8':
        return np.dtype(np.uint8), 3
    if normalized == 'rgb8':
        return np.dtype(np.uint8), 3
    if normalized == 'bgra8':
        return np.dtype(np.uint8), 4
    if normalized == 'rgba8':
        return np.dtype(np.uint8), 4
    if normalized in {'mono8', '8uc1'}:
        return np.dtype(np.uint8), 1
    if normalized in {'mono16', '16uc1'}:
        return np.dtype(np.uint16), 1

    match = re.fullmatch(r'(\d+)([usf])c(\d+)', normalized)
    if match is None:
        raise ValueError(f'Unsupported image encoding: {encoding}')

    bit_width = int(match.group(1))
    family = match.group(2)
    channel_count = int(match.group(3))
    if bit_width == 8 and family == 'u':
        return np.dtype(np.uint8), channel_count
    if bit_width == 16 and family == 'u':
        return np.dtype(np.uint16), channel_count
    if bit_width == 16 and family == 's':
        return np.dtype(np.int16), channel_count
    if bit_width == 32 and family == 'f':
        return np.dtype(np.float32), channel_count
    if bit_width == 32 and family == 'u':
        return np.dtype(np.uint32), channel_count
    raise ValueError(f'Unsupported image encoding: {encoding}')


def convert_image_for_png(image_array: np.ndarray, encoding: str) -> np.ndarray:
    normalized = str(encoding).strip().lower()
    if normalized == 'rgb8':
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    if normalized == 'rgba8':
        return cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    return image_array


def point_fields_to_json(fields: list[Any]) -> list[dict[str, Any]]:
    result = []
    for field in fields:
        result.append(
            {
                'name': str(getattr(field, 'name', '')),
                'offset': int(getattr(field, 'offset', 0)),
                'datatype': int(getattr(field, 'datatype', 0)),
                'count': int(getattr(field, 'count', 0)),
            }
        )
    return result


def select_pcd_fields(fields: list[Any]) -> tuple[list[SelectedPointField], list[str]]:
    fields_by_name = {
        str(getattr(field, 'name', '')): field
        for field in fields
    }
    selected_names = ['x', 'y', 'z']
    if 'rgb' in fields_by_name:
        selected_names.append('rgb')
    elif 'rgba' in fields_by_name:
        selected_names.append('rgba')

    missing_xyz = [name for name in ('x', 'y', 'z') if name not in fields_by_name]
    if missing_xyz:
        raise ValueError(
            f'PointCloud2 is missing required xyz fields: {", ".join(missing_xyz)}'
        )

    selected_fields = [
        SelectedPointField(
            name=name,
            offset=int(getattr(fields_by_name[name], 'offset', 0)),
            datatype=int(getattr(fields_by_name[name], 'datatype', 0)),
            count=int(getattr(fields_by_name[name], 'count', 1)),
        )
        for name in selected_names
    ]
    ignored_fields = [
        str(getattr(field, 'name', ''))
        for field in fields
        if str(getattr(field, 'name', '')) not in selected_names
    ]
    return selected_fields, ignored_fields


def pointcloud_message_to_structured_array(
    message: Any,
) -> tuple[np.ndarray, list[SelectedPointField], list[str]]:
    width = int(getattr(message, 'width', 0))
    height = int(getattr(message, 'height', 0))
    point_step = int(getattr(message, 'point_step', 0))
    row_step = int(getattr(message, 'row_step', 0))
    is_bigendian = bool(getattr(message, 'is_bigendian', False))
    fields = list(getattr(message, 'fields', []))
    selected_fields, ignored_fields = select_pcd_fields(fields)
    point_count = width * height
    if point_count <= 0:
        raise ValueError('PointCloud2 width*height must be positive.')
    if point_step <= 0 or row_step <= 0:
        raise ValueError('PointCloud2 point_step/row_step must be positive.')

    input_dtype = build_pointcloud_input_dtype(
        selected_fields=selected_fields,
        point_step=point_step,
        is_bigendian=is_bigendian,
    )
    raw_records = read_pointcloud_records(
        data=getattr(message, 'data'),
        width=width,
        height=height,
        row_step=row_step,
        point_step=point_step,
        dtype=input_dtype,
    )
    output_dtype = build_pcd_output_dtype(selected_fields)
    output_array = np.empty(point_count, dtype=output_dtype)
    for field in selected_fields:
        output_array[field.name] = raw_records[field.name].astype(
            output_dtype.fields[field.name][0],
            copy=False,
        )
    return output_array, selected_fields, ignored_fields


def build_pointcloud_input_dtype(
    *,
    selected_fields: list[SelectedPointField],
    point_step: int,
    is_bigendian: bool,
) -> np.dtype:
    names = []
    formats = []
    offsets = []
    for field in selected_fields:
        if int(field.count) != 1:
            raise ValueError(f'PCD export only supports count=1 fields, got {field.name}.')
        names.append(field.name)
        formats.append(
            point_field_datatype_to_numpy(int(field.datatype), is_bigendian=is_bigendian)
        )
        offsets.append(int(field.offset))
    return np.dtype(
        {
            'names': names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': int(point_step),
        }
    )


def build_pcd_output_dtype(selected_fields: list[SelectedPointField]) -> np.dtype:
    names = []
    formats = []
    for field in selected_fields:
        names.append(field.name)
        formats.append(point_field_datatype_to_numpy(int(field.datatype), is_bigendian=False))
    return np.dtype({'names': names, 'formats': formats})


def point_field_datatype_to_numpy(
    datatype: int,
    *,
    is_bigendian: bool,
) -> np.dtype:
    base_dtype_map = {
        POINT_FIELD_INT8: np.int8,
        POINT_FIELD_UINT8: np.uint8,
        POINT_FIELD_INT16: np.int16,
        POINT_FIELD_UINT16: np.uint16,
        POINT_FIELD_INT32: np.int32,
        POINT_FIELD_UINT32: np.uint32,
        POINT_FIELD_FLOAT32: np.float32,
        POINT_FIELD_FLOAT64: np.float64,
    }
    if datatype not in base_dtype_map:
        raise ValueError(f'Unsupported PointField datatype: {datatype}')
    base_dtype = np.dtype(base_dtype_map[datatype])
    if base_dtype.itemsize == 1:
        return base_dtype
    return base_dtype.newbyteorder('>' if is_bigendian else '<')


def read_pointcloud_records(
    *,
    data: Any,
    width: int,
    height: int,
    row_step: int,
    point_step: int,
    dtype: np.dtype,
) -> np.ndarray:
    byte_view = memoryview(data)
    required_bytes = row_step * height
    if len(byte_view) < required_bytes:
        raise ValueError(
            f'PointCloud2 data is truncated: {len(byte_view)} bytes < expected {required_bytes}.'
        )

    expected_row_bytes = width * point_step
    if row_step == expected_row_bytes:
        return np.frombuffer(byte_view[:required_bytes], dtype=dtype, count=width * height)

    row_records = []
    for row_index in range(height):
        row_start = row_index * row_step
        row_end = row_start + expected_row_bytes
        row_records.append(
            np.frombuffer(
                byte_view[row_start:row_end],
                dtype=dtype,
                count=width,
            )
        )
    return np.concatenate(row_records, axis=0)


def build_pcd_header(
    *,
    point_count: int,
    width: int,
    height: int,
    selected_fields: list[SelectedPointField],
) -> str:
    field_names = [field.name for field in selected_fields]
    sizes = []
    types = []
    counts = []
    for field in selected_fields:
        size, type_char = point_field_datatype_to_pcd_spec(field.datatype)
        sizes.append(str(size))
        types.append(type_char)
        counts.append(str(int(field.count)))
    lines = [
        '# .PCD v0.7 - Point Cloud Data file format',
        'VERSION 0.7',
        f'FIELDS {" ".join(field_names)}',
        f'SIZE {" ".join(sizes)}',
        f'TYPE {" ".join(types)}',
        f'COUNT {" ".join(counts)}',
        f'WIDTH {int(width)}',
        f'HEIGHT {int(height)}',
        'VIEWPOINT 0 0 0 1 0 0 0',
        f'POINTS {int(point_count)}',
        'DATA binary',
    ]
    return '\n'.join(lines) + '\n'


def point_field_datatype_to_pcd_spec(datatype: int) -> tuple[int, str]:
    mapping = {
        POINT_FIELD_INT8: (1, 'I'),
        POINT_FIELD_UINT8: (1, 'U'),
        POINT_FIELD_INT16: (2, 'I'),
        POINT_FIELD_UINT16: (2, 'U'),
        POINT_FIELD_INT32: (4, 'I'),
        POINT_FIELD_UINT32: (4, 'U'),
        POINT_FIELD_FLOAT32: (4, 'F'),
        POINT_FIELD_FLOAT64: (8, 'F'),
    }
    if datatype not in mapping:
        raise ValueError(f'Unsupported PointField datatype: {datatype}')
    return mapping[datatype]


def write_binary_pcd(
    *,
    file_path: Path,
    point_count: int,
    width: int,
    height: int,
    structured_array: np.ndarray,
    selected_fields: list[SelectedPointField],
) -> None:
    header = build_pcd_header(
        point_count=point_count,
        width=width,
        height=height,
        selected_fields=selected_fields,
    )
    with file_path.open('wb') as file_obj:
        file_obj.write(header.encode('ascii'))
        file_obj.write(structured_array.tobytes(order='C'))


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
