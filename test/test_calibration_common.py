import json
import math

import numpy as np
import pytest

from je_software.calibration_common import (
    invert_transform,
    load_eye_in_hand_calibration_result,
    make_transform,
    transform_point,
    transform_to_json_dict,
)


def _rotation_x(angle_rad: float) -> np.ndarray:
    c_value = math.cos(angle_rad)
    s_value = math.sin(angle_rad)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c_value, -s_value],
            [0.0, s_value, c_value],
        ],
        dtype=np.float64,
    )


def _rotation_y(angle_rad: float) -> np.ndarray:
    c_value = math.cos(angle_rad)
    s_value = math.sin(angle_rad)
    return np.asarray(
        [
            [c_value, 0.0, s_value],
            [0.0, 1.0, 0.0],
            [-s_value, 0.0, c_value],
        ],
        dtype=np.float64,
    )


def _rotation_z(angle_rad: float) -> np.ndarray:
    c_value = math.cos(angle_rad)
    s_value = math.sin(angle_rad)
    return np.asarray(
        [
            [c_value, -s_value, 0.0],
            [s_value, c_value, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_consistency_equation_is_self_consistent() -> None:
    t_gripper_cam = make_transform(
        _rotation_z(math.radians(25.0)) @ _rotation_x(math.radians(-10.0)),
        np.asarray([0.04, -0.03, 0.12], dtype=np.float64),
    )
    t_base_target = make_transform(
        _rotation_y(math.radians(18.0)) @ _rotation_z(math.radians(5.0)),
        np.asarray([0.55, -0.10, 0.21], dtype=np.float64),
    )

    for index in range(6):
        angle = math.radians(8.0 + index * 11.0)
        t_base_gripper = make_transform(
            _rotation_z(angle) @ _rotation_y(angle * 0.5),
            np.asarray([0.3 + index * 0.02, -0.2 + index * 0.01, 0.4], dtype=np.float64),
        )
        t_cam_target = (
            invert_transform(t_gripper_cam)
            @ invert_transform(t_base_gripper)
            @ t_base_target
        )
        recovered = t_base_gripper @ t_gripper_cam @ t_cam_target
        assert np.allclose(recovered, t_base_target, atol=1e-9)


def test_point_check_error_formula_recovers_manual_offset() -> None:
    t_base_gripper_lock = make_transform(
        _rotation_x(math.radians(15.0)),
        np.asarray([0.25, -0.15, 0.38], dtype=np.float64),
    )
    t_gripper_cam = make_transform(
        _rotation_z(math.radians(-30.0)),
        np.asarray([0.02, 0.01, 0.09], dtype=np.float64),
    )
    t_cam_target_lock = make_transform(
        _rotation_y(math.radians(12.0)),
        np.asarray([0.18, -0.02, 0.54], dtype=np.float64),
    )
    point_t = np.asarray([0.11, 0.08, 0.0], dtype=np.float64)

    p_base_target = transform_point(
        t_base_gripper_lock @ t_gripper_cam @ t_cam_target_lock,
        point_t,
    )
    expected_offset = np.asarray([0.0015, -0.0020, 0.0008], dtype=np.float64)
    p_base_measured = p_base_target + expected_offset
    error = p_base_measured - p_base_target

    assert np.allclose(error, expected_offset, atol=1e-12)


def test_load_eye_in_hand_calibration_result(tmp_path) -> None:
    t_gripper_cam = make_transform(
        _rotation_z(math.radians(20.0)),
        np.asarray([0.03, 0.01, 0.10], dtype=np.float64),
    )
    payload = {
        'pose_frame_id': 'base_link',
        'gripper_frame': 'gripper_link',
        'camera_frame_id': 'camera_color_optical_frame',
        'board': {
            'type': 'charuco',
            'squares_x': 7,
            'squares_y': 5,
            'square_length_m': 0.05,
            'marker_length_m': 0.03,
            'aruco_dictionary': 'DICT_4X4_50',
            'charuco_config_path': None,
        },
        'best_result': {
            't_gripper_cam': transform_to_json_dict(t_gripper_cam),
        },
    }
    result_path = tmp_path / 'eye_in_hand_result.json'
    result_path.write_text(json.dumps(payload), encoding='utf-8')

    result = load_eye_in_hand_calibration_result(result_path)

    assert result.pose_frame_id == 'base_link'
    assert result.gripper_frame == 'gripper_link'
    assert result.camera_frame_id == 'camera_color_optical_frame'
    assert np.allclose(result.t_gripper_cam, t_gripper_cam, atol=1e-12)
    assert np.allclose(
        result.t_cam_gripper @ result.t_gripper_cam,
        np.eye(4, dtype=np.float64),
        atol=1e-12,
    )


def test_load_eye_in_hand_calibration_result_requires_best_transform(
    tmp_path,
) -> None:
    result_path = tmp_path / 'broken_eye_in_hand_result.json'
    result_path.write_text(
        json.dumps(
            {
                'pose_frame_id': 'base_link',
                'gripper_frame': 'gripper_link',
                'camera_frame_id': 'camera_color_optical_frame',
                'board': {
                    'type': 'charuco',
                    'squares_x': 7,
                    'squares_y': 5,
                    'square_length_m': 0.05,
                    'marker_length_m': 0.03,
                    'aruco_dictionary': 'DICT_4X4_50',
                },
                'best_result': {},
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError):
        load_eye_in_hand_calibration_result(result_path)
