import json
import math

import numpy as np
import pytest

from je_software.calibration_common import (
    CharucoBoardHelper,
    invert_transform,
    load_eye_in_hand_calibration_result,
    load_eye_to_hand_calibration_result,
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


def test_eye_to_hand_consistency_equation_is_self_consistent() -> None:
    t_base_cam = make_transform(
        _rotation_y(math.radians(-12.0)) @ _rotation_z(math.radians(8.0)),
        np.asarray([0.62, -0.18, 0.84], dtype=np.float64),
    )
    t_gripper_target = make_transform(
        _rotation_x(math.radians(9.0)) @ _rotation_z(math.radians(-22.0)),
        np.asarray([0.08, 0.03, 0.11], dtype=np.float64),
    )

    for index in range(6):
        angle = math.radians(10.0 + index * 9.0)
        t_base_gripper = make_transform(
            _rotation_z(angle) @ _rotation_y(angle * 0.4),
            np.asarray([0.22 + index * 0.03, -0.16 + index * 0.015, 0.42], dtype=np.float64),
        )
        t_cam_target = (
            invert_transform(t_base_cam)
            @ t_base_gripper
            @ t_gripper_target
        )
        recovered = invert_transform(t_base_gripper) @ t_base_cam @ t_cam_target
        assert np.allclose(recovered, t_gripper_target, atol=1e-9)


def test_eye_to_hand_point_check_formula_matches_visual_and_robot_chains() -> None:
    t_base_cam = make_transform(
        _rotation_z(math.radians(14.0)),
        np.asarray([0.58, -0.11, 0.81], dtype=np.float64),
    )
    t_base_gripper = make_transform(
        _rotation_y(math.radians(-18.0)),
        np.asarray([0.24, -0.17, 0.39], dtype=np.float64),
    )
    t_gripper_target = make_transform(
        _rotation_x(math.radians(11.0)),
        np.asarray([0.09, 0.01, 0.12], dtype=np.float64),
    )
    point_t = np.asarray([0.13, 0.07, 0.0], dtype=np.float64)

    t_cam_target = (
        invert_transform(t_base_cam)
        @ t_base_gripper
        @ t_gripper_target
    )
    p_base_from_camera = transform_point(t_base_cam @ t_cam_target, point_t)
    p_base_from_robot = transform_point(t_base_gripper @ t_gripper_target, point_t)

    assert np.allclose(p_base_from_camera, p_base_from_robot, atol=1e-12)

    expected_offset = np.asarray([0.0010, -0.0015, 0.0006], dtype=np.float64)
    error = (p_base_from_robot + expected_offset) - p_base_from_camera
    assert np.allclose(error, expected_offset, atol=1e-12)


def test_default_point_set_applies_uniform_z_offset() -> None:
    helper = CharucoBoardHelper.from_board_config(
        {
            'type': 'charuco',
            'squares_x': 7,
            'squares_y': 5,
            'square_length_m': 0.05,
            'marker_length_m': 0.03,
            'aruco_dictionary': 'DICT_4X4_50',
            'charuco_config_path': None,
        }
    )

    points = helper.default_point_set(
        point_set='center_corners',
        extra_charuco_ids=[1, 8],
        point_z_offset_m=0.042,
    )

    assert points
    for point in points:
        assert math.isclose(float(point.point_t[2]), 0.042, abs_tol=1e-12)


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


def test_load_eye_to_hand_calibration_result(tmp_path) -> None:
    t_base_cam = make_transform(
        _rotation_y(math.radians(-15.0)),
        np.asarray([0.55, -0.12, 0.76], dtype=np.float64),
    )
    t_gripper_target = make_transform(
        _rotation_z(math.radians(18.0)),
        np.asarray([0.07, 0.02, 0.09], dtype=np.float64),
    )
    payload = {
        'pose_frame_id': 'base_link',
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
            't_base_cam': transform_to_json_dict(t_base_cam),
            't_gripper_target': transform_to_json_dict(t_gripper_target),
        },
    }
    result_path = tmp_path / 'eye_to_hand_result.json'
    result_path.write_text(json.dumps(payload), encoding='utf-8')

    result = load_eye_to_hand_calibration_result(result_path)

    assert result.pose_frame_id == 'base_link'
    assert result.camera_frame_id == 'camera_color_optical_frame'
    assert np.allclose(result.t_base_cam, t_base_cam, atol=1e-12)
    assert np.allclose(result.t_gripper_target, t_gripper_target, atol=1e-12)
    assert np.allclose(
        result.t_cam_base @ result.t_base_cam,
        np.eye(4, dtype=np.float64),
        atol=1e-12,
    )
    assert np.allclose(
        result.t_target_gripper @ result.t_gripper_target,
        np.eye(4, dtype=np.float64),
        atol=1e-12,
    )


def test_load_eye_to_hand_calibration_result_requires_best_transforms(
    tmp_path,
) -> None:
    result_path = tmp_path / 'broken_eye_to_hand_result.json'
    result_path.write_text(
        json.dumps(
            {
                'pose_frame_id': 'base_link',
                'camera_frame_id': 'camera_color_optical_frame',
                'board': {
                    'type': 'charuco',
                    'squares_x': 7,
                    'squares_y': 5,
                    'square_length_m': 0.05,
                    'marker_length_m': 0.03,
                    'aruco_dictionary': 'DICT_4X4_50',
                },
                'best_result': {
                    't_base_cam': transform_to_json_dict(np.eye(4, dtype=np.float64)),
                },
            }
        ),
        encoding='utf-8',
    )

    with pytest.raises(ValueError):
        load_eye_to_hand_calibration_result(result_path)
