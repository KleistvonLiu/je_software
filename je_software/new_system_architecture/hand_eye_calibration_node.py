"""Backward-compatible alias for the eye-to-hand calibration node."""

from .eye_to_hand_calibration_node import (
    EyeToHandCalibrationNode,
    main,
)


HandEyeCalibrationNode = EyeToHandCalibrationNode
