"""Tests for ``game_settings`` camera helpers."""

from __future__ import annotations

import math
import os
import unittest
from unittest import mock

from game_cam_env import (
    GAME_CAM_DEFAULT_INIT_EYE_Y,
    GAME_CAM_DEFAULT_INIT_PITCH_DEG,
    GAME_CAM_DEFAULT_PIVOT_Y,
    GAME_CAM_DEFAULT_INIT_DIST,
    GAME_CAM_DEFAULT_MAX_DIST,
    initial_camera_yaw_pitch_deg,
    orbit_pitch_deg_for_eye,
)

_BASE_CAM_ENV = {
    "GAME_CAM_INIT_LOOK_DEG": "",
    "GAME_CAM_INIT_YAW_DEG": "",
    "GAME_CAM_INIT_PITCH_DEG": "",
    "GAME_CAM_INIT_HEIGHT": "",
    "GAME_CAM_PIVOT_Y": "",
    "GAME_CAM_INIT_DIST": "",
    "GAME_CAM_MAX_DIST": "",
}


class TestInitialCamera(unittest.TestCase):
    def test_default_pitch_when_no_camera_env(self) -> None:
        with mock.patch.dict(os.environ, {**_BASE_CAM_ENV}, clear=False):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertIsNone(yaw)
        self.assertEqual(pitch, GAME_CAM_DEFAULT_INIT_PITCH_DEG)
        self.assertAlmostEqual(
            pitch,
            orbit_pitch_deg_for_eye(
                GAME_CAM_DEFAULT_INIT_EYE_Y,
                GAME_CAM_DEFAULT_PIVOT_Y,
                GAME_CAM_DEFAULT_INIT_DIST,
                GAME_CAM_DEFAULT_MAX_DIST,
            ),
            places=6,
        )

    def test_look_deg_one_number_sets_default_pitch(self) -> None:
        with mock.patch.dict(os.environ, {**_BASE_CAM_ENV, "GAME_CAM_INIT_LOOK_DEG": "12"}, clear=False):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertEqual(yaw, 12.0)
        self.assertEqual(pitch, GAME_CAM_DEFAULT_INIT_PITCH_DEG)

    def test_explicit_pitch_overrides_default(self) -> None:
        with mock.patch.dict(os.environ, {**_BASE_CAM_ENV, "GAME_CAM_INIT_PITCH_DEG": "5"}, clear=False):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertIsNone(yaw)
        self.assertEqual(pitch, 5.0)

    def test_init_height_overrides_pitch_deg(self) -> None:
        """HEIGHT wins over explicit GAME_CAM_INIT_PITCH_DEG; default pivot 1.4, default dist 7."""
        with mock.patch.dict(
            os.environ,
            {**_BASE_CAM_ENV, "GAME_CAM_INIT_PITCH_DEG": "5", "GAME_CAM_INIT_HEIGHT": "1.9"},
            clear=False,
        ):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertIsNone(yaw)
        expected_rad = math.asin((1.9 - 1.4) / GAME_CAM_DEFAULT_INIT_DIST)
        self.assertAlmostEqual(pitch, math.degrees(expected_rad), places=5)

    def test_init_height_overrides_look_deg_pitch(self) -> None:
        with mock.patch.dict(
            os.environ,
            {**_BASE_CAM_ENV, "GAME_CAM_INIT_LOOK_DEG": "12,-10", "GAME_CAM_INIT_HEIGHT": "1.4"},
            clear=False,
        ):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertEqual(yaw, 12.0)
        self.assertAlmostEqual(pitch, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
