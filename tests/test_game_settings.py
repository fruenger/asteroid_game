"""Tests for ``game_settings`` camera helpers."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from game_cam_env import GAME_CAM_DEFAULT_INIT_PITCH_DEG, initial_camera_yaw_pitch_deg


class TestInitialCamera(unittest.TestCase):
    def test_default_pitch_when_no_camera_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GAME_CAM_INIT_LOOK_DEG": "",
                "GAME_CAM_INIT_YAW_DEG": "",
                "GAME_CAM_INIT_PITCH_DEG": "",
            },
            clear=False,
        ):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertIsNone(yaw)
        self.assertEqual(pitch, GAME_CAM_DEFAULT_INIT_PITCH_DEG)
        self.assertEqual(pitch, -30.0)

    def test_look_deg_one_number_sets_default_pitch(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GAME_CAM_INIT_LOOK_DEG": "12",
                "GAME_CAM_INIT_YAW_DEG": "",
                "GAME_CAM_INIT_PITCH_DEG": "",
            },
            clear=False,
        ):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertEqual(yaw, 12.0)
        self.assertEqual(pitch, -30.0)

    def test_explicit_pitch_overrides_default(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "GAME_CAM_INIT_LOOK_DEG": "",
                "GAME_CAM_INIT_YAW_DEG": "",
                "GAME_CAM_INIT_PITCH_DEG": "5",
            },
            clear=False,
        ):
            yaw, pitch = initial_camera_yaw_pitch_deg()
        self.assertIsNone(yaw)
        self.assertEqual(pitch, 5.0)


if __name__ == "__main__":
    unittest.main()
