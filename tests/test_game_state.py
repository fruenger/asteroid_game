"""Unit tests for Ursina-free game_state (Phase A)."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from game_state import (
    AsteroidGameState,
    FrameHints,
    KeysHeld,
    KeysInput,
    wrap_text,
)
from game_catalog_common import load_catalog_entries
from game_i18n import ensure_locale_env
from game_strings import toast_time_window_reject_text


class TestCatalogFile(unittest.TestCase):
    def test_load_five_field_line(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as wf:
                wf.write("2020-01-01 12:00:00, ObjA, TeamB, 1.500000, 2.250000\n")
            rows = load_catalog_entries(path)
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0][3], 1.5)
            self.assertAlmostEqual(rows[0][4], 2.25)
        finally:
            os.unlink(path)


class TestWrapText(unittest.TestCase):
    def test_wrap_respects_width(self) -> None:
        s = wrap_text("one two three four five", width=10)
        lines = s.split("\n")
        self.assertTrue(all(len(line) <= 10 for line in lines))


class TestPhaseAFlow(unittest.TestCase):
    def setUp(self) -> None:
        ensure_locale_env()
        self._env = mock.patch.dict(os.environ, {"GAME_START_BEFORE_WINDOW": "0"}, clear=False)
        self._env.start()
        self.rng = np.random.default_rng(42)
        self.obj_dir = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.gs = AsteroidGameState(
            min_time=0.1,
            max_time=0.9,
            object_dir_cartesian=self.obj_dir,
            rng=self.rng,
            imsize=32,
        )

    def tearDown(self) -> None:
        self._env.stop()

    def test_space_outside_window_rejected(self) -> None:
        self.gs.time_now = 0.05
        r = self.gs.handle_discrete_input("space", wall_t=5.0)
        self.assertIn("time_window_reject", r["events"])
        self.assertEqual(self.gs.step, 1)
        self.assertEqual(self.gs.toast_message, toast_time_window_reject_text())
        self.assertGreater(self.gs.toast_until_wall_t, 5.0)

    def test_toast_in_tick_after_early_space(self) -> None:
        self.gs.time_now = 0.05
        self.gs.handle_discrete_input("space", wall_t=5.0)
        out = self.gs.tick(dt=0.01, wall_t=5.5, keys=KeysInput(), hints=FrameHints())
        self.assertEqual(out.get("toast"), toast_time_window_reject_text())

    def test_space_inside_window_advances(self) -> None:
        self.gs.time_now = 0.5
        r = self.gs.handle_discrete_input("space", wall_t=50.0)
        self.assertIn("time_window_ok", r["events"])
        self.assertEqual(self.gs.step, 2)
        self.assertTrue(self.gs.time_stopped)

    def test_z_opens_observatory(self) -> None:
        self.gs.step = 2
        r = self.gs.handle_discrete_input("z", wall_t=0.0)
        self.assertIn("observatory_opening_started", r["events"])
        self.assertEqual(self.gs.step, 3)

    def test_telescope_aligns_after_integration(self) -> None:
        import rig_kinematics as rk

        self.gs.step = 3
        self.gs.observatory_open_t = 11.0
        self.gs.ra_deg = 12.0
        self.gs.dec_deg = 5.0
        u = np.asarray(rk.optical_axis_world_unit(self.gs.ra_deg, self.gs.dec_deg), dtype=np.float64).reshape(3)
        u = u / float(np.linalg.norm(u))
        self.gs.object_dir_cartesian = u
        self.gs.object_dir_visual_cartesian = u.copy()
        keys = KeysInput()
        out = self.gs.tick(dt=0.016, wall_t=100.0, keys=keys, hints=FrameHints())
        self.assertIn("telescope_aligned", out["events"])
        self.assertEqual(self.gs.step, 4)

    def test_deferred_align_only_via_apply(self) -> None:
        import rig_kinematics as rk

        self.gs.step = 3
        self.gs.observatory_open_t = 11.0
        self.gs.ra_deg = 12.0
        self.gs.dec_deg = 5.0
        u = np.asarray(rk.optical_axis_world_unit(self.gs.ra_deg, self.gs.dec_deg), dtype=np.float64).reshape(3)
        u = u / float(np.linalg.norm(u))
        self.gs.object_dir_cartesian = u.copy()
        self.gs.object_dir_visual_cartesian = u.copy()
        out = self.gs.tick(
            dt=0.016, wall_t=100.0, keys=KeysInput(), hints=FrameHints(defer_telescope_target_alignment=True)
        )
        self.assertNotIn("telescope_aligned", out["events"])
        self.assertEqual(self.gs.step, 3)
        evs = self.gs.apply_telescope_target_alignment(u)
        self.assertIn("telescope_aligned", evs)
        self.assertEqual(self.gs.step, 4)

    def test_dome_clear_advances_to_stage4_boot(self) -> None:
        self.gs.step = 4
        keys = KeysInput()
        hints = FrameHints(dome_ray_exit_distance=150.0)
        t0 = 200.0
        out = self.gs.tick(dt=0.01, wall_t=t0, keys=keys, hints=hints)
        self.assertIn("entered_stage4", out["events"])
        self.assertEqual(self.gs.step, 5)
        self.assertGreater(self.gs._stage4_exposure_unlock_wall_t, t0)
        self.assertTrue(self.gs.image_panel_enabled)

    def test_stage4_no_exposure_during_boot(self) -> None:
        self.gs.step = 5
        t0 = 300.0
        self.gs._stage4_exposure_unlock_wall_t = t0 + 5.0
        self.gs.image_array.fill(0.0)
        keys = KeysInput(held=KeysHeld(space=True))
        self.gs.tick(dt=1.0, wall_t=t0 + 1.0, keys=keys, hints=FrameHints())
        self.assertTrue(np.all(self.gs.image_array == 0))

    def test_three_exposures_complete(self) -> None:
        self.gs.step = 5
        self.gs._stage4_exposure_unlock_wall_t = 0.0
        self.gs.exposure_time = 0.0
        self.gs.all_images = []
        wall = 400.0
        self.gs.tick(dt=0.01, wall_t=wall, keys=KeysInput(held=KeysHeld(space=False)), hints=FrameHints())
        wall += 0.01
        self.gs.tick(dt=0.01, wall_t=wall, keys=KeysInput(held=KeysHeld(space=True)), hints=FrameHints())
        wall += 0.01
        while len(self.gs.all_images) < 3 and self.gs.step == 5:
            self.gs.tick(dt=0.5, wall_t=wall, keys=KeysInput(held=KeysHeld(space=True)), hints=FrameHints())
            wall += 0.5
        self.assertEqual(len(self.gs.all_images), 3)
        self.assertEqual(self.gs.step, 6)

    def test_help_pauses_tick(self) -> None:
        self.gs.step = 3
        self.gs.handle_discrete_input("h", wall_t=0.0)
        t_before = self.gs.time_now
        self.gs.tick(dt=0.1, wall_t=999.0, keys=KeysInput(), hints=FrameHints())
        self.assertEqual(self.gs.time_now, t_before)

    def test_step5_pick_near_target(self) -> None:
        self.gs.step = 6
        self.gs.all_images = [np.zeros((32, 32)), np.zeros((32, 32)), np.zeros((32, 32))]
        pix = (
            float(self.gs.image_locations[0, 1]),
            float(self.gs.image_locations[1, 1]),
        )
        hints = FrameHints(step5_pick_pixel_xy=pix)
        out = self.gs.tick(dt=0.02, wall_t=500.0, keys=KeysInput(), hints=hints)
        self.assertIn("asteroid_picked", out["events"])
        self.assertEqual(self.gs.step, 7)
        self.assertTrue(self.gs.image_panel_enabled)
        self.assertTrue(self.gs.step6_object_name.startswith("OST/"))

    def test_step6_catalog_keys(self) -> None:
        self.gs.step = 7
        self.gs._reset_step6_form()
        self.gs.step6_focus_idx = 0
        self.gs.step6_key_action("char", "X")
        self.assertIn("X", self.gs.step6_object_name)
        self.gs.step6_key_action("tab", "")
        self.assertEqual(self.gs.step6_focus_idx, 1)
        self.gs.step6_key_action("char", "Z")
        self.assertIn("Z", self.gs.step6_discoverer)
        self.gs.step6_key_action("backspace", "")
        self.assertFalse(self.gs.step6_discoverer.endswith("Z"))

    def test_step6_identified_marker_pixels(self) -> None:
        self.gs.step = 7
        row, col, rad = self.gs.step6_identified_marker_pixels()
        self.assertGreater(rad, 0.0)
        self.assertGreaterEqual(row, 0.0)
        self.assertGreaterEqual(col, 0.0)
        self.assertLess(row, float(self.gs.imsize))
        self.assertLess(col, float(self.gs.imsize))

    def test_step7_panel_marker_follows_animation_index(self) -> None:
        self.gs.step = 7
        self.gs.all_images = [np.zeros((32, 32)), np.zeros((32, 32)), np.zeros((32, 32))]
        for idx in range(3):
            self.gs.image_shown = idx
            row, col, rad = self.gs.step7_panel_marker_pixels()
            self.assertGreater(rad, 0.0)
            self.assertGreaterEqual(row, 0.0)
            self.assertGreaterEqual(col, 0.0)

    def test_step6_save_appends_catalog_line(self) -> None:
        self.gs.step = 7
        self.gs._reset_step6_form()
        self.gs.step6_object_name = "TestObj"
        self.gs.step6_discoverer = "TeamX"
        self.gs.step6_orbit_a_au = 2.5
        self.gs.step6_orbit_p_yr = 4.0
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        try:
            with mock.patch.dict(os.environ, {"GAME_USER_SAVES_PATH": path}):
                r = self.gs.step6_key_action("save", "")
            self.assertFalse(r.get("process_exit"))
            self.assertEqual(self.gs.step, 8)
            with open(path, encoding="utf-8") as rf:
                body = rf.read()
            self.assertIn("TestObj", body)
            self.assertIn("TeamX", body)
            self.assertIn("2.500000", body)
            self.assertIn("4.000000", body)
        finally:
            os.unlink(path)

    def test_step6_save_fills_discoverer_when_empty(self) -> None:
        self.gs.step = 7
        self.gs._reset_step6_form()
        self.gs.step6_object_name = "SoloObj"
        self.gs.step6_discoverer = ""
        self.gs.step6_orbit_a_au = 0.0
        self.gs.step6_orbit_p_yr = 0.0
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        try:
            with mock.patch.dict(os.environ, {"GAME_USER_SAVES_PATH": path}):
                self.gs.step6_key_action("save", "")
            self.assertEqual(self.gs.step, 8)
            self.assertTrue(self.gs.step6_discoverer.strip())
            with open(path, encoding="utf-8") as rf:
                body = rf.read()
            self.assertIn("SoloObj", body)
            self.assertIn(self.gs.step6_discoverer.strip(), body)
        finally:
            os.unlink(path)

    def test_step7_finish_exits(self) -> None:
        self.gs.step = 8
        r = self.gs.step7_key_action("finish")
        self.assertTrue(r.get("handled"))
        self.assertTrue(r.get("process_exit"))


if __name__ == "__main__":
    unittest.main()
