"""Smoke test for touch_game_bridge (needs poliastro / full venv)."""

from __future__ import annotations

import unittest


class TestTouchGameBridge(unittest.TestCase):
    def test_session_tick_smoke(self) -> None:
        import touch_game_bridge as tg

        tg.session_init(verbose_scenario=False)
        st = tg.status()
        self.assertTrue(st["ok"])
        self.assertEqual(st["step"], 1)

        out = tg.tick(0.016, 100.0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertTrue(out["ok"])
        self.assertIn("paused_help", out)
        self.assertIn("locale", out)
        self.assertIn(out["locale"], ("de", "en", "es"))
        self.assertIn("help_overlay_kind", out)
        self.assertEqual(out["help_overlay_kind"], "none")
        self.assertIn("target_guides", out)
        tgdict = out["target_guides"]
        self.assertIsInstance(tgdict, dict)
        self.assertIn("ox", tgdict)
        self.assertAlmostEqual(tgdict["ox"] ** 2 + tgdict["oy"] ** 2 + tgdict["oz"] ** 2, 1.0, places=5)
        self.assertEqual(int(tgdict["draw_laser"]), 0)
        self.assertIn("sun_x", out)
        self.assertIn("help_body", out)
        self.assertIn("help_story_plain", out)
        self.assertIn("catalog_lines", out)
        self.assertIn("panel_enabled", out)
        self.assertIn("panel_w", out)
        self.assertIn("panel_h", out)
        self.assertEqual(out["step"], 1)
        self.assertEqual(out["catalog_lines"], [])
        self.assertIn("chrome_ui", out)
        self.assertIn("quit", out["chrome_ui"])
        self.assertIn("chrome_disabled", out)
        self.assertIn("exposure_primary", out["chrome_disabled"])
        self.assertIsInstance(out["chrome_disabled"]["exposure_primary"], bool)

    def test_align_dome_to_telescope(self) -> None:
        import touch_game_bridge as tg

        tg.session_init(verbose_scenario=False)
        r = tg.align_dome_to_telescope()
        self.assertTrue(r.get("ok"))
        self.assertIn("dome_az_deg", r)

    def test_set_dome_az_deg(self) -> None:
        import touch_game_bridge as tg

        tg.session_init(verbose_scenario=False)
        r = tg.set_dome_az_deg(-37.25)
        self.assertTrue(r.get("ok"))
        self.assertAlmostEqual(float(r["dome_az_deg"]), -37.25, places=4)

    def test_set_locale_from_host(self) -> None:
        import touch_game_bridge as tg

        tg.session_init(verbose_scenario=False)
        r = tg.set_locale_from_host("en")
        self.assertTrue(r.get("ok"))
        self.assertEqual(r.get("locale"), "en")
        out = tg.tick(0.016, 100.0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.assertEqual(out.get("locale"), "en")
        tg.set_locale_from_host("de")


if __name__ == "__main__":
    unittest.main()
